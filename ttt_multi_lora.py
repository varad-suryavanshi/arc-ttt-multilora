#!/usr/bin/env python3
import os
import json
import random
import argparse
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch.nn.functional as F
import numpy as np
import math

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MAX_CONTEXT = 8192  # for Qwen2-7B


# ============================================================
# ARC UTILITY FUNCTIONS  (same as your original script)
# ============================================================

def rotate_grid(grid, k):
    return np.rot90(np.array(grid), k).tolist()

def flip_grid(grid, axis):
    return np.flip(np.array(grid), axis=axis).tolist()

def permute_colors(grid):
    arr = np.array(grid)
    colors = np.unique(arr)
    non_black = [c for c in colors if c != 0]
    permuted = np.random.permutation(non_black)
    perm = {0: 0}
    perm.update({orig: new for orig, new in zip(non_black, permuted)})
    return np.vectorize(lambda v: perm[v])(arr).tolist()

def sample_augment_params():
    k = random.choice([0, 1, 2, 3])
    flip_h = random.random() < 0.5
    flip_v = random.random() < 0.5
    non_black = list(range(1, 10))
    permuted = np.random.permutation(non_black)
    color_mapping = {0: 0}
    color_mapping.update({orig: int(new) for orig, new in zip(non_black, permuted)})
    return k, flip_h, flip_v, color_mapping

def apply_augment(grid, params):
    k, flip_h, flip_v, color_mapping = params
    arr = np.array(grid)
    arr = np.rot90(arr, k)
    if flip_h:
        arr = np.flip(arr, axis=0)
    if flip_v:
        arr = np.flip(arr, axis=1)
    vec_map = np.vectorize(lambda v: color_mapping.get(int(v), int(v)))
    arr = vec_map(arr)
    return arr.tolist()

def build_pseudo_tasks(problem, m=10):
    x_tr = [ex['input']  for ex in problem['train']]
    y_tr = [ex['output'] for ex in problem['train']]
    N = len(x_tr)
    tasks = []
    for i in range(N):
        x_test_f, y_test_f = x_tr[i], y_tr[i]
        x_tr_p = x_tr[:i] + x_tr[i+1:]
        y_tr_p = y_tr[:i] + y_tr[i+1:]
        for _ in range(m):
            params = sample_augment_params()
            aug_x_tr = [apply_augment(x, params) for x in x_tr_p]
            aug_y_tr = [apply_augment(y, params) for y in y_tr_p]
            ax_test  = apply_augment(x_test_f, params)
            ay_test  = apply_augment(y_test_f, params)
            tasks.append({
                'train': [{'input': xx, 'output': yy}
                          for xx, yy in zip(aug_x_tr, aug_y_tr)],
                'test':  [{'input': ax_test, 'output': ay_test}]
            })
    return tasks


# ============================================================
# PROMPT BUILDING  (unchanged from your original script)
# ============================================================

color_map = {
    0:"Black",1:"Blue",2:"Red",3:"Green",4:"Yellow",
    5:"Grey",6:"Fuchsia",7:"Orange",8:"Teal",9:"Brown"
}

# def grid_to_str(grid):
#     return "\n".join(" ".join(color_map[c] for c in row) for row in grid)

def grid_to_str(grid):
    """
    Convert a 2D integer grid (0–9) into a compact digit string:
    each row is a line, cells are separated by spaces.

    Example:
      [[1, 0, 3],
       [4, 4, 9]]

    becomes:
      "1 0 3\n4 4 9"
    """
    return "\n".join(
        " ".join(str(int(c)) for c in row)
        for row in grid
    )





def make_prompt_and_answer(task):
    """
    Build a short, digit-based ARC prompt.

    Format:

    You solve ARC-like puzzles. Grids are digits 0–9.
    Train examples:
    Example 1 input:
    <grid>
    Example 1 output:
    <grid>
    ...
    Test input:
    <grid>
    Test output:
    <--- model should complete with the output grid only
    """

    lines = []
    lines.append("You solve ARC-like puzzles.")
    lines.append("Grids are 2D arrays of digits 0-9.")
    lines.append("Cells are separated by spaces, rows by newlines.")
    lines.append("You are given train input/output pairs and one test input.")
    lines.append("Infer the rule and output the correct test output grid.")
    lines.append("Train examples:")

    # Train examples
    for i, ex in enumerate(task["train"], 1):
        lines.append(f"Example {i} input:")
        lines.append(grid_to_str(ex["input"]))
        lines.append(f"Example {i} output:")
        lines.append(grid_to_str(ex["output"]))

    # Test example
    ti = task["test"][0]["input"]
    lines.append("Test input:")
    lines.append(grid_to_str(ti))
    lines.append("Test output:")

    # Everything above is the "prompt"
    prompt = "\n".join(lines)

    # The "answer" is ONLY the test output grid as digits
    to = task["test"][0]["output"]
    answer = "\n" + grid_to_str(to)

    return prompt, answer



def tokenize_example(ex, tokenizer, max_len=MAX_CONTEXT):
    prompt, answer = make_prompt_and_answer(ex)
    enc = tokenizer(prompt + answer, truncation=True, padding=False, max_length=max_len)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    labels = enc["input_ids"].copy()
    labels[:len(p_ids)] = [-100] * len(p_ids)
    enc["labels"] = labels
    return enc


# ============================================================
# MULTI-LORA MODEL  
# ============================================================

class MultiLoRAModel(nn.Module):

    def __init__(self, base_model, tokenizer, k=8, r=64, alpha=64):
        super().__init__()
        self.k = k
        self.tokenizer = tokenizer
        self.base = base_model

        # freeze base model
        for p in self.base.parameters():
            p.requires_grad = False

        hidden_size = self.base.config.hidden_size
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 🔥 IMPORTANT: use fp32 for adapters (more stable than fp16)
        self.adapter_dtype = torch.float32

        self.A = nn.ParameterList()
        self.B = nn.ParameterList()
        for _ in range(k):
            A = nn.Parameter(torch.empty(hidden_size, r, dtype=self.adapter_dtype))
            B = nn.Parameter(torch.empty(r, hidden_size, dtype=self.adapter_dtype))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.zeros_(B)
            # shrink initial weights so delta is tiny at start
            A.data *= 0.05
            self.A.append(A)
            self.B.append(B)

        print(
            f"Initialized MultiLoRAModel with {k} adapters, "
            f"hidden_size={hidden_size}, r={r}"
        )

    def forward(self, input_ids_list, attention_masks_list):
        """
        input_ids_list: list of K tensors [1, T]
        attention_masks_list: list of K tensors [1, T]
        We only need one base forward; use the first.
        """
        base_out = self.base.model(
            input_ids=input_ids_list[0],
            attention_mask=attention_masks_list[0],
            output_hidden_states=True,
            return_dict=True
        )
        hidden = base_out.hidden_states[-1]  # [1, T, H], dtype usually float16

        # upcast to fp32 for stable adapter math
        hidden_fp32 = hidden.to(self.adapter_dtype)  # [1, T, H]

        logits_list = []
        for i in range(self.k):
            A = self.A[i]  # [H, r] (fp32)
            B = self.B[i]  # [r, H] (fp32)

            # [1, T, H] -> [1, T, r] -> [1, T, H] in fp32
            delta_fp32 = hidden_fp32 @ A @ B  # [1, T, H]

            # clip delta to avoid huge explosions
            delta_fp32 = torch.clamp(delta_fp32, -5.0, 5.0)

            # apply scaling and cast back to base dtype (fp16)
            hidden_i = hidden + self.scaling * delta_fp32.to(hidden.dtype)  # [1, T, H]

            # logits in base model dtype
            logits = self.base.lm_head(hidden_i)  # [1, T, V]
            logits_list.append(logits)

        return logits_list





def pad_to_max_length(tensors, pad_id, add_batch_dim=False, max_len=MAX_CONTEXT):
    """
    Pad a list of 1D tensors [T] to same length, with an optional batch dim.
    Also hard-clamps all sequences to <= max_len.
    """
    processed = []
    for t in tensors:
        if t.size(0) > max_len:
            t = t[:max_len]
        processed.append(t)

    max_seq_len = max(t.size(0) for t in processed)
    padded = []
    masks = []

    for t in processed:
        pad_len = max_seq_len - t.size(0)
        padded_t = torch.cat(
            [t, torch.full((pad_len,), pad_id, dtype=torch.long)],
            dim=0
        )
        mask = torch.cat(
            [torch.ones_like(t), torch.zeros(pad_len, dtype=torch.long)],
            dim=0
        )

        if add_batch_dim:
            padded_t = padded_t.unsqueeze(0)  # [1, T]
            mask = mask.unsqueeze(0)          # [1, T]

        padded.append(padded_t)
        masks.append(mask)

    return padded, masks




def build_tokenized_pseudo_tasks(problem, tokenizer, num_pseudo=200):
    """
    Build (input_ids_list, labels_list) for all pseudo-tasks of a single ARC problem.
    """
    pseudo = build_pseudo_tasks(problem, m=num_pseudo)
    all_ids = []
    all_labels = []

    for ex in pseudo:
        enc = tokenize_example(ex, tokenizer)
        ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        labs = torch.tensor(enc["labels"], dtype=torch.long)
        all_ids.append(ids)
        all_labels.append(labs)

    return all_ids, all_labels


def sample_batch_for_lora(input_ids_list, labels_list):
    """Sample a random pseudo-task for a single LoRA adapter."""
    idx = random.randint(0, len(input_ids_list) - 1)
    return input_ids_list[idx], labels_list[idx]


def make_multi_lora_batch(problems, tokenizer, num_pseudo):
    """
    Construct a batch for K LoRAs from K problems.
    Returns lists:
        padded_ids:       [K x T]
        attention_masks:  [K x T]
        padded_labels:    [K x T]
    """
    K = len(problems)
    raw_ids = []
    raw_labels = []

    for problem in problems:
        ids_list, labels_list = build_tokenized_pseudo_tasks(problem, tokenizer, num_pseudo)
        ids, labs = sample_batch_for_lora(ids_list, labels_list)
        # --- TRUNCATE to TinyLlama max context ---
        max_len = MAX_CONTEXT
        ids = ids[:max_len]
        labs = labs[:max_len]
        raw_ids.append(ids)
        raw_labels.append(labs)

    # pad_id = tokenizer.pad_token_id

    # padded_ids, masks = pad_to_max_length(raw_ids, pad_id)
    # padded_labels, _ = pad_to_max_length(raw_labels, -100)

    # return padded_ids, masks, padded_labels
    pad_id = tokenizer.pad_token_id

    # inputs & attention masks: [1, T]
    padded_ids, masks = pad_to_max_length(raw_ids, pad_id, add_batch_dim=True)
    # labels: [T] (no batch dim)
    padded_labels, _ = pad_to_max_length(raw_labels, -100, add_batch_dim=False)

    return padded_ids, masks, padded_labels



# ============================================================
# TRAINING LOOP
# ============================================================


def get_lora_parameters(model):
    """Return trainable low-rank adapter parameters (A & B)."""
    params = []
    for p in model.A:
        params.append(p)
    for p in model.B:
        params.append(p)
    return params

def train_multi_lora(model, tokenizer, problems, num_steps=50,
                     lr=1e-4, num_pseudo=200, device="cuda"):
    """
    Train K LoRAs in parallel using K ARC problems.
    """
    model.train()
    model.to(device)

    # Optimizer on LoRA weights only
    lora_params = get_lora_parameters(model)
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    print(f"[TRAIN] Multi-LoRA group: {model.k} adapters, {num_steps} steps")
    print(f"[TRAIN] Total LoRA params: {sum(p.numel() for p in lora_params)}")

    for step in range(num_steps):

        # ----------------------------
        # 1. Build multi-LoRA batch
        # ----------------------------
        ids_list, masks_list, labels_list = make_multi_lora_batch(
            problems, tokenizer, num_pseudo
        )

        ids_list = [x.to(device) for x in ids_list]
        masks_list = [x.to(device) for x in masks_list]
        labels_list = [x.to(device) for x in labels_list]

        # ----------------------------
        # 2. forward
        # ----------------------------
        logits_list = model(ids_list, masks_list)

        # 🔥 NEW: sanitize logits to avoid NaNs / Infs / giant values
        cleaned_logits_list = []
        for i, logits in enumerate(logits_list):
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"[WARN] NaNs/Infs in logits for adapter {i} at step {step+1}, clamping.")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            logits = torch.clamp(logits, -50.0, 50.0)  # keep logits in a sane range
            cleaned_logits_list.append(logits)
        logits_list = cleaned_logits_list


        # ----------------------------
        # 3. Multi-loss
        # ----------------------------
        # 3. Multi-loss
        total_loss = 0.0
        valid_losses = 0
        for i in range(model.k):
            logits = logits_list[i]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_list[i][..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] loss for adapter {i} is {loss} at step {step+1}, skipping this adapter in total_loss.")
                continue

            total_loss = total_loss + loss
            valid_losses += 1

        if valid_losses == 0:
            print(f"[WARN] All adapter losses NaN/Inf at step {step+1}, skipping update.")
            optimizer.zero_grad()
            continue

        total_loss = total_loss / valid_losses  # optional: average over valid adapters

        # 4. backward + update
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[WARN] total_loss is {total_loss} at step {step+1}, skipping update.")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()

        if (step + 1) % 5 == 0:
            print(f"[Step {step+1}/{num_steps}] Loss = {total_loss.item():.4f}")


    print("[TRAIN] Finished multi-LoRA training batch.")
    return model




@torch.no_grad()
def generate_with_lora(model, tokenizer, problem, adapter_idx,
                       max_new_tokens=256, device="cuda"):
    """
    Perform full autoregressive generation for test input,
    applying the chosen low-rank adapter at every step.

    Returns ONLY the model's continuation after the prompt
    (i.e., the predicted output grid text).
    """

    model.eval()
    model.to(device)

    # ----- Build real task prompt -----
    real_task = {
        "train": problem["train"],
        "test":  [problem["test"][0]]
    }
    prompt, _ = make_prompt_and_answer(real_task)

    # Reserve room so total length <= 2048
    max_input_len = MAX_CONTEXT - max_new_tokens
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len
    ).to(device)

    input_ids = inputs["input_ids"]     # [1, T]
    attn_mask = inputs["attention_mask"]

    unk_id = tokenizer.unk_token_id

    for _ in range(max_new_tokens):

        # 1. Base forward
        base_out = model.base.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden = base_out.hidden_states[-1]  # [1, T, H], usually fp16

        #  upcast to adapter dtype (fp32), apply A/B, clamp, then back to hidden.dtype
        hidden_fp32 = hidden.to(model.adapter_dtype)  # [1, T, H]
        A = model.A[adapter_idx]                      # [H, r] (fp32)
        B = model.B[adapter_idx]                      # [r, H] (fp32)

        delta_fp32 = hidden_fp32 @ A @ B              # [1, T, H]
        delta_fp32 = torch.clamp(delta_fp32, -5.0, 5.0)
        hidden_i = hidden + model.scaling * delta_fp32.to(hidden.dtype)  # [1, T, H]

        # 3. Next-token logits
        logits = model.base.lm_head(hidden_i)  # [1, T, V]

        # forbid <unk> to avoid collapsing into <unk><unk>...
        if unk_id is not None:
            logits[:, :, unk_id] = -1e9

        # Greedy: pick argmax at last position
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [1, 1]

        # 4. Append token
        input_ids = torch.cat([input_ids, next_token], dim=1)
        new_mask = torch.ones_like(next_token)
        attn_mask = torch.cat([attn_mask, new_mask], dim=1)

        # 5. Stop if EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    # ----- Decode ONLY the continuation after the prompt -----
    prompt_len = inputs["input_ids"].size(1)
    suffix_ids = input_ids[0][prompt_len:]  # tokens after prompt

    completion = tokenizer.decode(
        suffix_ids,
        skip_special_tokens=True
    ).strip()

    generated_len = suffix_ids.size(0)
    print(f"[DEBUG] Adapter {adapter_idx}: generated {generated_len} new tokens, text_len={len(completion)}")

    if completion == "":
        completion = "[EMPTY_COMPLETION]"

    return completion




# ============================================================
# LOAD MODEL (TinyLlama 4-bit)
# ============================================================

# def load_tinyllama_4bit(model_dir):
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#         llm_int8_has_fp16_weight=False
#     )

#     print(f"[LOAD] Loading TinyLlama from: {model_dir}")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_dir,
#         quantization_config=bnb_config,
#         device_map="auto",
#         trust_remote_code=True
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_dir,
#         trust_remote_code=True,
#         use_fast=False
#     )
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     return model, tokenizer


def load_qwen_7b_4bit(model_dir):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_has_fp16_weight=False,
    )

    print(f"[LOAD] Loading Qwen 7B from: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_fast=False,
    )

    # Make sure pad/eos/unk are usable
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer



# ============================================================
# GROUP ARC PROBLEMS INTO CHUNKS OF K=8
# ============================================================

def chunk_list(lst, chunk_size):
    """Split a list into equally sized chunks."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to TinyLlama model folder")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Folder with ARC .json files")
    parser.add_argument("--k", type=int, default=8,
                        help="Number of LoRA adapters (and problems per group)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs per group")
    parser.add_argument("--steps", type=int, default=50,
                        help="Training steps per LoRA group")
    parser.add_argument("--num_pseudo", type=int, default=200,
                        help="Pseudo-task augmentations per problem")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="arc_predictions",
                        help="Where to save predicted outputs")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------------------------------------
    # Load TinyLlama 4-bit
    # ---------------------------------------------
    # base_model, tokenizer = load_tinyllama_4bit(args.model_dir)
    base_model, tokenizer = load_qwen_7b_4bit(args.model_dir)


    # ---------------------------------------------
    # Load ARC evaluation problems
    # ---------------------------------------------
    all_files = sorted(os.listdir(args.data_dir))
    problems = []
    problem_names = []

    for fname in all_files:
        if fname.endswith(".json"):
            path = os.path.join(args.data_dir, fname)
            problem = json.load(open(path))
            problems.append(problem)
            problem_names.append(fname)

    print(f"[DATA] Loaded {len(problems)} ARC problems.")

    # ---------------------------------------------
    # Group problems into batches of K=8
    # ---------------------------------------------
    groups = chunk_list(list(zip(problem_names, problems)), args.k)
    print(f"[RUN] Total groups to process: {len(groups)}")

    # ---------------------------------------------
    # Process each group
    # ---------------------------------------------
    for group_idx, group in enumerate(groups):

        print(f"\n=============================")
        print(f"GROUP {group_idx+1}/{len(groups)}: {len(group)} problems")
        print(f"=============================\n")

        # Unzip names & problems
        names, prob_batch = zip(*group)

        # -----------------------------------------
        # Create Multi-LoRA model for this group
        # -----------------------------------------
        model = MultiLoRAModel(
            base_model=base_model,
            tokenizer=tokenizer,
            k=len(prob_batch),
            r=64,
            alpha=64
        )

        # -----------------------------------------
        # Train Multi-LoRA adapters
        # -----------------------------------------
        for epoch in range(args.epochs):
            print(f"\n[EPOCH {epoch+1}/{args.epochs}]")
            train_multi_lora(
                model=model,
                tokenizer=tokenizer,
                problems=prob_batch,
                num_steps=args.steps,
                lr=args.lr,
                num_pseudo=args.num_pseudo,
                device="cuda"
            )

        # -----------------------------------------
        # Inference for each problem in this group
        # -----------------------------------------
        for adapter_idx, (fname, problem) in enumerate(group):

            print(f"[INFER] Problem: {fname} (adapter {adapter_idx})")

            completion = generate_with_lora(
                model=model,
                tokenizer=tokenizer,
                problem=problem,
                adapter_idx=adapter_idx,
                max_new_tokens=args.max_new_tokens,
                device="cuda"
            )

            out_path = os.path.join(args.out_dir, fname.replace(".json", ".txt"))
            true_grid = grid_to_str(problem["test"][0]["output"])

            with open(out_path, "w") as f:
                f.write("MODEL OUTPUT:\n")
                f.write(completion)
                f.write("\n\nTRUE OUTPUT:\n")
                f.write(true_grid)

            print(f"[SAVE] Wrote prediction to {out_path}")

    print("\n DONE — All ARC problems processed.")


if __name__ == "__main__":
    main()
