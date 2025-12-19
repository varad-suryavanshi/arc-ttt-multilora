#!/usr/bin/env python3
import argparse
import json

from transformers import AutoTokenizer

# Reuse augmentation + prompt + tokenization from main script
from ttt_multi_lora import (
    build_pseudo_tasks,
    make_prompt_and_answer,
    tokenize_example,
)


def grid_to_str_digits(grid):
    """Pretty-print a 2D grid of ints."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to tokenizer / model folder (HF format).",
    )
    parser.add_argument(
        "--problem_path",
        type=str,
        required=True,
        help="Path to a single ARC JSON problem.",
    )
    parser.add_argument(
        "--num_pseudo",
        type=int,
        default=5,
        help="How many pseudo-tasks to generate for this problem.",
    )
    parser.add_argument(
        "--pseudo_index",
        type=int,
        default=0,
        help="Which pseudo-task index to inspect (0-based).",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=2048,
        help="Max sequence length for tokenization.",
    )
    parser.add_argument(
        "--max_tokens_show",
        type=int,
        default=200,
        help="How many tokens to print in the table.",
    )
    args = parser.parse_args()

    print(f"[LOAD] Loading tokenizer from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[DATA] Loading ARC problem from: {args.problem_path}")
    with open(args.problem_path, "r") as f:
        problem = json.load(f)

    print(f"[AUG] Building {args.num_pseudo} pseudo-tasks for this problem...")
    pseudo_tasks = build_pseudo_tasks(problem, m=args.num_pseudo)

    idx = args.pseudo_index
    if idx < 0 or idx >= len(pseudo_tasks):
        raise ValueError(
            f"pseudo_index {idx} out of range (0..{len(pseudo_tasks)-1})"
        )

    task = pseudo_tasks[idx]
    print(f"[AUG] Inspecting pseudo-task index {idx}\n")

    # ---- Show augmented grids ----
    print("===== AUGMENTED TRAIN EXAMPLES =====\n")
    for i, ex in enumerate(task["train"], 1):
        print(f"--- Train example {i} INPUT ---")
        print(grid_to_str_digits(ex["input"]))
        print(f"--- Train example {i} OUTPUT ---")
        print(grid_to_str_digits(ex["output"]))
        print()

    print("===== AUGMENTED TEST EXAMPLE =====")
    print("--- Test INPUT ---")
    print(grid_to_str_digits(task["test"][0]["input"]))
    print("--- Test OUTPUT (GROUND TRUTH) ---")
    print(grid_to_str_digits(task["test"][0]["output"]))
    print()

    # ---- Prompt & answer ----
    prompt, answer = make_prompt_and_answer(task)
    print("===== PROMPT (WHAT MODEL READS AS INPUT) =====")
    print(prompt)
    print("\n===== ANSWER (GROUND TRUTH GRID TEXT) =====")
    print(answer)
    print()

    # ---- Tokenization & labels ----
    enc = tokenize_example(task, tokenizer, max_len=args.max_len)
    input_ids = enc["input_ids"]
    labels = enc["labels"]

    total_tokens = len(input_ids)
    num_ignored = sum(1 for x in labels if x == -100)
    num_targets = total_tokens - num_ignored

    print("===== TOKEN / LABEL SUMMARY =====")
    print(f"Total tokens:      {total_tokens}")
    print(f"Ignored (-100):    {num_ignored}")
    print(f"Target tokens:     {num_targets}")
    print()

    to_show = min(args.max_tokens_show, total_tokens)
    print(
        f"===== FIRST {to_show} TOKENS (idx, id, label, flag, token text) ====="
    )
    for i in range(to_show):
        tok_id = input_ids[i]
        lab = labels[i]
        text = tokenizer.decode([tok_id])
        flag = "IGN" if lab == -100 else "TGT"
        print(f"{i:4d} | id={tok_id:5d} | label={lab:5d} | {flag:3s} | {text!r}")

    print("\n✅ Debug done. Now you can see:")
    print("  • the augmented grids")
    print("  • the prompt and answer strings")
    print("  • which tokens are ignored (-100) vs used as training targets (TGT)")


if __name__ == "__main__":
    main()

