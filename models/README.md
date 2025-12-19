````markdown
# Models for ARC TTT + Multi-LoRA

This folder stores the **base LLM checkpoints** used for test-time training (TTT)
with Multi-LoRA.

The main script, `ttt_multi_lora.py`, never downloads models by itself – it just
expects a **local directory** passed via `--model_dir`, e.g.:

```bash
python ttt_multi_lora.py \
  --model_dir models/qwen2-7b \
  --data_dir data/evaluation \
  ...
````


---

## 1. Expected Directory Structure

From the repository root:

```text
arc_ttt_multilora/
├── ttt_multi_lora.py
├── debug_arc_tokenization.py
├── requirements.txt
├── models/
│   ├── README.md          # (this file)
│   ├── qwen2-7b/          # example: Qwen 7B checkpoint
│   └── tinyllama/         # optional: TinyLlama 1.1B checkpoint
└── data/
    └── evaluation/
        ├── 00576224.json
        ├── ...
```

You can choose any folder names you like (`qwen2-7b`, `tinyllama`, etc.); just
make sure you pass the same path to `--model_dir`.

---

## 2. Recommended Base Model: Qwen2-7B (4-bit)

In our experiments, we use a **Qwen2-7B–style model** in **4-bit quantization**
via `bitsandbytes`. The code assumes:

* A standard Hugging Face Transformers layout (what you get from `from_pretrained`).
* Causal LM architecture (decoder-only, `AutoModelForCausalLM`).
* Context length ~4096 tokens (we usually set `max_len=4096` in debug scripts).

### 2.1. Downloading the Model (Hugging Face)

On a machine with internet access:

```bash
# Example: download a Qwen 7B model snapshot to a local folder
huggingface-cli download Qwen/Qwen2-7B-Instruct \
  --local-dir qwen2-7b \
  --local-dir-use-symlinks False
```

Then copy the resulting `qwen2-7b/` folder into:

```text
arc_ttt_multilora/models/qwen2-7b/
```

Inside `qwen2-7b/` you should see files like:

```text
config.json
generation_config.json
model.safetensors   (or pytorch_model-00001-of-00002.bin, etc.)
tokenizer.json
tokenizer.model
special_tokens_map.json
tokenizer_config.json
...
```

### 2.2. Using Qwen2-7B in the Code

In all our scripts, we simply point `--model_dir` to this folder:

```bash
python ttt_multi_lora.py \
  --model_dir models/qwen2-7b \
  --data_dir data/evaluation \
  --k 2 \
  --steps 10 \
  --num_pseudo 20 \
  --out_dir arc_predictions_qwen_debug
```

Under the hood, the loader does:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_has_fp16_weight=False,
    ),
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
```

---

## 3. Optional: TinyLlama 1.1B

You can also use a **small baseline model** such as TinyLlama 1.1B, for quick
debugging or smaller GPUs.

Example layout:

```text
models/
  ├── qwen2-7b/
  └── tinyllama/
      ├── config.json
      ├── pytorch_model.bin (or safetensors)
      ├── tokenizer.json
      ├── ...
```

Then run:

```bash
python ttt_multi_lora.py \
  --model_dir models/tinyllama \
  --data_dir data/evaluation \
  --k 2 \
  --steps 10 \
  --num_pseudo 20 \
  --out_dir arc_predictions_tinyllama_debug
```

Note: TinyLlama typically has a shorter max context length (e.g. 2048), so the
scripts clamp sequences accordingly (via `max_len=2048`).

---

## 4. GPU / Quantization Notes

* We use **4-bit quantization** via `bitsandbytes` to make Qwen2-7B fit on a
  single A100/RTX-class GPU.
* The LoRA / low-rank adapters are stored in regular FP16/FP32 PyTorch tensors,
  while the base model stays quantized.
* Make sure:

  * `bitsandbytes` is installed (see `requirements.txt`),
  * and your GPU/driver supports it.

If you see errors like:

```text
RuntimeError: expected mat1 and mat2 to have the same dtype...
```

it usually means the adapter weights (`A`, `B`) were created in a different
dtype than the model’s hidden states. In this repo we explicitly align the
adapter dtype with `base.lm_head.weight.dtype` (typically `torch.float16`).

---

## 5. Changing the Base Model

You can switch to any other compatible HF model by:

1. Downloading it into a new subfolder under `models/`
   (e.g., `models/my-custom-llm/`).
2. Passing that folder to `--model_dir` when running the scripts.
3. Making sure:

   * It is a causal LM and works with `AutoModelForCausalLM`.
   * It has a reasonable context length (`config.max_position_embeddings`).
   * You adjust `max_len` in `tokenize_example` / debug scripts if needed.

Example:

```bash
python ttt_multi_lora.py \
  --model_dir models/my-custom-llm \
  --data_dir data/evaluation \
  ...
```

---

In short:

* Put your **HF-style model folder** under `models/`.
* Point `--model_dir` to that folder.
* The scripts will handle 4-bit loading, tokenization, and Multi-LoRA training
  on ARC problems.

```

::contentReference[oaicite:0]{index=0}
```

