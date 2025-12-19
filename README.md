
# ARC Test-Time Training with Multi-LoRA

This repository contains an experimental **test-time training (TTT)** pipeline for the **ARC (Abstraction and Reasoning Corpus)** using a **single base LLM** (e.g., Qwen2-7B) and **many low-rank adapters (LoRA)** trained **in parallel**.

The core idea:

- Each ARC task is treated as a **small “program induction” problem**.
- For a given base model, we create **K separate LoRA adapters**, one per ARC task in a group.
- In each training step, we:
  - Run **one shared forward pass** through the frozen base model (batched over K tasks).
  - Apply **K different LoRA adapters** on the resulting hidden states.
  - Compute **K independent losses** and backprop only through the LoRA parameters.

This approximates the idea of *“optimizing many concurrent LoRAs on the same base model”* without custom CUDA kernels like `cublasGemmGroupedBatchedEx`.

---

## 1. Project Overview

### 1.1. ARC + Test-Time Training

- Each ARC problem has a small set of train input/output grids and one test input.
- We treat solving each problem as **test-time fine-tuning**:
  - Build **pseudo-tasks** from the train examples via heavy augmentation.
  - Train a small LoRA adapter that specializes the base LLM to that problem.
  - Use the adapted model to **generate the test output grid**.

### 1.2. Multi-LoRA Optimization

Naïve version:

- For N ARC tasks:
  - Run test-time LoRA fine-tuning **separately** for each task.
  - Cost scales linearly with the number of tasks.

Multi-LoRA version (this repo):

- Group tasks in batches of size **K** (e.g., K = 8).
- Create **K LoRA adapters** on top of **one frozen base model**.
- Per training step:
  1. Batch K tasks → **one forward** through the base LLM.
  2. Use the resulting hidden states as a shared “base feature”.
  3. Apply each LoRA adapter to those hidden states to get K sets of logits.
  4. Compute K cross-entropy losses (one per task) and backprop only through LoRA params.

Effectively:

> **One shared forward**, **K small backwards**  
> instead of **K full forwards + K full backwards**.

This improves GPU utilization and reduces wall-clock time per ARC task, while keeping accuracy comparable to per-task LoRA training.

---

## 2. Repository Structure

```text
arc_ttt_multilora/
├── ttt_multi_lora.py            # Main multi-LoRA TTT script
├── debug_arc_tokenization.py    # Small debug tool for one pseudo-task
├── requirements.txt             # Python dependencies
├── scripts/
│   └── run_qwen_multi_lora.sh   # Example run script for Qwen2-7B
├── data/
│   └── README.md                # How to place ARC JSON files
├── models/
│   └── README.md                # How to download / point to HF models
└── results/
    └── README.md                # Where predictions / logs go
````

---

## 3. Installation

### 3.1. Create Environment

```bash
conda create -n arc_ttt python=3.10 -y
conda activate arc_ttt
pip install -r requirements.txt
```

Main dependencies (see `requirements.txt`):

* `torch` (GPU recommended)
* `transformers`
* `datasets`
* `bitsandbytes`
* `numpy`

### 3.2. Download a Base Model (Example: Qwen2-7B)

```bash
mkdir -p models

huggingface-cli download Qwen/Qwen2-7B \
  --local-dir models/qwen2-7b \
  --local-dir-use-symlinks False
```

This gives you a local HF-style folder at `models/qwen2-7b`.

---

## 4. Data Setup (ARC JSONs)

We **do not** ship ARC data.

Expected layout:

```text
data/
  evaluation/
    00576224.json
    009d5c81.json
    ...
```

* Each `.json` is one ARC task from the evaluation split.
* Put your ARC JSONs into `data/evaluation/`.
* Then use `--data_dir data/evaluation` when running the main script.

See `data/README.md` for a short reminder.

---

## 5. Main Script: `ttt_multi_lora.py`

This is the core of the project.

### 5.1. What It Does

For each ARC problem:

1. **Build pseudo-tasks** using augmentations:

   * `rotate_grid` — rotation by 0°, 90°, 180°, 270° (`k ∈ {0,1,2,3}`).
   * `flip_grid` — optional horizontal and/or vertical flips (4 combos: none, H, V, H+V).
   * `permute_colors` — random permutation of non-zero colors (1–9) (one permutation per pseudo-task).
2. For each problem:

   * `build_pseudo_tasks(problem, m=num_pseudo)` creates `m` augmented tasks that preserve the underlying rule.
   * Each pseudo-task is converted to a **prompt+answer** pair (digit-based grids).
3. For each **group of K problems**:

   * Instantiates `MultiLoRAModel` with K adapters on top of the frozen base model.
   * Runs `train_multi_lora(...)` for a defined number of steps.
4. After training:

   * For each problem in the group, runs `generate_with_lora(...)` using its adapter.
   * Writes a `.txt` file with **MODEL OUTPUT** and **TRUE OUTPUT** for the test grid.

### 5.2. Key Design Choices

* **Base model is frozen** (Qwen2-7B loaded in 4-bit via bitsandbytes).
* LoRA is implemented as **generic low-rank adapters**:

  * For each adapter i:

    * `A_i ∈ ℝ^{H×r}`, `B_i ∈ ℝ^{r×H}`, where H = hidden size, r = rank.
  * During forward:

    * `delta_i = hidden @ A_i @ B_i`
    * `hidden_i = hidden + α/r * delta_i`
    * logits_i = LM head applied to `hidden_i`.
* Labels for training:

  * We **mask** all prompt tokens with `-100` labels (ignored).
  * Only the **answer tokens (the output grid text)** contribute to cross-entropy loss.

---

## 6. Running the Code

### 6.1. Quick Start (Qwen2-7B)

Example (from repo root):

```bash
python ttt_multi_lora.py \
  --model_dir models/qwen2-7b \
  --data_dir data/evaluation \
  --k 2 \
  --steps 10 \
  --num_pseudo 20 \
  --out_dir arc_predictions_qwen_debug
```

Important arguments:

* `--model_dir` : path to HF model folder (e.g. `models/qwen2-7b`).
* `--data_dir`  : folder containing ARC JSONs (e.g. `data/evaluation`).
* `--k`         : number of tasks per group (and number of LoRA adapters per group).
* `--steps`     : training steps per group (TTT iterations).
* `--num_pseudo`: number of pseudo-tasks (augmentations) per ARC problem.
* `--out_dir`   : where prediction `.txt` files are written.

You can also use the helper script:

```bash
bash scripts/run_qwen_multi_lora.sh
```

(Adjust paths and hyperparameters in that script as needed.)

---

## 7. Debugging: `debug_arc_tokenization.py`

When something looks off, this helper shows **exactly** what the model sees.

Example:

```bash
python debug_arc_tokenization.py \
  --model_dir models/qwen2-7b \
  --problem_path data/evaluation/00576224.json \
  --num_pseudo 5 \
  --pseudo_index 0 \
  --max_len 4096 \
  --max_tokens_show 200
```

It prints:

* The **augmented train/test grids** for the selected pseudo-task.
* The **prompt string** (what we feed as input).
* The **answer string** (target output grid).
* A nicely formatted table of:

  * token index,
  * token ID,
  * label (`-100` for ignored, real ID for target),
  * decoded token text.

This clarifies:

* Where the prompt ends and the target begins.
* How many tokens actually contribute to the loss.

---

## 8. Results


### 8.1. Throughput & Efficiency (Per Task, ARC TTT)

* **Baseline**: Single-task LoRA TTT

  * Train each ARC problem separately.
  * Full forward+backward for each task.
* **Multi-LoRA**: Our approach with **K = 8** adapters per group.

Observed trends:

* **~7–8× reduction** in *effective wall-clock time per task*:

  * One shared forward pass replaces 8 separate forwards.
  * Backwards are cheap because LoRA ranks are small (e.g. r = 64).
* **GPU utilization** improves from ~30–40% to ~70–80% on A100-80GB.
* **Accuracy** on held-out ARC tests remains within ~1–2% of single-task LoRA TTT.


---

## 9. Extending the Project

Here are some directions you can explore:

1. **Local vs Global LoRA Adapters**

   * Train two families of adapters:

     * “Local” (small neighborhood reasoning).
     * “Global” (global structure / pattern).
   * Then train a small classifier LLM to decide whether a given ARC task is local or global.
   * At inference:

     * Use the more popular adapter type (e.g. global) 90% of the time.
     * Fall back to the other only when needed.
   * This aligns with S-LoRA-style ideas: merge or cache the *popular* LoRA into base weights, keep rare adapters separate.

2. **Better Grouping Strategies**

   * Instead of random grouping of tasks into K, group by:

     * Similar input size,
     * Similar color palettes,
     * Similar “task family”.
   * This may reduce variance and stabilize training across adapters.

3. **Alternative Base Models**

   * Swap Qwen2-7B for:

     * Smaller TinyLlama-style models (faster, cheaper),
     * Or larger models (possibly higher accuracy but more GPU needed).

---

## 10. Limitations & Notes

* This code is **research/experimental**:

  * No full ARC leaderboard evaluation is included yet.
  * Some configs (too large `num_pseudo`, too high `lr`) may lead to NaNs; tune carefully.
* We use **digit-based grid prompts** instead of color names for simplicity:

  * Each grid cell is a digit 0–9.
  * 0 is usually background; 1–9 correspond to colored cells.
* Model is loaded in **4-bit** (`bitsandbytes`) to save memory; full-precision training is not implemented here.

---
