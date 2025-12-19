````markdown
# Data Layout for ARC TTT + Multi-LoRA

This folder contains the **ARC (Abstraction and Reasoning Corpus)** tasks used for
test-time training (TTT) with Multi-LoRA.

The code in `ttt_multi_lora.py` expects a directory of `.json` files, where each file
is **one ARC problem** with the usual structure:

- `train`: list of input/output grid pairs
- `test`: list of test input grids (and sometimes outputs, if you have them)

We do **not** redistribute the ARC dataset here. You must download it yourself
from an official source and place the files in the structure described below.

---

## 1. Expected Directory Structure

From the repository root:

```text
arc_ttt_multilora/
├── ttt_multi_lora.py
├── debug_arc_tokenization.py
├── ...
└── data/
    ├── README.md        # (this file)
    └── evaluation/
        ├── 00576224.json
        ├── 009d5c81.json
        ├── 00dbd492.json
        ├── ...
````

* `data/evaluation/`
  Contains the ARC **evaluation** split tasks, each as a separate `.json` file.

You can also add other subfolders if you like:

```text
data/
  ├── evaluation/        # main split used in ttt_multi_lora.py
  ├── training/          # (optional) ARC training split
  └── debug/             # (optional) hand-picked tasks for debugging
```

But by default, the main script assumes:

```bash
--data_dir data/evaluation
```

---

## 2. How to Obtain ARC JSON Files

1. Go to an official ARC / ARC-AGI source (e.g., the original ARC dataset
   or Kaggle mirrors).
2. Download the JSON files for the **evaluation** split.
3. Place all `.json` files into:

```bash
arc_ttt_multilora/data/evaluation/
```

After that, you should see something like:

```bash
ls data/evaluation
00576224.json  009d5c81.json  00dbd492.json  03560426.json  ...
```

---

## 3. Using the Data with the Code

Example command from the repo root:

```bash
python ttt_multi_lora.py \
  --model_dir models/qwen2-7b \
  --data_dir data/evaluation \
  --k 2 \
  --steps 10 \
  --num_pseudo 20 \
  --out_dir arc_predictions_qwen_debug
```

* `--data_dir` should point to the folder **containing** the `.json` files
  (e.g., `data/evaluation`).
* The script will automatically:

  * List all `.json` files in that folder.
  * Load them as ARC problems.
  * Group them into batches of size `k` for Multi-LoRA training.

If you get an error like:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'data/evaluation'
```

then:

* Ensure you are running the command from the **repo root**, and
* That `data/evaluation/` exists and contains `.json` files.

---

## 4. Debugging a Single Problem

You can inspect tokenization and augmentation for a single problem with:

```bash
python debug_arc_tokenization.py \
  --model_dir models/qwen2-7b \
  --problem_path data/evaluation/00576224.json \
  --num_pseudo 5 \
  --pseudo_index 0 \
  --max_len 4096 \
  --max_tokens_show 200
```

This will print:

* The **augmented train/test grids** for that pseudo-task
* The **prompt** and **answer** text
* Which tokens are ignored (`label = -100`) vs used as targets

This is useful to verify that your data files are correctly formatted and
compatible with the training code.

---

```
::contentReference[oaicite:0]{index=0}
```

