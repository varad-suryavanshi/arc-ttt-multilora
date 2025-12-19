#!/usr/bin/env bash

python ttt_multi_lora.py \
  --model_dir models/qwen2-7b \
  --data_dir data/evaluation \
  --k 8 \
  --steps 50 \
  --num_pseudo 200 \
  --out_dir arc_predictions_qwen

