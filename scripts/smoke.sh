#!/bin/bash
# Capped smoke run of UniPELT (APL) on STS-B K=1000 (identical on GPU and TPU).
# Usage: smoke.sh <code_dir> <out_dir> <wrapper.py>   (python env must be active)
# Caps: 315 optimizer steps (5 epochs of 63), logging every 10 steps, one dev eval + checkpoint save at step 315,
# reload of the best checkpoint, then dev eval and the full 1,500-pair test eval. Seed 42, data seed 111.
CODE=$1; OUT=$2; WRAP=$3
cd $CODE && mkdir -p $OUT
echo "SMOKE_START $(date -u +%FT%TZ)"
python $WRAP --model_name_or_path bert-base-uncased --task_name stsb \
  --do_train --do_eval --do_predict \
  --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 5e-4 --num_train_epochs 50 \
  --max_train_samples 1000 --data_seed 111 --seed 42 \
  --add_enc_prefix True --train_adapter --add_lora True \
  --evaluation_strategy steps --eval_steps 315 --save_steps 315 --load_best_model_at_end \
  --metric_for_best_model spearmanr --greater_is_better True --save_total_limit 2 \
  --max_steps 315 --logging_steps 10 --output_dir $OUT/out --overwrite_output_dir > $OUT/smoke.log 2>&1
echo "SMOKE_EXIT $? $(date -u +%FT%TZ)"
