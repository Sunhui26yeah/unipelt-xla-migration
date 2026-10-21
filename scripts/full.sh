#!/bin/bash
# Full run of UniPELT (APL) on STS-B K=1000 (identical on GPU and TPU): the SELECTION.md command.
# Usage: full.sh <code_dir> <out_dir> <wrapper.py>   (python env must be active)
CODE=$1; OUT=$2; WRAP=$3
cd $CODE && mkdir -p $OUT
echo "FULL_START $(date -u +%FT%TZ)"
python $WRAP --model_name_or_path bert-base-uncased --task_name stsb \
  --do_train --do_eval --do_predict \
  --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 5e-4 --num_train_epochs 50 \
  --max_train_samples 1000 --data_seed 111 --seed 42 \
  --add_enc_prefix True --train_adapter --add_lora True \
  --evaluation_strategy epoch --save_strategy epoch --load_best_model_at_end \
  --metric_for_best_model spearmanr --greater_is_better True --save_total_limit 2 \
  --logging_steps 10 --output_dir $OUT/out --overwrite_output_dir > $OUT/full.log 2>&1
echo "FULL_EXIT $? $(date -u +%FT%TZ)"
