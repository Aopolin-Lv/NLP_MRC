export SQUAD_DIR=./SQuAD2.0/dataset

python Code/run_squad.py \
  --model_typ bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --version_2_with_negative \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --no_cuda \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/
