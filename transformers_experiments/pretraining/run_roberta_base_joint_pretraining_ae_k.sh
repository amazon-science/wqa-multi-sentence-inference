# change datasets path and output folder as needed. We suggest to run this experiment on a single P4
python -m transformers_framework \
    --model RobertaJointMLMAndClassification \
    --devices 8 \
    --accelerator gpu --strategy deepspeed_stage_2 \
    --precision 16 \
    --pre_trained_model roberta-base \
    --name roberta-base-joint-ie-k \
    --output_dir outputs/joint-pretraining \
    \
    --adapter JointwiseArrowAdapter \
    --batch_size 64 \
    --train_filepath /path/to/datasets \
    --field_names premise consequence \
    --label_name label \
    \
    --log_every_n_steps 100 \
    --accumulate_grad_batches 8 \
    --max_sequence_length 64 \
    -k 5 --selection random --separated \
    --learning_rate 1e-04 \
    --max_steps 100000 \
    --weight_decay 0.01 \
    --num_warmup_steps 10000 \
    --num_workers 8 \
    --head_type AE_k \
    --seed 1337
