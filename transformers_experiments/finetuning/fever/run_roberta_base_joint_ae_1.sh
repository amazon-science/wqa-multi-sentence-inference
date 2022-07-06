python -m transformers_framework \
    --model RobertaJointFactChecking \
    --devices 2 --accelerator gpu --strategy ddp \
    --pre_trained_model <path-to-pretrained-model> \
    --name roberta-base-joint-fever-AE-1 \
    --output_dir outputs/joint-fever \
    \
    --adapter JointwiseArrowAdapter \
    --batch_size 32 --val_batch_size 128 --test_batch_size 128 \
    --train_filepath <path-to-fever-dataset> --train_split train \
    --valid_filepath <path-to-fever-dataset> --valid_split validation \
    --field_names claim evidence \
    --label_name label \
    --key_name key \
    \
    --accumulate_grad_batches 1 \
    --max_sequence_length 64 \
    -k 5 --selection all --separated --force_load_dataset_in_memory --reduce_labels \
    --learning_rate 1e-05 \
    --max_epochs 15 \
    --early_stopping \
    --patience 8 \
    --weight_decay 0.0 \
    --num_warmup_steps 1000 \
    --monitor validation/accuracy \
    --val_check_interval 0.5 \
    --num_workers 8 \
    --shuffle_candidates --reload_dataloaders_every_n_epoch 1 \
    --num_labels 3 \
    --head_type AE_1 \
