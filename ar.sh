# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# accelerate launch ar.py \
accelerate launch --main_process_port 29501 ar.py \
--wandb_proj "ar" \
--exp_name "scale_up" \
--output_dir "experiment/scale_up" \
--sd_path "./ckpts/zsnr" \
--data_dir "path_to_data_directory" \
--weighting "snr" \
--uncon_ratio 0.5 \
--learning_rate 0.00005 \
--mixed_precision "fp16" \
--max_train_steps 500000 \
--train_batch_size 5 \
--gradient_accumulation_steps 1 \
--gc \
--sample_every -1 \
--save_every 10000 \
--report_to "wandb"