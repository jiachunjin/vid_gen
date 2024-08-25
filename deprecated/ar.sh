# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com
# accelerate launch ar.py \
accelerate launch --main_process_port 29505 ar.py \
--wandb_proj "ar" \
--exp_name "2k_doublenoise" \
--output_dir "experiment/2k_doublenoise" \
--sd_path "./ckpts/zsnr" \
--data_dir "/data/vsd_data/captioned_OpenVid/" \
--weighting "snr" \
--sigma 1 \
--uncon_ratio 0.5 \
--learning_rate 0.00005 \
--mixed_precision "fp16" \
--max_train_steps 100000 \
--train_batch_size 15 \
--gradient_accumulation_steps 1 \
--gc \
--sample_every -1 \
--save_every 2000 \
--report_to "wandb"