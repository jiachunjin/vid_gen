# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# accelerate launch ar.py \
accelerate launch --main_process_port 29501 ar.py \
--wandb_proj "ar" \
--exp_name "8fps_4096_uncon0.5" \
--output_dir "experiment/8fps_4096_uncon0.5" \
--sd_path "/home/jiachun/codebase/vsd/ckpts/zsnr" \
--data_dir "/data/vsd_data/captioned_8s_64f_motion" \
--weighting "snr" \
--uncon_ratio 0.5 \
--learning_rate 0.00005 \
--mixed_precision "fp16" \
--max_train_steps 100000 \
--train_batch_size 5 \
--gradient_accumulation_steps 1 \
--gc \
--sample_every -1 \
--save_every 10000 \
--report_to "wandb"