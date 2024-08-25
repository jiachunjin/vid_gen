export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# accelerate launch train.py \
accelerate launch --main_process_port 29505 train.py \
--wandb_proj "scale_up" \
--exp_name "notxt" \
--output_dir "experiment/notxt" \
--sd_path "/home/jiachun/codebase/vsd/ckpts/zsnr" \
--data_dir "/data/vsd_data/captioned_8s_64f_motion/" \
--weighting "snr" \
--con_type "qformer" \
--con_depth 2 \
--con_nframes 64 \
--p_uncond 1.0 \
--learning_rate 0.0001 \
--mixed_precision "fp16" \
--max_train_steps 100000 \
--gc \
--train_batch_size 5 \
--gradient_accumulation_steps 1 \
--save_every 1000 \
--push_to_hub \
--report_to "no"