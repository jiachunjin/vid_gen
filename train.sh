# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# accelerate launch train.py \
accelerate launch --main_process_port 29505 train.py \
--wandb_proj "scale_up" \
--exp_name "scale_up" \
--output_dir "experiment/scale_up" \
--sd_path "/home/jiachun/codebase/vsd/ckpts/zsnr" \
--data_dir "path to the processed data" \
--resume_path "path to the resume checkpoint" \
--weighting "snr" \
--con_type "qformer" \
--interface "new" \
--con_depth 20 \
--con_numq 512 \
--con_nframes 64 \
--con_heads 16 \
--con_dim_head 256 \
--con_patch_size 8 \
--p_uncond 0.5 \
--learning_rate 0.00005 \
--mixed_precision "fp16" \
--max_train_steps 200000 \
--gc \
--train_batch_size 1 \
--gradient_accumulation_steps 1 \
--save_every 1000 \
--push_to_hub \
--report_to "wandb"