import os
import torch
import random
import logging
import argparse

import transformers
import diffusers
import torch.nn.functional as F

from diffusers.training_utils import cast_training_params, compute_snr
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from tqdm.auto import tqdm
from huggingface_hub import login, logout
from safetensors.torch import load_file


from utils.dataset import captioned_video

import os

def parse_args():
    parser = argparse.ArgumentParser(description="vsd parser")
    # sd
    parser.add_argument("--interface", type=str, choices=["old", "new"])
    parser.add_argument("--sd_path", type=str, default="/home/jiachun/codebase/vsd/ckpts/zsnr")
    # conditioner
    parser.add_argument("--con_type", type=str, choices=["qformer", "avgpool"], default="qformer")
    parser.add_argument("--con_depth", type=int, default=4)
    parser.add_argument("--con_dim", type=int, default=1024)
    parser.add_argument("--con_nframes", type=int, default=32)
    parser.add_argument("--con_numq", type=int, default=512)
    parser.add_argument("--con_patch_size", type=int, default=8)
    parser.add_argument("--con_heads", type=int, default=16)
    parser.add_argument("--con_dim_head", type=int, default=88)
    parser.add_argument("--con_num_temporal_state", type=int, default=8)
    parser.add_argument("--con_num_spatial_state", type=int, default=4)
    # training
    parser.add_argument("--weighting", choices=["uniform", "snr", "minsnr", "maxsnr"])
    parser.add_argument("--data_dir", type=str, default="/data/vsd_data/captioned_OpenVid")
    parser.add_argument("--p_uncond", type=float, default=1.0)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--gc", action="store_true", help="Enable gradient checkpointing or not.")
    parser.add_argument("--num_train_samples", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=1000000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # experiment
    parser.add_argument("--wandb_proj", type=str, default="wandb_proj")
    parser.add_argument("--exp_name", type=str, default="exp_name")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--output_dir", type=str, default="experiment/run")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--push_to_hub", action="store_true")

    args = parser.parse_args()
    return args    


def get_logger_accelerator(args):
    logger = get_logger(__name__, log_level="INFO")
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "no" else args.report_to,
        project_config=accelerator_project_config,
    )
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    return accelerator, logger


def get_models(args):
    from diffusers import DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from models.unet_context import UNet_context
    from models.unet_distributed import UNet_distributed
    if args.interface == "old":
        unet = UNet_context.from_pretrained(os.path.join(args.sd_path, "unet"))
        unet.eval()
        unet.requires_grad_(False)
        unet.hack(
            args.con_type,
            args.con_dim,
            args.con_depth,
            args.con_numq,
            args.con_nframes,
            args.con_patch_size,
            args.con_heads,
            args.con_dim_head,
        )
    elif args.interface == "new":
        unet = UNet_distributed.from_pretrained_2d(
            os.path.join(args.sd_path, "unet"),
            con_type = args.con_type,
            con_dim = args.con_dim,
            con_depth = args.con_depth,
            con_numq = args.con_numq,
            con_nframes = args.con_nframes,
            con_patch_size = args.con_patch_size,
            con_dim_head = args.con_dim_head,
            con_heads = args.con_heads,
        )
        unet.requires_grad_(False)

    noise_scheduler = DDPMScheduler.from_pretrained(os.path.join(args.sd_path, "scheduler"))
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(args.sd_path, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.sd_path, "text_encoder"))
    text_encoder.requires_grad_(False)

    return unet, noise_scheduler, tokenizer, text_encoder


def main():
    args = parse_args()
    accelerator, logger = get_logger_accelerator(args)
    unet, noise_scheduler, tokenizer, text_encoder = get_models(args)
    if args.resume_path:
        unet_state_path = os.path.join(args.resume_path, "diffusion_pytorch_model.safetensors")
        unet_state_dict = load_file(unet_state_path)

        model_dict = unet.state_dict()
        pretrained_dict = {k: v for k, v in unet_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)

        unet.load_state_dict(model_dict, strict=False)

        # global_step = int(args.resume_path.split("/")[-1].split("-")[-1][:-1]) * 1000
        # logger.info(f"Resume training from {args.resume_path}, global step: {global_step}")
        # if accelerator.is_main_process: print(f"resume training from {args.resume_path}, global step: {global_step}")
        global_step = 0
    else:
        global_step = 0

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    unet_params = []
    unet_params = unet.return_extra_parameters()

    for p in unet_params:
        p.requires_grad = True

    params_to_learn = unet_params

    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    if args.gc:
        unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    total_params = sum(p.numel() for p in optimizer.param_groups[0]['params'])
    num_param_conditioner = sum(p.numel() for p in unet.conditioner.parameters())

    train_dataloader = torch.utils.data.DataLoader(
        captioned_video(args.data_dir, subset=args.num_train_samples),
        shuffle=True,
        collate_fn=None,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if accelerator.is_main_process:
        logger.info(f"Total optimized parameters: {total_params}, conditioner: {num_param_conditioner}, dataset size: {len(train_dataloader.dataset)}")

    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(args.wandb_proj, config=vars(args))
        if args.push_to_hub:
            login(token="")

    done = False
    epoch = 0
    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    while not done:
        train_loss = 0.0
        uncon_loss = 0.0
        con_loss = 0.0
        acc_steps = 0
        for _, batch in enumerate(train_dataloader):
            unet.train()
            with accelerator.accumulate(unet):
                latents, video_length, caption = batch

                latents = latents * 0.18215
                b, f, c, h, w = latents.shape

                num_to_replace = int(b * args.p_uncond)
                indices_to_replace = random.sample(range(b), num_to_replace)
                for idx in indices_to_replace:
                    caption[idx] = ""

                text_inputs = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                text_input_ids = text_inputs.input_ids
                with torch.no_grad():
                    txt_embedding = text_encoder(text_input_ids.to(accelerator.device), return_dict=False)[0].detach()

                timesteps = torch.randint(0, 1000, (b,), device=latents.device)
                timesteps = timesteps.long()

                snr = compute_snr(noise_scheduler, timesteps)
                if args.weighting == "uniform":
                    mse_loss_weights = 1 / (snr + 1)
                elif args.weighting == "snr":
                    mse_loss_weights = 1
                elif args.weighting == "minsnr":
                    mse_loss_weights = torch.stack([snr, 5.0 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    mse_loss_weights = mse_loss_weights / (snr + 1)
                elif args.weighting == "maxsnr":
                    mse_loss_weights = torch.stack([snr, 1.0 * torch.ones_like(timesteps)], dim=1).max(dim=1)[0]
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                noise = torch.randn_like(latents, device=accelerator.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                random_frame_indices = torch.zeros((b,), dtype=torch.long, device=accelerator.device)
                for i, l in enumerate(video_length):
                    random_frame_indices[i] = torch.randint(0, l, (1,))
                noisy_frames = torch.stack([noisy_latents[i, idx] for i, idx in enumerate(random_frame_indices)])

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = torch.stack([noise[i, idx] for i, idx in enumerate(random_frame_indices)])
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    selected_frames = torch.stack([latents[i, idx] for i, idx in enumerate(random_frame_indices)])
                    frame_noise = torch.stack([noise[i, idx] for i, idx in enumerate(random_frame_indices)])
                    target = noise_scheduler.get_velocity(selected_frames, frame_noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(
                    sample=noisy_frames,
                    timestep=timesteps,
                    encoder_hidden_states=txt_embedding,
                    contexts=noisy_latents,
                    random_frame_indices=random_frame_indices,
                ).sample

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")

                cond_indices = [index for index, value in enumerate(caption) if value != ""]
                uncond_indices = [index for index, value in enumerate(caption) if value == ""]
                loss_cond = loss[cond_indices].mean()
                loss_uncond = loss[uncond_indices].mean()

                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item()

                avg_loss_cond = accelerator.gather(loss_cond).mean().item()
                avg_loss_uncond = accelerator.gather(loss_uncond).mean().item()

                acc_steps += 1
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = params_to_learn
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                train_loss /= acc_steps
                acc_steps = 0
                accelerator.log({"train_loss": train_loss, "loss_cond": avg_loss_cond, "loss_uncond": avg_loss_uncond}, step=global_step)
                logs = {"step_loss": train_loss}
                progress_bar.set_postfix(**logs)
                train_loss = 0.0
                accelerator.wait_for_everyone()

                if global_step % args.save_every == 0 or global_step == 1000:
                    unet.eval()
                    if accelerator.is_main_process:
                        unwarped_unet = accelerator.unwrap_model(unet)
                        unwarped_unet.save_pretrained(os.path.join(args.output_dir, f"{args.exp_name}-unet-{int(global_step / 1000)}k"))
                        if args.push_to_hub:
                            unwarped_unet.push_to_hub("orres/vid_gen")
                            logger.info(f"pushed to hugging face at step {global_step}")
                accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                done = True
                break
        epoch += 1
    if accelerator.is_main_process:
        if args.push_to_hub:
            logout()

    accelerator.end_training()


if __name__ == "__main__":
    main()