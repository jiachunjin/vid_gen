import os
import torch
import logging
import argparse

# import datasets
import transformers
import diffusers
import torch.nn.functional as F
from diffusers.training_utils import cast_training_params
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.training_utils import compute_snr
from tqdm.auto import tqdm
from einops import rearrange
import pandas as pd
import random

from dataset import captioned_video

def parse_args():
    parser = argparse.ArgumentParser(description="vsd parser")
    # sd
    parser.add_argument("--sd_path", type=str, default="/home/jiachun/codebase/vsd/ckpts/zsnr")
    # conditioner
    parser.add_argument("--con_depth", type=int, default=2)
    parser.add_argument("--con_lenq", type=int, default=512)
    parser.add_argument("--con_heads", type=int, default=16)
    parser.add_argument("--con_dim_head", type=int, default=256)
    parser.add_argument("--con_num_media_embeds", type=int, default=64)
    # training
    parser.add_argument("--uncon_ratio", type=float, default=0.5)
    parser.add_argument("--weighting", choices=["**", "minsnr", "no", "snr", "maxsnr"])
    parser.add_argument("--data_dir", type=str, default="/data/vsd_data/captioned_OpenVid")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--gc", action="store_true", help="Enable gradient checkpointing or not.")
    parser.add_argument("--num_train_samples", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=1000000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--sample_nframes", type=int, default=32)
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
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
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
    from models import UNet_context, Conditioner

    conditioner = Conditioner(
        dim              = 1024,
        depth            = args.con_depth,
        num_latents      = args.con_lenq,
        heads            = args.con_heads,
        dim_head         = args.con_dim_head,
        num_media_embeds = args.con_num_media_embeds,
    )

    unet = UNet_context.from_pretrained(os.path.join(args.sd_path, "unet"))
    unet.eval()
    unet.requires_grad_(False)
    unet.hack()
    
    noise_scheduler = DDPMScheduler.from_pretrained(os.path.join(args.sd_path, "scheduler"))
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(args.sd_path, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.sd_path, "text_encoder"))
    text_encoder.requires_grad_(False)

    return conditioner, unet, noise_scheduler, tokenizer, text_encoder


def main():
    args = parse_args()
    accelerator, logger = get_logger_accelerator(args)
    conditioner, unet, noise_scheduler, tokenizer, text_encoder = get_models(args)
    logger.info(f"UNet initialized from {args.sd_path}")
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location="cpu")
        m, u = unet.load_state_dict(ckpt['unet'], strict=False)
        m, u = conditioner.load_state_dict(ckpt['conditioner'])
        if "global_step" in ckpt.keys():
            global_step = ckpt["global_step"]
        del ckpt
        if accelerator.is_main_process: print(f"resume training from {args.resume_path}, global step: {global_step}")
    else:
        global_step = 0

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    conditioner.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    unet_params = []
    unet_params = unet.return_extra_parameters()

    for p in unet_params:
        p.requires_grad = True

    params_to_learn = unet_params + list(conditioner.parameters())

    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)
        cast_training_params(conditioner, dtype=torch.float32)

    if args.gc:
        unet.enable_gradient_checkpointing()
        # conditioner.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    total_params = sum(p.numel() for p in optimizer.param_groups[0]['params'])
    conditioner_num_train = sum(p.numel() for p in conditioner.parameters() if p.requires_grad)
    unet_num_train = sum(p.numel() for p in unet_params if p.requires_grad)
    
    logger.info(f"Total optimized parameters: {total_params}, conditioner: {conditioner_num_train}, unet: {unet_num_train}")

    train_dataloader = torch.utils.data.DataLoader(
        captioned_video(args.data_dir, subset=args.num_train_samples),
        shuffle=True,
        collate_fn=None,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    unet, conditioner, optimizer, train_dataloader = accelerator.prepare(
        unet, conditioner, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(args.wandb_proj, config=vars(args))

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
        acc_steps = 0
        for _, batch in enumerate(train_dataloader):
            unet.train()
            conditioner.train()
            with accelerator.accumulate(unet, conditioner):
                latents, video_length, caption = batch
                caption = list(caption)
                latents = latents * 0.18215
                # latents = latents.repeat(2, 1, 1, 1, 1)
                # video_length = video_length.repeat(2)
                b, f, c, h, w = latents.shape
                
                drop_indices = random.sample(range(b), int(b * args.uncon_ratio))
                for idx in drop_indices:
                    caption[idx] = ""
                # print(caption, drop_indices, args.uncon_ratio)
                # exit(0)

                text_inputs = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                text_input_ids = text_inputs.input_ids
                txt_embedding = text_encoder(text_input_ids.to(accelerator.device), return_dict=False)[0].detach()

                timesteps = torch.randint(0, 1000, (b,), device=latents.device)
                timesteps = timesteps.long()

                snr = compute_snr(noise_scheduler, timesteps)
                if args.weighting == "no":
                    mse_loss_weights = 1 / (snr + 1)
                elif args.weighting == "snr":
                    mse_loss_weights = 1
                elif args.weighting == "minsnr":
                    mse_loss_weights = torch.stack([snr, 5.0 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    mse_loss_weights = mse_loss_weights / (snr + 1)
                elif args.weighting == "maxsnr":
                    mse_loss_weights = torch.stack([snr, 1.0 * torch.ones_like(timesteps)], dim=1).max(dim=1)[0]
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                random_frame_indices = torch.zeros((b,), dtype=torch.long, device=accelerator.device)
                for i, l in enumerate(video_length):
                    random_frame_indices[i] = torch.randint(1, l, (1,))

                selected_frames = torch.stack([latents[i, idx] for i, idx in enumerate(random_frame_indices)])
                noise = torch.randn_like(selected_frames, device=accelerator.device)

                noisy_latents = noise_scheduler.add_noise(selected_frames, noise, timesteps)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    # selected_frames = torch.stack([latents[i, idx] for i, idx in enumerate(random_frame_indices)])
                    target = noise_scheduler.get_velocity(selected_frames, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # target = noise_scheduler.get_velocity(selected_frames, noise, timesteps)
                # exit(0)
                context = conditioner(latents, random_frame_indices) # (b, l, d)
                # mask = (torch.rand(b) > args.p_uncond).float().unsqueeze(1).unsqueeze(2).to(accelerator.device)
                # mask = mask.expand(-1, context.shape[1], context.shape[2])
                # context = context * mask
                # context = conditioner(noisy_latents, timesteps)
                # print(context.shape, noisy_latents.shape)
                # exit(0)
                model_pred = unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=txt_embedding,
                    context_tokens=context
                ).sample

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item()

                acc_steps += 1
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     params_to_clip = params_to_learn
                #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                train_loss /= acc_steps
                acc_steps = 0
                accelerator.log({"train_loss": train_loss}, step=global_step)
                logs = {"step_loss": train_loss}
                progress_bar.set_postfix(**logs)
                train_loss = 0.0
                accelerator.wait_for_everyone()

                if global_step % args.save_every == 0 or global_step == 1000:
                    unet.eval()
                    conditioner.eval()
                    if accelerator.is_main_process:
                        # save_path = os.path.join(args.output_dir, f"{args.exp_name}-{global_step}-ckpt")
                        unwarped_unet = accelerator.unwrap_model(unet)
                        unwarped_conditioner = accelerator.unwrap_model(conditioner)
                        # torch.save(
                        #     {
                        #         "unet": unwarped_unet.state_dict(),
                        #         "conditioner": unwarped_conditioner.state_dict(),
                        #         "global_step": global_step,
                        #     },
                        #     save_path
                        # )
                        unwarped_unet.save_pretrained(os.path.join(args.output_dir, f"{args.exp_name}-unet-{int(global_step / 1000)}"))
                        unwarped_unet.push_to_hub("orres/vid_gen")
                        unwarped_conditioner.save_pretrained(os.path.join(args.output_dir, f"{args.exp_name}-conditioner-{int(global_step / 1000)}"))
                        unwarped_conditioner.push_to_hub("orres/vid_gen")
                        logger.info(f"pushed to hugging face at step {global_step}")

                accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                done = True
                break
        epoch += 1
    accelerator.end_training()


if __name__ == "__main__":
    main()
