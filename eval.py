import os
import torch
import numpy as np
from einops import rearrange
from tqdm.auto import tqdm, trange

from thop import profile
from thop import clever_format


def sample_ar(ckpt_path, data_path, id=None, prompt=None):
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    from ar import get_models, parse_args
    device = torch.device("cuda:5")

    args = parse_args()
    # args.sd_path = "/home/jiachun/codebase/vsd/ckpts/sd2-1"
    args.sd_path = "/home/jiachun/codebase/vsd/ckpts/zsnr"
    # args.sd_path = "/home/jiachun/codebase/vsd/ckpts/sd2-1"
    # args.sd_path = "/home/jiachun/codebase/vsd/ckpts/sd2-1-v"
    # args.con_dim_head = 88
    # args.con_num_media_embeds = 32
    conditioner, unet, scheduler, tokenizer, text_encoder = get_models(args)
    vae = AutoencoderKL.from_pretrained(os.path.join(args.sd_path, "vae"))
    image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)

    exp_name = ckpt_path.split("/")[-1].split("-")[0]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    global_step = ckpt["global_step"]
    print(f"global_step: {global_step}")

    unet.load_state_dict(ckpt['unet'])
    conditioner.load_state_dict(ckpt['conditioner'])
    del ckpt
    vae = vae.to(device)
    unet = unet.to(device)
    conditioner = conditioner.to(device)
    text_encoder = text_encoder.to(device)

    unet.eval()
    conditioner.eval()

    f = 64
    inference_steps = 20
    scheduler.set_timesteps(inference_steps)

    with torch.no_grad(), torch.cuda.amp.autocast():
        sampled_latents = torch.zeros((f, 4, 64, 64), device=device)
        if prompt is None:
            first_frame = torch.load(os.path.join(data_path, str(id)))["latent"][0].to(device) * 0.18215
            sampled_latents[0, ...] = first_frame
            caption = torch.load(os.path.join(data_path, str(id)))["caption"]

            print(caption)
            text_inputs = tokenizer([caption], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids
            txt_embedding = text_encoder(text_input_ids.to(device), return_dict=False)[0].detach()
        else:
            print(prompt)
            text_input_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
            null_input_ids = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids

            txt_embedding = text_encoder(text_input_ids.to(device), return_dict=False)[0].detach()
            null_embedding = text_encoder(null_input_ids.to(device), return_dict=False)[0].detach()
            latents = torch.randn((1, 4, 64, 64), device=device)
            for t in scheduler.timesteps:
                t_sample = torch.as_tensor([t]*2, device=device)
                model_pred = unet(
                    sample = torch.cat([latents] * 2),
                    timestep = t_sample,
                    encoder_hidden_states = torch.cat([txt_embedding, null_embedding], dim=0),
                ).sample

                noise_pred, noise_pred_uncond = torch.chunk(model_pred, 2)
                model_pred = noise_pred_uncond + 10 * (noise_pred - noise_pred_uncond)

                latents = scheduler.step(model_pred, t, latents).prev_sample
            sampled_latents[0, ...] = latents

        for i in trange(1, f):
            sampled_frame = sample_next_frame(
                frame_id=i,
                unet=unet,
                conditioner=conditioner,
                scheduler=scheduler,
                current_latents=sampled_latents,
                txt_embedding=txt_embedding,
                device=device,
            )
            sampled_latents[i, ...] = sampled_frame

        latents = 1 / vae.config.scaling_factor * sampled_latents.squeeze()
        vae.enable_slicing()
        image = vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        images = image_processor.numpy_to_pil(image)
        if prompt is None:
            tag = id
        else:
            tag = prompt
        images[0].save(f"./samples/ar_{exp_name}_{int(global_step/1000)}k_{tag}.gif", save_all=True, append_images=images[1:], loop=0, duration=125)
        

def sample_next_frame(frame_id, unet, conditioner, scheduler, current_latents, txt_embedding, device):
    latents = torch.randn((1, 4, 64, 64), device=device)
    context = current_latents # 0.8 * torch.randn_like(current_latents, device=device) + 0.2 * current_latents
    latents *= scheduler.init_noise_sigma
    context_tokens = conditioner(context.unsqueeze(dim=0), torch.as_tensor([frame_id], device=device, dtype=torch.long))
    for t in scheduler.timesteps:
        latents = scheduler.scale_model_input(latents, t)
        t_sample = torch.as_tensor([t], device=device)
        model_pred = unet(
            sample                = latents,
            timestep              = t_sample,
            encoder_hidden_states = txt_embedding,
            context_tokens        = context_tokens,
        ).sample
        # break
        latents = scheduler.step(model_pred, t, latents).prev_sample
    # return -model_pred
    return latents
    

if __name__ == "__main__":
    for i in range(10):
        sample_ar(
            ckpt_path="/home/jiachun/codebase/vsd/ar/experiment/8fps_4096_uncon0.5/8fps_4096_uncon0.5-40000-ckpt",
            data_path="/data/vsd_data/captioned_8s_64f_motion",

            # ckpt_path="/home/jiachun/codebase/vsd/cog/experiment/2k/2k-35000-ckpt",
            # data_path="/data/vsd_data/captioned_OpenVid/",

            # ckpt_path="/home/jiachun/codebase/vsd/ar/experiment/8fps_both/8fps_both-30000-ckpt",
            # data_path="/data/vsd_data/captioned_8s_64f_motion",
            # data_path="/data/jiachun/captioned_video/",
            id=i,
            # prompt="A man running on the grass.",
        )