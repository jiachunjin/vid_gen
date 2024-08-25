import os
import torch
import numpy as np
from einops import rearrange
from tqdm.auto import tqdm, trange
from safetensors.torch import load_file

# from thop import profile
# from thop import clever_format


def sample_ar(unet_path, data_path, id=None, guidance_scale=12.5, prompt=None, iter=None):
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    from ar import get_models, parse_args
    device = torch.device("cuda:7")

    args = parse_args()
    # args.sd_path = "/home/jiachun/codebase/vsd/ckpts/sd2-1"
    args.sd_path = "/home/jiachun/codebase/vsd/ckpts/zsnr"
    # args.sd_path = "/home/jiachun/codebase/vsd/ckpts/sd2-1"
    # args.sd_path = "/home/jiachun/codebase/vsd/ckpts/sd2-1-v"
    # args.con_dim_head = 88
    # args.con_num_media_embeds = 32
    unet, scheduler, tokenizer, text_encoder = get_models(args)
    vae = AutoencoderKL.from_pretrained(os.path.join(args.sd_path, "vae"))
    image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)

    # exp_name = ckpt_path.split("/")[-1].split("-")[0]
    # ckpt = torch.load(ckpt_path, map_location="cpu")
    # global_step = ckpt["global_step"]
    # print(f"global_step: {global_step}")

    unet_state_path = os.path.join(unet_path, "diffusion_pytorch_model.safetensors")
    unet_state_dict = load_file(unet_state_path)
    unet.load_state_dict(unet_state_dict)

    # conditioner = conditioner.from_pretrained(conditioner_path)

    vae = vae.to(device)
    unet = unet.to(device)
    # conditioner = conditioner.to(device)
    text_encoder = text_encoder.to(device)

    unet.eval()
    # conditioner.eval()

    f = 64
    inference_steps = 30
    scheduler.set_timesteps(inference_steps)

    with torch.no_grad(), torch.cuda.amp.autocast():
        sampled_latents = torch.zeros((f, 4, 64, 64), device=device)
        if prompt is None:
            first_frame = torch.load(os.path.join(data_path, str(id)))["latent"][0].to(device) * 0.18215
            sampled_latents[0, ...] = first_frame
            caption = torch.load(os.path.join(data_path, str(id)))["caption"]

            print(caption)
            text_inputs = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids
            null_input_ids = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
            txt_embedding = text_encoder(text_input_ids.to(device), return_dict=False)[0].detach()
            null_embedding = text_encoder(null_input_ids.to(device), return_dict=False)[0].detach()
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
                model_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                latents = scheduler.step(model_pred, t, latents).prev_sample
            sampled_latents[0, ...] = latents

        for i in trange(1, f):
            sampled_frame = sample_next_frame(
                frame_id=i,
                unet=unet,
                # conditioner=conditioner,
                scheduler=scheduler,
                current_latents=sampled_latents,
                txt_embedding=txt_embedding,
                null_embedding=null_embedding,
                guidance_scale=guidance_scale,
                device=device,
            )
            sampled_latents[i, ...] = sampled_frame
        
        examine_coherence(unet_path, sampled_latents)

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
        images[0].save(f"./samples/samples_{tag}_{int(iter/1000)}k.gif", save_all=True, append_images=images[1:], loop=0, duration=30)
        torch.cuda.empty_cache()
        

def sample_next_frame(frame_id, unet, scheduler, current_latents, txt_embedding, null_embedding, guidance_scale, device):
    latents = torch.randn((1, 4, 64, 64), device=device)
    context = current_latents # 0.8 * torch.randn_like(current_latents, device=device) + 0.2 * current_latents
    latents *= scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        latents = scheduler.scale_model_input(latents, t)
        # with cfg
        # t_sample = torch.as_tensor([t]*2, device=device)
        # model_pred = unet(
        #     sample = torch.cat([latents] * 2),
        #     timestep = t_sample,
        #     encoder_hidden_states = torch.cat([txt_embedding, null_embedding], dim=0),
        #     latents               = context.unsqueeze(dim=0).repeat(2, 1, 1, 1, 1),
        #     random_frame_indices  = torch.as_tensor([frame_id], device=device, dtype=torch.long).repeat(2)
        # ).sample
        # noise_pred, noise_pred_uncond = torch.chunk(model_pred, 2)
        # model_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
        # without cfg
        latents = scheduler.scale_model_input(latents, t)
        t_sample = torch.as_tensor([t], device=device)
        model_pred = unet(
            sample                = latents,
            timestep              = t_sample,
            encoder_hidden_states = txt_embedding,
            latents               = context.unsqueeze(dim=0),
            random_frame_indices  = torch.as_tensor([frame_id], device=device, dtype=torch.long)
        ).sample

        model_pred_uncond = unet(
            sample                = latents,
            timestep              = t_sample,
            encoder_hidden_states = txt_embedding,
        ).sample

        model_pred = model_pred_uncond + 0.9 * (model_pred - model_pred_uncond)

        latents = scheduler.step(model_pred, t, latents).prev_sample

    return latents


def examine_coherence(unet_path, latents=None):
    import torch
    import torch.nn.functional as F
    from ar import get_models, parse_args
    from tqdm.auto import trange
    device = torch.device("cuda:7")

    args = parse_args()
    args.sd_path = "/home/jiachun/codebase/vsd/ckpts/zsnr"
    unet, scheduler, tokenizer, text_encoder = get_models(args)

    unet_state_path = os.path.join(unet_path, "diffusion_pytorch_model.safetensors")
    unet_state_dict = load_file(unet_state_path)
    unet.load_state_dict(unet_state_dict)
    unet = unet.to(device)

    diff_list = torch.zeros((1, 64))
    norm_list = torch.zeros((1, 64))
    with torch.no_grad():
        for i in trange(1):
            # latents = torch.load(f"/data/vsd_data/captioned_8s_64f_motion/{i}")["latent"].to(device)
            for j in range(2, 64):
                ct_cur = unet.conditioner(latents.unsqueeze(0), torch.as_tensor([j]))
                torch.cuda.empty_cache()
                ct_pre = unet.conditioner(latents.unsqueeze(0), torch.as_tensor([j-1]))
                diff = torch.norm(ct_cur.flatten() - ct_pre.flatten())
                norm = torch.norm(ct_cur.flatten())
                diff_list[i, j] = diff
                norm_list[i, j] = norm
        print(diff_list)
        diff = diff_list.mean(dim=0)[2:].mean()
        norm = norm_list.mean(dim=0)[2:].mean()
        print(diff, norm, diff.shape)

def sanity_check(unet_path, data_path, id, k):
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    from ar import get_models, parse_args
    device = torch.device("cuda:7")

    args = parse_args()
    args.sd_path = "/home/jiachun/codebase/vsd/ckpts/zsnr"

    unet, scheduler, tokenizer, text_encoder = get_models(args)
    vae = AutoencoderKL.from_pretrained(os.path.join(args.sd_path, "vae"))
    image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)

    unet_state_path = os.path.join(unet_path, "diffusion_pytorch_model.safetensors")
    unet_state_dict = load_file(unet_state_path)
    unet.load_state_dict(unet_state_dict)
    vae = vae.to(device)
    unet = unet.to(device)
    text_encoder = text_encoder.to(device)

    unet.eval()
    f = k+1
    sampled_latents = torch.zeros((f, 4, 64, 64), device=device)
    first_k_frames = torch.load(os.path.join(data_path, str(id)))["latent"][:k].to(device) * 0.18215
    sampled_latents[:k, ...] = first_k_frames
    caption = torch.load(os.path.join(data_path, str(id)))["caption"]

    with torch.no_grad():
        print(caption)
        text_inputs = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        null_input_ids = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
        txt_embedding = text_encoder(text_input_ids.to(device), return_dict=False)[0].detach()
        null_embedding = text_encoder(null_input_ids.to(device), return_dict=False)[0].detach()

        inference_steps = 30
        scheduler.set_timesteps(inference_steps)

        for i in trange(k, k+1):
            sampled_frame = sample_next_frame(
                frame_id=i,
                unet=unet,
                scheduler=scheduler,
                current_latents=sampled_latents,
                txt_embedding=txt_embedding,
                null_embedding=null_embedding,
                guidance_scale=1,
                device=device,
            )
            sampled_latents[i, ...] = sampled_frame


        latents = 1 / vae.config.scaling_factor * sampled_latents.squeeze()
        vae.enable_slicing()
        image = vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        images = image_processor.numpy_to_pil(image)
        return images[-1]
        # grid_image = image_grid(images, rows=1, cols=k+1)
        # grid_image.save(f"./samples/{k}.png")
        # images[0].save(f"./samples/tmp.gif", save_all=True, append_images=images[1:], loop=0, duration=500)
        # torch.cuda.empty_cache()


def image_grid(images, rows, cols):
    from PIL import Image
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    
    return grid

if __name__ == "__main__":
    images = []
    for k in trange(1, 32):
        pred_image = sanity_check(
            unet_path="/home/jiachun/codebase/vid_gen/experiment/10/10-unet-52",
            data_path="/data/vsd_data/captioned_8s_64f_motion",
            id=3,
            k=k,
        )
        images.append(pred_image)
    images[0].save(f"./samples/tmp_3_30.gif", save_all=True, append_images=images[1:], loop=0, duration=30)
    images[0].save(f"./samples/tmp_3_62.gif", save_all=True, append_images=images[1:], loop=0, duration=62)
    # for i in range(10):
    #     sample_ar(
    #         unet_path="/home/jiachun/codebase/vid_gen/experiment/10/10-unet-52",
    #         guidance_scale = 8,
    #         # data_path="/data/vsd_data/captioned_OpenVid/",
    #         data_path="/data/vsd_data/captioned_8s_64f_motion",
    #         # data_path="/data/jiachun/captioned_nonoverlap/",
    #         # data_path="/data/vsd_data/captioned_OpenVid_p2/",
    #         id=3,
    #         # prompt="A boat sailing in the sea",
    #         # prompt="a train moving slowly on a bridge.",
    #         # prompt="Two individuals are walking in the snowy woods.",
    #         iter=40000,
    #     )