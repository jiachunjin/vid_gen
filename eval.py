import os
import torch
from safetensors.torch import load_file

def sample_with_guidance(ckpt_path, prompt="", timesteps=30, num_samples=5, guidance_scale_context=None, guidance_scale_txt=None, examine_x0=False):
    from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler
    from diffusers.image_processor import VaeImageProcessor
    from train import get_models, parse_args
    from tqdm.auto import tqdm

    device = torch.device("cuda:0")
    args = parse_args()
    args.con_type = "qformer"
    args.con_depth = 20
    args.con_nframes = 64
    args.con_heads = 16
    args.con_numq = 512
    args.con_dim_head = 256
    args.interface = "new"

    unet, _, tokenizer, text_encoder = get_models(args)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(args.sd_path, "scheduler"))
    vae = AutoencoderKL.from_pretrained(os.path.join(args.sd_path, "vae"))
    image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)

    unet_state_path = os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors")
    unet_state_dict = load_file(unet_state_path)
    unet.load_state_dict(unet_state_dict)
    # ckpt_path_comp = ckpt_path.split("-")
    # global_steps = ckpt_path_comp[-1]

    dtype = torch.float16
    unet = unet.to(device, dtype)
    text_encoder = text_encoder.to(device, dtype)
    vae = vae.to(device, dtype)
    unet.eval()

    text_input_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
    txt_embedding = text_encoder(text_input_ids.to(device), return_dict=False)[0].detach()

    null_input_ids = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
    null_embedding = text_encoder(null_input_ids.to(device), return_dict=False)[0].detach()

    f = 64

    # from diffusers import DDPMScheduler
    # denoising_start_index = 399
    # noise_scheduler = DDPMScheduler.from_pretrained("/home/jiachun/codebase/diffusers/examples/text_to_image/ckpts/", subfolder="scheduler")
    # latents = torch.load(f"/data/vsd_data/captioned_OpenVid_p2/9")["latent"].to(device) * 0.18215
    # print(latents.shape)
    # noise = torch.randn_like(latents, device=device)
    # timesteps = torch.as_tensor([denoising_start_index], device=device, dtype=torch.int64)
    # latents = noise_scheduler.add_noise(latents, noise, timesteps)

    for j in range(num_samples):
        torch.cuda.empty_cache()
        scheduler.set_timesteps(timesteps)
        print(f"scheduler: {scheduler.__class__.__name__}")
        print(f"prompt: {prompt}")
        latents = torch.randn((f, 4, 64, 64), device=device)
        latents *= scheduler.init_noise_sigma
        for t in tqdm(scheduler.timesteps):
            # if t > denoising_start_index:
            #     continue
            latents = scheduler.scale_model_input(latents, t)
            with torch.no_grad(), torch.cuda.amp.autocast():
                t_sample = torch.as_tensor([t], device=device)

                noise_pred = unet(
                    sample=latents,
                    timestep=t_sample.repeat(f),
                    encoder_hidden_states=txt_embedding.repeat(f, 1, 1).to(device),
                    contexts=latents.unsqueeze(dim=0),
                ).sample

                noise_pred_null = unet(
                    sample=latents,
                    timestep=t_sample.repeat(f),
                    encoder_hidden_states=null_embedding.repeat(f, 1, 1).to(device),
                    contexts=latents.unsqueeze(dim=0),
                ).sample

                pred_null = unet(
                    sample=torch.cat((latents, latents), dim=0),
                    timestep=t_sample.repeat(2*f),
                    encoder_hidden_states=torch.cat((txt_embedding.repeat(f, 1, 1), null_embedding.repeat(f, 1, 1))).to(device),
                ).sample
                noise_pred_uncond, noise_pred_uncond_null = pred_null.chunk(2)
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # ##### test #####

                # noise_pred_uncond_null = unet(
                #     sample=latents,
                #     timestep=t_sample.repeat(f),
                #     encoder_hidden_states=null_embedding.repeat(f, 1, 1).to(device),
                # ).sample

                # noise_pred_null = noise_pred_uncond_null + 1.5 * guidance_scale * (noise_pred_null - noise_pred_uncond_null)

                # noise_pred = noise_pred_null + guidance_scale * (noise_pred - noise_pred_null)
                gd_1 = gd_2 = guidance_scale_context
                gd_3 = guidance_scale_txt
                noise_pred = (1 - gd_2 - gd_3 + gd_2 * gd_3) * noise_pred_uncond_null + \
                (gd_3 - gd_1 * gd_3) * noise_pred_uncond + \
                (gd_2 - gd_2 * gd_3) * noise_pred_null + \
                (gd_1 * gd_3) * noise_pred
                
                # ##### test end #####

                latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / vae.config.scaling_factor * latents.squeeze()
        vae.enable_slicing()
        image = vae.decode(latents, return_dict=False)[0]
        torch.cuda.empty_cache()
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        images = image_processor.numpy_to_pil(image)
        if len(prompt) > 50:
            prompt_out = prompt[:50]
        else:
            prompt_out = prompt
        images[0].save(f"./samples/new_{prompt_out}_{timesteps}_{j}.gif", save_all=True, append_images=images[1:], loop=0, duration=65)


if __name__ == "__main__":
    with torch.no_grad(), torch.cuda.amp.autocast():
        sample_with_guidance(
            "/home/jiachun/codebase/vid_gen/experiment/scale_up/scale_up-unet-0k",
            # "/home/jiachun/codebase/vid_gen/experiment/new_interface/new_interface-unet-78k",
            # prompt = "",
            prompt = "A sailing boat on the sea, under a red sky.",
            timesteps = 50,
            # guidance_scale_context = 1.2,
            guidance_scale_context = 1.1,
            guidance_scale_txt = 8.,
            examine_x0 = False,
        )

# 1. context tokens not enough
# 2. too deep for the conditioner
# 3. weighting of large t is too small
