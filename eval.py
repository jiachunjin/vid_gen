import os
import torch
from safetensors.torch import load_file

def sample_with_guidance(ckpt_path, prompt="", guidance_scale=None, examine_x0=False, j=None):
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    from train import get_models, parse_args
    from tqdm.auto import tqdm

    device = torch.device("cuda:7")
    args = parse_args()
    args.con_type = "qformer"
    args.con_depth = 2
    args.con_nframes = 64

    unet, scheduler, tokenizer, text_encoder = get_models(args)
    vae = AutoencoderKL.from_pretrained(os.path.join(args.sd_path, "vae"))
    image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)

    unet_state_path = os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors")
    unet_state_dict = load_file(unet_state_path)
    unet.load_state_dict(unet_state_dict)

    unet = unet.to(device)
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    unet.eval()

    text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_input_ids = text_inputs.input_ids
    txt_embedding = text_encoder(text_input_ids.to(device), return_dict=False)[0].detach()

    timesteps = 50
    scheduler.set_timesteps(timesteps)
    f = 64
    latents = torch.randn((f, 4, 64, 64), device=device)
    latents *= scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps):
        latents = scheduler.scale_model_input(latents, t)
        with torch.no_grad(), torch.cuda.amp.autocast():
            t_sample = torch.as_tensor([t], device=device)

            noise_pred = unet(
                sample=latents,
                timestep=t_sample.repeat(f),
                encoder_hidden_states=txt_embedding.repeat(f, 1, 1).to(device),
                contexts=latents.unsqueeze(dim=0),
            ).sample
            if examine_x0:
                break
            noise_pred_uncond = unet(
                sample=latents,
                timestep=t_sample.repeat(f),
                encoder_hidden_states=txt_embedding.repeat(f, 1, 1).to(device),
            ).sample

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    print(f"{torch.cuda.max_memory_allocated(device)/1024/1024} MB")

    latents = 1 / vae.config.scaling_factor * latents.squeeze()
    vae.enable_slicing()
    image = vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()

    images = image_processor.numpy_to_pil(image)
    
    prefix = "sample" if not examine_x0 else "mean"

    images[0].save(f"./samples/{prefix}_{guidance_scale}_{prompt}_{j}.gif", save_all=True, append_images=images[1:], loop=0, duration=30)

if __name__ == "__main__":
    with torch.no_grad():
        for j in range(10):
            sample_with_guidance(
                "/home/jiachun/codebase/vid_gen/experiment/q2000_00txt/q2000_00txt-unet-3k",
                guidance_scale = 1.5,
                examine_x0 = False,
                j=j,
            )