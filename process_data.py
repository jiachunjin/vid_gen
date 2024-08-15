import os
import torch
import imageio
from torchvision import transforms
from diffusers import AutoencoderKLCogVideoX, AutoencoderKL

device = torch.device("cuda:7")
path_2d = "/home/jiachun/codebase/vsd/ckpts/zsnr"
path_3d = "/home/jiachun/codebase/vsd/ckpts/CogVideoX-2b"
vae = AutoencoderKL.from_pretrained(path_2d, subfolder="vae", torch_dtype=torch.float16).to(device)
vae_3d = AutoencoderKLCogVideoX.from_pretrained(path_3d, subfolder="vae", torch_dtype=torch.float16).to(device)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
])

video_list = os.listdir("/home/jiachun/codebase/vsd/data/new_caption")
print(video_list)

for index, video in enumerate(video_list):
    caption = video.split(".")[0]
    print(caption)
    video_path = os.path.join("/home/jiachun/codebase/vsd/data/new_caption", video)
    frames = []
    video_reader = imageio.get_reader(video_path, "ffmpeg")
    for i, frame in enumerate(video_reader):
        if i % 4 != 0: continue
        frame = transform(frame)
        frames.append(frame)
        if len(frames) == 9:
            break
    video_reader.close()
    condition_frames = frames[:-1]
    condition_frames = torch.stack(condition_frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(torch.float16)
    with torch.no_grad():
        condition = vae_3d.encode(condition_frames)[0].sample().squeeze()
        frame = frames[-1].unsqueeze(0).to(torch.float16).to(device)
        frame = frame * 2. - 1.
        frame = vae.encode(frame).latent_dist.mean
    # print(condition.shape, frame.shape, condition.device, frame.device)
    # print("==================")
    data = {
        "frame": frame.cpu(),
        "condition": condition.cpu(),
        "caption": caption,
    }
    torch.save(data, f'./data/{index}')