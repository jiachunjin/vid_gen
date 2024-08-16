import os
import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
from diffusers import AutoencoderKL
import tqdm
import math
import pandas as pd
import pickle
import torch.multiprocessing as multiprocessing
import time

progress_bar = None
varlock = multiprocessing.Lock()

target_fps = 16
target_frames = 64
target_seconds = 4
resolution = (512, 512)

def read_video_to_tensor(file_path):
    cap = cv2.VideoCapture(file_path)
    curr_fps = cap.get(cv2.CAP_PROP_FPS)
    curr_frames = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    curr_seconds = curr_frames / curr_fps
    frames_to_read = math.ceil(curr_fps * target_seconds)
    
    frames = []
    if cap.isOpened():
        if curr_frames >= frames_to_read:
            for _ in range(frames_to_read):
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame)
                frames.append(frame_tensor)
        else:
            for _ in range(curr_frames):
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame)
                frames.append(frame_tensor)
        

    cap.release()

    if curr_frames >= frames_to_read:
        picked_indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
    else:
        picked_indices = np.linspace(0, len(frames) - 1, math.floor(target_fps * curr_seconds), dtype=int)
        
    frames = [frames[i] for i in picked_indices]
    len_v = len(frames)
    frames = torch.stack(frames)    
    frames = frames.permute(0, 3, 1, 2)

    transform = transforms.Compose([
        transforms.Resize(resolution),
    ])
    frames = transform(frames) / 255.
    frames = frames * 2. - 1.

    return frames, len_v

def process_video(data_path: str, caption_dict_path: str, vae_path: str, output_path: str, mp4s: list, current_index: int, captioned_csv, device: str = "cuda:7"):
    with torch.no_grad():
        caption_dict = pickle.load(open(caption_dict_path, "rb"))

        vae = AutoencoderKL.from_pretrained(vae_path)
        vae.requires_grad_(False)
        vae = vae.to(device)
        
        for mp4 in mp4s:
            caption = caption_dict[mp4]
            video_path = os.path.join(data_path, mp4)

            v, len_v = read_video_to_tensor(video_path)
            v = v.to(device)

            def encode_video(l, r):
               
                if len_v >= r:
                    encoder_posterior = vae.encode(v[l:r]).latent_dist
                    encoder_posterior = encoder_posterior.mean
                elif len_v < r and len_v >= l:
                    encoder_posterior1 = vae.encode(v[l:]).latent_dist
                    encoder_posterior1 = encoder_posterior1.mean
                    encoder_posterior2 = torch.zeros((r - len_v, 4, 64, 64), device=device)
                    encoder_posterior = torch.cat((encoder_posterior1, encoder_posterior2), dim=0)
                else:
                    encoder_posterior = torch.zeros((r - l, 4, 64, 64), device=device)
            
                return encoder_posterior

            step = 16
            encoder_posteriors = [encode_video(i, i + step) for i in range(0, 64, step)]
            
            mode = torch.cat(encoder_posteriors, dim=0).cpu()

            data = {
                "latent": mode,
                "caption": caption,
                "length": len_v,
            }

            with varlock:
                # print(f"work done by {device}, shape of mode: {mode.shape}")
                with current_index.get_lock():
                    torch.save(data, os.path.join(output_path, f"{current_index.value}"))

                    captioned_csv.append((current_index.value, caption, mp4, len_v))

                    with open(os.path.join(output_path, "captioned_OpenVid.txt"), "a") as f:
                        f.write(f"{current_index.value},\"{caption}\",{mp4}, {len_v}\n")

                    current_index.value += 1
                    progress_bar.n = current_index.value
                    progress_bar.refresh()
        

def main():
    ### Configurations
    cudas = [0, 1, 2] # iterable[int]
    data_path = "--data_path" # /xxx/OpenVid-1M.csv
    vae_path = "--vae_path" # /zsnr/vae/
    caption_dict_path = "--caption_dict_path" # /xxx/caption_dict.pkl
    output_path = "--output_path"
    ###

    st = time.time()

    cudas = [f"cuda:{num}" for num in cudas]
    threads_count = len(cudas)
    print(f"Threads count: {threads_count}")

    # Read video list
    mp4s = sorted(os.listdir(data_path))
    
    total_tasks = len(mp4s)
    global progress_bar
    progress_bar = tqdm.tqdm(total=total_tasks, desc="Processing videos", position=0) 

    # Clear the tracking file
    with open(os.path.join(output_path, "captioned_OpenVid.txt"), "w") as f:
        f.write("filename,caption,mp4_filename,length\n")

    # Encode videos
    with multiprocessing.Manager() as manager:
        current_index = multiprocessing.Value("i", 0)

        processes = []
        captioned_csv = manager.list()
        for i in range(threads_count):
            p = multiprocessing.Process(target=process_video, args=(data_path, 
                                                                    caption_dict_path, 
                                                                    vae_path, 
                                                                    output_path, 
                                                                    mp4s[i::threads_count], 
                                                                    current_index,
                                                                    captioned_csv,
                                                                    cudas[i]))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print("Saving captioned csv...")

        captioned_csv = list(captioned_csv)
        df = pd.DataFrame(captioned_csv, columns=["filename", "caption", "mp4_filename", "length"])
        df.to_csv(os.path.join(output_path, "captioned_OpenVid.csv"), index=False)

        print("Captioned csv saved.")
        print(f"Time taken: {time.time() - st} seconds, threads: {threads_count}")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()