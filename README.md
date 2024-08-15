# Video Gen README

## Dataset Downloading
Dataset: OpenVid-1M, https://github.com/NJU-PCALab/OpenVid-1M/tree/main

Download OpenVid-1M with the official instructions on GitHub. Please refer to the ```Preparation``` part in https://github.com/NJU-PCALab/OpenVid-1M/blob/main/README.md . Install the environment and run the official download script to download and unzip the OpenVid-1M dataset.


## Environment Setup
Download the base stable-diffusion checkpoints with
```
cd ckpts/
export HF_ENDPOINT=https://hf-mirror.com (用于中国大陆服务器) 
huggingface-cli download --resume-download ByteDance/sd2.1-base-zsnr-laionaes5 --local-dir zsnr
```
Setup the Python environment:
```
conda env create -f environment.yaml
conda activate vid_gen
```

## Preprocessing
There are two steps in preprocessing part: Use ```llama-3``` to rewrite the caption and encode the videos into ```vae latents```.

1. 


## Experiment Launching
1. Run ```accelerate config``` to setup distributed training
    - "multi-GPU", "fp16"
2. Init wandb
    - ```wandb.login()```
    - Jiachun's wandb token is ```96131f8aede9a09cdcdaecc19c054f804e330d3d```
3. Fill ```ar.sh``` with the data path
    - ```data_dir``` in ```ar.sh``` should be the directory to the processed data in the "Preprocessing" section
    - modify ```train_batch_size``` to the proper size to better utilize the GPUs
5. Start to train
    - ```nohup sh ar.sh > log.txt 2>&1 &```

## Evluation
Sending the saved checkpoints to Jiachun? Do we have better options?