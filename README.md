# Video Gen README

## Dataset Downloading
Dataset: OpenVid-1M, https://github.com/NJU-PCALab/OpenVid-1M/tree/main

Download OpenVid-1M with the official instructions on GitHub. Please refer to the ```Preparation``` part in https://github.com/NJU-PCALab/OpenVid-1M/blob/main/README.md . Install the environment and run the official download script to download and unzip the OpenVid-1M dataset.


## Environment Setup
Download the base stable-diffusion checkpoints with
```
cd ckpts/
export HF_ENDPOINT=https://hf-mirror.com (ignore this line if the server can visit huggingface directly) 
huggingface-cli download --resume-download ByteDance/sd2.1-base-zsnr-laionaes5 --local-dir zsnr
```
Setup the Python environment:
```
conda env create -f environment.yaml
conda activate vid_gen
```

## Preprocessing
There are two steps in preprocessing part: Use ```llama-3``` to rewrite the caption and encode the videos into ```vae latents```.

**Step 1:** ```preprocess_prompts.py```
Before executing, config ```cudas``` and ```csv_path``` in ```line 67``` and ```line 68```.
Execute ```python preprocess_prompts.py``` to use ```llama-3``` to generate rewrite ```caption_dict.pkl```. 

**Remaining questions:** Is it possible to run llama-3 on batch-parallel level? One model requires 16+G VRAM, then is it possible to run multiple models on one GPU and boost the preprocessing?

Bingde: If you do not have a ```llama-3``` access token, you may apply for one on https://huggingface.co/meta-llama/Meta-Llama-3-8B or pass an argument```token=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``` in ```line 24, AutoModelForCausalLM.from_pretrained``` to use my token.

**Step 2:** ```preprocess_OpenVid.py```
This step involves ```vae``` to encode videos into ```latents```.
[the code will be pushed later]

## Experiment Launching
1. Run ```accelerate config``` to setup distributed training
    - "multi-GPU", "fp16", others remain default
2. Init wandb
    - ```wandb.login()```
    - Jiachun's wandb token is ```96131f8aede9a09cdcdaecc19c054f804e330d3d```
3. Fill ```ar.sh``` with the data path
    - ```data_dir``` in ```ar.sh``` should be the directory to the processed data in the "Preprocessing" section
    - comment line 3 ```export HF_ENDPOINT=https://hf-mirror.com``` if the server can visit huggingface directly
    - modify ```train_batch_size``` to the proper size to better utilize the GPUs
4. Start to train
    - ```nohup sh ar.sh > log.txt 2>&1 &```

## Evaluation
We will use the checkpoints uploaded to huggingface to evaluate.