
## Update 24.09.14
1. login huggingface with token: `xxx`
2. download the previous ckpt with:

``````bash
cd ckpts/
huggingface-cli download --resume-download ByteDance/sd2.1-base-zsnr-laionaes5 --local-dir zsnr
huggingface-cli download --resume-download orres/vid_gen --local-dir 112k
``````

3. modify `train.sh`
   - --sd_path "the path to zsnr/"
   - --data_dir "the path to the processed Open-VID"
   - --resume_path "the path to 112k/"
   - --train_batch_size (调整到合适的大小)
4. 过滤data_dir中的latents确保`len_v` == 64
5. https://github.com/jiachunjin/vid_gen/blob/60b3d66d1fa2a73d0168219c2ea66c9fe4c65074/train.py#L216 填入huggingface token `xxx`



## What's New?
We switch to plan A (more details in docs/doc.pdf)
1. Now we should filter out videos that are shorter than 64 frames (i.e. videos with padding in the end)
2. Command to launch experiments changes to ```nohup sh train.sh > log.txt 2>&1 &```

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

**Step 1:** ```preprocess_llama.py```

Before executing, config ```cudas``` to utilize the available GPUs and ```csv_path``` to access ```OpenVid-1M.csv```, which is acquired in **Dataset Downloading** part. (```line 67``` and ```line 68```)
Execute ```python preprocess_llama.py``` to use ```llama-3``` to generate recaption ```caption_dict.pkl```. 

**Remaining questions:** Is it possible to run llama-3 on batch-parallel level? One model requires 16+G VRAM, then is it possible to run multiple models on one GPU and boost the preprocessing? (an update: see the last sentence of this paragraph.)

Bingde: Currently the script is using my llama-3 token via passing the argument```token="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"``` in ```line 24```. Alternatively, you may apply for one on https://huggingface.co/meta-llama/Meta-Llama-3-8B.

**Step 2:** ```preprocess_vae_16fps.py```

This step involves ```vae``` to encode videos into ```latents``` with ```16fps``` and ```64 frames``` in total. Before executing, 
1. config ```cudas``` to utilize the available GPUs. (```line 123```)
2. config ```data_path``` to access the videos. There should be a folder containing all ```*.mp4``` files. (```line 124```)
3. config ```vae_path``` to access the pretrained ```vae``` model downloaded in **Environment Setup** part. (```line 125```)
4. config ```caption_dict_path``` to access ```caption_dict.pkl``` generated in **Step 1**. (```line 126```)
5. config a proper ```output_path``` to save all the ```vae latents```. (```line 127```)

**Remaining questions:** The variable ```step``` in ```line 95``` may affect the processing speed of ```vae```. A larger value like 32 or 64 may be faster (more VRAM required). 

By several preliminary testings, running multiple threads on one GPU or increasing the value of ```step``` are unlikely to boost the processing.


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