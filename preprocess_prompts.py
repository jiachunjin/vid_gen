from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import tqdm
import pickle
import os
import torch.multiprocessing as multiprocessing
import pandas as pd

progress_bar = None
varlock = multiprocessing.Lock()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

instruction = '''
    You are expected to rewrite the following prompt to make it no longer than 45 words while keeping the information and the essence of the original prompt.
    You should directly give the rewritten prompt in the chat without any additional information.
    That is, no such words like "Here is the rewritten prompt:".
    The original prompt is as follows: 
'''

def process_prompts(shared_dict, mp4s_and_caps: list, current_index, device: str = None):
    with torch.no_grad():
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16,).to(device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        for mp4, cap in mp4s_and_caps:
            content = instruction + cap

            messages = [{"role": "An AI Copilot", "content": content},]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model.device)

            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
            )

            with varlock:
                response = outputs[0][input_ids.shape[-1]:]
                response = tokenizer.decode(response, skip_special_tokens=True)
                response = response.strip("\" \n")
                shared_dict[mp4] = response
                
            with current_index.get_lock():
                current_index.value += 1
                progress_bar.n = current_index.value
                progress_bar.refresh()

def main():
    ### Configurations
    cudas = [0, 1, 2] # iterable[int]
    csv_path = "--csv_path" # /xxx/OpenVid-1M.csv
    ###

    st = time.time()

    cudas = [f"cuda:{num}" for num in cudas]
    threads_count = len(cudas)
    print(f"Threads count: {threads_count}")

    # Read CSV and convert to list
    df = pd.read_csv(csv_path)

    mp4s = df["path"]
    captions = df["text"]

    mp4s_and_caps = list(zip(mp4s, captions))

    # Progress bar
    total_tasks = len(captions)
    global progress_bar
    progress_bar = tqdm.tqdm(total=total_tasks, desc="Processing captions", position=0)
    
    # Process captions
    with multiprocessing.Manager() as manager:
        shared_dict = manager.dict()
        current_index = multiprocessing.Value("i", 0)

        processes = []
        for i in range(threads_count):
            p = multiprocessing.Process(target=process_prompts, args=(shared_dict, mp4s_and_caps[i::threads_count], current_index, cudas[i]))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()

        print("All threads finished.")

        # Save the shared_dict
        caption_dict = dict(shared_dict)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_dir, "caption_dict.pkl"), "wb") as f:
            pickle.dump(caption_dict, f)
        print("Caption rewrite dict saved.")
    
    print(f"Time taken: {time.time() - st} seconds, threads: {threads_count}")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main() 