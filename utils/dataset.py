import os
import torch
import pickle
import time
import torch.multiprocessing as multiprocessing

SAVE_NAME = "index_mapping.pkl"
varlock = multiprocessing.Lock()

def scan_atomic(files, path, accept, mapping, expected_frames=64):
    for file in files:
        if torch.load(os.path.join(path, file))["length"] == expected_frames:
        # if torch.load(os.path.join(path, file)):
            with accept.get_lock():
                mapping[accept.value] = file
                accept.value += 1

class captioned_video():
    def __init__(self, path, subset=None, expected_frames=64, threads_count=20) -> None:
        self.path = path
        current_dir = os.path.dirname(os.path.realpath(__file__))

        if os.path.exists(os.path.join(current_dir, SAVE_NAME)):
            print(f"Dataloader: Found index mapping file, loading...")
            self.mapping = pickle.load(open(os.path.join(current_dir, SAVE_NAME), "rb"))
            self.capacity = len(self.mapping)
            print(f"Dataloader: {self.capacity} acceptable files found")
        else:
            with multiprocessing.Manager() as manager:
                self.mapping = manager.dict()

                accept = multiprocessing.Value("i", 0)
                files = sorted(os.listdir(self.path))

                processes = []
                for i in range(threads_count):
                    p = multiprocessing.Process(target=scan_atomic, args=(files[i::threads_count], self.path, accept, self.mapping))
                    processes.append(p)
                    p.start()
                
                for p in processes:
                    p.join()

                self.capacity = accept.value
                print(f"Dataloader: {self.capacity} acceptable files found")

                self.mapping = dict(self.mapping)
                pickle.dump(self.mapping, open(os.path.join(current_dir, SAVE_NAME), "wb"))
                print(f"Dataloader: Index mapping saved to {SAVE_NAME}")
        
        if subset is not None:
            if subset > self.capacity:
                raise ValueError(f"Dataloader: Expect {subset} files, but only found {self.capacity} files")
            self.length = subset
        else:
            self.length = self.capacity

    def __getitem__(self, index):
        target_index = self.mapping[index]
        z = torch.load(os.path.join(self.path, str(target_index)))
        latent = z["latent"]
        caption = z["caption"]
        len_v = z["length"]
        return latent, len_v, caption

    def __len__(self):
        return self.length

if __name__ == "__main__":
    path = "/data/vsd_data/captioned_OpenVid_768"

    st = time.time()
    dataset = captioned_video(path, subset=None, threads_count=10)
    print(f"Time elapsed: {time.time() - st} seconds")

### Previous version

# class captioned_video():
#     def __init__(self, path, subset=None, expected_frames=64) -> None:
#         self.path = path
#         self.mapping = {}
#         if subset is not None:
#             accept = 0
#             curr_index = 0
#             while accept < subset:
#                 if os.path.exists(os.path.join(self.path, str(curr_index))):
#                     if torch.load(os.path.join(self.path, str(curr_index)))["length"] == expected_frames:
#                         self.mapping[accept] = curr_index
#                         accept += 1
#                 else:
#                     raise ValueError(f"Dataloader: Expect {subset} files, but only found {accept} files")
#                 curr_index += 1
#             self.length = subset
#             print(f"Dataloader: {self.length} acceptable files found, {curr_index} files checked")
#         else:
#             accept = 0
#             curr_index = 0
#             while True:
#                 if os.path.exists(os.path.join(self.path, str(curr_index))):
#                     if torch.load(os.path.join(self.path, str(curr_index)))["length"] == expected_frames:
#                         self.mapping[accept] = curr_index
#                         accept += 1
#                 else:
#                     break
#                 curr_index += 1
#             self.length = accept
#             print(f"Dataloader: {self.length} acceptable files found")
    
#     def __getitem__(self, index):
#         index = self.mapping[index]
#         z = torch.load(os.path.join(self.path, str(index)))
#         latent = z["latent"]
#         caption = z["caption"]
#         len_v = z["length"]
#         return latent, len_v, caption

#     def __len__(self):
#         return self.length