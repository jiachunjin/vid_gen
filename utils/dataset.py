import os
import torch

class captioned_video():
    def __init__(self, path, subset=None, expected_frames=64) -> None:
        self.path = path
        self.mapping = {}
        if subset is not None:
            accept = 0
            curr_index = 0
            while accept < subset:
                if os.path.exists(os.path.join(self.path, str(curr_index))):
                    if torch.load(os.path.join(self.path, str(curr_index)))["length"] == expected_frames:
                        self.mapping[accept] = curr_index
                        accept += 1
                else:
                    raise ValueError(f"Dataloader: Expect {subset} files, but only found {accept} files")
                curr_index += 1
            self.length = subset
            print(f"Dataloader: {self.length} acceptable files found, {curr_index} files checked")
        else:
            accept = 0
            curr_index = 0
            while True:
                if os.path.exists(os.path.join(self.path, str(curr_index))):
                    if torch.load(os.path.join(self.path, str(curr_index)))["length"] == expected_frames:
                        self.mapping[accept] = curr_index
                        accept += 1
                else:
                    break
                curr_index += 1
            self.length = accept
            print(f"Dataloader: {self.length} acceptable files found")
    
    def __getitem__(self, index):
        index = self.mapping[index]
        z = torch.load(os.path.join(self.path, str(index)))
        latent = z["latent"]
        caption = z["caption"]
        len_v = z["length"]
        return latent, len_v, caption

    def __len__(self):
        return self.length


# import os
# import torch

# class captioned_video():
#     def __init__(self, path, subset=None) -> None:
#         self.path = path
#         if subset is not None:
#             self.length = subset
#         else:
#             self.length = len(os.listdir(self.path))
    
#     def __getitem__(self, index):
#         z = torch.load(os.path.join(self.path, str(index)))
#         latent = z["latent"]
#         caption = z["caption"]
#         return latent, len(latent), caption

#     def __len__(self):
#         return self.length