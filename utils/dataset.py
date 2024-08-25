import os
import torch

class captioned_video():
    def __init__(self, path, subset=None) -> None:
        self.path = path
        if subset is not None:
            self.length = subset
        else:
            self.length = len(os.listdir(self.path))
    
    def __getitem__(self, index):
        z = torch.load(os.path.join(self.path, str(index)))
        latent = z["latent"]
        caption = z["caption"]
        return latent, len(latent), caption

    def __len__(self):
        return self.length