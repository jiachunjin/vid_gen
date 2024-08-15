import os
import torch
import math
import numpy as np
import torch.nn as nn

class captioned_video():
    def __init__(self, path, subset=None) -> None:
        self.path = path
        if subset is not None:
            self.length = subset
        else:
            self.length = len(os.listdir(self.path))
        print(f"dataset size: {self.length}")
    
    def __getitem__(self, index):
        z = torch.load(os.path.join(self.path, str(index))) # 2500
        latent = z["latent"]
        caption = z["caption"]
        return latent, len(latent), caption

    def __len__(self):
        return self.length
