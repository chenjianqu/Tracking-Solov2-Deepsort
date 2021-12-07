import os
import torch
import numpy as np

model_path="./ckpt.t7"

state_dict = torch.load(model_path)["net_dict"]

with open("ckpt.bin","wb") as f:
    for key in state_dict.keys():
        f.write(state_dict[key].cpu().numpy())

