#########################################################
# Copyright (C) 2022, Chen Jianqu, Shanghai University
#
# This file is part of interact_slam.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#########################################################

import os
import torch
import numpy as np




model_path="./ckpt.t7"

state_dict = torch.load(model_path)["net_dict"]

with open("ckpt.bin","wb") as f:
    for key in state_dict.keys():
        f.write(state_dict[key].cpu().numpy())

