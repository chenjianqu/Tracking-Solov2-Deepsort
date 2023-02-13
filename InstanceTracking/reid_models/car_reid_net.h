/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_CAR_REIDNET_H
#define DYNAMIC_VINS_CAR_REIDNET_H

#include <vector>
#include <string>
#include <torch/torch.h>

struct CarReIdNetImpl : torch::nn::Module {
public:
    CarReIdNetImpl();

    torch::Tensor forward(torch::Tensor x);

    void load_form(const std::string &bin_path);

private:
    torch::nn::Sequential conv1{nullptr}, conv2{nullptr};
};

TORCH_MODULE(CarReIdNet);


#endif //DYNAMIC_VINS_REIDNET_H
