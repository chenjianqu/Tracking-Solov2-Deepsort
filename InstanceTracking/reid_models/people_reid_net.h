//
// Created by chen on 2023/2/13.
//

#ifndef TRACKING_SOLOV2_DEEPSORT_PEOPLE_REID_NET_H
#define TRACKING_SOLOV2_DEEPSORT_PEOPLE_REID_NET_H

#include <vector>
#include <string>
#include <torch/torch.h>

struct NetImpl : torch::nn::Module {
public:
    NetImpl();

    torch::Tensor forward(torch::Tensor x);

    void load_form(const std::string &bin_path);

private:
    torch::nn::Sequential conv1{nullptr}, conv2{nullptr};
};

TORCH_MODULE(Net);


#endif //TRACKING_SOLOV2_DEEPSORT_PEOPLE_REID_NET_H
