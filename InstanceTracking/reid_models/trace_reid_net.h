//
// Created by chen on 2023/2/13.
//

#ifndef TRACKING_SOLOV2_DEEPSORT_TRACE_REID_NET_H
#define TRACKING_SOLOV2_DEEPSORT_TRACE_REID_NET_H

#include <vector>
#include <string>
#include <torch/torch.h>

class TraceReidNet {
public:
    torch::Tensor forward(torch::Tensor x);

    torch::Tensor operator()(torch::Tensor x);

    void load_form(const std::string &bin_path);

private:
    torch::jit::Module model;

};


#endif //TRACKING_SOLOV2_DEEPSORT_TRACE_REID_NET_H
