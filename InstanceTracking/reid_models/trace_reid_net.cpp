//
// Created by chen on 2023/2/13.
//

#include "trace_reid_net.h"
#include <torch/script.h>





torch::Tensor TraceReidNet::forward(torch::Tensor x){
    return model.forward({ x }).toTensor();
}

torch::Tensor TraceReidNet::operator()(torch::Tensor x){
    return forward(x);
}


void TraceReidNet::load_form(const std::string &model_path){
    model = torch::jit::load(model_path);
    model.eval();
    model.to(at::kCUDA);
}



