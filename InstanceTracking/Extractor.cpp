/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Tracking_Solov2_Deepsort.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "Extractor.h"
#include "../parameters.h"


Extractor::Extractor() {
    if(Config::use_trace_model){
        trace_net.load_form(Config::EXTRACTOR_MODEL_PATH);
    }
    else{
        net->load_form(Config::EXTRACTOR_MODEL_PATH);
        net->to(torch::kCUDA);
        net->eval();
    }
}

torch::Tensor Extractor::extract(vector<cv::Mat> input) {
    if (input.empty()) {
        return torch::empty({0, 512});
    }

    torch::NoGradGuard no_grad;

    static const auto MEAN = torch::tensor({0.485f, 0.456f, 0.406f}).view({1, -1, 1, 1}).cuda();
    static const auto STD = torch::tensor({0.229f, 0.224f, 0.225f}).view({1, -1, 1, 1}).cuda();

    vector<torch::Tensor> resized;
    for (auto &x:input) {
        cv::resize(x, x, {Config::kReidImgWidth, Config::kReidImgHeight});
        cv::cvtColor(x, x, cv::COLOR_RGB2BGR);
        x.convertTo(x, CV_32F, 1.0 / 255);
        resized.push_back(torch::from_blob(x.data, {Config::kReidImgHeight,Config::kReidImgWidth, 3}));
    }
    auto tensor = torch::stack(resized).cuda().permute({0, 3, 1, 2}).sub_(MEAN).div_(STD);

    //sgLogger->debug("input net tensor:{}x{}x{}x{}",tensor.size(0),tensor.size(1),tensor.size(2),tensor.size(3));

    if(Config::use_trace_model){
        return trace_net(tensor);
    }
    else{
        return net(tensor);
    }
}