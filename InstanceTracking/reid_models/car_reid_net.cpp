/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "car_reid_net.h"
#include <fstream>

#include "../../parameters.h"
#include "../../utils.h"

namespace nn = torch::nn;

namespace {
    struct BasicBlockImpl : nn::Module {
        explicit BasicBlockImpl(int64_t c_in, int64_t c_out, bool is_downsample = false) {
            conv = register_module(
                    "conv",
                    nn::Sequential(
                            nn::Conv2d(nn::Conv2dOptions(c_in, c_out, 3)
                            .stride(is_downsample ? 2 : 1)
                            .padding(1).bias(false)),
                            nn::BatchNorm2d(c_out),
                            nn::Functional(torch::relu),
                            nn::Conv2d(nn::Conv2dOptions(c_out, c_out, 3)
                            .stride(1).padding(1).bias(false)),
                            nn::BatchNorm2d(c_out)));

            if (is_downsample) {
                downsample = register_module(
                        "downsample",
                        nn::Sequential(nn::Conv2d(nn::Conv2dOptions(c_in, c_out, 1)
                        .stride(2).bias(false)),
                                       nn::BatchNorm2d(c_out)));
            } else if (c_in != c_out) {
                downsample = register_module(
                        "downsample",
                        nn::Sequential(nn::Conv2d(nn::Conv2dOptions(c_in, c_out, 1)
                        .stride(1).bias(false)),
                                       nn::BatchNorm2d(c_out)));
            }
        }

        torch::Tensor forward(torch::Tensor x) {
            auto y = conv->forward(x);
            if (!downsample.is_empty()) {
                x = downsample->forward(x);
            }
            return torch::relu(x + y);
        }

        nn::Sequential conv{nullptr}, downsample{nullptr};
    };

    TORCH_MODULE(BasicBlock);

    void load_tensor(torch::Tensor t, std::ifstream &fs) {
        fs.read(static_cast<char *>(t.data_ptr()), t.numel() * sizeof(float));
    }

    void load_Conv2d(nn::Conv2d m, std::ifstream &fs) {
        load_tensor(m->weight, fs);
        if (m->options.bias()) {
            load_tensor(m->bias, fs);
        }
    }

    void load_BatchNorm(nn::BatchNorm2d m, std::ifstream &fs) {
        load_tensor(m->weight, fs);
        load_tensor(m->bias, fs);
        load_tensor(m->running_mean, fs);
        load_tensor(m->running_var, fs);
    }

    void load_Sequential(nn::Sequential s, std::ifstream &fs) {
        if (s.is_empty()) return;
        for (auto &m:s->children()) {
            if (auto c = std::dynamic_pointer_cast<nn::Conv2dImpl>(m)) {
                load_Conv2d(c, fs);
            } else if (auto b = std::dynamic_pointer_cast<nn::BatchNorm2dImpl>(m)) {
                load_BatchNorm(b, fs);
            }
        }
    }

    nn::Sequential make_layers(int64_t c_in, int64_t c_out, size_t repeat_times, bool is_downsample = false) {
        nn::Sequential ret;
        for (size_t i = 0; i < repeat_times; ++i) {
            ret->push_back(BasicBlock(i == 0 ? c_in : c_out, c_out, i == 0 ? is_downsample : false));
        }
        return ret;
    }
}


CarReIdNetImpl::CarReIdNetImpl() {
    conv1 = register_module("conv1",
                            nn::Sequential(
                                    nn::Conv2d(nn::Conv2dOptions(3, 64, 3)
                                    .stride(1).padding(1)),
                                    nn::BatchNorm2d(64),
                                    nn::Functional(torch::relu)));
    conv2 = register_module("conv2", nn::Sequential());
    conv2->extend(*make_layers(64, 64, 2, false));
    conv2->extend(*make_layers(64, 128, 2, true));
    conv2->extend(*make_layers(128, 256, 2, true));
    conv2->extend(*make_layers(256, 512, 2, true));
}


torch::Tensor CarReIdNetImpl::forward(torch::Tensor x) {
    sgLogger->debug("forword0 {}",DimsToStr(x.sizes()));
    x = conv1->forward(x);
    sgLogger->debug("forword1 {}",DimsToStr(x.sizes()));
    x = torch::max_pool2d(x, 3, 2, 1);
    sgLogger->debug("forword2 {}",DimsToStr(x.sizes()));
    x = conv2->forward(x);
    sgLogger->debug("forword3 {}",DimsToStr(x.sizes()));
    x = torch::avg_pool2d(x, {4, 8}, 1);
    sgLogger->debug("forword4 {}",DimsToStr(x.sizes()));
    x = x.view({x.size(0), -1});
    sgLogger->debug("forword5 {}",DimsToStr(x.sizes()));
    x.div_(x.norm(2, 1, true));
    sgLogger->debug("forword6 {}",DimsToStr(x.sizes()));
    return x;
}

void CarReIdNetImpl::load_form(const std::string &bin_path) {
    std::ifstream fs(bin_path, std::ios_base::binary);

    load_Sequential(conv1, fs);

    for (auto &m:conv2->children()) {
        auto b = std::static_pointer_cast<BasicBlockImpl>(m);
        load_Sequential(b->conv, fs);
        load_Sequential(b->downsample, fs);
    }

    fs.close();
}



