/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Tracking_Solov2_Deepsort.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef INSTANCE_SEGMENT_BUFFER_H
#define INSTANCE_SEGMENT_BUFFER_H

#include <string>
#include <memory>
#include <NvInfer.h>


class MyBuffer{
public:
    using Ptr = std::shared_ptr<MyBuffer>;
    explicit MyBuffer(nvinfer1::ICudaEngine& engine);
    ~MyBuffer();

    void cpyInputToGPU();
    void cpyOutputToCPU();

    cudaStream_t stream{};
    int binding_num;
    std::string names[12];
    nvinfer1::Dims dims[12]{};
    int size[12]{};
    float **cpu_buffer = new float* [12];//指针数组，包含输入和输出的buffer
    void *gpu_buffer[12]{}; //定义指针数组，用于指定GPU上的输入输出缓冲区
};


#endif //DYNAMIC_VINS_BUFFER_H
