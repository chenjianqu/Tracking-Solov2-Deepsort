/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Tracking_Solov2_Deepsort.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/



#include <spdlog/logger.h>
#include <cuda_runtime_api.h>

#include "buffer.h"
#include "common.h"
#include "../parameters.h"


MyBuffer::MyBuffer(nvinfer1::ICudaEngine& engine){
    ///申请输出buffer
    binding_num=engine.getNbBindings();
    for(int i=0;i<binding_num;++i){
        auto dim=engine.getBindingDimensions(i);
        dims[i]=dim;
        int buffer_size=dim.d[0]*dim.d[1]*dim.d[2]*dim.d[3]*sizeof(float);
        size[i]=buffer_size;
        cpu_buffer[i]=(float *)malloc(buffer_size);
        if(auto s=cudaMalloc(&gpu_buffer[i], buffer_size);s!=cudaSuccess)
            throw std::runtime_error(fmt::format("cudaMalloc failed, status:{}",s));
        names[i]=engine.getBindingName(i);
    }
    if(auto s=cudaStreamCreate(&stream);s!=cudaSuccess)
        throw std::runtime_error(fmt::format("cudaStreamCreate failed, status:{}",s));
}


MyBuffer::~MyBuffer(){
    cudaStreamDestroy(stream);
    for(int i=0;i<binding_num;++i){
        if(auto s=cudaFree(gpu_buffer[i]);s!=cudaSuccess)
            sgLogger->error("cudaFree failed, status:{}",s);
        delete cpu_buffer[i];
    }
    delete[] cpu_buffer;
}


void MyBuffer::cpyInputToGPU(){
    if(auto status = cudaMemcpyAsync(gpu_buffer[0], cpu_buffer[0], size[0], cudaMemcpyHostToDevice, stream);
        status != cudaSuccess)
        throw std::runtime_error(fmt::format("cudaMemcpyAsync failed, status:{}",status));

}

void MyBuffer::cpyOutputToCPU(){
    for(int i=1;i<binding_num;++i){
        if(auto status = cudaMemcpyAsync(cpu_buffer[i],gpu_buffer[i], size[i], cudaMemcpyDeviceToHost, stream);
        status != cudaSuccess)
            throw std::runtime_error(fmt::format("cudaMemcpyAsync failed, status:{}",status));
    }
    if(auto status=cudaStreamSynchronize(stream);status != cudaSuccess)
        throw std::runtime_error(fmt::format("cudaStreamSynchronize failed, status:{}",status));
}