//
// Created by chen on 2021/11/7.
//

#ifndef INSTANCE_SEGMENT_INFER_H
#define INSTANCE_SEGMENT_INFER_H

#include <optional>
#include <memory>

#include "common.h"
#include "pipeline.h"
#include "solo.h"
#include "buffer.h"

#include <NvInfer.h>



struct InferDeleter{
    template <typename T>
    void operator()(T* obj) const{
        if (obj)
            obj->destroy();
    }
};

class Infer {
public:
    using Ptr = std::shared_ptr<Infer>;
    Infer();
    void forward_tensor(cv::Mat &img,torch::Tensor &mask_tensor,std::vector<InstInfo> &insts);
    void visualizeResult(cv::Mat &input,cv::Mat &mask,std::vector<InstInfo> &insts);

private:
    MyBuffer::Ptr buffer;
    Pipeline::Ptr pipeline;
    Solov2::Ptr solo;

    std::unique_ptr<nvinfer1::IRuntime,InferDeleter> runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<IExecutionContext, InferDeleter> context;

    double infer_time{0};
};


#endif //
