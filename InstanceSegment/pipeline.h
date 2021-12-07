//
// Created by chen on 2021/11/7.
//

#ifndef INSTANCE_SEGMENT_PIPELINE_H
#define INSTANCE_SEGMENT_PIPELINE_H

#include <iostream>

#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torchvision/vision.h>

#include "../parameters.h"
#include "../utils.h"

class Pipeline {
public:
    using Ptr=std::shared_ptr<Pipeline>;

    Pipeline(){

    }

    template<typename ImageType>
    std::tuple<float,float> getXYWHS(const ImageType &img);
    void* setInputTensorCuda(cv::Mat &img);

    cv::Mat processMask(cv::Mat &mask,std::vector<InstInfo> &insts);

    ImageInfo imageInfo;
    torch::Tensor input_tensor;

private:
};


#endif //
