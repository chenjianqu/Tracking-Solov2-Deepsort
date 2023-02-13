/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Tracking_Solov2_Deepsort.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "reid_models/car_reid_net.h"
#include "reid_models/people_reid_net.h"
#include "reid_models/trace_reid_net.h"

class Extractor {
public:
    Extractor();

    torch::Tensor extract(std::vector<cv::Mat> input); // return GPUTensor

private:
    Net net;
    //CarReIdNet net;
    TraceReidNet trace_net;
};


#endif //EXTRACTOR_H
