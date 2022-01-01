/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Tracking_Solov2_Deepsort.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DEFINES_H
#define DEFINES_H

#include <opencv2/opencv.hpp>

struct Track {
    int id;
    cv::Rect2f box;
};


#endif //DEFINES_H
