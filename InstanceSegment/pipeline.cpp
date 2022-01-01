/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Tracking_Solov2_Deepsort.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <iostream>
#include <opencv2/cudaimgproc.hpp>

#include "pipeline.h"


using namespace std;
using namespace torch::indexing;
using InterpolateFuncOptions=torch::nn::functional::InterpolateFuncOptions;


template<typename ImageType>
std::tuple<float,float> Pipeline::getXYWHS(const ImageType &img)
{
    imageInfo.originH = img.rows;
    imageInfo.originW = img.cols;

    int w, h, x, y;
    float r_w = Config::inputW / (img.cols*1.0f);
    float r_h = Config::inputH / (img.rows*1.0f);
    if (r_h > r_w) {
        w = Config::inputW;
        h = r_w * img.rows;
        if(h%2==1)h++;//这里确保h为偶数，便于后面的使用
        x = 0;
        y = (Config::inputH - h) / 2;
    } else {
        w = r_h* img.cols;
        if(w%2==1)w++;
        h = Config::inputH;
        x = (Config::inputW - w) / 2;
        y = 0;
    }

    imageInfo.rect_x = x;
    imageInfo.rect_y = y;
    imageInfo.rect_w = w;
    imageInfo.rect_h = h;

    return {r_h,r_w};
}





void* Pipeline::setInputTensorCuda(cv::Mat &img)
{
    TicToc tt;

    auto [r_h,r_w] = getXYWHS(img);

    cv::Mat img_float;
    img.convertTo(img_float,CV_32FC3);
    sgLogger->debug("setInputTensorCuda convertTo: {} ms",tt.toc_then_tic());
    input_tensor = torch::from_blob(img_float.data, { imageInfo.originH,imageInfo.originW ,3 }, torch::kFloat32).to(torch::kCUDA);


    sgLogger->debug("setInputTensorCuda from_blob:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    ///bgr->rgb
    input_tensor = torch::cat({
        input_tensor.index({"...",2}).unsqueeze(2),
        input_tensor.index({"...",1}).unsqueeze(2),
        input_tensor.index({"...",0}).unsqueeze(2)
        },2);
    sgLogger->debug("setInputTensorCuda bgr->rgb:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    ///hwc->chw
    input_tensor = input_tensor.permute({2,0,1});
    sgLogger->debug("setInputTensorCuda hwc->chw:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    ///norm
    static torch::Tensor mean_t=torch::from_blob(SOLO_IMG_MEAN,{3,1,1},torch::kFloat32).to(torch::kCUDA).expand({3,imageInfo.originH,imageInfo.originW});
    static torch::Tensor std_t=torch::from_blob(SOLO_IMG_STD,{3,1,1},torch::kFloat32).to(torch::kCUDA).expand({3,imageInfo.originH,imageInfo.originW});
    input_tensor = ((input_tensor-mean_t)/std_t);
    sgLogger->debug("setInputTensorCuda norm:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    ///resize
    static auto options=InterpolateFuncOptions().mode(torch::kBilinear).align_corners(true);
    options=options.size(std::vector<int64_t>({imageInfo.rect_h,imageInfo.rect_w}));
    input_tensor = torch::nn::functional::interpolate(input_tensor.unsqueeze(0),options).squeeze(0);
    sgLogger->debug("setInputTensorCuda resize:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    ///拼接图像边缘
    static auto op = torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat32);
    static cv::Scalar mag_color(SOLO_IMG_MEAN[2],SOLO_IMG_MEAN[1],SOLO_IMG_MEAN[0]);
    if (r_h > r_w) { //在图像顶部和下部拼接空白图像
        int cat_w = Config::inputW;
        int cat_h = (Config::inputH-imageInfo.rect_h)/2;
        torch::Tensor cat_t = torch::zeros({3,cat_h,cat_w},op);
        input_tensor = torch::cat({cat_t,input_tensor,cat_t},1);
    } else {
        int cat_w= (Config::inputW-imageInfo.rect_w)/2;
        int cat_h=Config::inputH;
        torch::Tensor cat_t = torch::zeros({3,cat_h,cat_w},op);
        input_tensor = torch::cat({cat_t,input_tensor,cat_t},2);
    }
    sgLogger->debug("setInputTensorCuda cat:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    input_tensor = input_tensor.contiguous();
    sgLogger->debug("setInputTensorCuda contiguous:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    return input_tensor.data_ptr();
}




cv::Mat Pipeline::processMask(cv::Mat &mask,std::vector<InstInfo> &insts)
{
    cv::Mat rect_img = mask(cv::Rect(imageInfo.rect_x, imageInfo.rect_y, imageInfo.rect_w, imageInfo.rect_h));
    cv::Mat out;
    cv::resize(rect_img, out, cv::Size(imageInfo.originW,imageInfo.originH), 0, 0, cv::INTER_LINEAR);

    ///调整包围框
    float factor_x = out.cols *1.f / rect_img.cols;
    float factor_y = out.rows *1.f / rect_img.rows;
    for(auto &inst : insts){
        inst.min_pt.x -= imageInfo.rect_x;
        inst.min_pt.y -= imageInfo.rect_y;
        inst.max_pt.x -= imageInfo.rect_x;
        inst.max_pt.y -= imageInfo.rect_y;

        inst.min_pt.x *= factor_x;
        inst.min_pt.y *= factor_y;
        inst.max_pt.x *= factor_x;
        inst.max_pt.y *= factor_y;
    }


    return out;
}


