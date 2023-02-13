/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Tracking_Solov2_Deepsort.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_UTILS_H
#define DYNAMIC_VINS_UTILS_H

#include <string>
#include <vector>
#include <chrono>
#include <random>

#include <spdlog/logger.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "parameters.h"



class TicToc{
public:
    TicToc(){
        tic();
    }

    void tic(){
        start = std::chrono::system_clock::now();
    }

    double toc(){
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

    double toc_then_tic(){
        auto t=toc();
        tic();
        return t;
    }

    void toc_print_tic(const char* str){
        cout<<str<<":"<<toc()<<" ms"<<endl;
        tic();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};



struct InstInfo{
    std::string name;
    int label_id;
    int id;
    int track_id;
    cv::Point2f min_pt,max_pt;
    cv::Rect2f rect;
    float prob;

    cv::Point2f mask_center;

    cv::Mat mask_cv;
    cv::cuda::GpuMat mask_gpu;
    torch::Tensor mask_tensor;
};




struct ImageInfo{
    int originH,originW;
    ///图像的裁切信息
    int rect_x, rect_y, rect_w, rect_h;
};




template <typename T>
static std::string dims2str(torch::ArrayRef<T> list){
    int i = 0;
    std::string text= "[";
    for(auto e : list) {
        if (i++ > 0) text+= ", ";
        text += std::to_string(e);
    }
    text += "]";
    return text;
}


inline cv::Point2f operator*(const cv::Point2f &lp,const cv::Point2f &rp)
{
    return {lp.x * rp.x,lp.y * rp.y};
}

template<typename MatrixType>
inline std::string eigen2str(const MatrixType &m){
    std::string text;
    for(int i=0;i<m.rows();++i){
        for(int j=0;j<m.cols();++j){
            text+=fmt::format("{:.2f} ",m(i,j));
        }
        if(m.rows()>1)
            text+="\n";
    }
    return text;
}

template<typename T>
inline std::string vec2str(const Eigen::Matrix<T,3,1> &vec){
    return eigen2str(vec.transpose());
}




void draw_text(cv::Mat &img, const std::string &str,const cv::Scalar &color, const cv::Point& pos,  float scale=1.f, int thickness=1,bool reverse = false);

void draw_bbox(cv::Mat &img, const cv::Rect2f& bbox,const std::string &label = "", const cv::Scalar &color = {0, 0, 0});



 float getBoxIoU(const cv::Point2f &box1_minPt, const cv::Point2f &box1_maxPt,
                       const cv::Point2f &box2_minPt, const cv::Point2f &box2_maxPt);

 float getBoxIoU(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt);

 cv::Scalar color_map(int64_t n);

 inline cv::Scalar_<unsigned int> getRandomColor(){
     static std::default_random_engine rde;
     static std::uniform_int_distribution<unsigned int> color_rd(0,255);
     return {color_rd(rde),color_rd(rde),color_rd(rde)};
 }



 template <typename T>
 static std::string DimsToStr(torch::ArrayRef<T> list){
     int i = 0;
     std::string text= "[";
     for(auto e : list) {
         if (i++ > 0) text+= ", ";
         text += std::to_string(e);
     }
     text += "]";
     return text;
 }




#endif //DYNAMIC_VINS_UTILS_H
