/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Tracking_Solov2_Deepsort.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef INSTANCE_SEGMENT_PARAMETER_H
#define INSTANCE_SEGMENT_PARAMETER_H

#include <vector>
#include <fstream>
#include <map>
#include <iostream>
#include <exception>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::pair;
using std::vector;

using namespace std::chrono_literals;
namespace fs=std::filesystem;

constexpr int INFER_IMAGE_LIST_SIZE=30;

constexpr double FOCAL_LENGTH = 460.0;
constexpr int WINDOW_SIZE = 10;
constexpr int NUM_OF_F = 1000;

constexpr int INSTANCE_FEATURE_SIZE=500;
constexpr int SIZE_SPEED=6;
constexpr int SIZE_BOX=3;


//图像归一化参数，注意是以RGB的顺序排序
inline float SOLO_IMG_MEAN[3]={123.675, 116.28, 103.53};
inline float SOLO_IMG_STD[3]={58.395, 57.12, 57.375};

constexpr int BATCH_SIZE=1;
constexpr int SOLO_TENSOR_CHANNEL=128;//张量的输出通道数应该是128

inline std::vector<float> SOLO_NUM_GRIDS={40, 36, 24, 16, 12};//各个层级划分的网格数
inline std::vector<float> SOLO_STRIDES={8, 8, 16, 32, 32};//各个层级的预测结果的stride


inline std::map<int,std::string> CocoLabelMap={
        {1, "person"}, {2, "bicycle"}, {3, "car"}, {4, "motorcycle"}, {5, "airplane"},
        {6, "bus"}, {7, "train"}, {8, "truck"}, {9, "boat"}, {10, "traffic light"},
        {11, "fire hydrant"}, {13, "stop sign"}, {14, "parking meter"}, {15, "bench"},
        {16, "bird"}, {17, "cat"}, {18, "dog"}, {19, "horse"}, {20, "sheep"}, {21, "cow"},
        {22, "elephant"}, {23, "bear"}, {24, "zebra"}, {25, "giraffe"}, {27, "backpack"},
        {28, "umbrella"}, {31, "handbag"}, {32, "tie"}, {33, "suitcase"}, {34, "frisbee"},
        {35, "skis"}, {36, "snowboard"}, {37, "sports ball"}, {38, "kite"}, {39, "baseball bat"},
        {40, "baseball glove"}, {41, "skateboard"}, {42, "surfboard"}, {43, "tennis racket"},
        {44, "bottle"}, {46, "wine glass"}, {47, "cup"}, {48, "fork"}, {49, "knife"}, {50, "spoon"},
        {51, "bowl"}, {52, "banana"}, {53, "apple"}, {54, "sandwich"}, {55, "orange"},
        {56, "broccoli"}, {57, "carrot"}, {58, "hot dog"}, {59, "pizza"}, {60, "donut"},
        {61, "cake"}, {62, "chair"}, {63, "couch"}, {64, "potted plant"}, {65, "bed"}, {67, "dining table"},
        {70, "toilet"}, {72, "tv"}, {73, "laptop"}, {74, "mouse"}, {75, "remote"}, {76, "keyboard"},
        {77, "cell phone"}, {78, "microwave"}, {79, "oven"}, {80, "toaster"},{ 81, "sink"},
        {82, "refrigerator"}, {84, "book"}, {85, "clock"},{ 86, "vase"}, {87, "scissors"},
        {88, "teddy bear"}, {89, "hair drier"}, {90, "toothbrush"}
};


inline std::shared_ptr<spdlog::logger> sgLogger;



inline std::vector<std::vector<int>> TENSOR_QUEUE_SHAPE{
        {1, 128, 12, 12},
        {1, 128, 16, 16},
        {1, 128, 24, 24},
        {1, 128, 36, 36},
        {1, 128, 40, 40},
        {1, 80, 12, 12},
        {1, 80, 16, 16},
        {1, 80, 24, 24},
        {1, 80, 36, 36},
        {1, 80, 40, 40},
        {1, 128, 96, 288}
};



enum SIZE_PARAMETERIZATION{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};



class Config {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr=std::shared_ptr<Config>;

    explicit Config(const std::string &file_name);

    inline static std::string DETECTOR_ONNX_PATH;
    inline static std::string DETECTOR_SERIALIZE_PATH;

    inline static int inputH,inputW,inputC;
    inline static int imageH,imageW;

    inline static std::vector<std::string> CocoLabelVector;

    inline static std::string SEGMENTOR_LOG_PATH;
    inline static std::string SEGMENTOR_LOG_LEVEL;
    inline static std::string SEGMENTOR_LOG_FLUSH;

    inline static int SOLO_NMS_PRE;
    inline static int SOLO_MAX_PER_IMG;
    inline static std::string SOLO_NMS_KERNEL;
    inline static float SOLO_NMS_SIGMA;
    inline static float SOLO_SCORE_THR;
    inline static float SOLO_MASK_THR;
    inline static float SOLO_UPDATE_THR;

    inline static std::atomic_bool ok{true};

    inline static string DATASET_DIR;
    inline static string WARN_UP_IMAGE_PATH;

    inline static int TRACKING_MAX_AGE;
    inline static int TRACKING_N_INIT;
    inline static string EXTRACTOR_MODEL_PATH;

    inline static bool use_trace_model{false};

    inline static int kReidImgWidth;
    inline static int kReidImgHeight;

};


#endif

