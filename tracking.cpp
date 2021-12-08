//
// Created by chen on 2021/11/9.
//
#include <opencv2/opencv.hpp>

#include "InstanceSegment/infer.h"
#include "InstanceTracking/DeepSORT.h"

#include "utils.h"
#include "parameters.h"


int main(int argc, char **argv)
{
    if(argc != 2){
        cerr<<"please input: [config file]"<<endl;
        return 1;
    }
    string config_file = argv[1];
    fmt::print("config_file:{}\n",argv[1]);

    Infer::Ptr infer;
    DeepSORT::Ptr tracker;

    try{
        Config cfg(config_file);
        infer.reset(new Infer);
        std::array<int64_t, 2> orig_dim{int64_t(Config::imageH), int64_t(Config::imageW)};
        tracker = std::make_unique<DeepSORT>(orig_dim);
    }
    catch(std::runtime_error &e){
        sgLogger->critical(e.what());
        cerr<<e.what()<<endl;
        return -1;
    }


    std::unordered_map<int , cv::Scalar_<unsigned int>> color_map;


    int cnt=0;
    cout<<"循环"<<endl;

    ///主线程，图像分割
    TicToc ticToc;

    float time_delta=0.1;
    for(int index=0;index<1000;++index)
    {
        char name[64];
        sprintf(name,"%06d.png",index);
        std::string img0_path=Config::DATASET_DIR+name;
        fmt::print("Read Image:{}\n",img0_path);

        cv::Mat img0=cv::imread(img0_path);
        if(img0.empty()){
            cerr<<"Read:"<<img0_path<<" failure"<<endl;
            break;
        }
        ticToc.tic();
        torch::Tensor mask_tensor;
        std::vector<InstInfo> insts_info;
        infer->forward_tensor(img0,mask_tensor,insts_info);

        fmt::print("insts_info.size():{}\n",insts_info.size());

        double delta_time=0;

        cv::Mat img_raw = img0.clone();

        if(!insts_info.empty()){
            std::vector<cv::Rect_<float>> dets;
            auto trks = tracker->update(insts_info, img0);
            delta_time=ticToc.toc();

            auto mask_size=cv::Size(img_raw.cols,img_raw.rows);
            mask_tensor = mask_tensor.to(torch::kInt8).abs().clamp(0,1);
            ///计算合并的mask
            auto merge_tensor = (mask_tensor.sum(0).clamp(0,1)*255).to(torch::kUInt8).to(torch::kCPU);
            auto mask = cv::Mat(mask_size,CV_8UC1,merge_tensor.data_ptr()).clone();
            cv::cvtColor(mask,mask,CV_GRAY2BGR);
            cv::scaleAdd(mask,0.5,img_raw,img_raw);

            for(auto &inst: trks){
                if(color_map.count(inst.track_id)==0){
                    color_map.insert({inst.track_id,getRandomColor()});
                }
                auto color = color_map[inst.track_id];

                draw_text(img_raw,fmt::format("{}:{:.2f}",Config::CocoLabelVector[inst.label_id],inst.prob),
                          color,inst.rect.tl());
                auto box_center = (inst.min_pt + inst.max_pt)/2.;
                cv::putText(img_raw, std::to_string(inst.track_id), box_center, cv::FONT_HERSHEY_SIMPLEX,
                            1, color,2);
                cv::rectangle(img_raw,inst.min_pt,inst.max_pt,color,1);
            }
        }
        else{
            delta_time=ticToc.toc();
        }

        fmt::print("tracker time:{} ms\n",delta_time);
        cv::putText(img_raw, fmt::format("{} ms",delta_time), cv::Point2d(20,20),
                    cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(255,0,0),2);

        cv::imshow("raw", img_raw);

        if(auto order=(cv::waitKey(1) & 0xFF); order == 'q')
            break;
        else if(order==' ')
            cv::waitKey(0);
    }

    return 0;
}

