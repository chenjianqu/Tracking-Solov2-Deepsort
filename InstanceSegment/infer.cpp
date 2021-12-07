//
// Created by chen on 2021/11/7.
//
#include <iostream>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>

#include "infer.h"

#include "../parameters.h"
#include "../utils.h"




using namespace std;


std::optional<int> getQueueShapeIndex(int c,int h,int w)
{
    int index=-1;
    for(int i=0;i< (int)TENSOR_QUEUE_SHAPE.size();++i){
        if(c==TENSOR_QUEUE_SHAPE[i][1] && h==TENSOR_QUEUE_SHAPE[i][2] && w==TENSOR_QUEUE_SHAPE[i][3]){
            index=i;
            break;
        }
    }
    if(index==-1)
        return std::nullopt;
    else
        return index;
}



Infer::Infer()
{
    ///注册预定义的和自定义的插件
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(),"");
    sgLogger->info("Read model param");
    std::string model_str;
    if(std::ifstream ifs(Config::DETECTOR_SERIALIZE_PATH);ifs.is_open()){
        while(ifs.peek() != EOF){
            std::stringstream ss;
            ss<<ifs.rdbuf();
            model_str.append(ss.str());
        }
        ifs.close();
    }
    else{
        auto msg=fmt::format("Can not open the DETECTOR_SERIALIZE_PATH:{}",Config::DETECTOR_SERIALIZE_PATH);
        sgLogger->critical(msg);
        throw std::runtime_error(msg);
    }

    sgLogger->info("createInferRuntime");

    ///创建runtime
    runtime=std::unique_ptr<nvinfer1::IRuntime,InferDeleter>(
            nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));

    sgLogger->info("deserializeCudaEngine");

    ///反序列化模型
    engine=std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(model_str.data(),model_str.size()) ,InferDeleter());

    sgLogger->info("createExecutionContext");

    ///创建执行上下文
    context=std::unique_ptr<nvinfer1::IExecutionContext,InferDeleter>(
            engine->createExecutionContext());

    if(!context){
        throw std::runtime_error("can not create context");
    }

    ///创建输入输出的内存
    buffer = std::make_shared<MyBuffer>(*engine);

    Config::inputH=buffer->dims[0].d[2];
    Config::inputW=buffer->dims[0].d[3];
    Config::inputC=3;

    pipeline=std::make_shared<Pipeline>();
    solo = std::make_shared<Solov2>();


    //cv::Mat warn_up_input(cv::Size(1226,370),CV_8UC3,cv::Scalar(128));
    cv::Mat warn_up_input = cv::imread(Config::WARN_UP_IMAGE_PATH);

    if(warn_up_input.empty()){
        sgLogger->error("Can not open warn up image:{}", Config::WARN_UP_IMAGE_PATH);
        return;
    }

    cv::resize(warn_up_input,warn_up_input,cv::Size(Config::imageW,Config::imageH));

    sgLogger->warn("warn up model,path:{}",Config::WARN_UP_IMAGE_PATH);

    //[[maybe_unused]] auto result = forward(warn_up_input);

    [[maybe_unused]] torch::Tensor mask_tensor;
    [[maybe_unused]] std::vector<InstInfo> insts_info;
    forward_tensor(warn_up_input,mask_tensor,insts_info);

    //if(insts_info.empty())throw std::runtime_error("model not init");

    sgLogger->info("infer init finished");
}



void Infer::forward_tensor(cv::Mat &img,torch::Tensor &mask_tensor,std::vector<InstInfo> &insts)
{
    TicToc ticToc,tt;

    ///将图片数据复制到输入buffer,同时实现了图像的归一化
    buffer->gpu_buffer[0] = pipeline->setInputTensorCuda(img);

    sgLogger->info("forward_tensor prepare:{} ms",tt.toc_then_tic());

    ///推断
    context->enqueue(BATCH_SIZE, buffer->gpu_buffer, buffer->stream, nullptr);

    sgLogger->info("forward_tensor enqueue:{} ms",tt.toc_then_tic());

    std::vector<torch::Tensor> outputs(TENSOR_QUEUE_SHAPE.size());

    auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat);
    for(int i=1;i<buffer->binding_num;++i){
        torch::Tensor tensor=torch::from_blob(
                buffer->gpu_buffer[i],
                {buffer->dims[i].d[0],buffer->dims[i].d[1],buffer->dims[i].d[2],buffer->dims[i].d[3]},
                opt);
        std::optional<int> index = getQueueShapeIndex(buffer->dims[i].d[1],buffer->dims[i].d[2],buffer->dims[i].d[3]);
        if(index){
            outputs[*index] = tensor.to(torch::kCUDA);
        }
        else{
            auto msg=fmt::format("getQueueShapeIndex failed:({},{},{},{})",buffer->dims[i].d[0],buffer->dims[i].d[1],buffer->dims[i].d[2],buffer->dims[i].d[3]);
            sgLogger->error(msg);
            throw std::runtime_error(msg);
        }
    }

    sgLogger->info("forward_tensor push_back:{} ms",tt.toc_then_tic());

    solo->getSegTensor(outputs,pipeline->imageInfo,mask_tensor,insts);

    sgLogger->info("forward_tensor getSegTensor:{} ms",tt.toc_then_tic());
    sgLogger->info("forward_tensor inst number:{}",insts.size());

    infer_time = ticToc.toc();
}



void Infer::visualizeResult(cv::Mat &input,cv::Mat &mask,std::vector<InstInfo> &insts)
{
    if(mask.empty()){
        cv::imshow("test",input);
        cv::waitKey(1);
    }
    else{
        cout<<mask.size<<endl;
        mask = pipeline->processMask(mask,insts);

        cv::Mat image_test;
        cv::add(input,mask,image_test);
        for(auto &inst : insts){
            if(inst.prob < 0.2)
                continue;
            inst.name = CocoLabelMap[inst.label_id + 1];
            cv::Point2i center = (inst.min_pt + inst.max_pt)/2;
            std::string show_text = fmt::format("{} {:.2f}",inst.name,inst.prob);
            cv::putText(image_test,show_text,center,CV_FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(255,0,0),2);
            cv::rectangle(image_test, inst.min_pt, inst.max_pt, cv::Scalar(255, 0, 0), 2);
        }
        cv::putText(image_test,fmt::format("{:.2f} ms",infer_time),cv::Point2i(20,20),CV_FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,255,255));

        cv::imshow("test",image_test);
        cv::waitKey(1);
    }
}
