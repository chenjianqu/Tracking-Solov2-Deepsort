#include <filesystem>

#include "parameters.h"


void initLogger()
{
    auto reset_log_file=[](const std::string &path){
        if(!fs::exists(path)){
            std::ifstream file(path);//创建文件
            file.close();
        }
        else{
            std::ofstream file(path,std::ios::trunc);//清空文件
            file.close();
        }
    };

    auto get_log_level=[](const std::string &level_str){
        if(level_str=="debug")
            return spdlog::level::debug;
        else if(level_str=="info")
            return spdlog::level::info;
        else if(level_str=="warn")
            return spdlog::level::warn;
        else if(level_str=="error" || level_str=="err")
            return spdlog::level::err;
        else if(level_str=="critical")
            return spdlog::level::critical;
        else{
            cerr<<"log level not right, set default warn"<<endl;
            return spdlog::level::warn;
        }
    };

    reset_log_file(Config::SEGMENTOR_LOG_PATH);
    sgLogger = spdlog::basic_logger_mt("segmentor_log",Config::SEGMENTOR_LOG_PATH);
    sgLogger->set_level(get_log_level(Config::SEGMENTOR_LOG_LEVEL));
    sgLogger->flush_on(get_log_level(Config::SEGMENTOR_LOG_FLUSH));
}



Config::Config(const std::string &file_name)
{
    cv::FileStorage fs(file_name, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(fmt::format("ERROR: Wrong path to settings:{}\n",file_name));
    }

    fs["image_height"]>>imageH;
    fs["image_width"]>>imageW;

    fs["segmentor_log_path"]>>SEGMENTOR_LOG_PATH;
    fs["segmentor_log_level"]>>SEGMENTOR_LOG_LEVEL;
    fs["segmentor_log_flush"]>>SEGMENTOR_LOG_FLUSH;

    cout<<"SEGMENTOR_LOG_PATH:"<<SEGMENTOR_LOG_PATH<<endl;
    cout<<"SEGMENTOR_LOG_LEVEL:"<<SEGMENTOR_LOG_LEVEL<<endl;
    cout<<"SEGMENTOR_LOG_FLUSH:"<<SEGMENTOR_LOG_FLUSH<<endl;

    fs["onnx_path"] >> DETECTOR_ONNX_PATH;
    fs["serialize_path"] >> DETECTOR_SERIALIZE_PATH;

    fs["SOLO_NMS_PRE"]>>SOLO_NMS_PRE;
    fs["SOLO_MAX_PER_IMG"]>>SOLO_MAX_PER_IMG;
    fs["SOLO_NMS_KERNEL"]>>SOLO_NMS_KERNEL;
    fs["SOLO_NMS_SIGMA"]>>SOLO_NMS_SIGMA;
    fs["SOLO_SCORE_THR"]>>SOLO_SCORE_THR;
    fs["SOLO_MASK_THR"]>>SOLO_MASK_THR;
    fs["SOLO_UPDATE_THR"]>>SOLO_UPDATE_THR;

    fs["DATASET_DIR"]>>DATASET_DIR;
    fs["WARN_UP_IMAGE_PATH"]>>WARN_UP_IMAGE_PATH;

    fs["EXTRACTOR_MODEL_PATH"]>>EXTRACTOR_MODEL_PATH;
    fs["TRACKING_N_INIT"]>>TRACKING_N_INIT;
    fs["TRACKING_MAX_AGE"]>>TRACKING_MAX_AGE;

    fs.release();

    CocoLabelVector.reserve(CocoLabelMap.size());
    for(auto &pair : CocoLabelMap){
        CocoLabelVector.push_back(pair.second);
    }


    initLogger();
}




