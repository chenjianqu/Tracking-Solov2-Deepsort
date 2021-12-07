# Tracking-Solov2-Deepsort
This project implement the Multi-Object-Tracking(MOT) base on SOLOv2 and DeepSORT with C++。
The instance segmentation model SOLOv2 has deploy to TensorRT, and the postprocess implement with 
Libtorch. Therefore, the frame rate of detection and tracking can exceed 40 FPS。
Test video was showed [here](https://www.bilibili.com/video/BV1Ki4y1Z7PC)


## Requirements
* Ubuntu
* Cuda10.2
* cudnn8
* GCC >=9
* TensorRT8
* Opencv3.4
* Libtorch1.8.2
* CMake3.20

## Acknowledge
[SOLO](https://github.com/wxinlong/solo/)  
[SOLOv2.tensorRT](https://github.com/zhangjinsong3/SOLOv2.tensorRT)  
[Yolov5_DeepSort_Pytorch]()  
[libtorch-yolov3-deepsort](https://github.com/weixu000/libtorch-yolov3-deepsort) 


## Geting Started
### 1.Install Solov2
see [Solov2-TensorRT-CPP](https://github.com/chenjianqu/Solov2-TensorRT-CPP)  

### 2.Install DeepSORT
download the deepsort model `ckpt.t7` from [here](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6).
Then use the script `conv_model_format.py` convert model format from `ckpt.t7` to `ckpt.bin`.

### 3. Run Demo
Firstly edit the `config.yaml` to right setting. Then compile the project:
```
mkdir build && cd build
cmake ..
```
Run the demo
```
./tracking ../config/config.yaml
```


