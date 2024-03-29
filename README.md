# MemoryNet:Memory augument is All You Need for image restoration
Memory augument is All You Need for image restoration
Memory是一个即插即用的记忆模块，随时提点！
Paper link https://arxiv.org/abs/2309.01377


## The structure of MemoryNet
![image](https://github.com/zhangbaijin/MPRNet-Cloud-removal/blob/main/structure.png)

## Results of Shadow removal on ISTD dataset
![image](https://github.com/zhangbaijin/MPRNet-Cloud-removal/blob/main/shadow-results.jpg)

## Quick Run

To test the pre-trained models of [Decloud](https://drive.google.com/drive/folders/1hJQVQopWMD0WazeQzZC2eDbtirXkGILO?usp=sharing) on your own images, run 
```
python demo.py --task Task_Name --input_dir path_to_images --result_dir save_images_here
```

## Pretrained model


1. Download the pretrained model [cloud-removal](https://drive.google.com/drive/folders/1hJQVQopWMD0WazeQzZC2eDbtirXkGILO?usp=sharing)

2.Baidu Drive: 链接：https://pan.baidu.com/s/1nBNEsRLIFS2VVtHl8O14Rw 提取码：5mli

# Dataset 
Download datasets RICE from [here](https://github.com/BUPTLdy/RICE_DATASET), and ISTD dataset from [here](https://github.com/nhchiu/Shadow-Removal-ISTD)

#### To reproduce PSNR/SSIM scores of the paper, run MATLAB script
```
evaluate_PSNR_SSIM.m
```
# ACKNOLAGEMENT
The code is updated on https://github.com/swz30/MPRNet
