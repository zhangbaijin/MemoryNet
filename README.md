# Shadow removal and cloud removal using MPRNet
shadow removal and cloud removal
## Results of Cloud removal on RICE dataset

![image](https://github.com/zhangbaijin/MPRNet-Cloud-removal/blob/main/cloud-results.jpg)

## Results of Cloud removal on RICE dataset
![image](https://github.com/zhangbaijin/MPRNet-Cloud-removal/blob/main/shadow-results.jpg)

### Quick run 
- Download SIDD Validation Data and Ground Truth from [here](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php) and place them in `./Datasets/SIDD/test/`
- Run
```
python test_SIDD.py --save_images
```
## Pretrained model

1. Download the pretrained model [cloud-removal](https://drive.google.com/drive/folders/1hJQVQopWMD0WazeQzZC2eDbtirXkGILO?usp=sharing)

2.Baidu Drive: 链接：https://pan.baidu.com/s/1nBNEsRLIFS2VVtHl8O14Rw 提取码：5mli

# Dataset 
Download datasets RICE from [here](https://github.com/BUPTLdy/RICE_DATASET), and ISTD dataset from [here](https://github.com/nhchiu/Shadow-Removal-ISTD)


#### To reproduce PSNR/SSIM scores of the paper, run
```

#### To reproduce PSNR/SSIM scores of the paper, run MATLAB script
```
evaluate_SIDD.m
```
