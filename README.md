# Shadow removal and cloud removal using MPRNet
shadow removal and cloud removal
## Results of Cloud removal on RICE dataset

![image](https://github.com/zhangbaijin/MPRNet-Cloud-removal/blob/main/cloud-results.jpg)

## Results of Cloud removal on RICE dataset
![image](https://github.com/zhangbaijin/MPRNet-Cloud-removal/blob/main/shadow-results.jpg)

#### Quick run 
- Download SIDD Validation Data and Ground Truth from [here](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php) and place them in `./Datasets/SIDD/test/`
- Run
```
python test_SIDD.py --save_images
```
## Evaluation

1. Download the pretrained model [cloud-removal](https://drive.google.com/drive/folders/1hJQVQopWMD0WazeQzZC2eDbtirXkGILO?usp=sharing)

Baidu Drive: 链接：https://pan.baidu.com/s/1nBNEsRLIFS2VVtHl8O14Rw 提取码：5mli

2. Download test datasets (RICE, ISTD) from [here](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs?usp=sharing) and place them in `./Datasets/Synthetic_Rain_Datasets/test/`

3. Run
```
python test.py
```

#### To reproduce PSNR/SSIM scores of the paper, run
```

#### To reproduce PSNR/SSIM scores of the paper, run MATLAB script
```
evaluate_SIDD.m
```
