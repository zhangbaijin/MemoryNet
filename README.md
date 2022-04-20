
# Results of Cloud removal on RICE dataset

![image](https://github.com/zhangbaijin/MPRNet-Cloud-removal/blob/main/cloud-results.jpg)

# Results of Cloud removal on RICE dataset
![image]()

#### Quick run 
- Download SIDD Validation Data and Ground Truth from [here](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php) and place them in `./Datasets/SIDD/test/`
- Run
```
python test_SIDD.py --save_images
```
#### Testing on DND dataset
- Download DND Benchmark Data from [here](https://noise.visinf.tu-darmstadt.de/downloads/) and place it in `./Datasets/DND/test/`
- Run
```
python test_DND.py --save_images
```

#### To reproduce PSNR/SSIM scores of the paper, run MATLAB script
```
evaluate_SIDD.m
```
