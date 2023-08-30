# SViT on ImageNet1k
This folder contains code for SViT on classification, and is build on [EViT codebase](https://github.com/youweiliang/evit/tree/master).

## Preparation
Besides `torch` and `timm`, the following additional packages should be installed for classification:
```
lmdb==1.2.1
pyarrow==5.0.0
torchprofile==0.0.4
einops==0.4.1
tensorboardX==2.4
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Result

We provide classification results on ImageNet1k for SViT-S.

| Model | Acc@1 | Acc@5 | Throughput (img/s) | #Params | Download | log |
| --- | --- | --- | --- | --- | --- | --- |
| DeiT | 79.8 | 95.0 | 1524 | 22.1M | [model](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) | - |
| SViT | 79.4 | 94.6 | 2246 | 22.1M | [model](https://drive.google.com/file/d/11qIVZe9LItjmyfVQwcqy-qm_KfWwKz-o/view?usp=drive_link) | [log](https://drive.google.com/file/d/1-Y4evVc9RpxVINe5afeA4EUAcLA-4Fy8/view?usp=drive_link)|


## Evaluation
To evaluate a SViT model on ImageNet val with a single GPU run:
```
python main.py --model svit_s --eval --resume pretrained/svit.pth --data-path ../data/imagenet --ratio-loss-lambda 4
```
<img width="1493" alt="image" src="https://github.com/kaikai23/svit-test/assets/71677542/2d9bbcc6-6d2f-41fb-840d-6c9bb102b88c">


## Throughput
We provide a script to measure models' throughput under different batch sizes.
```shell
python speed_test.py
```
Note that nested_tensor's speed can drop when CPU is busy, so it is recommended to test the speed when there are no other running jobs.

## Visualization
We provide a script to visualize the token pruning:
```
python visualize.py
```
<img width="800" alt="image" src="https://github.com/kaikai23/svit-test/assets/71677542/58c5e631-efb7-4344-8281-9fcf3f220791">


# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Acknowledgement
We would like to think the authors of [DeiT](https://github.com/facebookresearch/deit), [EViT](https://github.com/youweiliang/evit/tree/master) and [timm](https://github.com/rwightman/pytorch-image-models), based on which this codebase was built.
