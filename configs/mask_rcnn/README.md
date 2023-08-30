# Mask R-CNN

> [Mask R-CNN](https://arxiv.org/abs/1703.06870)

<!-- [ALGORITHM] -->

## Introduction

Mask R-CNN is a conceptually simple, flexible, and general framework for object instance segmentation. It efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. And it extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. 

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143967081-c2552bed-9af2-46c4-ae44-5b3b74e5679f.png"/>
</div>
