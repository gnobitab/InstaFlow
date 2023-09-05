# InstaFlow

This is the official page of paper
## [InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation]()
by *Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, Qiang Liu* from [Helixon Research](https://www.helixon.com/) and UT Austin

## Introduction

InstaFlow is an one-step model dervied from pre-trained Stable Diffusion. Our key technique is a text-conditioned ```Reflow``` procedure which is the core of the [Rectified Flow](https://github.com/gnobitab/RectifiedFlow) pipeline. ```Reflow``` straightens the trajectories of probability flows, refines the coupling between noises and images, and facilitates the distillation process with student models. However, the effectiveness of reflow has only been examined on small datasets like CIFAR10. 

## Gallery

### One-step generation with InstaFlow-0.9B in 0.09s

<p align="middle">
  <img src='github_misc/gallery/09B_img_1.png' width='192'>
  <img src='github_misc/gallery/09B_img_2.png' width='192'>
  <img src='github_misc/gallery/09B_img_3.png' width='192'>
  <img src='github_misc/gallery/09B_img_4.png' width='192'>
</p>

### One-step generation with InstaFlow-1.7B in 0.12s

<p align="middle">
  <img src='github_misc/gallery/17B_img_1.png' width='192'>
  <img src='github_misc/gallery/17B_img_2.png' width='192'>
  <img src='github_misc/gallery/17B_img_3.png' width='192'>
  <img src='github_misc/gallery/17B_img_4.png' width='192'>
</p>

### InstaFlow-0.9B one-step generation refined with SDXL-Refiner

<p align="middle">
  <img src='github_misc/gallery/09B_refine.png' width='512'>
</p>

## Comparison with SD 1.5 in Inference Time

For an intuitive understanding, we use the same computer and took screenshots of random generation. The text prompt is *"A photograph of a snowy mountain near a beautiful lake under sunshine."*


                   InstaFlow-0.9B                                   Stable Diffusion 1.5

![](github_misc/comparison.gif)

## Related Materials

We provide several related links and readings here

* The official Rectified Flow github repo (https://github.com/gnobitab/RectifiedFlow)

* An introduction of Rectified Flow in English (https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html)

* An introduction of Rectified Flow in Chinese--Zhihu (https://zhuanlan.zhihu.com/p/603740431)

* FlowGrad: Controlling the Output of Generative ODEs With Gradients (https://github.com/gnobitab/FlowGrad)

* Fast Point Cloud Generation with Straight Flows (https://github.com/klightz/PSF)

## Thanks

Our training scripts are modified from [one of the fine-tuning examples in Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)
