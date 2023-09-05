# RectifiedFlow-SD

This is the official implementation of paper
## [Pushing Stable Diffusion to One-Step Generation with Rectified Flow]()
by *Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, Qiang Liu* from [Helixon Research](https://www.helixon.com/) and UT Austin

## Introduction

[Rectified Flow](https://github.com/gnobitab/RectifiedFlow) is a novel method for generative modeling. The crux of the Rectified Flow pipeline is a special **reflow** procedure which straightens the trajectories of probability flows and eventually reaches **one-step** generation.

In this work, we prove that the Rectified Flow framework works not only on small-scale problems, but also on large foundation models like Stable Diffusion.
By adopting text-conditioned reflow and distillation, we obtain the first one-step SD model that can generate high-quality images, with **pure supervised learning**. Based on the pre-trained SD 1.5, our one-step model only costs **199 A100 GPU days** for training.


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
