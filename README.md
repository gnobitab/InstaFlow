# RectifiedFlow-SD

This is the official implementation of paper 
## [Pushing Stable Diffusion to One-Step Generation with Rectified Flow]() 
by *Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, Qiang Liu* from [Helixon Research](https://www.helixon.com/) and UT Austin

## Introduction

[Rectified Flow](https://github.com/gnobitab/RectifiedFlow) is a novel method for generative modeling. The crux of the Rectified Flow pipeline is a special **reflow** procedure which straightens the trajectories of probability flows and eventually reaches **one-step** generation. 

In this work, we prove that the Rectified Flow framework works not only on small-scale problems, but also on large foundation models like Stable Diffusion.
By adopting text-conditioned reflow and distillation, we obtain the first one-step SD model that can generate high-quality images, with **pure supervised learning**. Based on the pre-trained SD 1.5, our one-step model only costs **199 A100 GPU days** for training.

![](github_misc/comparison.gif)

![](github_misc/fig1.png)

## Interactive Colab Notebook

We provide an interactive Colab notebook to help you play with our one-step model. Click [Here]() 

## Gradio User Interface


## Related Materials

We provide several related links and readings here

* The official Rectified Flow github repo (https://github.com/gnobitab/RectifiedFlow)

* An introduction of Rectified Flow in English (https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html)

* An introduction of Rectified Flow in Chinese--Zhihu (https://zhuanlan.zhihu.com/p/603740431)

* FlowGrad: Controlling the Output of Generative ODEs With Gradients (https://github.com/gnobitab/FlowGrad)

* Fast Point Cloud Generation with Straight Flows (https://github.com/klightz/PSF) 

## Thanks

