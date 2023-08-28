# RectifiedFlow-SD

This is the official implementation of paper 
## [Pushing Stable Diffusion to One-Step Generation with Rectified Flow]() 
by *Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, Qiang Liu* from [Helixon Research](https://www.helixon.com/) and UT Austin

## Introduction

[Rectified Flow](https://github.com/gnobitab/RectifiedFlow) is a novel method for generative modeling. In its framework, a special **reflow** procedure is invented to straighten the trajectories of probability flows and eventually reach **one-step** generation. 

In this work, we prove that the Rectified Flow framework works not only on small-scale problems, but also on large foundation models like Stable Diffusion.
By adopting text-conditioned reflow and distillation, we obtain the first one-step SD model that can generate high-quality images, with pure supervised learning. Based on Pre-trained SD 1.5, our training only costs 199 A100 GPU days.

In this repo, we release the evaluation code along with pre-trained models.

![](github_misc/fig1.png)

