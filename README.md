# RectifiedFlow-SD

This is the official implementation of paper 
## [Pushing Stable Diffusion to One-Step Generation with Rectified Flow]() 
by *Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, Qiang Liu* from [Helixon Research](https://www.helixon.com/) and UT Austin

## Introduction

[Rectified Flow](https://github.com/gnobitab/RectifiedFlow) is a novel method for generative modeling. The crux of the Rectified Flow pipeline is a special **reflow** procedure which straightens the trajectories of probability flows and eventually reaches **one-step** generation. 

In this work, we prove that the Rectified Flow framework works not only on small-scale problems, but also on large foundation models like Stable Diffusion.
By adopting text-conditioned reflow and distillation, we obtain the first one-step SD model that can generate high-quality images, with **pure supervised learning**. Based on the pre-trained SD 1.5, our one-step model only costs **199 A100 GPU days** for training.

## Comparison with SD 1.5 in Inference Time

For an intuitive understanding, we use the same computer and took screenshots of random generation. The text prompt is *"A photograph of a snowy mountain near a beautiful lake under sunshine."*


                   InstaFlow-0.9B                                   Stable Diffusion 1.5

![](github_misc/comparison.gif)

## Interactive Colab Notebook

We provide an interactive Colab notebook to help you play with our one-step model. Click [Here]() 

## Gradio User Interface

We provide a minimal Gradio Interface for inference with our one-step model and refine with SDXL-Refiner-1.0.

### Environment

Install the environment with the following commands:

```
pip install accelerate transformers invisible-watermark "numpy>=1.17" "PyWavelets>=1.1.1" "opencv-python>=4.1.0.25" safetensors "gradio==3.11.0"
pip install git+https://github.com/huggingface/diffusers.git
```

### Usage

First clone the repo:

```
git clone xxx

cd xxx
```

Then we download the checkpoint ```instaflow_0_9B.pt``` and put it in ```./```. It is the one-step InstaFlow-0.9B model. Download [Here]()

Finally, use the following commands to start the Gradio Interface:

```
python gradio_interface.py
```

The interface should appear like this:

add image


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

## Related Materials

We provide several related links and readings here

* The official Rectified Flow github repo (https://github.com/gnobitab/RectifiedFlow)

* An introduction of Rectified Flow in English (https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html)

* An introduction of Rectified Flow in Chinese--Zhihu (https://zhuanlan.zhihu.com/p/603740431)

* FlowGrad: Controlling the Output of Generative ODEs With Gradients (https://github.com/gnobitab/FlowGrad)

* Fast Point Cloud Generation with Straight Flows (https://github.com/klightz/PSF) 

## Thanks

