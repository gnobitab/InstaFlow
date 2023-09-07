<div align="center">

# InstaFlow! One-Step Stable Diffusion with Rectified Flow

[[Paper]]()

by *Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, Qiang Liu* from [Helixon Research](https://www.helixon.com/) and UT Austin 

</div>

## Introduction

Diffusion models have demonstrated remarkable promises in text-to-image generation. However, their efficacy is still largely hindered by computational constraints stemming from the need of iterative numerical solvers at the inference time for solving the diffusion/flow processes. 

**InstaFlow** is an ```ultra-fast```, ```one-step``` image generator that achieves image quality close to Stable Diffusion, while requiring significantly reduced computational resources. This efficiency is made possible through a recent [Rectified Flow](https://github.com/gnobitab/RectifiedFlow) technique, which trains probability flows with straight trajectories, hence inherently requiring only a single step for fast inference.

**InstaFlow** has several advantages: 
- ```Ultra-Fast Inference```: **InstaFlow** models are **one-step generators**, which directly map noises to images and avoid multi-step sampling of diffusion models. On our machine with A100 GPU, the inference time is around 0.1 second, saving ~90% of the inference time compared to the original Stable Diffusion.
- ```High-Quality```: **InstaFlow** generates images with intricate details like Stable Diffusion, and have similar FID on MS COCO 2014 as state-of-the-art GAN, [StyleGAN-T](https://github.com/autonomousvision/stylegan-t).
- ```Simple and Efficient Training```: The training process of **InstaFlow** merely involves **supervised training**. Leveraging pre-trained Stable Diffusion, it only takes **199 A100 GPU days** to get **InstaFlow-0.9B**.  

![](github_misc/n_step.png)

## Gallery

### One-step generation with InstaFlow-0.9B (0.09s per image, $512 \times 512$)

<p align="middle">
  <img src='github_misc/gallery/09B_img_1.png' width='192'>
  <img src='github_misc/gallery/09B_img_2.png' width='192'>
  <img src='github_misc/gallery/09B_img_3.png' width='192'>
  <img src='github_misc/gallery/09B_img_4.png' width='192'>
</p>

### One-step generation with InstaFlow-1.7B (0.12s per image, $512 \times 512$)

<p align="middle">
  <img src='github_misc/gallery/17B_img_1.png' width='192'>
  <img src='github_misc/gallery/17B_img_2.png' width='192'>
  <img src='github_misc/gallery/17B_img_3.png' width='192'>
  <img src='github_misc/gallery/17B_img_4.png' width='192'>
</p>

### InstaFlow-0.9B one-step generation + SDXL-Refiner

<p align="middle">
  <img src='github_misc/gallery/09B_refine.png' width='384'>
</p>

## Comparison with SD 1.5 on our A100 machine

For an intuitive understanding, we used the same A100 server and took screenshots from the Gridio interface of random generation with different models. InstaFlow-0.9B is one-step, while SD 1.5 adopts 25-step [DPMSolver](https://github.com/LuChengTHU/dpm-solver). It takes around 0.3 second to download the image from the server. The text prompt is *"A photograph of a snowy mountain near a beautiful lake under sunshine."*

```
                   InstaFlow-0.9B                               Stable Diffusion 1.5
```

![](github_misc/comparison.gif)

## Related Materials

We provide several related links and readings here:

* The official Rectified Flow github repo (https://github.com/gnobitab/RectifiedFlow)

* An introduction of Rectified Flow (https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html)

* An introduction of Rectified Flow in Chinese--Zhihu (https://zhuanlan.zhihu.com/p/603740431)

* FlowGrad: Controlling the Output of Generative ODEs With Gradients (https://github.com/gnobitab/FlowGrad)

* Fast Point Cloud Generation with Straight Flows (https://github.com/klightz/PSF)

## Citation

TODO

## Thanks

Our training scripts are modified from [one of the fine-tuning examples in Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py).
Other parts of our work also heavily relies on the [🤗 Diffusers](https://github.com/huggingface/diffusers) library.

