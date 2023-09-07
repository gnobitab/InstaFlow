<div align="center">

# InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation

[[Paper]]()

by *Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, Qiang Liu* from [Helixon Research](https://www.helixon.com/) and UT Austin 

</div>

## Introduction

Diffusion models are known to be slow due to the need of iterative inference. InstaFlow is an ```ultra-fast```, ```one-step``` image generator derived from pre-trained Stable Diffusion. InstaFlow is enabled by [Rectified Flow](https://github.com/gnobitab/RectifiedFlow).

**InstaFlow** has several advantages: 
- ```Ultra-Fast Inference```: **InstaFlow** models are **one-step generators**, which directly map noises to images and avoid multi-step sampling. On our computer, the inference time is around 0.1 second, saving ~90% of the inference time compared to the original SD.
- ```High-Quality```: **InstaFlow** is the **first** one-step generator derived from diffusion models that has **GAN-level generation quality**.
- ```Simple and Efficient Training```: The training process of **InstaFlow** merely involves **supervised training**, which is much less trickier than adversarial training. Leveraging pre-trained Stable Diffusion, it only takes us **199 A100 GPU days** to get **InstaFlow-0.9B**.  

**TODO: add Figure** 

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

## Comparison with SD 1.5 on Real Computer

For an intuitive understanding, we use the same computer and took screenshots of random generation with different models. InstaFlow-0.9B is one-step, while SD 1.5 adopts 25-step [DPMSolver](https://github.com/LuChengTHU/dpm-solver). The text prompt is *"A photograph of a snowy mountain near a beautiful lake under sunshine."*


                   InstaFlow-0.9B                                              Stable Diffusion 1.5

![](github_misc/comparison.gif)

## Related Materials

We provide several related links and readings here:

* The official Rectified Flow github repo (https://github.com/gnobitab/RectifiedFlow)

* An introduction of Rectified Flow in English (https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html)

* An introduction of Rectified Flow in Chinese--Zhihu (https://zhuanlan.zhihu.com/p/603740431)

* FlowGrad: Controlling the Output of Generative ODEs With Gradients (https://github.com/gnobitab/FlowGrad)

* Fast Point Cloud Generation with Straight Flows (https://github.com/klightz/PSF)

## Citation

TODO

## Thanks

Our training scripts are modified from [one of the fine-tuning examples in Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py).
Other parts of our work also heavily relies on the [🤗 Diffusers](https://github.com/huggingface/diffusers) library.

