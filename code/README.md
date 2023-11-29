# InstaFlow - Code

This folder contains the scripts for training and inference of InstaFlow. 

## Dependencies


```
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate gradio
pip install controlnet_aux
```

If PyTorch doesn't work correctly, please follow the [official site](https://pytorch.org/get-started/locally/) of PyTorch to fix it on your platform.

**NOTE: The pre-trained models are subject to CC-BY-NC-4.0 liscence, which is non-commercial.**

They are also subject to the original CreativeML Open RAIL-M liscence in Stable Diffusion 1.5.

## Inference: 2-Rectified Flow (Few-Step Generation)

Below is a simple examplary script for generating images with pre-trained 2-Rectified Flow:

```py
from pipeline_rf import RectifiedFlowPipeline
import torch

pipe = RectifiedFlowPipeline.from_pretrained("XCLiu/2_rectified_flow_from_sd_1_5", torch_dtype=torch.float16) 
### switch to torch.float32 for higher quality

pipe.to("cuda")  ### if GPU is not available, comment this line

prompt = "A hyper-realistic photo of a cute cat."

### 2-rectified flow is a multi-step text-to-image generative model.
### It can generate with extremely few steps, e.g, 2-8
### For guidance scale, the optimal range is [1.0, 2.0], which is smaller than normal Stable Diffusion.
### You may set negative_prompts like normal Stable Diffusion
images = pipe(prompt=prompt, 
              negative_prompt="painting, unreal, twisted", 
              num_inference_steps=25, 
              guidance_scale=1.5).images 
images[0].save("./image.png")
```

It will automatically download the checkpoints from Hugging Face and inference with the user-defined hyper-parameters. The generated image will be stored in ```./image.png```. 

## Inference: InstaFlow-0.9B (One-Step Generation)

To generate images with pre-trained one-step InstaFlow-0.9B, you may refer to the following code snippet:
```py
from pipeline_rf import RectifiedFlowPipeline
import torch

pipe = RectifiedFlowPipeline.from_pretrained("XCLiu/instaflow_0_9B_from_sd_1_5", torch_dtype=torch.float16) 
### switch to torch.float32 for higher quality

pipe.to("cuda")  ### if GPU is not available, comment this line

prompt = "A hyper-realistic photo of a cute cat."

### InstaFlow-0.9B is a one-step text-to-image generative model.
### It only allows num_inference_steps=1 and guidance_scale=0.0; it does not support negative prompts (for now)
images = pipe(prompt=prompt, 
              num_inference_steps=1, 
              guidance_scale=0.0).images 
images[0].save("./image.png")
```

It will automatically download the checkpoints from Hugging Face and inference with the user-defined hyper-parameters. The generated image will be stored in ```./image.png```

## Adding ControlNet to 2-Rectified Flow or InstaFlow-0.9B
Pre-trained ControlNets are compatible with few-step 2-Rectified Flow and one-step InstaFlow-0.9B.

Here are three examples:
```py
python3 rf_controlnet.py --seed 0 \
    --ctrl_type 'depth' \
    --ref_img 'assets/vermeer.png' \
    --instaflow

python3 rf_controlnet.py --seed 0 \
    --ctrl_type 'canny' \
    --ref_img 'assets/astronaut.png' \
    --instaflow

python3 rf_controlnet.py --seed 0 \
    --ctrl_type 'openpose' \
    --ref_img 'assets/tom.png' \
```

