import gradio as gr

from pipeline_rf import RectifiedFlowPipeline

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F

from diffusers import StableDiffusionXLImg2ImgPipeline
import time
import copy
import numpy as np

def merge_dW_to_unet(pipe, dW_dict, alpha=1.0):
    _tmp_sd = pipe.unet.state_dict()
    for key in dW_dict.keys():
        _tmp_sd[key] += dW_dict[key] * alpha
    pipe.unet.load_state_dict(_tmp_sd, strict=False)
    return pipe

def get_dW_and_merge(pipe_rf, lora_path='Lykon/dreamshaper-7', save_dW = False, base_sd='runwayml/stable-diffusion-v1-5', alpha=1.0):    
    # get weights of base sd models
    from diffusers import DiffusionPipeline
    _pipe = DiffusionPipeline.from_pretrained(
        base_sd, 
        torch_dtype=torch.float16,
        safety_checker = None,
    )
    sd_state_dict = _pipe.unet.state_dict()
    
    # get weights of the customized sd models, e.g., the aniverse downloaded from civitai.com    
    _pipe = DiffusionPipeline.from_pretrained(
        lora_path, 
        torch_dtype=torch.float16,
        safety_checker = None,
    )
    lora_unet_checkpoint = _pipe.unet.state_dict()
    
    # get the dW
    dW_dict = {}
    for key in lora_unet_checkpoint.keys():
        dW_dict[key] = lora_unet_checkpoint[key] - sd_state_dict[key]
    
    # return and save dW dict
    if save_dW:
        save_name = lora_path.split('/')[-1] + '_dW.pt'
        torch.save(dW_dict, save_name)
        
    pipe_rf = merge_dW_to_unet(pipe_rf, dW_dict=dW_dict, alpha=alpha)
    pipe_rf.vae = _pipe.vae
    pipe_rf.text_encoder = _pipe.text_encoder
    
    return dW_dict



pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")

insta_pipe = RectifiedFlowPipeline.from_pretrained("XCLiu/instaflow_0_9B_from_sd_1_5", torch_dtype=torch.float16) 
dW_dict = get_dW_and_merge(insta_pipe, lora_path="Lykon/dreamshaper-7", save_dW=False, alpha=1.0)     
insta_pipe.to("cuda")

global img

@torch.no_grad()
def set_new_latent_and_generate_new_image(seed, prompt, randomize_seed, num_inference_steps=1, guidance_scale=0.0):
    print('Generate with input seed')
    global img
    negative_prompt=""
    if randomize_seed:
        seed = np.random.randint(0, 2**32)
    seed = int(seed)
    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale)
    print(seed, num_inference_steps, guidance_scale)

    t_s = time.time()
    generator = torch.manual_seed(seed)
    images = insta_pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, generator=generator).images 
    inf_time = time.time() - t_s 

    img = copy.copy(np.array(images[0]))

    return images[0], inf_time, seed

@torch.no_grad()
def refine_image_512(prompt):
    print('Refine with SDXL-Refiner (512)')
    global img

    t_s = time.time()
    img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2) / 255.0 
    img = img.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
    new_image = pipe(prompt, image=img).images[0] 
    print('time consumption:', time.time() - t_s) 
    new_image = np.array(new_image) * 1.0 / 255.

    img = copy.copy(new_image)

    return new_image


with gr.Blocks() as gradio_gui:
    gr.Markdown(
    """
    # InstaFlow! One-Step Stable Diffusion with Rectified Flow [[paper]](https://arxiv.org/abs/2309.06380)
    ## This is a demo of one-step InstaFlow-0.9B with [dreamshaper-7](https://huggingface.co/Lykon/dreamshaper-7) (a LoRA that improves image quality) and measures the inference time.  
    """)

    with gr.Row():
        with gr.Column(scale=0.4):
            with gr.Group():
                gr.Markdown("Generation from InstaFlow-0.9B")
                im = gr.Image()
        
        with gr.Column(scale=0.4):
            inference_time_output = gr.Textbox(value='0.0', label='Inference Time with One-Step InstaFlow (Second)')
            seed_input = gr.Textbox(value='101098274', label="Random Seed") 
            randomize_seed = gr.Checkbox(label="Randomly Sample a Random Seed", value=True)
            prompt_input = gr.Textbox(value='A high-resolution photograph of a waterfall in autumn; muted tone', label="Prompt")

            new_image_button = gr.Button(value="One-Step Generation with InstaFlow and the Random Seed")
            new_image_button.click(set_new_latent_and_generate_new_image, inputs=[seed_input, prompt_input, randomize_seed], outputs=[im, inference_time_output, seed_input])

            refine_button_512 = gr.Button(value="Refine One-Step Generation with SDXL Refiner (Resolution: 512)")
            refine_button_512.click(refine_image_512, inputs=[prompt_input], outputs=[im])


gradio_gui.launch()   

