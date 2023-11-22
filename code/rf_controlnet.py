import argparse, pathlib
import numpy as np
from PIL import Image
import torch
from diffusers.utils import load_image
from diffusers import ControlNetModel
from controlnet_aux import CannyDetector, OpenposeDetector, MidasDetector, HEDdetector

from pipeline_rf_ctrl import RectifiedFlowCtrlPipeline

def main(args):
    instaflow = args.instaflow
    ctrl_type = args.ctrl_type
    seed = args.seed
    save_dir = args.save_dir
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    ## Prepare conditioning signals
    prompt = args.prompt
    
    if ctrl_type == 'depth':
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", 
            torch_dtype=torch.float16,
            # local_files_only=True,
        )
        preprocessor = MidasDetector.from_pretrained("lllyasviel/Annotators")
    elif ctrl_type == 'canny':
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", 
            torch_dtype=torch.float16,
            # local_files_only=True,
        )
        preprocessor = CannyDetector()
        
    elif ctrl_type == 'openpose':
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose", 
            torch_dtype=torch.float16,
            # local_files_only=True,
        )
        preprocessor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    else:
        assert False

    ref_image = np.array(Image.open(args.ref_img))
    ref_image = preprocessor(ref_image, image_resolution=512, output_type="np")
    ref_img = Image.fromarray(ref_image)
    ref_name = args.ref_img.split('/')[-1].split('.')[0]
    ref_img.save(f'{save_dir}/{ref_name}_{ctrl_type}.png')


    ## Define pipeline
    if instaflow:
        pipe = RectifiedFlowCtrlPipeline.from_pretrained(
            "XCLiu/instaflow_0_9B_from_sd_1_5", 
            torch_dtype=torch.float16,
            controlnet=controlnet, 
            # local_files_only=True,
            ) 
    else:
        pipe = RectifiedFlowCtrlPipeline.from_pretrained(
            "XCLiu/2_rectified_flow_from_sd_1_5", 
            torch_dtype=torch.float16,
            controlnet=controlnet, 
            # local_files_only=True,
            )
    pipe.to("cuda")


    ## Sampling
    generator = torch.manual_seed(seed)
    W, H = ref_img.size
    if instaflow:
        n_step = 1
        images = pipe(
            prompt, 
            height=H, width=W, 
            num_inference_steps=n_step, 
            guidance_scale=1.0, 
            generator = generator,
            image=ref_img,
        ).images
    else:
        n_step = args.n_step
        ## 2-rectified flow is a multi-step text-to-image generative model.
        ## It can generate with extremely few steps, e.g, 2-8
        ## For guidance scale, the optimal range is [1.0, 2.0], which is smaller than normal Stable Diffusion.
        ## You may set negative_prompts like normal Stable Diffusion
        images = pipe(prompt=prompt, 
                    height=H, width=W,
                    negative_prompt="", 
                    num_inference_steps=n_step, 
                    guidance_scale=1.5,
                    generator = generator,
                    image=ref_img,
                    ).images 
    
    
    prompt = '-'.join(prompt.replace(',', ' ').split(' '))
    if instaflow:
        save_name = f"{save_dir}/{ref_name}-{ctrl_type}_step-{n_step}-seed-{seed}_{prompt}_insta.png"
    else:
        save_name = f"{save_dir}/{ref_name}-{ctrl_type}_step-{n_step}-seed-{seed}_{prompt}.png"
    print(f'save to: {save_name}')
    images[0].save(save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",  type=str, default="tmp")
    parser.add_argument("--seed",  type=int, default=0)
    parser.add_argument("--n_step",  type=int, default=25)
    parser.add_argument("--instaflow",  action="store_true")
    parser.add_argument("--ctrl_type",  type=str, default="openpose")
    parser.add_argument("--prompt",  type=str, default="A woman.")
    parser.add_argument("--ref_img",  type=str, default="assets/tom.png")
    
    args = parser.parse_args()
    main(args)
