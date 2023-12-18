import copy
import torch
from safetensors import safe_open
from pipeline_rf import RectifiedFlowPipeline
import argparse

def merge_dW_to_unet(pipe, dW_dict, alpha=1.0):
    _tmp_sd = pipe.unet.state_dict()
    for key in dW_dict.keys():
        _tmp_sd[key] += dW_dict[key] * alpha
    pipe.unet.load_state_dict(_tmp_sd, strict=False)
    return pipe

def load_hf_hub_lora(pipe_rf, lora_path='Lykon/dreamshaper-7', save_dW = False, base_sd='runwayml/stable-diffusion-v1-5', alpha=1.0):    
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

def load_civitai_lora(pipeline, checkpoint_path, multiplier, device, dtype):
    ### See https://github.com/huggingface/diffusers/issues/3064
    from safetensors.torch import load_file
    from collections import defaultdict

    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline

def main(args):

    ## define pipeline
    if args.instaflow:
        pipe = RectifiedFlowPipeline.from_pretrained(
        "XCLiu/instaflow_0_9B_from_sd_1_5", 
        torch_dtype=torch.float16,
        safety_checker=None,
        )
    else:
        pipe = RectifiedFlowPipeline.from_pretrained(
            "XCLiu/2_rectified_flow_from_sd_1_5", 
            torch_dtype=torch.float16,
            safety_checker=None,
            ) 
    

    ## load lora weights
    if args.lora_path != "":
        if args.lora_type == 'hf':
            dW_dict = load_hf_hub_lora(pipe, lora_path=args.lora_path, save_dW=False, alpha=1.0)
            pipe.to("cuda")  ### if GPU is not available, comment this line
        elif args.lora_type == 'civitai':
            pipe.to("cuda")
            pipe = load_civitai_lora(pipe, args.lora_path, 1.0, 'cuda', torch.float16)  
            
            ### NOTE: below is code snippet to combine together two loras
            # dW_dict = load_hf_hub_lora(pipe, lora_path='Lykon/dreamshaper-7', save_dW=False, alpha=1.0)
            # pipe.to("cuda")
            # pipe = load_civitai_lora(pipe, 'civitai/V1.1-17SciencefictioncityonMars.safetensors' , 1.0, 'cuda', torch.float16)
        else:
            raise NotImplementedError
    else:
        pipe.to("cuda")  ### if GPU is not available, comment this line
        


    ## sampling 
    generator = torch.manual_seed(args.seed)
    if args.instaflow:
        n_step = 1
        images = pipe(
            args.prompt, 
            num_inference_steps=n_step, 
            guidance_scale=1.0, 
            generator = generator,
        ).images
    else:
        n_step = args.n_step
        images = pipe(prompt=args.prompt, 
                    negative_prompt="painting, unreal, twisted", 
                    num_inference_steps=n_step, 
                    guidance_scale=1.5,
                    generator = generator,
                    ).images 

    if args.lora_path != "":
        lora_name = args.lora_path.split('/')[-1]
        if args.instaflow:
            lora_name = 'insta_' + lora_name
        images[0].save(f"{args.save_dir}/lora-{lora_name}_step-{n_step}_seed-{args.seed}_{args.prompt}.png")
    else:
        if args.instaflow:
            images[0].save(f"{args.save_dir}/no_lora_insta_step-{n_step}-seed-{args.seed}_{args.prompt}.png")
        else:
            images[0].save(f"{args.save_dir}/no_lora_step-{n_step}-seed-{args.seed}_{args.prompt}.png")
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",  type=str, default="tmp")
    parser.add_argument("--seed",  type=int, default=0)
    parser.add_argument("--n_step",  type=int, default=25)
    parser.add_argument("--instaflow",  action="store_true")
    parser.add_argument("--prompt",  type=str, default="A photo of a cute dog;masterpiece")
    parser.add_argument("--lora_type", type=str, choices=['civitai', 'hf'], default='hf')
    parser.add_argument("--lora_path", type=str, default='Lykon/dreamshaper-7')

    args = parser.parse_args()
    main(args)
