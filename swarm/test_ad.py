
import os
import json
import torch
import random

from glob import glob
from datetime import datetime
from safetensors import safe_open

from diffusers import AutoencoderKL
from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from .animatediff.models.unet import UNet3DConditionModel
from .animatediff.pipelines.pipeline_animation import AnimationPipeline
from .animatediff.utils.util import save_videos_grid
from .animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from .animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora 


def animate():
    unet_additional_kwargs = {
        "unet_use_cross_frame_attention": False,
        "unet_use_temporal_attention": False,
        "use_motion_module": True,
        "motion_module_resolutions": [1, 2, 4, 8],
        "motion_module_mid_block": False,
        "motion_module_decoder_only": False,
        "motion_module_type": "Vanilla",
        "motion_module_kwargs": {
            "num_attention_heads": 8,
            "num_transformer_block": 1,
            "attention_block_types": ["Temporal_Self", "Temporal_Self"],
            "temporal_position_encoding": True,
            "temporal_position_encoding_max_len": 24,
            "temporal_attention_dim_div": 1
        }
    }    
    noise_scheduler_kwargs = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "linear"   
    } 

    stable_diffusion_dropdown = "E:\\animatediff\\models\\StableDiffusion\\stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(stable_diffusion_dropdown, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_dropdown, subfolder="text_encoder").cuda()    
    vae = AutoencoderKL.from_pretrained(stable_diffusion_dropdown, subfolder="vae").cuda()
    unet = UNet3DConditionModel.from_pretrained_2d(stable_diffusion_dropdown, subfolder="unet", **unet_additional_kwargs).cuda()
    scheduler = EulerDiscreteScheduler.from_config(**noise_scheduler_kwargs)
    pipeline = AnimationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler
    ).to("cuda")
    

if __name__ == "__main__":
    animate()   