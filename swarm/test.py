from .worker import startup, do_work
from .job_arguments import format_args
import asyncio
from . import __version__
from .gpu.device import Device
from .settings import load_settings

test_job = {
    "id": "__test__",
    "model_name": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "prompt": "A picture of Cthulhu, the many testicled god of the deep",
    "num_inference_steps": 25,
    "outputs": ["primary"],
}

txt2audio_job = {
    "id": "__test__",
    "model_name": "cvssp/audioldm",
    "workflow": "txt2audio",
    "prompt": "Techno music with a strong, upbeat tempo and high melodic riffs",
    "num_inference_steps": 10,
    "outputs": ["primary"],
}

vid2vid_job = {
    "id": "__test__",
    "model_name": "timbrooks/instruct-pix2pix",
    "prompt": "make it sunny",
    "negative_prompt": "ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, blurry, bad anatomy, bad proportions, cloned face, disfigured, out of frame, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, long neck",
    "num_inference_steps": 10,
    "workflow": "vid2vid",
    "video_uri": "https://nftfactory.blob.core.windows.net/images/Pexels%20Videos%202795750.mp4",
    "outputs": ["primary"],
}

txt2vidJob = {
    "id": "__test__",
    "model_name": "timbrooks/instruct-pix2pix",
    "prompt": "dogs dancing",
    "negative_prompt": "ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, blurry, bad anatomy, bad proportions, cloned face, disfigured, out of frame, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, long neck",
    "num_inference_steps": 10,
    "workflow": "txt2vid",
    "outputs": ["primary"],
}

txt2vidJob2 = {
    "id": "__test__",
    "model_name": "damo-vilab/text-to-video-ms-1.7b",
    "prompt": "Darth Vader surfing a wave",
    "workflow": "txt2vid",
    "outputs": ["primary"],
    "num_frames": 24,
    "content_type": "video/webm",
}

txt2vidZeroscopeJob = {
    "id": "__test__",
    "model_name": "cerspense/zeroscope_v2_576w",
    "prompt": "A marmot dancing in a disco",
    "workflow": "txt2vid",
    "outputs": ["primary"],
    "height": 320,
    "width": 576,
    "num_frames": 36,
    "content_type": "video/webm",
    "upscale": False,
}

bark_job = {
    "id": "__test__",
    "model_name": "suno/bark",
    "prompt": "Hola, mi nombre es Pepe'. Y, eh, y me gusta pepe' coin. [laughs] Pero me gustaría que apostara si tuviera más marmotas",
    "workflow": "txt2audio",
    "outputs": ["primary"],
}

if_job = {
    "id": "__test__",
    "model_name": "DeepFloyd/IF-II-L-v1.0",
    "prompt": 'a photo of a green frog wearing blue sunglasses standing in front of the eiffel tower holding a sign that says "i shill chia"',
    "workflow": "txt2img",
    "outputs": ["primary"],
}

kandinsky_job = {
    "id": "__test__",
    "model_name": "kandinsky-community/kandinsky-2-2-decoder",
    "prompt": "A picture of Cthulhu, the many testicled god of the deep",
    "negative_prompt": "low quality, bad quality",
    "workflow": "txt2img",
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "outputs": ["primary"],
    "parameters": {
        "pipeline_type": "AutoPipelineForText2Image",
        "prior_guidance_scale": 1.0,
    },
}
kandinsky_img2img_job = {
    "id": "__test__",
    "model_name": "kandinsky-community/kandinsky-2-1",
    "start_image_uri": "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg",
    "prompt": "A fantasy landscape, Cinematic lighting",
    "negative_prompt": "low quality, bad quality",
    "workflow": "img2img",
    "num_inference_steps": 30,
    "outputs": ["primary"],
    "parameters": {
        "pipeline_type": "AutoPipelineForImage2Image",
        "prior_guidance_scale": 1.0,
    },
}
kandinsky_controlnet_job = {
    "id": "__test__",
    "model_name": "kandinsky-community/kandinsky-2-2-controlnet-depth",
    "start_image_uri": "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/cat.png",
    "prompt": "A robot, 4k photo",
    "negative_prompt": "low quality, bad quality",
    "workflow": "img2img",
    "num_inference_steps": 30,
    "outputs": ["primary"],
    "parameters": {
        "pipeline_type": "KandinskyV22ControlnetPipeline",
        "pipeline_prior_type": "KandinskyV22PriorPipeline",
        "prior_model_name": "kandinsky-community/kandinsky-2-2-prior",
        "default_height": 768,
        "default_width": 768,
    },
}
kandinsky3_job = {
    "id": "__test__",
    "model_name": "kandinsky-community/kandinsky-3",
    "prompt": "A fantasy landscape, Cinematic lighting",
    "negative_prompt": "low quality, bad quality",
    "workflow": "txt2img",
    "variant": "fp16",
    "num_inference_steps": 20,
    "guidance_scale": 4.0,
    "outputs": ["primary"],
    "parameters": {
        "large_model": True,
        "always_offload": True,
        "use_bfloat16": True,
        "pipeline_type": "AutoPipelineForText2Image",
        "allow_user_scheduler": False
    },
}
animatelcm_job = {
    "id": "__test__",
    "model_name": "emilianJR/epiCRealism",
    "prompt": "A dancing marmot, 4k, high resolution",
    "negative_prompt": "bad quality, worse quality, low resolution",
    "workflow": "txt2vid",
    "num_inference_steps": 6,
    "guidance_scale": 2.0,
    "outputs": ["primary"],
    "num_frames": 32,
    "content_type": "image/gif",
    "parameters": {
        "pipeline_type": "AnimateDiffPipeline",
        "allow_user_scheduler": False,
        "scheduler_type": "LCMScheduler",
        "motion_adapter":
        {
            "model_name": "wangfuyun/AnimateLCM"
        },
        "lora" :
        {
            "model_name": "wangfuyun/AnimateLCM",
            "weight_name": "AnimateLCM_sd15_t2v_lora.safetensors",
            "adapter_name": "lcm-lora",
            "weight": 0.8
        },
        "scheduler_args":
        {
            "beta_schedule": "linear"
        }
    },
}

animatediff_job = {
    "id": "__test__",
    "model_name": "emilianJR/epiCRealism",
    "prompt": "A dancing marmot, 4k, high resolution",
    "negative_prompt": "bad quality, worse quality, low resolution",
    "workflow": "txt2vid",
    "num_inference_steps": 6,
    "guidance_scale": 2.0,
    "outputs": ["primary"],
    "num_frames": 32,
    "content_type": "image/gif",
    "parameters": {
        "pipeline_type": "AnimateDiffPipeline",
        "allow_user_scheduler": False,
        "scheduler_type": "LCMScheduler",
        "motion_adapter":
        {
            "model_name": "ByteDance/AnimateDiff-Lightning",
            "num_inference_steps": 4,
            "checkpoint_file": "animatediff_lightning_4step_diffusers.safetensors",
        },
        "scheduler_args": {
            "scheduler_type": "EulerDiscreteScheduler",
            "beta_schedule": "linear",
            "timestep_spacing": "trailing"
        },
    },
}

qr_monster_job = {
    "id": "__test__",
    "workflow": "img2img",
    "model_name": "SG161222/Realistic_Vision_V5.1_noVAE",
    "prompt": "a badger",
    "negative_prompt": "disfigured, ugly, misshapen, low quality, nsfw, blurry",
    "guidance_scale": 7.5,
    "num_inference_steps": 40,
    "strength": 0.95,
    "content_type": "image/jpeg",
    "start_image_uri": "",
    "num_images_per_prompt": 1,
    "parameters": {
        "scheduler_type": "EulerAncestralDiscreteScheduler",
        "controlnet": {
            "type": "qrcode",
            "controlnet_model_name": "monster-labs/control_v1p_sd15_qrcode_monster",
            "preprocess": True,
            "controlnet_conditioning_scale": 0.88,
            "qr_code_contents": "https://api.mintgarden.io/collections/col1wsnmtt8egrgd4zw3kd53zx2wczzdxdkmf9wgntg9atnrhvma4mpqgzcjdh/offers/random/bech32",
            "parameters": {
                "controlnet_model_type": "ControlNetModel",
                "controlnet_prepipeline_type": "StableDiffusionControlNetPipeline"
            }
        },
        "vae": "stabilityai/sd-vae-ft-mse"
    },
    "outputs": [
        "primary"
    ],

    "revision": "fp16"
}

flux_job = {
    "id": "__test__",
    "workflow": "txt2img",
    "model_name": "black-forest-labs/FLUX.1-dev",
    "prompt": "A picture of Cthulhu, the many testicled god of the deep",
    "guidance_scale": 1.7,
    "num_inference_steps": 40,
    "content_type": "image/jpeg",
    "num_images_per_prompt": 1,
    "parameters": {
        "large_model": True,
        "always_offload_sequential": True,
        "use_bfloat16": True,
        "pipeline_type": "FluxPipeline",
        "allow_user_scheduler": False,
        "max_sequence_length": 512,
        "default_height": 1024,
        "default_width": 1024,
    },
    # "lora": "alvdansen/flux-koda/araminta_k_flux_koda.safetensors",
    "outputs": [
        "primary"
    ],
}

flux_fast_job = {
    "id": "__test__",
    "workflow": "txt2img",
    "model_name": "black-forest-labs/FLUX.1-schnell",
    "prompt": "a marmot holding a sign that says 'feed me mojos!'",
    "guidance_scale": 0,
    "num_inference_steps": 5,
    "content_type": "image/jpeg",
    "num_images_per_prompt": 1,
    "parameters": {
        "large_model": True,
        "always_offload_sequential": True,
        "use_bfloat16": True,
        "pipeline_type": "FluxPipeline",
        "allow_user_scheduler": False,
        "max_sequence_length": 256
    },
    #"lora": "alvdansen/flux-koda/araminta_k_flux_koda.safetensors",
    "outputs": [
        "primary"
    ],
}

cog_video_job = {
    "id": "__test__",
    "workflow": "txt2vid",
    "model_name": "THUDM/CogVideoX-2b",
    "prompt": "Tracking shot,late afternoon light casting long shadows,a cyclist in athletic gear pedaling down a scenic mountain road,winding path with trees and a lake in the background,invigorating and adventurous atmosphere.",
    "guidance_scale": 6,
    "num_inference_steps": 50,
    "content_type": "image/mp4",
    "num_frames": 49,
    "parameters": {
        "large_model": True,
        "always_offload_sequential": True,
        "use_bfloat16": True,
        "pipeline_type": "CogVideoXPipeline",
        "allow_user_scheduler": False,
    },
    "outputs": [
        "primary"
    ],
}

sd35_job = {
    "id": "__test__",
    "workflow": "txt2img",
    "model_name": "stabilityai/stable-diffusion-3.5-large",
    "prompt": "A picture of Cthulhu, the many testicled god of the deep",
    "guidance_scale": 3.5,
    "num_inference_steps": 25,
    "content_type": "image/jpeg",
    "num_images_per_prompt": 1,
    "parameters": {
        "large_model": True,
        "always_offload": True,
        "use_bfloat16": True,
        "pipeline_type": "StableDiffusion3Pipeline",
    },
    # "lora": "alvdansen/flux-koda/araminta_k_flux_koda.safetensors",
    "outputs": [
        "primary"
    ],
}

settings = load_settings()


async def run_test(job):
    await startup()
    try:
        func, args = await format_args(job, settings, "cuda")
        result = await do_work(Device(0), func, args)

        if "error" in result["pipeline_config"]:
            print(result["pipeline_config"]["error"])
        else:
            print("ok")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    asyncio.run(run_test(flux_fast_job))
