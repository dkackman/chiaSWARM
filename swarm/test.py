from .worker import startup, do_work
from .job_arguments import format_args
import asyncio
from . import __version__
from .gpu.device import Device
from .settings import load_settings

test_job = {
    "id": "__test__",
    "model_name": "stabilityai/stable-diffusion-2-1",
    "prompt": "spoons",
    "num_inference_steps": 10,
    "outputs": ["primary", "inference_image_strip"],
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
    "prompt": "A fantasy landscape, Cinematic lighting",
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
    "prompt": "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.",
    "guidance_scale": 3.5,
    "num_inference_steps": 50,
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
    "outputs": [
        "primary"
    ],
}

flux_fast_job = {
    "id": "__test__",
    "workflow": "txt2img",
    "model_name": "black-forest-labs/FLUX.1-schnell",
    "prompt": "a cartoon picture of someone being stabbed with spoons",
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
        "max_sequence_length": 256,
        "default_height": 1024,
        "default_width": 1024,
    },
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
