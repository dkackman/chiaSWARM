import torch
from diffusers import DiffusionPipeline, StableDiffusionLatentUpscalePipeline
from PIL import Image


def upscale_image(
    image, device_identifier, prompt, negative_prompt, num_images_per_prompt, generator
):
    print("Upscaling...")
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "stabilityai/sd-x2-latent-upscaler",
        torch_dtype=torch.float16,
    ).to(device_identifier)

    if num_images_per_prompt > 1:
        prompt = [prompt] * num_images_per_prompt

    image = upscaler(
        prompt=prompt,
        image=image,
        num_inference_steps=20,
        guidance_scale=0,
        negative_prompt=negative_prompt,
        generator=generator,
    ).images[0]

    return [image]


def upscale_video(prompt, scheduler_type, video_frames, strength=0.6):
    upscaler = DiffusionPipeline.from_pretrained(
        "cerspense/zeroscope_v2_XL", torch_dtype=torch.float16
    )
    upscaler.scheduler = scheduler_type.from_config(
        upscaler.scheduler.config, use_karras_sigmas=True
    )
    upscaler.enable_vae_slicing()

    video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]
    return upscaler(prompt, video=video, strength=strength).frames
