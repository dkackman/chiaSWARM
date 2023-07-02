import torch
from diffusers import StableDiffusionLatentUpscalePipeline
from ..type_helpers import has_method


def upscale_image(
    image, device_identifier, prompt, negative_prompt, num_images_per_prompt, generator
):
    print("Upscaling...")
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "stabilityai/sd-x2-latent-upscaler",
        torch_dtype=torch.float16,
    )

    upscaler = upscaler.to(device_identifier)
    # upscaler.enable_attention_slicing()
    # upscaler.enable_sequential_cpu_offload()
    if has_method(upscaler, "enable_xformers_memory_efficient_attention"):
        upscaler.enable_xformers_memory_efficient_attention()

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
