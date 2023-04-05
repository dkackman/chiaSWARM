import torch
from diffusers import StableDiffusionLatentUpscalePipeline
from ..type_helpers import has_method


def upscale_latents(
    low_res_latents, device_id, prompt, num_images_per_prompt, generator
):
    print("Upscaling...")
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "stabilityai/sd-x2-latent-upscaler",
        torch_dtype=torch.float16,
    )

    upscaler = upscaler.to(f"cuda:{device_id}")  # type: ignore
    upscaler.enable_attention_slicing()
    upscaler.enable_sequential_cpu_offload()  # type: ignore
    if has_method(upscaler, "enable_xformers_memory_efficient_attention"):
        upscaler.enable_xformers_memory_efficient_attention()

    if num_images_per_prompt > 1:
        prompt = [prompt] * num_images_per_prompt

    image = upscaler(  # type: ignore
        prompt=prompt,
        image=low_res_latents,
        num_inference_steps=20,
        guidance_scale=0,
        generator=generator,
    ).images[  # type: ignore
        0
    ]  # type: ignore

    return [image]