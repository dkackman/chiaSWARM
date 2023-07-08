import torch
from diffusers import StableDiffusionLatentUpscalePipeline


def upscale_image(
    image, device_identifier, prompt, negative_prompt, num_images_per_prompt, generator, preserve_vram
):
    print("Upscaling...")
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "stabilityai/sd-x2-latent-upscaler",
        torch_dtype=torch.float16,
        use_safe_tensors=True,
    ).to(device_identifier)

    if preserve_vram:
        upscaler.enable_attention_slicing()
        
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
