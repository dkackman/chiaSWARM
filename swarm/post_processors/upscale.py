import torch
from diffusers import StableDiffusionLatentUpscalePipeline


def upscale_image_sdx2(
    image,
    device_identifier,
    prompt,
    negative_prompt,
    num_images_per_prompt,
    generator,
    preserve_vram,
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


def upscale_image(samples, upscale_method, scale_by):
    width = round(samples["images"].shape[3] * scale_by)
    height = round(samples["images"].shape[2] * scale_by)
    s = common_upscale(samples["images"], width, height, upscale_method, "disabled")
    return s


def common_upscale(samples, width, height, upscale_method, crop=False):
    if crop == "center":
        old_width = samples.shape[3]
        old_height = samples.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples[:, :, y : old_height - y, x : old_width - x]
    else:
        s = samples

    return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)
