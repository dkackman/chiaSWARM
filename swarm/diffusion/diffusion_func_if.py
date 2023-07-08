import torch
from diffusers import (
    DiffusionPipeline,
)
from diffusers.utils import pt_to_pil
from ..post_processors.output_processor import OutputProcessor, is_nsfw
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch


def diffusion_if_callback(device_identifier, model_name, **kwargs):
    pipe = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-M-v1.0",
        variant="fp16",
        text_encoder=None,
        torch_dtype=torch.float16,
    )
    pipe.to(device_identifier)
    pipe_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-M-v1.0",
        variant="fp16",
        text_encoder=None,
        torch_dtype=torch.float16,
    )
    pipe_2.to(device_identifier)
    pipe_3 = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16
    )
    pipe_3.to(device_identifier)

    prompt = kwargs.get("prompt", "")
    negative_prompt = kwargs.get("prompt", None)
    prompt_embeds = torch.randn((1, 2, 4096), dtype=torch.float16)
    negative_embeds = torch.randn((1, 2, 4096), dtype=torch.float16)
    # prompt_embeds, negative_embeds = pipe.encode_prompt(prompt, negative_prompt)

    generator = kwargs.pop("generator")
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        generator=generator,
        output_type="pt",
    ).images
    pt_to_pil(image)[0].save("./if_stage_I.png")

    image = pipe_2(
        image=image,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        generator=generator,
        output_type="pt",
    ).images
    pt_to_pil(image)[0].save("./if_stage_II.png")

    images = pipe_3(
        prompt=prompt, image=image, generator=generator, noise_level=100
    ).images
    images[0].save("./if_stage_III.png")

    pipeline_config = {}
    pipeline_config["nsfw"] = is_nsfw(p)

    output_processor = OutputProcessor(
        kwargs.pop("outputs", ["primary"]),
        kwargs.pop("content_type", "image/jpeg"),
    )
    output_processor.add_outputs(images)
    return (output_processor.get_results(), pipeline_config)  # type: ignore
