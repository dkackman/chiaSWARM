import torch
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from ..output_processor import OutputProcessor
from .upscale import upscale_image
from ..type_helpers import has_method
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

def diffusion_if_callback(device_identifier, model_name, **kwargs):
    # stage 1
    stage_1 = DiffusionPipeline.from_pretrained(model_name, variant="fp16", torch_dtype=torch.float16).to(device_identifier)
    stage_1.enable_model_cpu_offload()

    # stage 2
    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
    ).to(device_identifier)
    #stage_2.enable_model_cpu_offload()

    # stage 3
    safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
    stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16).to(device_identifier)
    #stage_3.enable_model_cpu_offload()

    prompt = kwargs.get("prompt", "")
    negative_prompt = kwargs.get("prompt", None)
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt, negative_prompt)

    generator = kwargs.pop("generator")
    image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images

    image = stage_2(
        image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
    ).images

    images = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
    images[0].save("./if_stage_III.png")

    pipeline_config = {}
    # if any image is nsfw, flag the entire result
    if (
        hasattr(stage_3, "nsfw_content_detected")
        and stage_3.nsfw_content_detected is not None
        and (
            (isinstance(stage_3.nsfw_content_detected, bool) and stage_3.nsfw_content_detected)
            or (
                isinstance(stage_3.nsfw_content_detected, list)
                and len(stage_3.nsfw_content_detected) >= 1
            )
        )
    ):
        for _ in filter(lambda nsfw: nsfw, stage_3.nsfw_content_detected):  # type: ignore
            pipeline_config["nsfw"] = True

    output_processor = OutputProcessor(
        kwargs.pop("outputs", ["primary"]),
        kwargs.pop("content_type", "image/jpeg"),
    )
    output_processor.add_outputs(images)
    return (output_processor.get_results(), pipeline_config)  # type: ignore
