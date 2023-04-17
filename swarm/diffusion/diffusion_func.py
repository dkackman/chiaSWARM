import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from ..output_processor import OutputProcessor
from .upscale import upscale_image
from ..type_helpers import has_method


def diffusion_callback(device_identifier, model_name, **kwargs):
    scheduler_type = kwargs.pop("scheduler_type", DPMSolverMultistepScheduler)
    pipeline_type = kwargs.pop("pipeline_type", StableDiffusionPipeline)
    num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)
    upscale = kwargs.pop("upscale", False)
    textual_inversion = kwargs.pop("textual_inversion", None)
    supports_xformers = kwargs.pop("supports_xformers", True)

    pipeline = pipeline_type.from_pretrained(
        model_name,
        revision=kwargs.pop("revision", "main"),
        torch_dtype=torch.float16,
    )
    pipeline = pipeline.to(device_identifier)  # type: ignore

    if textual_inversion is not None:
        try:
            pipeline.load_textual_inversion(textual_inversion)
        except Exception:
            raise ValueError(
                f"Textual inversion\n{textual_inversion}\nis incompatible with\n{model_name}"
            )

    if has_method(pipeline, "scheduler"):
        pipeline.scheduler = scheduler_type.from_config(  # type: ignore
            pipeline.scheduler.config, use_karras_sigmas=True  # type: ignore
        )

    output_processor = OutputProcessor(
        kwargs.pop("outputs", ["primary"]),
        kwargs.pop("content_type", "image/jpeg"),
    )

    if output_processor.need_intermediates():
        print("Capturing latents")

        def latents_callback(i, t, latents):
            output_processor.add_latents(pipeline, latents)  # type: ignore

        kwargs["callback"] = latents_callback
        kwargs["callback_steps"] = 5

    mem_info = torch.cuda.mem_get_info(device_identifier)
    # if we're upscaling or mid-range on mem, preserve memory vs performance
    if (
        (
            upscale and mem_info[1] < 16000000000
        )  # for 3090's etc just letem go full bore
        or num_images_per_prompt > 1
        or mem_info[1] < 12000000000
    ):
        # not all pipelines share these methods, so check first
        if has_method(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        if supports_xformers and has_method(
            pipeline, "enable_xformers_memory_efficient_attention"
        ):
            pipeline.enable_xformers_memory_efficient_attention()
        if has_method(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()  # type: ignore
        if has_method(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()  # type: ignore
        if has_method(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()  # type: ignore

    p = pipeline(**kwargs)  # type: ignore

    # if any image is nsfw, flag the entire result
    if (
        hasattr(p, "nsfw_content_detected")
        and p.nsfw_content_detected is not None  # type: ignore
        and len(p.nsfw_content_detected) >= 1  # type: ignore
    ):
        for _ in filter(lambda nsfw: nsfw, p.nsfw_content_detected):  # type: ignore
            pipeline.config["nsfw"] = True

    images = p.images  # type: ignore
    if upscale:
        images = upscale_image(
            images,
            device_identifier,
            kwargs["prompt"],
            num_images_per_prompt,
            kwargs["generator"],
        )

    output_processor.add_outputs(images)
    return (output_processor.get_results(), pipeline.config)  # type: ignore
