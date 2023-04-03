import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionLatentUpscalePipeline,
)
from .output_processor import OutputProcessor


def diffusion_callback(device_id, model_name, **kwargs):
    scheduler_type = kwargs.pop("scheduler_type", DPMSolverMultistepScheduler)
    pipeline_type = kwargs.pop("pipeline_type", StableDiffusionPipeline)
    num_images_per_prompt = kwargs.pop("num_images_per_prompt", 1)
    upscale = kwargs.pop("upscale", False)
    if upscale:  # if upscaling stay in latent space
        kwargs["output_type"] = "latent"

    pipeline = pipeline_type.from_pretrained(
        model_name,
        revision=kwargs.pop("revision", "main"),
        torch_dtype=torch.float16,
    )
    pipeline = pipeline.to(f"cuda:{device_id}")  # type: ignore

    pipeline.scheduler = scheduler_type.from_config(  # type: ignore
        pipeline.scheduler.config  # type: ignore
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

    mem_info = torch.cuda.mem_get_info(device_id)
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
        if has_method(pipeline, "enable_xformers_memory_efficient_attention"):
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
        images = upscale_latents(
            images,
            device_id,
            kwargs["prompt"],
            num_images_per_prompt,
            kwargs["generator"],
        )

    output_processor.add_outputs(images)
    return (output_processor.get_results(), pipeline.config)  # type: ignore


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


def has_method(o, name):
    return callable(getattr(o, name, None))
