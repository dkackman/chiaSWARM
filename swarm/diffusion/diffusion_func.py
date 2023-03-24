import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from .output_processor import OutputProcessor


def diffusion_callback(device_id, model_name, **kwargs):
    scheduler_type = kwargs.pop("scheduler_type", DPMSolverMultistepScheduler)
    pipeline_type = kwargs.pop("pipeline_type", StableDiffusionPipeline)

    pipeline = pipeline_type.from_pretrained(
        model_name,
        revision=kwargs.pop("revision"),
        torch_dtype=torch.float16,
    ).to( # type: ignore
        f"cuda:{device_id}"
    ) 

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
        
    p = pipeline(**kwargs)  # type: ignore

    # if any image is nsfw, flag the entire result
    if (
        hasattr(p, "nsfw_content_detected")
        and p.nsfw_content_detected is not None  # type: ignore
        and len(p.nsfw_content_detected) >= 1  # type: ignore
    ):
        for _ in filter(lambda nsfw: nsfw, p.nsfw_content_detected):  # type: ignore
            pipeline.config["nsfw"] = True

    output_processor.add_outputs(p.images)  # type: ignore
    return (output_processor.get_results(), pipeline.config)  # type: ignore
