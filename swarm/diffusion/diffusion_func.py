import torch
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    ControlNetModel,
)
from ..output_processor import OutputProcessor
from .upscale import upscale_image
from ..type_helpers import has_method, run_compile


def diffusion_callback(device_identifier, model_name, **kwargs):
    # these arguments won't be passed directly to the pipeline
    # everything else in kwargs gets passed through
    scheduler_type = kwargs.pop("scheduler_type", DPMSolverMultistepScheduler)
    pipeline_type = kwargs.pop("pipeline_type", DiffusionPipeline)
    num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)
    upscale = kwargs.pop("upscale", False)
    textual_inversion = kwargs.pop("textual_inversion", None)
    lora = kwargs.pop("lora", None)
    cross_attention_scale = kwargs.pop("cross_attention_scale", 1.0)

    output_processor = OutputProcessor(
        kwargs.pop("outputs", ["primary"]),
        kwargs.pop("content_type", "image/jpeg"),
    )

    if "controlnet_model_name" in kwargs:
        controlnet = ControlNetModel.from_pretrained(
            kwargs.pop("controlnet_model_name"),
            revision=kwargs.pop("controlnet_revision", "main"),
            torch_dtype=torch.float16,
        ).to(device_identifier)

        if kwargs.pop("save_preprocessed_input", False):
            output_processor.add_other_outputs(
                "preprocessed_input", [kwargs.get("control_image")]
            )

    pipeline = pipeline_type.from_pretrained(
        model_name,
        revision=kwargs.pop("revision", "main"),
        variant=kwargs.pop("variant", None),
        torch_dtype=torch.float16,
        controlnet=controlnet if "controlnet" in locals() else None
    )

    if textual_inversion is not None:
        try:
            pipeline.load_textual_inversion(textual_inversion)
        except Exception as e:
            raise ValueError(
                f"Textual inversion\n{textual_inversion}\nis incompatible with\n{model_name}\n{lora}\n\n{e}"
            ) from e

    pipeline = pipeline.to(device_identifier)

    # (UnCLIPPipeline) doesn't have a unet
    if hasattr(pipeline, "unet") and pipeline.unet is not None:
        pipeline.unet.to(memory_format=torch.channels_last)

    if hasattr(pipeline, "controlnet") and pipeline.controlnet is not None:
        pipeline.controlnet.to(memory_format=torch.channels_last)

    if run_compile and hasattr(pipeline, "unet") and pipeline.unet is not None:
        pipeline.unet = torch.compile(
            pipeline.unet, mode="reduce-overhead", fullgraph=True
        )
        if hasattr(pipeline, "controlnet") and pipeline.controlnet is not None:
            pipeline.controlnet = torch.compile(
                pipeline.controlnet, mode="reduce-overhead", fullgraph=True
            )

    if lora is not None and hasattr(pipeline, "unet") and pipeline.unet is not None:
        try:
            pipeline.unet.load_attn_procs(lora)
            kwargs["cross_attention_kwargs"] = {"scale": cross_attention_scale}

        except Exception as e:
            raise ValueError(
                f"Could not load lora \n{lora}\nIt might be incompatible with {model_name}\n{e}"
            ) from e

    # not all pipelines use a scheduler, so check first (UnCLIPPipeline)
    if has_method(pipeline, "scheduler"):
        pipeline.scheduler = scheduler_type.from_config(
            pipeline.scheduler.config, use_karras_sigmas=True
        )

    mem_info = torch.cuda.mem_get_info(device_identifier)
    # if we're upscaling or mid-range on mem, preserve memory vs performance
    if num_images_per_prompt > 1 or mem_info[1] < 12000000000:
        # not all pipelines share these methods, so check first
        if has_method(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()

    p = pipeline(**kwargs)

    # if any image is nsfw, flag the entire result
    if (
        hasattr(p, "nsfw_content_detected")
        and p.nsfw_content_detected is not None
        and (
            (isinstance(p.nsfw_content_detected, bool) and p.nsfw_content_detected)
            or (
                isinstance(p.nsfw_content_detected, list)
                and len(p.nsfw_content_detected) >= 1
            )
        )
    ):
        for _ in filter(lambda nsfw: nsfw, p.nsfw_content_detected):
            pipeline.config["nsfw"] = True

    images = p.images
    if upscale:
        images = upscale_image(
            images,
            device_identifier,
            kwargs.get("prompt", ""),
            kwargs.get("negative_prompt", ""),
            num_images_per_prompt,
            kwargs["generator"],
        )

    output_processor.add_outputs(images)
    return (output_processor.get_results(), pipeline.config)
