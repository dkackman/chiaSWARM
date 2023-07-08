import torch
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    ControlNetModel,
)
from ..type_helpers import has_method
from ..post_processors.output_processor import OutputProcessor, is_nsfw
from ..post_processors.upscale import upscale_image


def diffusion_callback(device_identifier, model_name, **kwargs):
    # these arguments won't be passed directly to the pipeline
    # everything else in kwargs gets passed through
    scheduler_type = kwargs.pop("scheduler_type", DPMSolverMultistepScheduler)
    pipeline_type = kwargs.pop("pipeline_type", DiffusionPipeline)
    upscale = kwargs.pop("upscale", False)
    textual_inversion = kwargs.pop("textual_inversion", None)
    lora = kwargs.pop("lora", None)
    cross_attention_scale = kwargs.pop("cross_attention_scale", 1.0)

    # set output_type if already there or upscale is selected (we use the latent upscaler)
    output_type = kwargs.pop("output_type", "latent" if upscale else None)
    if output_type is not None:
        kwargs["output_type"] = output_type

    use_safe_tensors = kwargs.pop("use_safe_tensors", None)

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
        controlnet=controlnet if "controlnet" in locals() else None,
        use_safe_tensors=use_safe_tensors,
    )

    if textual_inversion is not None:
        try:
            pipeline.load_textual_inversion(textual_inversion)
        except Exception as e:
            raise ValueError(
                f"Textual inversion\n{textual_inversion}\nis incompatible with\n{model_name}\n{lora}\n\n{e}"
            ) from e

    pipeline = pipeline.to(device_identifier)

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
        ).to(device_identifier)

    mem_info = torch.cuda.mem_get_info(device_identifier)
    # if we're mid-range on mem, preserve memory vs performance
    preserve_vram = (
        kwargs.get("num_images_per_prompt", 1) > 1 and mem_info[1] < 12000000000
    ) or (kwargs.pop("large_model", False) and mem_info[1] < 16000000000)

    if preserve_vram:
        # not all pipelines share these methods, so check first
        if has_method(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if has_method(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()

    refiner = kwargs.pop("refiner", None)

    p = pipeline(**kwargs)
    images = p.images
    pipeline.config["nsfw"] = is_nsfw(p)

    if refiner is not None:
        refiner_pipeline = DiffusionPipeline.from_pretrained(
            refiner["model_name"],
            variant=refiner.get("variant", None),
            revision=refiner.get("revision", "main"),
            torch_dtype=torch.float16,
            use_safetensors=refiner.get("use_safetensors", True),
        ).to(device_identifier)

        if preserve_vram and has_method(refiner_pipeline, "enable_model_cpu_offload"):
            refiner_pipeline.enable_model_cpu_offload()

        images = refiner_pipeline(
            image=images,
            prompt=kwargs.get("prompt", ""),
            negative_prompt=kwargs.get("negative_prompt", None),
            generator=kwargs["generator"],
        ).images

    if upscale:
        images = upscale_image(
            images,
            device_identifier,
            kwargs.get("prompt", ""),
            kwargs.get("negative_prompt", None),
            kwargs.get("num_images_per_prompt", 1),
            kwargs["generator"],
            preserve_vram,
        )

    output_processor.add_outputs(images)
    return (output_processor.get_results(), pipeline.config)
