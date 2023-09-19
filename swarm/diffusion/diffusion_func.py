import torch
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    ControlNetModel,
    AutoencoderKL,
)
from ..type_helpers import has_method
from ..post_processors.output_processor import OutputProcessor, is_nsfw
from .pipeline_steps import prior_pipeline, refiner_pipeline, upscale_pipeline


def diffusion_callback(device_identifier, model_name, **kwargs):
    # these arguments won't be passed directly to the pipeline
    # because they are not used by the pipelines directly
    # everything else in kwargs gets passed through
    scheduler_type = kwargs.pop("scheduler_type", DPMSolverMultistepScheduler)
    pipeline_type = kwargs.pop("pipeline_type", DiffusionPipeline)
    upscale = kwargs.pop("upscale", False)
    textual_inversion = kwargs.pop("textual_inversion", None)
    lora = kwargs.pop("lora", None)
    cross_attention_scale = kwargs.pop("cross_attention_scale", 1.0)
    refiner = kwargs.pop("refiner", None)

    # set output_type if already there or upscale is/ selected (we use the latent upscaler)
    output_type = kwargs.pop("output_type", "latent" if upscale else None)
    if output_type is not None:
        kwargs["output_type"] = output_type

    output_processor = OutputProcessor(
        kwargs.pop("outputs", ["primary"]),
        kwargs.pop("content_type", "image/jpeg"),
    )

    load_pipeline_args = {}
    load_pipeline_args["revision"] = kwargs.pop("revision", "main")
    load_pipeline_args["variant"] = kwargs.pop("variant", None)
    load_pipeline_args["torch_dtype"] = torch.float16
    load_pipeline_args["use_safe_tensors"] = kwargs.pop("use_safe_tensors", None)

    if "vae" in kwargs:
        load_pipeline_args["vae"] = AutoencoderKL.from_pretrained(
            kwargs.pop("vae"), torch_dtype=torch.float16
        ).to(device_identifier)

    if "controlnet_model_name" in kwargs:
        load_pipeline_args["controlnet"] = ControlNetModel.from_pretrained(
            kwargs.pop("controlnet_model_name"),
            revision=kwargs.pop("controlnet_revision", "main"),
            torch_dtype=torch.float16,
        ).to(device_identifier)

        if kwargs.pop("save_preprocessed_input", False):
            if "control_image" not in kwargs:
                output_processor.add_other_outputs(
                    "preprocessed_input", [kwargs.get("image")]
                )
            else:
                output_processor.add_other_outputs(
                    "preprocessed_input", [kwargs.get("control_image")]
                )

    main_pipeline = pipeline_type.from_pretrained(model_name, **load_pipeline_args)

    if textual_inversion is not None:
        try:
            main_pipeline.load_textual_inversion(textual_inversion)
        except Exception as e:
            raise ValueError(
                f"Textual inversion\n{textual_inversion}\nis incompatible with\n{model_name}\n{lora}\n\n{e}"
            ) from e

    main_pipeline = main_pipeline.to(device_identifier)

    if lora is not None and has_method(main_pipeline, "load_lora_weights"):
        try:
            main_pipeline.load_lora_weights(
                lora["lora"],
                weight_name=lora["weight_name"],
                subfolder=lora["subfolder"],
                use_safetensors=True,
            )
            kwargs["cross_attention_kwargs"] = {"scale": cross_attention_scale}

        except Exception as e:
            raise ValueError(
                f"Could not load lora \n{lora}\nIt might be incompatible with {model_name}\n{e}"
            ) from e

    # not all pipelines use a scheduler, so check first (UnCLIPPipeline)
    if has_method(main_pipeline, "scheduler"):
        main_pipeline.scheduler = scheduler_type.from_config(
            main_pipeline.scheduler.config, use_karras_sigmas=True
        ).to(device_identifier)

    mem_info = torch.cuda.mem_get_info(device_identifier)
    # if we're mid-range on mem, preserve memory vs performance
    preserve_vram = (
        kwargs.get("num_images_per_prompt", 1) > 1 and mem_info[1] < 12000000000
    ) or (kwargs.pop("large_model", False) and mem_info[1] < 12000000000)

    if preserve_vram:
        # not all pipelines share these methods, so check first
        if has_method(main_pipeline, "enable_vae_slicing"):
            main_pipeline.enable_vae_slicing()
        if has_method(main_pipeline, "enable_model_cpu_offload"):
            main_pipeline.enable_model_cpu_offload()

    prior_pipeline(kwargs, device_identifier)

    images = main_pipeline(**kwargs).images

    images = refiner_pipeline(refiner, images, device_identifier, preserve_vram, kwargs)

    images = upscale_pipeline(upscale, images, device_identifier, kwargs)

    main_pipeline.config["nsfw"] = is_nsfw(main_pipeline)
    output_processor.add_outputs(images)
    return (output_processor.get_results(), main_pipeline.config)
