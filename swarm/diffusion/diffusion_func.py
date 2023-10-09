import torch
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    ControlNetModel,
    AutoencoderKL,
)
from ..type_helpers import has_method
from ..pre_processors.image_utils import center_crop_resize
from ..post_processors.output_processor import OutputProcessor, is_nsfw
from ..post_processors.upscale import upscale_image
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

    # if there is a controlnet load and configure it
    if "controlnet_model_name" in kwargs:
        controlnet_model_type = kwargs.pop("controlnet_model_type", ControlNetModel)

        load_pipeline_args["controlnet"] = controlnet_model_type.from_pretrained(
            kwargs.pop("controlnet_model_name"),
            revision=kwargs.pop("controlnet_revision", "main"),
            torch_dtype=torch.float16,
        ).to(device_identifier)

        if kwargs.pop("save_preprocessed_input", False):
            if "control_image" in kwargs:
                output_processor.add_other_outputs(
                    "control_image", [kwargs.get("control_image")]
                )
            elif "hint" in kwargs:
                output_processor.add_other_outputs(
                    "control_image", [kwargs.get("hint")]
                )
            elif "image" in kwargs:
                output_processor.add_other_outputs(
                    "control_image", [kwargs.get("image")]
                )

    # if there is a controlnet prepipeline execute it
    # this is how QR code monster works. it runs a prepipeline to get the latent
    # and then runs the main pipeline with the latent
    if "controlnet_prepipeline_type" in kwargs:
        controlnet_prepipeline_type = kwargs.pop("controlnet_prepipeline_type")
        prepipeline = controlnet_prepipeline_type.from_pretrained(
            model_name,
            controlnet=load_pipeline_args["controlnet"],
            vae=load_pipeline_args.get("vae", None),
            torch_dtype=torch.float16,
        ).to(device_identifier)
        # take out the original control_image
        control_image = kwargs.pop("control_image", None)

        args = kwargs.copy()
        args.pop("strength", None)
        image = prepipeline(output_type="latent", **args)

        # the start_image was center cropped and sized to 512x512
        # this will make is 1024x1024
        upscaled_latents = upscale_image(image, "nearest-exact", 2)
        # put the control_image back upscaled to match the latent dimensions
        if control_image is not None:
            kwargs["control_image"] = center_crop_resize(control_image, (1024, 1024))
            
        kwargs["image"] = upscaled_latents
        load_pipeline_args["unet"] = prepipeline.unet

    main_pipeline = pipeline_type.from_pretrained(model_name, **load_pipeline_args)

    if textual_inversion is not None:
        try:
            main_pipeline.load_textual_inversion(textual_inversion)
        except Exception as e:
            raise ValueError(
                f"Textual inversion\n{textual_inversion}\nis incompatible with\n{model_name}\n{lora}\n\n{e}"
            ) from e

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

    main_pipeline = main_pipeline.to(device_identifier)

    # not all pipelines use a scheduler, so check first (UnCLIPPipeline)
    if main_pipeline.scheduler is not None:
        main_pipeline.scheduler = scheduler_type.from_config(
            main_pipeline.scheduler.config, use_karras_sigmas=True
        )

    mem_info = torch.cuda.mem_get_info(device_identifier)
    # if we're mid-range on mem (12GB or less), preserve memory vs performance
    preserve_vram = (
        kwargs.get("num_images_per_prompt", 1) > 1 and mem_info[1] < 13884377600
    ) or (kwargs.pop("large_model", False) and mem_info[1] < 13884377600)

    if preserve_vram:
        if main_pipeline.unet is not None:
            main_pipeline.unet.to(memory_format=torch.channels_last)
        # not all pipelines share these methods, so check first
        if has_method(main_pipeline, "enable_vae_slicing"):
            main_pipeline.enable_vae_slicing()
        if has_method(main_pipeline, "enable_vae_tiling"):
            main_pipeline.enable_vae_tiling()
        if has_method(main_pipeline, "enable_model_cpu_offload"):
            main_pipeline.enable_model_cpu_offload()

    # prior pipeline is used by the Kandinsky and others
    prior_pipeline(kwargs, device_identifier)

    images = main_pipeline(**kwargs).images

    # SDXL uses a refiner pipeline
    images = refiner_pipeline(
        refiner, images, device_identifier, preserve_vram, kwargs.copy()
    )

    images = upscale_pipeline(upscale, images, device_identifier, kwargs.copy())

    main_pipeline.config["nsfw"] = is_nsfw(main_pipeline)
    output_processor.add_outputs(images)
    return (output_processor.get_results(), main_pipeline.config)
