import torch
import logging
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available
from .output_processor import OutputProcessor


def diffusion_callback(device_id, model_name, **kwargs):
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_name,
        subfolder="scheduler",
    )

    pipeline = get_pipeline(
        device_id,
        model_name,
        kwargs.pop("revision"),
        kwargs.pop("torch_dtype", torch.float16),
        scheduler,
        kwargs.pop("pipeline_type", DiffusionPipeline),
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


def get_pipeline(
    device_id: int,
    model_name: str,
    revision: str,
    torch_dtype,
    scheduler,
    pipeline_type,
):
    logging.debug(
        f"Loading {model_name} to device {device_id} - {torch.cuda.get_device_name(device_id)}"
    )

    # load the pipeline and send it to the gpu
    pipeline = pipeline_type.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch_dtype,
        scheduler=scheduler,
    ).to(
        f"cuda:{device_id}"
    )  # type: ignore

    try:
        pipeline.enable_attention_slicing()
    except:
        print("error enable_attention_slicing")

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    return pipeline
