import torch
from diffusers import (
    DiffusionPipeline,
    MotionAdapter,
    LCMScheduler
)
from ..type_helpers import has_method
from ..post_processors.output_processor import make_result
from ..toolbox.video_helpers import export_to_video
from PIL import Image


def txt2vid_diffusion_callback(device_identifier, model_name, **kwargs):
    scheduler_type = kwargs.pop("scheduler_type", LCMScheduler)
    pipeline_type = kwargs.pop("pipeline_type", DiffusionPipeline)
    kwargs["num_frames"] = kwargs.pop("num_frames", 25)
    content_type = kwargs.pop("content_type", "video/mp4")
    upscale = kwargs.pop("upscale", False)
    lora = kwargs.pop("lora", None)
    kwargs.pop("outputs", ["primary"])
    torch_dtype = torch.bfloat16 if kwargs.pop("use_bfloat16", False) else torch.float16

    motion_adapter = None
    if "motion_adapter" in kwargs:
        motion_adapter = MotionAdapter.from_pretrained(kwargs["motion_adapter"]["model_name"], torch_dtype=torch_dtype)

    pipeline = pipeline_type.from_pretrained(
        model_name,
        revision=kwargs.pop("revision", "main"),
        variant=kwargs.pop("variant", None),
        motion_adapter=motion_adapter,
        torch_dtype=torch_dtype,
    )

    if lora is not None:
        pipeline.load_lora_weights(lora["model_name"], weight_name=lora["weight_name"], adapter_name=lora["adapter_name"])
        pipeline.set_adapters([lora["adapter_name"]], [lora["weight"]])

    pipeline = pipeline.to(device_identifier)

    pipeline.scheduler = scheduler_type.from_config(
        pipeline.scheduler.config, beta_schedule="linear"
    )

    # not all pipelines share these methods, so check first
    if has_method(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()

    if has_method(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()

    p = pipeline(**kwargs)
    video_frames = p.frames[0]

    if upscale:
        upscaler = DiffusionPipeline.from_pretrained(
            "cerspense/zeroscope_v2_XL", torch_dtype=torch.float16
        )
        upscaler.scheduler = scheduler_type.from_config(
            upscaler.scheduler.config, use_karras_sigmas=True
        )
        upscaler.enable_vae_slicing()

        video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]
        video_frames = upscaler(kwargs["prompt"], video=video, strength=0.6).frames

    thumbnail, video_buffer = export_to_video(content_type, video_frames)

    results = {"primary": make_result(video_buffer, thumbnail, content_type)}
    return (results, pipeline.config)
