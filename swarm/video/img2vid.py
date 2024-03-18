import torch
from diffusers import StableVideoDiffusionPipeline
from ..type_helpers import has_method
from ..post_processors.output_processor import make_result
from ..toolbox.video_helpers import export_to_video


def img2vid_diffusion_callback(device_identifier, model_name, **kwargs):
    kwargs.pop("scheduler_type", None)
    pipeline_type = kwargs.pop("pipeline_type", StableVideoDiffusionPipeline)
    kwargs["num_frames"] = kwargs.pop("num_frames", 12)
    kwargs.pop("guidance_scale", 25)
    content_type = kwargs.pop("content_type", "video/mp4")
    kwargs.pop("outputs", ["primary"])

    torch_dtype = torch.bfloat16 if kwargs.pop("use_bfloat16", False) else torch.float16

    pipeline = pipeline_type.from_pretrained(
        model_name,
        revision=kwargs.pop("revision", "main"),
        variant=kwargs.pop("variant", None),
        torch_dtype=torch_dtype,
    )
    pipeline = pipeline.to(device_identifier)

    if has_method(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if has_method(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    if has_method(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()

    p = pipeline(**kwargs)
    video_frames = p.frames[0]

    thumbnail, video_buffer = export_to_video(content_type, video_frames)
    results = {"primary": make_result(video_buffer, thumbnail, content_type)}
    return (results, pipeline.config)
