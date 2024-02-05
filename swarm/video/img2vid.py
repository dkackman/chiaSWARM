import torch
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from ..type_helpers import has_method
import tempfile
import pathlib
from io import BytesIO
from ..post_processors.output_processor import make_result
from ..toolbox.video_helpers import get_frame
import cv2
from typing import List
import numpy as np
import shutil
from diffusers.utils import export_to_gif

def img2vid_diffusion_callback(device_identifier, model_name, **kwargs):
    scheduler_type = kwargs.pop("scheduler_type", DPMSolverMultistepScheduler)
    pipeline_type = kwargs.pop("pipeline_type", DiffusionPipeline)
    kwargs.pop("num_frames", 25)
    kwargs.pop("guidance_scale", 25)
    kwargs["num_inference_steps"] = 1
    kwargs["decode_chunk_size"] = 1
    content_type = kwargs.pop("content_type", "video/mp4")
    kwargs.pop("outputs", ["primary"])

    pipeline = pipeline_type.from_pretrained(
        model_name,
        revision=kwargs.pop("revision", "main"),
        variant=kwargs.pop("variant", None),
        torch_dtype=torch.float16,
    )
    pipeline = pipeline.to(device_identifier)

    pipeline.scheduler = scheduler_type.from_config(
        pipeline.scheduler.config, use_karras_sigmas=True
    )

    # mem_info = torch.cuda.mem_get_info(device_identifier)
    # if we're doing a long video or mid-range on mem, preserve memory vs performance
    # if (
    #     kwargs["num_frames"] > 30
    #     and mem_info[1] < 16000000000  # for 3090's etc just let em go full bore
    # ):
    #     # not all pipelines share these methods, so check first
    #     if has_method(pipeline, "enable_vae_slicing"):
    #         pipeline.enable_vae_slicing()

    # if has_method(pipeline, "enable_model_cpu_offload"):
    #     pipeline.enable_model_cpu_offload()

    p = pipeline(**kwargs)
    video_frames = p.frames

    if content_type.startswith("video"):
        media_info = ("mp4", "XVID") if content_type == "video/mp4" else ("webm", "VP90")

        # convert to video
        with tempfile.TemporaryDirectory() as tmpdirname:
            final_filepath = export_to_video(
                video_frames,
                pathlib.Path(tmpdirname).joinpath(f"video.{media_info[0]}").__str__(),
                media_info[1],
            )
            with open(final_filepath, "rb") as video_file:
                video_buffer = BytesIO(video_file.read())

            shutil.copy(final_filepath, "./video.mp4")
            thumbnail = get_frame(final_filepath, 0)

        results = {"primary": make_result(video_buffer, thumbnail, content_type)}
        return (results, pipeline.config)

    with tempfile.TemporaryDirectory() as tmpdirname:
        final_filepath = export_to_gif(
            video_frames[0],
            pathlib.Path(tmpdirname).joinpath(f"video.gif").__str__()
        )
        with open(final_filepath, "rb") as video_file:
            video_buffer = BytesIO(video_file.read())

        shutil.copy(final_filepath, "./video.gif")
        thumbnail = get_frame(final_filepath, 0)

    results = {"primary": make_result(video_buffer, thumbnail, content_type)}
    return (results, pipeline.config)


def export_to_video(
    video_frames: List[np.ndarray], output_video_path: str, codec
) -> str:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=8, frameSize=(w, h))
    for video_frame in video_frames:
        img = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(img)

    return output_video_path
