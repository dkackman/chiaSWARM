import torch
from diffusers import StableVideoDiffusionPipeline
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
from PIL import Image

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

    if content_type.startswith("video"):
        media_info = ("mp4", "XVID") if content_type == "video/mp4" else ("webm", "VP90")

        # convert to video
        with tempfile.TemporaryDirectory() as tmpdirname:
            final_filepath = export_to_video_pil(
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
            video_frames,
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

    video_writer.release()  # Ensure to release the video writer

    return output_video_path


def export_to_video_pil(
    video_frames: List[Image.Image],  # Expect a list of PIL images
    output_video_path: str,
    codec: str,
    fps: int = 8  # Added fps as a parameter for flexibility
) -> str:
    # Convert the first PIL image to NumPy to get dimensions
    first_frame = np.array(video_frames[0])
    h, w, _ = first_frame.shape
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frameSize=(w, h))
    
    for video_frame in video_frames:
        # Convert each PIL image to a NumPy array
        img_array = np.array(video_frame)
        # Convert RGB (PIL's default) to BGR (OpenCV's default)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    
    video_writer.release()  # Ensure to release the video writer
    return output_video_path