import shutil
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from io import BytesIO
from PIL import Image
from typing import List
from diffusers.utils import export_to_gif
import tempfile
import pathlib
import numpy as np
import cv2


def get_frame(mp4_file_path, frame_index=0):
    # Load the video clip
    try:
        clip = VideoFileClip(mp4_file_path)

        # Get the first frame of the clip as an image
        frame = clip.get_frame(frame_index)

        # Convert the frame to a JPEG image in memory
        frame_image = Image.fromarray(frame)
        _buffer = BytesIO()
        frame_image.save(_buffer, format="JPEG")
        # Close the buffer and the clip
        clip.close()
        # Get the bytes of the JPEG image from the buffer
        return _buffer

    except Exception as e:
        print(e)
        return None


def make_video(images, duration_seconds):
    frames = [np.array(img) for img in images]
    fps = len(frames) / duration_seconds
    clip = ImageSequenceClip.ImageSequenceClip(frames, fps=fps)
    with tempfile.NamedTemporaryFile(suffix=".webm") as tmp:
        clip.write_videofile(
            filename=tmp.name,
            codec="libvpx",  # webm codec
            fps=fps,
            threads=2,
            preset="veryslow",  # veryslow == smallest file
        )

        # return the last image (used for thumbnail) and the video bytes
        return images[len(images) - 1], BytesIO(tmp.read())


def export_to_video(content_type, video_frames, save_debug = False, fps: int = 8):    
    if content_type.startswith("video"):
        media_info = ("mp4", "XVID") if content_type == "video/mp4" else ("webm", "VP90")

        # convert to video
        with tempfile.TemporaryDirectory() as tmpdirname:
            final_filepath = export_to_video_pil(
                video_frames,
                pathlib.Path(tmpdirname).joinpath(f"video.{media_info[0]}").__str__(),
                media_info[1],
                fps
            )
            with open(final_filepath, "rb") as video_file:
                video_buffer = BytesIO(video_file.read())

            if save_debug:
                shutil.copy(final_filepath, "./video.mp4")

            thumbnail = get_frame(final_filepath, 0)

        return (thumbnail, video_buffer)

    with tempfile.TemporaryDirectory() as tmpdirname:
        final_filepath = export_to_gif(
            video_frames,
            pathlib.Path(tmpdirname).joinpath(f"video.gif").__str__(),
            fps
        )
        with open(final_filepath, "rb") as video_file:
            video_buffer = BytesIO(video_file.read())

        if save_debug:
            shutil.copy(final_filepath, "./video.gif")
            
        thumbnail = get_frame(final_filepath, 0)

        return (thumbnail, video_buffer)

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