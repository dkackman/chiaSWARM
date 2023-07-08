from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from io import BytesIO
from PIL import Image
import numpy as np
import tempfile


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
