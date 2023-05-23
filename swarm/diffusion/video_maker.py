import numpy as np
import moviepy.video.io.ImageSequenceClip
import tempfile
import io


def make_video(images, duration_seconds):
    frames = [np.array(img) for img in images]
    fps = len(frames) / duration_seconds
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=fps)
    with tempfile.NamedTemporaryFile(suffix=".webm") as tmp:
        clip.write_videofile(
            filename=tmp.name,
            codec="libvpx",  # webm codec
            fps=fps,
            threads=2,
            preset="veryslow",  # veryslow == smallest file
        )

        # return the last image (used for thumbnail) and the video bytes
        return images[len(images) - 1], io.BytesIO(tmp.read())
