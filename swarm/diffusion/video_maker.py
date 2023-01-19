import numpy as np
import moviepy.video.io.ImageSequenceClip


def make_video(images, duration_seconds):
    pils = []
    for img in images:
        pils.append(np.array(img))

    fps = len(pils) / duration_seconds
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(pils, fps=fps)
    clip.write_videofile("/home/don/my_video.webm")
    return {}
