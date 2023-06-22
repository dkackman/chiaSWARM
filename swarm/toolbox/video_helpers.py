from moviepy.video.io.VideoFileClip import VideoFileClip
from io import BytesIO
from PIL import Image


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
