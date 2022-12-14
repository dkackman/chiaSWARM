from .diffusion.device import Device
import io
from PIL import Image, ImageDraw


def generate_buffer(device: Device, **kwargs):
    format = kwargs.pop("format", "JPEG").upper()
    format = format if format != "JSON" else "JPEG"

    try:
        image, pipe_config = device(**kwargs)  # type: ignore
    except Exception as e:
        print(e)
        message = "error generating image"
        if len(e.args) > 0:
            message = e.args[0]

        image = image_from_text(message)
        pipe_config = {}

    buffer = image_to_buffer(image, format)

    # we return kwargs so that it can be used as metadata if needed
    return buffer, pipe_config, kwargs


def image_from_text(text):
    image = Image.new(mode="RGB", size=(512, 512))
    draw = ImageDraw.Draw(image)

    draw.text((5, 5), text, align="left")
    return image


def image_to_buffer(image, format):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)

    return buffer
