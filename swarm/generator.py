from .diffusion.device import Device
from urllib.parse import unquote
import io
from PIL import Image
import requests
from enum import Enum


class image_format_enum(str, Enum):
    jpeg = "jpeg"
    json = "json"
    png = "png"


class audio_format_enum(str, Enum):
    wav = "wav"
    json = "json"


def generate_buffer(device: Device, **kwargs):
    format = kwargs.pop("format", "JPEG").upper()
    format = format if format != "JSON" else "JPEG"
    try:
        if "prompt" in kwargs:
            kwargs["prompt"] = clean_prompt(kwargs["prompt"])

        image, pipe_config = device(**kwargs)  # type: ignore
    except Exception as e:
        print(e)

        if len(e.args) > 0:
            if e.args[0] == "busy":
                raise Exception(423)
            if e.args[0] == "NSFW":
                raise Exception(406)  # Not Acceptable

        raise Exception(500)

    buffer = image_to_buffer(image, format)

    # we return kwargs so that it can be used as metadata if needed
    return buffer, pipe_config, kwargs


def get_image(uri):
    head = requests.head(uri, allow_redirects=True)
    content_length = head.headers.pop("Content-Length", 0)

    print(f"Image size is {content_length} bytes")

    # enforce size limit here - need error communication back to end user too

    response = requests.get(uri, allow_redirects=True)

    # diffusers example resize everything to a square not sure if that a requiremnt or not
    image = Image.open(io.BytesIO(response.content)).convert("RGB").resize((512, 512))
    # maxzise = 512
    # if image.height > maxzise or image.width > maxzise:
    #    image.thumbnail((maxzise, maxzise), Image.Resampling.LANCZOS)

    return image


def image_to_buffer(image, format):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)

    return buffer


def clean_prompt(str):
    encoded = unquote(str).encode("utf8", "ignore")
    decoded = encoded.decode("utf8", "ignore")
    cleaned = decoded.replace('"', "").replace("'", "").strip()

    return cleaned
