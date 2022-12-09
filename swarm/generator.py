from .diffusion.device import Device
from urllib.parse import unquote
import io
from PIL import Image, ImageDraw
import requests
from enum import Enum
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline, StableDiffusionInpaintPipeline


class image_format_enum(str, Enum):
    jpeg = "jpeg"
    json = "json"
    png = "png"


class audio_format_enum(str, Enum):
    wav = "wav"
    json = "json"


def generate_buffer(device: Device, job, **kwargs):
    format = kwargs.pop("format", "JPEG").upper()
    format = format if format != "JSON" else "JPEG"

    try:
        # some workloads have different processing and arrguments - that happens here
        if kwargs["model_name"] == "stabilityai/stable-diffusion-x4-upscaler":
            kwargs["image"] = get_image(job["start_image_uri"]).resize((128, 128))
            kwargs["pipeline_type"] = StableDiffusionUpscalePipeline   

        elif kwargs["model_name"] == "stabilityai/stable-diffusion-2-inpainting":
            kwargs["image"] = get_image(job["start_image_uri"])
            kwargs["mask_image"] = get_image(job["mask_image_uri"])
            kwargs["pipeline_type"] = StableDiffusionInpaintPipeline   

        # start_image_uri signals to use the img2img workflow
        elif "start_image_uri" in job:
            kwargs["image"] = get_image(job["start_image_uri"])
            kwargs["strength"] = job.get("strength", 0.6)
            kwargs["pipeline_type"] = StableDiffusionImg2ImgPipeline

        if "prompt" in kwargs:
            kwargs["prompt"] = clean_prompt(kwargs["prompt"])

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


def get_image(uri):
    head = requests.head(uri, allow_redirects=True)
    content_length = head.headers.pop("Content-Length", 0)

    # to protect worker nodes, no external images over 2 MiB
    if int(content_length) > 2097152:
        raise Exception(
            f"Input image too large.\nMax size is 2097152 bytes.\nImage was {content_length}."
        )

    response = requests.get(uri, allow_redirects=True)

    # diffusers example resize everything to a square not sure if that a requiremnt or not
    image = Image.open(io.BytesIO(response.content)).convert("RGB").resize((512, 512))
    # maxzise = 512
    # if image.height > maxzise or image.width > maxzise:
    #    image.thumbnail((maxzise, maxzise), Image.Resampling.LANCZOS)

    return image


def image_from_text(text):
    image = Image.new(mode="RGB", size=(256, 256))
    draw = ImageDraw.Draw(image)

    draw.text((5, 5), text, align="left")
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
