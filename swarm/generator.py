from .diffusion.device import Device
from .diffusion.device_pool import remove_device_from_pool, add_device_to_pool
from .format_converter import image_to_buffer
from PIL import Image, ImageDraw
import base64
from .job_arguments import format_args
from . import __version__
import hashlib

async def do_work(job):
    device = remove_device_from_pool()

    try:
        content_type = job.pop("content_type", "image/jpeg")
        id = job.pop("id")
        try:
            kwargs = format_args(job)

            buffer, pipeline_config = generate_buffer(
                device,
                content_type,
                **kwargs,
            )
        except Exception as e:
            buffer, pipeline_config = exception_image(e, "image/jpeg")

        thumb_buffer = make_thumbnail(buffer)

        return {
            "id": id,
            "content_type": content_type,
            "blob": base64.b64encode(buffer.getvalue()).decode("UTF-8"),
            "thumbnail": base64.b64encode(thumb_buffer.getvalue()).decode("UTF-8"),
            "nsfw": pipeline_config.get("nsfw", False),
            "worker_version": __version__,
            "pipeline_config": pipeline_config,
            "sha256_hash": hashlib.sha256(buffer.getvalue()).hexdigest()
        }

    finally:
        add_device_to_pool(device)


def make_thumbnail(buffer):
    image = Image.open(buffer).convert("RGB")
    image.thumbnail((100, 100), Image.Resampling.LANCZOS)
    return image_to_buffer(image, "image/jpeg", "web_low")


def exception_image(e, content_type):
    print(e)
    message = "error generating image"
    if len(e.args) > 0:
        message = e.args[0]

    image = image_from_text(message)
    pipe_config = {"error": message}

    return image_to_buffer(image, content_type), pipe_config


def generate_buffer(device: Device, content_type, **kwargs):
    try:
        image, pipe_config = device(**kwargs)  # type: ignore

    except Exception as e:
        return exception_image(e, content_type)

    return image_to_buffer(image, content_type), pipe_config


def image_from_text(text):
    image = Image.new(mode="RGB", size=(512, 512))
    draw = ImageDraw.Draw(image)

    draw.text((5, 5), text, align="left")
    return image
