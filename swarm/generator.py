from .diffusion.device_pool import remove_device_from_pool, add_device_to_pool
from .diffusion.output_processor import make_result, image_to_buffer
from PIL import Image, ImageDraw
from .job_arguments import format_args
from . import __version__


async def do_work(job):
    device = remove_device_from_pool()

    try:
        id = job.pop("id")
        try:
            kwargs = format_args(job)
            artifacts, pipeline_config = device(**kwargs)  # type: ignore

        except Exception as e:
            content_type = job.get("content_type", "image/jpeg")
            artifacts, pipeline_config = exception_image(e, content_type)

        return {
            "id": id,
            "artifacts": artifacts,
            "nsfw": pipeline_config.get("nsfw", False),
            "worker_version": __version__,
            "pipeline_config": pipeline_config,
        }

    finally:
        add_device_to_pool(device)


def image_from_text(text):
    image = Image.new(mode="RGB", size=(512, 512))
    draw = ImageDraw.Draw(image)

    draw.multiline_text((5, 5), text)
    return image


def exception_image(e, content_type):
    print(e)
    message = "error generating image"
    if len(e.args) > 0:
        message = e.args[0]

    image = image_from_text(message)
    pipe_config = {"error": message}

    return {
        "primary": make_result(image_to_buffer(image, content_type), content_type)
    }, pipe_config
