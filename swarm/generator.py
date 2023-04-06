from .output_processor import make_result, image_to_buffer, make_text_result
from PIL import Image, ImageDraw
from .job_arguments import format_args
from . import __version__


async def do_work(job, device):
    id = job.pop("id")

    try:
        worker_function, kwargs = format_args(job)

    except Exception as e:
        # any error here is fatal (i.e. not something a worker could recover from)
        # the job should not be resubmitted as input args are wrong somehow
        content_type = job.get("content_type", "image/jpeg")
        print(e)
        if content_type.startswith("image/"):
            artifacts, pipeline_config = exception_image(e, content_type)
        else:
            artifacts, pipeline_config = exception_message(e)

        return {
            "id": id,
            "artifacts": artifacts,
            "nsfw": pipeline_config.get("nsfw", False),  # type ignore
            "fatal_error": True,
            "worker_version": __version__,
            "pipeline_config": pipeline_config,
        }            

    try:
        artifacts, pipeline_config = device(worker_function, **kwargs)  # type: ignore

    except Exception as e:
        content_type = job.get("content_type", "image/jpeg")
        print(e)
        if content_type.startswith("image/"):
            artifacts, pipeline_config = exception_image(e, content_type)
        else:
            artifacts, pipeline_config = exception_message(e)

    return {
        "id": id,
        "artifacts": artifacts,
        "nsfw": pipeline_config.get("nsfw", False),  # type ignore
        "worker_version": __version__,
        "pipeline_config": pipeline_config,
    }


def image_from_text(text):
    image = Image.new(mode="RGB", size=(512, 512))
    draw = ImageDraw.Draw(image)

    draw.multiline_text((5, 5), text)
    return image


def exception_image(e, content_type):
    message = "error generating image"
    if len(e.args) > 0:
        message = e.args[0]

    image = image_from_text(message)
    pipe_config = {"error": message}

    buffer = image_to_buffer(image, content_type)
    return {"primary": make_result(buffer, buffer, content_type)}, pipe_config


def exception_message(e):
    message = "error generating image"
    if len(e.args) > 0:
        message = e.args[0]

    pipe_config = {"error": message}

    return {"primary": make_text_result(str(e))}, pipe_config
