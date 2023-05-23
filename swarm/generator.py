from .output_processor import (
    make_result,
    image_to_buffer,
    make_text_result,
    image_from_text,
)
from .job_arguments import format_args
from . import __version__
import asyncio


async def do_work(job, device):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, synchronous_do_work_function, job, device)


def synchronous_do_work_function(job, device):
    id = job.pop("id")
    print(f"Processing {id} on {device.descriptor()}")

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

    # generation will throw this error if some is not-recoverable/fatal
    # (e.g. a textual-inversion not compatible with the base model)
    except ValueError as e:
        content_type = job.get("content_type", "image/jpeg")
        print(e)
        if content_type.startswith("image/"):
            artifacts, pipeline_config = exception_image(e, content_type)
        else:
            artifacts, pipeline_config = exception_message(e)

        return {
            "id": id,
            "artifacts": artifacts,
            "fatal_error": True,
            "nsfw": pipeline_config.get("nsfw", False),  # type ignore
            "worker_version": __version__,
            "pipeline_config": pipeline_config,
        }

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


def exception_image(e, content_type):
    message = e.args[0] if len(e.args) > 0 else "error generating image"
    image = image_from_text(message)
    pipe_config = {"error": message}

    _buffer = image_to_buffer(image, content_type)
    return {"primary": make_result(_buffer, _buffer, content_type)}, pipe_config


def exception_message(e):
    message = e.args[0] if len(e.args) > 0 else "error generating image"
    pipe_config = {"error": message}

    return {"primary": make_text_result(str(e))}, pipe_config
