import diffusers
import torch
import logging
from .settings import (
    load_settings,
    resolve_path,
)
from packaging import version
from .post_processors.output_processor import (
    exception_image,
    exception_message,
    fatal_exception_response,
)
from .log_setup import setup_logging
from . import __version__
from swarm.settings import load_settings

settings = load_settings()

def do_work(device, worker_function, kwargs):
    job_id = kwargs.pop("id")
    print(f"Processing {job_id} on {device.descriptor()}")

    try:
        artifacts, pipeline_config = device(worker_function, **kwargs)

    # generation will throw this error if some is not-recoverable/fatal
    # (e.g. a textual-inversion not compatible with the base model)
    except (ValueError, TypeError) as e:
        return fatal_exception_response(e, job_id, kwargs)

    except Exception as e:
        content_type = kwargs.get("content_type", "image/jpeg")
        print(e)
        if content_type.startswith("image/"):
            artifacts, pipeline_config = exception_image(e, content_type)
        else:
            artifacts, pipeline_config = exception_message(e)

    return {
        "id": job_id,
        "artifacts": artifacts,
        "nsfw": pipeline_config.get("nsfw", False),  # type ignore
        "worker_version": __version__,
        "pipeline_config": pipeline_config,
    }


def startup():
    if not torch.cuda.is_available():
        raise Exception("CUDA not present. Quitting.")

    if version.parse(torch.__version__) < version.parse("2.0.0"):
        raise Exception(
            f"Pytorch must be 2.0 or greater (found {torch.__version__}). Run install script. Quitting."
        )

    setup_logging(resolve_path(settings.log_filename), settings.log_level)
    logging.info(f"Version {__version__}")
    logging.debug(f"Torch version {torch.__version__}")
    diffusers.logging.set_verbosity_error()

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
