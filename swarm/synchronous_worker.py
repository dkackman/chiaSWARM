import diffusers
import torch
import logging
from .settings import (
    load_settings,
    resolve_path,
)
from packaging import version
from .log_setup import setup_logging
from .pipeline_processors.pipeline import run_pipeline
from . import __version__

settings = load_settings()

def do_work(job, output_dir):
    job_id = job.pop("id")
    print(f"Processing {job_id}")

    try:
        result = None
        for pipeline in job["pipelines"]:
            name = pipeline["name"]
            print(f"Running pipeline {name}")
            result = run_pipeline(pipeline, "cuda", result)

        
    except Exception as e:
        print(e)


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
