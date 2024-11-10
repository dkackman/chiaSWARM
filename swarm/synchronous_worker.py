import diffusers
import torch
import logging
import mimetypes
import os
import json
from .settings import (load_settings, resolve_path)
from packaging import version
from .log_setup import setup_logging
from .pipeline_processors.pipeline import run_pipeline
from .pipeline_processors.arguments import prepare_args

from . import __version__

settings = load_settings()

def do_work(job_id, input_job, output_dir):
    print(f"Processing {job_id}")

    try:
        result = None
        input_job["id"] = job_id
        job = prepare_args(input_job)
        default_seed = job.get("seed", 0)
        for pipeline in job["pipelines"]:
            name = pipeline["name"]
            print(f"Running pipeline {name}")

            # if the pipeline's configuration doesn't have a seed use the default from above
            configuration = pipeline["configuration"]
            configuration["seed"] = configuration["seed"] if "seed" in configuration else default_seed

            result = run_pipeline(pipeline, "cuda", result)

        content_type = job.get("content_type", "image/jpeg")
        extension = mimetypes.guess_extension(content_type)

        if result is not None:
            result.save(os.path.join(output_dir, f"{job_id}{extension}"))
            with open(os.path.join(output_dir, f"{job_id}.json"), 'w') as file:
                json.dump(input_job, file, indent=4)

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
