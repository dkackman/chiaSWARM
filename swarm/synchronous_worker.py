import diffusers
import torch
import logging
import mimetypes
import os
import json
import soundfile
from .settings import (load_settings, resolve_path)
from packaging import version
from .log_setup import setup_logging
from .pipeline_processors.pipeline import run_pipeline
from .pipeline_processors.arguments import prepare_args
from diffusers.utils import export_to_video
from . import __version__

settings = load_settings()

def do_work(job_id, input_job, output_dir):
    print(f"Processing {job_id}")

    default_seed = input_job.get("seed", torch.seed())
    input_job["id"] = job_id
    input_job["seed"] = default_seed
    job = prepare_args(input_job)

    results = []
    intermediate_results = {}
    for pipeline in job["pipelines"]:
        name = pipeline["name"]
        print(f"Running pipeline {name}")

        # if the pipeline's configuration doesn't have a seed use the default from above
        configuration = pipeline["configuration"]
        configuration["seed"] = configuration["seed"] if "seed" in configuration else default_seed

        result = run_pipeline(pipeline, "cuda", intermediate_results)
        if result is not None:
            results.extend(result)  

    with open(os.path.join(output_dir, f"{job_id}.json"), 'w') as file:
        json.dump(input_job, file, indent=4)

    content_type = job.get("content_type", "image/jpeg")
    extension = guess_extension(content_type)
    for i, result in enumerate(results):
        output_path = os.path.join(output_dir, f"{job_id}-{i}{extension}")
        if content_type.startswith("video"):
            export_to_video(result, output_path, fps=8)

        elif content_type.startswith("audio"):
            soundfile.write(output_path, result, 44100)

        elif hasattr(result, 'save'):
            result.save(output_path)


def guess_extension(content_type):
    ext = mimetypes.guess_extension(content_type)
    if ext is not None:
        return ext

    if content_type == "audio/wav":
        return ".wav"
    
    return ""


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
