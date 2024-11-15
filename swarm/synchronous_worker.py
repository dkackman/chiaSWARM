import diffusers
import torch
import logging
import os
import json
from .settings import (load_settings, resolve_path)
from packaging import version
from .log_setup import setup_logging
from .pipeline_processors.pipeline import run_pipeline
from .pipeline_processors.arguments import prepare_args
from . import __version__

settings = load_settings()

def do_work(input_job, output_dir):
    job_id = input_job["id"]
    print(f"Processing {job_id}")

    default_seed = input_job.get("seed", torch.seed())
    
    input_job["seed"] = default_seed
    job = prepare_args(input_job)

    # collections that are passed between pipelines to share state
    results = []
    intermediate_results = {}
    shared_components = {}
    for pipeline in job["pipelines"]:
        name = pipeline["name"]
        print(f"Running pipeline {name}")

        # if the pipeline's configuration doesn't have a seed use the default from above
        configuration = pipeline["configuration"]
        configuration["seed"] = configuration["seed"] if "seed" in configuration else default_seed

        result = run_pipeline(pipeline, "cuda", intermediate_results, shared_components)
        if result is not None:
            results.extend(result)  

    with open(os.path.join(output_dir, f"{job_id}.json"), 'w') as file:
        json.dump(input_job, file, indent=4)

    for i, result in enumerate(results):
        default_name = f"{job_id}-{i}{result.guess_extension()}"
        result.save(output_dir, default_name)


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
