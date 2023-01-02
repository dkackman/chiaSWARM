from .settings import (
    Settings,
    settings_exist,
    save_settings,
    get_settings_full_path,
    load_settings,
    load_settings,
    resolve_path,
    settings_exist,
)

import asyncio
import logging
from .log_setup import setup_logging
from diffusers import DiffusionPipeline
import torch
from . import __version__
import sys


async def init():
    logging.info("init_app")

    overwrite = False
    if settings_exist() and "--reset" in sys.argv:
        overwrite = True

    if not settings_exist() or overwrite:
        settings = Settings()

        print("Provide the following details for the intial configuration:\n")
        token = input("Huggingface API token: ").strip()
        if len(token) == 0:
            print("A Huggingface API token is required.")
            return

        sdaas_token = input("chiaSWARM token: ").strip()

        sdaas_uri = input(
            "chiaSWARM uri (https://chiaswarm-dev.azurewebsites.net): "
        ).strip()
        sdaas_uri = (
            "https://chiaswarm-dev.azurewebsites.net"
            if len(sdaas_uri) == 0
            else sdaas_uri
        )

        settings.huggingface_token = token
        settings.sdaas_token = sdaas_token
        settings.sdaas_uri = sdaas_uri

        save_settings(settings)
        print(f"Configuration saved to {get_settings_full_path()}")

    settings = load_settings()
    setup_logging(resolve_path(settings.log_filename), settings.log_level)
    logging.debug(f"Version {__version__}")
    logging.debug(f"Torch version {torch.__version__}")
    print("Preloading pipelines. This may take awhile...")

    known_models = [
        ("stabilityai/stable-diffusion-2-1", "fp16", None),
        ("stabilityai/stable-diffusion-2-1-base", "fp16", None),
        ("stabilityai/stable-diffusion-2-depth", "fp16", None),
        ("stabilityai/stable-diffusion-2-inpainting", "fp16", None),
        ("stabilityai/stable-diffusion-x4-upscaler", "fp16", None),
        ("nitrosocke/Future-Diffusion", "main", None),
        ("nitrosocke/Nitro-Diffusion", "main", None),
        ("nitrosocke/Ghibli-Diffusion", "main", None),
        ("nitrosocke/redshift-diffusion", "main", None),
        ("prompthero/openjourney", "main", None),
        ("riffusion/riffusion-model-v1", "main", None),
        ("runwayml/stable-diffusion-v1-5", "fp16", None),
        ("runwayml/stable-diffusion-inpainting", "fp16", None),
        ("Envvi/Inkpunk-Diffusion", "main", None),
    ]

    # this makes sure that all of the diffusers are downloaded and cached
    for model in known_models:
        print(f"Initializing {model[0]}/{model[1]}")
        DiffusionPipeline.from_pretrained(
            model[0],
            use_auth_token=settings.huggingface_token,
            revision=model[1],
            torch_dtype=torch.float16,
            custom_pipeline=model[2],
        )
    print("done")
    print("To be the swarm type 'python -m swarm.worker'")


asyncio.run(init())
