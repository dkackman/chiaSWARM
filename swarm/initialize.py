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
import requests


async def init():
    logging.info("init_app")

    overwrite = False
    if settings_exist() and "--reset" in sys.argv:
        overwrite = True

    if not settings_exist() or overwrite:
        settings = Settings()

        # print("Provide the following details for the intial configuration:\n")
        # token = input("Huggingface API token: ").strip()
        # if len(token) == 0:
        #     print("A Huggingface API token is required.")
        #     return

        sdaas_token = input("chiaSWARM token: ").strip()

        sdaas_uri = input("chiaSWARM uri (https://chiaswarm.ai): ").strip()
        sdaas_uri = "https://chiaswarm.ai" if len(sdaas_uri) == 0 else sdaas_uri

        # settings.huggingface_token = token
        settings.sdaas_token = sdaas_token
        settings.sdaas_uri = sdaas_uri

        save_settings(settings)
        print(f"Configuration saved to {get_settings_full_path()}")

    settings = load_settings()
    setup_logging(resolve_path(settings.log_filename), settings.log_level)
    logging.debug(f"Version {__version__}")
    logging.debug(f"Torch version {torch.__version__}")
    print("Preloading pipelines. This may take awhile...")
    known_models = get_models_from_hive(f"{settings.sdaas_uri.rstrip('/')}/")

    # this makes sure that all of the diffusers are downloaded and cached
    for model in known_models:
        model_name = model["model_name"]
        revision = model["revision"]
        print(f"Initializing {model_name}/{revision}")
        DiffusionPipeline.from_pretrained(
            model_name,
            # use_auth_token=settings.huggingface_token,
            revision=revision,
            torch_dtype=torch.float16,
        )
    print("done")
    print("To be the swarm type 'python -m swarm.worker'")


def get_models_from_hive(hive_uri):
    print(f"Fetching known model list from the hive at {hive_uri}...")

    try:
        response = requests.get(
            f"{hive_uri}data/models.json",
            timeout=10,
            headers={
                "user-agent": f"chiaSWARM.worker/{__version__}",
            },
        )
        data = response.json()
        print("done")

        return data["models"]
    except Exception as e:
        print(e)
        return []


asyncio.run(init())
