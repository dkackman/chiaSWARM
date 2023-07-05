import argparse
import asyncio
import torch
from diffusers import DiffusionPipeline, ControlNetModel
from . import __version__
from .log_setup import setup_logging
from .settings import (
    Settings,
    get_settings_full_path,
    load_settings,
    resolve_path,
    save_settings,
)
from .hive import get_models
from .type_helpers import get_type


async def init():
    print("init_app")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true", help="overwrite existing settings"
    )
    parser.add_argument("--silent", action="store_true", help="do not prompt for input")
    parser.add_argument(
        "--download",
        action="store_true",
        help="download all the models (will use a lot of disk space))",
    )
    args = parser.parse_args()

    settings_exist = load_settings() is not None
    overwrite = args.reset and settings_exist

    if not args.silent and (not settings_exist or overwrite):
        try:
            settings = Settings()

            sdaas_uri = (
                input("chiaSWARM uri (https://chiaswarm.ai): ").strip()
                or "https://chiaswarm.ai"
            )
            sdaas_token = input("chiaSWARM token: ").strip()

            settings.sdaas_token = sdaas_token
            settings.sdaas_uri = sdaas_uri

            save_settings(settings)
            print(f"Configuration saved to {get_settings_full_path()}")

        except Exception as e:
            print(f"Failed to save configuration: {e}")
            raise

    settings = load_settings()
    setup_logging(resolve_path(settings.log_filename), settings.log_level)
    print(f"Version {__version__}")
    print(f"Torch version {torch.__version__}")
    print("App initialization complete")

    if args.download:
        await download_diffusers(settings)

    print("To be the swarm type 'python -m swarm.worker'")


async def download_diffusers(settings):
    print("Downloading models...")
    known_models = await get_models(f"{settings.sdaas_uri.rstrip('/')}/")

    for model in known_models:
        model_name = model["model_name"]
        revision = model["revision"]
        variant = model.get("variant")
        print(f"Initializing {model_name}/{revision}")

        try:
            parameters = model.pop("parameters", {})

            if parameters.get("can_preload", True):
                loader = DiffusionPipeline

                if "controlnet_type" in model:
                    loader = ControlNetModel
                elif "model_type" in parameters:
                    loader = get_type("transformers", parameters["model_type"])

                # this will cause diffusers to fetch the latest model data
                loader.from_pretrained(
                    model_name,
                    revision=revision,
                    variant=variant,
                    torch_dtype=torch.float16,
                )
        except Exception as e:
            print(f"Failed to initialize {model_name}/{revision}: {e}")
            raise

    print("Model download complete")


if __name__ == "__main__":
    asyncio.run(init())
