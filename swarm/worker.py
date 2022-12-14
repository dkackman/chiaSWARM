from .diffusion.device import Device
from .diffusion.device_pool import add_device_to_pool, remove_device_from_pool
from .generator import generate_buffer
import torch
import asyncio
import logging
from .settings import (
    load_settings,
    resolve_path,
)
from .log_setup import setup_logging
import base64
import json
import requests
from datetime import datetime
from . import __version__
from .job_arguments import format_args

settings = load_settings()
hive_uri = f"{settings.sdaas_uri.rstrip('/')}/api"


async def run_worker():
    await startup()

    logging.info("worker")

    while True:
        try:
            print(f"{datetime.now()}: Asking for work from {settings.sdaas_uri}...")

            response = requests.get(
                f"{hive_uri}/work",
                params={
                    "worker_version": __version__,
                    "worker_name": settings.worker_name,
                },
                headers={
                    "Content-type": "application/json",
                    "Authorization": f"Bearer {settings.sdaas_token}",
                },
            )

            if response.ok:
                response_dict = response.json()

                wait_seconds = response_dict.pop("wait_seconds", 10)
                for job in response_dict["jobs"]:
                    await do_work(job)

                await asyncio.sleep(wait_seconds)

            elif response.status_code == 400:
                # this is when workers are not returning results within expectations
                response_dict = response.json()
                message = response_dict.pop("message", "bad worker")
                print(f"{hive_uri} says {message}")
                wait_seconds = response_dict.pop("wait_seconds", 120)
                print(f"sleeping for {wait_seconds} seconds")

                await asyncio.sleep(wait_seconds)

            else:
                print(f"{hive_uri} returned {response.status_code}")
                print("sleeping for 60 seconds")

                await asyncio.sleep(60)

        except Exception as e:
            print(e)  # this is if the work queue endpoint is unavailable
            print("sleeping for 60 seconds")
            await asyncio.sleep(60)


async def do_work(job):
    device = remove_device_from_pool()
    content_type = job.pop("content_type", "image/jpeg")
    id = job.pop("id")
    kwargs = format_args(job, content_type)

    try:
        buffer, pipeline_config, args = generate_buffer(
            device,
            **kwargs,
        )

        result = {
            "id": id,
            "content_type": content_type,
            "blob": base64.b64encode(buffer.getvalue()).decode("UTF-8"),
            "nsfw": pipeline_config.get("nsfw", False),
            "worker_name": settings.worker_name,
            "pipeline_config": pipeline_config,
        }

        requests.post(
            f"{hive_uri}/results",
            data=json.dumps(result),
            headers={
                "Content-type": "application/json",
                "Authorization": f"Bearer {settings.sdaas_token}",
            },
        )

    except Exception as e:
        print(e)
    finally:
        add_device_to_pool(device)


async def startup():
    if not torch.cuda.is_available():
        raise Exception("CUDA not present. Quitting.")

    setup_logging(resolve_path(settings.log_filename), settings.log_level)
    logging.info(f"Version {__version__}")
    logging.debug(f"Torch version {torch.__version__}")

    for i in range(0, torch.cuda.device_count()):
        logging.info(f"Adding cuda device {i} - {torch.cuda.get_device_name(i)}")
        add_device_to_pool(Device(i, settings.huggingface_token))


if __name__ == "__main__":
    asyncio.run(run_worker())
