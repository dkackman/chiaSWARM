from .gpu.device import Device
from .generator import do_work
from .log_setup import setup_logging
from . import __version__
from .gpu.device_pool import add_device_to_pool
from .settings import (
    load_settings,
    resolve_path,
)
import torch
import asyncio
import logging
import requests
from datetime import datetime
import json

settings = load_settings()
hive_uri = f"{settings.sdaas_uri.rstrip('/')}/api"


async def run_worker():
    await startup()

    logging.info("worker")

    while True:
        try:
            print(f"{datetime.now()}: Asking for work from the hive at {hive_uri}...")

            response = requests.get(
                f"{hive_uri}/work",
                timeout=10,
                params={
                    "worker_version": __version__,
                    "worker_name": settings.worker_name,
                },
                headers={
                    "Content-type": "application/json",
                    "Authorization": f"Bearer {settings.sdaas_token}",
                    "user-agent": f"chiaSWARM.worker/{__version__}",
                },
            )

            if response.ok:
                response_dict = response.json()

                wait_seconds = response_dict.pop("wait_seconds", 11)
                for job in response_dict["jobs"]:
                    print("Got work")

                    # main worker function
                    result = await do_work(job)

                    resultResponse = requests.post(
                        f"{hive_uri}/results",
                        data=json.dumps(result),
                        headers={
                            "Content-type": "application/json",
                            "Authorization": f"Bearer {settings.sdaas_token}",
                            "user-agent": f"chiaSWARM.worker/{__version__}",
                        },
                    )
                    if resultResponse.status_code == 500:
                        print(f"The hive returned an error: {resultResponse.reason}")
                    else:
                        print(resultResponse.json())

                await asyncio.sleep(wait_seconds)

            elif response.status_code == 400:
                # this is when workers are not returning results within expectations
                response_dict = response.json()
                message = response_dict.pop("message", "bad worker")
                print(f"{hive_uri} says {message}")
                wait_seconds = response_dict.pop("wait_seconds", 121)
                print(f"sleeping for {wait_seconds} seconds")

                await asyncio.sleep(wait_seconds)

            else:
                print(f"{hive_uri} returned {response.status_code}")
                print("sleeping for 120 seconds")

                await asyncio.sleep(121)

        except Exception as e:
            print(e)  # this is if the work queue endpoint is unavailable
            print("sleeping for 120 seconds")
            await asyncio.sleep(121)


async def startup():
    if not torch.cuda.is_available():
        raise Exception("CUDA not present. Quitting.")

    setup_logging(resolve_path(settings.log_filename), settings.log_level)
    logging.info(f"Version {__version__}")
    logging.debug(f"Torch version {torch.__version__}")

    for i in range(0, torch.cuda.device_count()):
        logging.info(f"Adding cuda device {i} - {torch.cuda.get_device_name(i)}")
        add_device_to_pool(Device(i))


if __name__ == "__main__":
    asyncio.run(run_worker())
