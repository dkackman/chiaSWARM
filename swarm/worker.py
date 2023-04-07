from .gpu.device import Device
from .generator import do_work
from .log_setup import setup_logging
from . import __version__
from .gpu.device_pool import add_device_to_pool, remove_device_from_pool
from .settings import (
    load_settings,
    resolve_path,
)
import torch
import asyncio
import logging
import aiohttp
from datetime import datetime
import json
from packaging import version


work_queue = asyncio.Queue(maxsize=torch.cuda.device_count())
result_queue = asyncio.Queue()

settings = load_settings()
hive_uri = f"{settings.sdaas_uri.rstrip('/')}/api"


async def run_worker():
    await startup()

    logging.info("worker")

    # Create a task for each device to process work
    device_tasks = []
    for i in range(torch.cuda.device_count()):
        device = remove_device_from_pool()
        device_tasks.append(asyncio.create_task(device_worker(device)))

    # Create a task for submitting results
    result_task = asyncio.create_task(result_worker())

    # Main loop to request work
    while True:
        await asyncio.sleep(11)
        await ask_for_work()


async def ask_for_work():
    print(
        f"{datetime.now()}: Asking for work from the hive at {hive_uri}..."
    )

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(
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
        ) as response:

            if response.status == 200:
                response_dict = await response.json()
                for job in response_dict["jobs"]:
                    id = job["id"]
                    print(f"Got job {id}")

                    await work_queue.put(job)

            elif response.status == 400:
                # this is when workers are not returning results within expectations
                response_dict = await response.json()
                message = response_dict.pop("message", "bad worker")
                print(f"{hive_uri} says {message}")
                response.raise_for_status()

            else:
                print(f"{hive_uri} returned {response.status}")
                response.raise_for_status()


async def device_worker(device: Device):
    while True:
        job = await work_queue.get()
        result = await do_work(job, device)
        await result_queue.put(result)
        work_queue.task_done()
        add_device_to_pool(device)


async def result_worker():
    while True:
        result = await result_queue.get()
        await submit_result(result)
        result_queue.task_done()


async def submit_result(result):
    print(f"Result complete")

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{hive_uri}/results",
            data=json.dumps(result),
            headers={
                "Content-type": "application/json",
                "Authorization": f"Bearer {settings.sdaas_token}",
                "user-agent": f"chiaSWARM.worker/{__version__}",
            },
        ) as resultResponse:

            if resultResponse.status == 500:
                print(f"The hive returned an error: {resultResponse.reason}")
            else:
                response_dict = await resultResponse.json()
                print(f"Restult {response_dict}")


async def startup():
    if not torch.cuda.is_available():
        raise Exception("CUDA not present. Quitting.")

    if version.parse(torch.__version__) < version.parse("2.0.0"):
        raise Exception(
            f"Pytorch must be 2.0 or greater (found {torch.__version__}). Run install script. Quitting."
        )

    setup_logging(resolve_path(settings.log_filename), settings.log_level)
    logging.info(f"Version {__version__}")
    logging.debug(f"Torch version {torch.__version__}")

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True  # type: ignore
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore

    for i in range(0, torch.cuda.device_count()):
        logging.info(
            f"Adding cuda device {i} - {torch.cuda.get_device_name(i)}")
        add_device_to_pool(Device(i))


if __name__ == "__main__":
    asyncio.run(run_worker())
