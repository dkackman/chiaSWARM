from .gpu.device import Device
from .generator import do_work
from .log_setup import setup_logging
from . import __version__
import diffusers
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


# producer/consumer queue for job retreived
# assigned in startup
work_queue:asyncio.Queue
available_gpus:asyncio.Semaphore

# producer consumer queue for results waiting to be uploaded
result_queue = asyncio.Queue()


settings = load_settings()
hive_uri = f"{settings.sdaas_uri.rstrip('/')}/api"


async def run_worker():
    await startup()

    logging.info("worker")

    # Create a task for each device to process work
    device_tasks = []
    for i in range(torch.cuda.device_count()):
        device = Device(i)
        device_tasks.append(asyncio.create_task(device_worker(device)))
        print(f"Started device {device.descriptor()}")

    # Create a task for submitting results
    result_task = asyncio.create_task(result_worker())

    # Main loop to request work
    while True:
        while work_queue.full():
            await asyncio.sleep(1)

        sleep_seconds = await ask_for_work()
        await asyncio.sleep(sleep_seconds)


async def ask_for_work():
    await available_gpus.acquire()
    print(
        f"{datetime.now()}: Asking for work from the hive at {hive_uri}..."
    )
    try:
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

                    found_work = False
                    for job in response_dict["jobs"]:
                        id = job["id"]
                        print(f"Got job {id}")
                        found_work = True
                        await work_queue.put(job)

                    # since there is work in the hive ask right away for more
                    return 1 if found_work else 11
                
                elif response.status == 400:
                    # this is when workers are not returning results within expectations
                    response_dict = await response.json()
                    message = response_dict.pop("message", "bad worker")
                    print(f"{hive_uri} says {message}")
                    response.raise_for_status()

                else:
                    print(f"{hive_uri} returned {response.status}")
                    response.raise_for_status()

    except Exception as e:
        logging.exception(e)
        print(e)
        return 121
    finally:
        available_gpus.release()

    return  11

async def device_worker(device: Device):
    while True:
        try:
            job = await work_queue.get()
            await available_gpus.acquire()
            result = await do_work(job, device)
            await result_queue.put(result)
        except Exception as e:
            logging.exception(e)

            print(f"device_worker {e}")
        finally:
            available_gpus.release()
            work_queue.task_done()

async def result_worker():
    while True:
        try:
            result = await result_queue.get()
            await submit_result(result)
        except Exception as e:
            logging.exception(e)

            print(f"result_worker {e}")
        finally:
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

    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPUs")
    global work_queue
    work_queue = asyncio.Queue(maxsize=gpu_count)

    global available_gpus
    available_gpus = asyncio.Semaphore(gpu_count)

    diffusers.logging.set_verbosity_error()

if __name__ == "__main__":
    asyncio.run(run_worker())
