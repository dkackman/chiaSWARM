import diffusers
import torch
import asyncio
import logging
from swarm.hive import ask_for_work, submit_result
from .settings import (
    load_settings,
    resolve_path,
)
from packaging import version
from .job_arguments import format_args
from .output_processor import (
    exception_image,
    exception_message,
    fatal_exception_response,
)
from .gpu.device import Device
from .log_setup import setup_logging
from . import __version__

# assigned in startup

# producer/consumer queue for job retrieved
work_queue: asyncio.Queue

# semaphore to limit the number of jobs running at once to the number of gpus
available_gpus: asyncio.Semaphore

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
        # spin wait if work queue is full
        while work_queue.full():
            await asyncio.sleep(1)

        await available_gpus.acquire()
        try:
            for job in await ask_for_work(settings, hive_uri):
                job_id = job["id"]
                print(f"Got job {job_id}")
                await work_queue.put(job)

            sleep_seconds = 11

        except Exception as e:
            logging.exception(e)
            print(e)
            sleep_seconds = 121

        finally:
            available_gpus.release()

        await asyncio.sleep(sleep_seconds)


async def device_worker(device: Device):
    while True:
        try:
            job = await work_queue.get()
            # we got work so acquire a gpu lock
            await available_gpus.acquire()
            worker_function, kwargs = await get_args(job)

            if worker_function is not None:
                result = await do_work(device, worker_function, kwargs)
                await result_queue.put(result)

        except Exception as e:
            logging.exception(e)
            print(f"device_worker {e}")

        finally:
            available_gpus.release()
            work_queue.task_done()


async def get_args(job):
    try:
        return await format_args(job)

    except Exception as e:
        # any error here is fatal (i.e. not something a worker could recover from)
        # the job should not be resubmitted as input args are wrong somehow
        await result_queue.put(fatal_exception_response(e, job["id"], job))

    return None, None


async def result_worker():
    while True:
        try:
            result = await result_queue.get()
            await submit_result(settings, hive_uri, result)

        except Exception as e:
            logging.exception(e)
            print(f"result_worker {e}")

        finally:
            result_queue.task_done()


async def do_work(device, worker_function, kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, synchronous_do_work, device, worker_function, kwargs
    )


def synchronous_do_work(device, worker_function, kwargs):
    job_id = kwargs.pop("id")
    print(f"Processing {job_id} on {device.descriptor()}")

    try:
        artifacts, pipeline_config = device(worker_function, **kwargs)  # type: ignore

    # generation will throw this error if some is not-recoverable/fatal
    # (e.g. a textual-inversion not compatible with the base model)
    except (ValueError, TypeError) as e:
        return fatal_exception_response(e, job_id, kwargs)

    except Exception as e:
        content_type = kwargs.get("content_type", "image/jpeg")
        print(e)
        if content_type.startswith("image/"):
            artifacts, pipeline_config = exception_image(e, content_type)
        else:
            artifacts, pipeline_config = exception_message(e)

    return {
        "id": job_id,
        "artifacts": artifacts,
        "nsfw": pipeline_config.get("nsfw", False),  # type ignore
        "worker_version": __version__,
        "pipeline_config": pipeline_config,
    }


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
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPUs")
    global work_queue
    work_queue = asyncio.Queue(maxsize=gpu_count)

    global available_gpus
    available_gpus = asyncio.Semaphore(gpu_count)

    diffusers.logging.set_verbosity_error()


if __name__ == "__main__":
    asyncio.run(run_worker())
