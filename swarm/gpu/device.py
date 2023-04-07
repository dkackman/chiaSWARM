import torch
import logging
from threading import Lock


class Device:
    def __init__(self, device_id: int) -> None:
        mem_info = torch.cuda.mem_get_info(device_id)
        if mem_info[1] < 8000000000:
            raise Exception(
                f"Not enough memory on device {device_id}. At least 8GB VRAM is required"
            )

        self.device_id = device_id
        self.mutex = Lock()

    def __call__(self, func, **kwargs):
        if not self.mutex.acquire(False):
            logging.error(f"Device {self.device_id} is busy but got invoked.")
            raise Exception("busy")

        try:
            self.log_device()

            model_name = kwargs.pop("model_name")
            seed = kwargs.pop("seed", None)
            if seed is None:
                seed = torch.seed()

            kwargs["generator"] = torch.Generator(
                device=f"cuda:{self.device_id}"
            ).manual_seed(seed)
            artifacts, pipeline_config = func(
                self.device_id, model_name, **kwargs)
            pipeline_config["seed"] = seed
            return artifacts, pipeline_config

        finally:
            self.mutex.release()

    def log_device(self):
        logging.debug(
            f"Using device# {self.device_id} - {torch.cuda.get_device_name(self.device_id)}"
        )
