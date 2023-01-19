import torch
import logging
from threading import Lock
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available
from .output_processor import OutputProcessor


class Device:
    device_id: int
    mutex: Lock

    def __init__(self, device_id: int) -> None:
        self.device_id = device_id
        # self.auth_token = auth_token
        self.mutex = Lock()

    def __call__(self, **kwargs):
        if not self.mutex.acquire(False):
            logging.error(f"Device {self.device_id} is busy but got invoked.")
            raise Exception("busy")

        try:
            if "prompt" in kwargs:
                logging.info(f"Prompt is {kwargs['prompt']}")
            self.log_device()

            model_name = kwargs.pop("model_name")

            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                model_name,
                # use_auth_token=self.auth_token,
                subfolder="scheduler",
            )

            pipeline = self.get_pipeline(
                model_name,
                kwargs.pop("revision"),
                kwargs.pop("custom_pipeline", None),
                kwargs.pop("torch_dtype", torch.float16),
                scheduler,
                kwargs.pop("pipeline_type", DiffusionPipeline),
            )

            seed = kwargs.pop("seed", None)
            if seed is None:
                seed = torch.seed()

            torch.manual_seed(seed)

            output_processor = OutputProcessor(
                kwargs.pop("outputs", ["primary"]),
                kwargs.pop("content_type", "image/jpeg"),
            )

            if output_processor.need_intermediates():
                print("Capturing latents")

                def latents_callback(i, t, latents):
                    output_processor.add_latents(pipeline, latents)  # type: ignore

                kwargs["callback"] = latents_callback
                kwargs["callback_steps"] = 4

            p = pipeline(**kwargs)  # type: ignore

            # if any image is nsfw, flag the entire result
            if (
                hasattr(p, "nsfw_content_detected")
                and p.nsfw_content_detected is not None  # type: ignore
                and len(p.nsfw_content_detected) >= 1  # type: ignore
            ):
                for _ in filter(lambda nsfw: nsfw, p.nsfw_content_detected):  # type: ignore
                    pipeline.config["nsfw"] = True

            pipeline.config["seed"] = seed

            output_processor.add_outputs(p.images)  # type: ignore

            return (output_processor.get_results(), pipeline.config)  # type: ignore

        finally:
            self.mutex.release()

    def get_pipeline(
        self,
        model_name: str,
        revision: str,
        custom_pipeline,
        torch_dtype,
        scheduler,
        pipeline_type,
    ):
        logging.debug(
            f"Loading {model_name} to device {self.device_id} - {torch.cuda.get_device_name(self.device_id)}"
        )

        # load the pipeline and send it to the gpu
        pipeline = pipeline_type.from_pretrained(
            model_name,
            # use_auth_token=self.auth_token,
            revision=revision,
            torch_dtype=torch_dtype,
            custom_pipeline=custom_pipeline,
            scheduler=scheduler,
        ).to(
            f"cuda:{self.device_id}"
        )  # type: ignore

        try:
            pipeline.enable_attention_slicing()
        except:
            print("error enable_attention_slicing")

        if is_xformers_available():
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )

        return pipeline

    def log_device(self):
        logging.debug(
            f"Using device# {self.device_id} - {torch.cuda.get_device_name(self.device_id)}"
        )
