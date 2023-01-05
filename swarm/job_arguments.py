import torch
from urllib.parse import unquote
import requests
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline,
)
from PIL import Image
import io

max_size = 1024


def format_args(job):
    if not "revision" in job.keys():
        job["revision"] = "fp16"

    # this is where all of the input arguments are raiotnalized and model specific
    args = job.copy()

    args["torch_dtype"] = torch.float16

    size = None
    if "height" in job and "width" in job:
        size = (job["height"], job["width"])
        if size[0] > max_size or size[1] > max_size:
            raise Exception(
                f"The max image size is (1024, 1024); got ({size[0]}, {size[1]})."
            )

    # some workloads have different processing and arguments - that happens here
    if args["model_name"] == "stabilityai/stable-diffusion-x4-upscaler":
        args["image"] = get_image(job["start_image_uri"], None)
        args["pipeline_type"] = StableDiffusionUpscalePipeline
        # this model will reject these two args
        args.pop("height", None)
        args.pop("width", None)

    elif (
        args["model_name"] == "stabilityai/stable-diffusion-2-inpainting"
        or args["model_name"] == "runwayml/stable-diffusion-inpainting"
    ):
        args["image"] = get_image(job["start_image_uri"], size)
        args["mask_image"] = get_image(job["mask_image_uri"], size)
        args["pipeline_type"] = StableDiffusionInpaintPipeline

    elif args["model_name"] == "stabilityai/stable-diffusion-2-depth":
        args["image"] = get_image(job["start_image_uri"], size)
        args["pipeline_type"] = StableDiffusionDepth2ImgPipeline
        args.pop("height", None)
        args.pop("width", None)

    # start_image_uri signals to use the img2img workflow for SD 1.5
    elif "start_image_uri" in job:
        args["image"] = get_image(job["start_image_uri"], size)
        args["pipeline_type"] = StableDiffusionImg2ImgPipeline
        # this model will reject these two args
        args.pop("height", None)
        args.pop("width", None)

    if "prompt" in args:
        args["prompt"] = clean_prompt(args["prompt"])

    if "negative_prompt" in args:
        args["negative_prompt"] = clean_prompt(args["negative_prompt"])

    # get rid of anything that shouldn't be passed to the pipeline
    args.pop("start_image_uri", None)
    args.pop("mask_image_uri", None)

    return args


def clean_prompt(str):
    encoded = unquote(str).encode("utf8", "ignore")
    decoded = encoded.decode("utf8", "ignore")
    cleaned = decoded.replace('"', "").replace("'", "").strip()

    return cleaned


def get_image(uri, size):
    head = requests.head(uri, allow_redirects=True)
    content_length = head.headers.pop("Content-Length", 0)
    content_type = head.headers.pop("Content-Type", "")

    if not content_type.startswith("image"):
        raise Exception(
            f"Input does not appear to be an image.\nContent type was {content_type}."
        )

    # to protect worker nodes, no external images over 2 MiB
    if int(content_length) > 2097152:
        raise Exception(
            f"Input image too large.\nMax size is 2097152 bytes.\nImage was {content_length}."
        )

    response = requests.get(uri, allow_redirects=True)

    image = Image.open(io.BytesIO(response.content)).convert("RGB")

    # if we have a desired size and the image is alrger than it, scale the image down
    if size != None and (image.height > size[0] or image.width > size[1]):
        image.thumbnail(size, Image.Resampling.LANCZOS)

    elif image.height > max_size or image.width > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    return image
