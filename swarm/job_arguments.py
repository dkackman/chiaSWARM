import torch
from urllib.parse import unquote
import requests
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline,
    StableDiffusionInstructPix2PixPipeline,
)
from PIL import Image, ImageOps

max_size = 1024


def format_args(job):
    args = job.copy()

    if not "revision" in args.keys():
        args["revision"] = "fp16"

    # this is where all of the input arguments are raiotnalized and model specific
    args["torch_dtype"] = torch.float16

    size = None
    if "height" in args and "width" in args:
        size = (args["height"], args["width"])
        if size[0] > max_size or size[1] > max_size:
            raise Exception(
                f"The max image size is (1024, 1024); got ({size[0]}, {size[1]})."
            )

    if "start_image_uri" in args:
        args["image"] = get_image(args.pop("start_image_uri"), size)

    if "mask_image_uri" in args:
        args["mask_image"] = get_image(args.pop("mask_image_uri"), size)

    # some workloads have different processing and arguments - that happens here
    # TODO data drive this from the model meta data (and schedulers)
    if args["model_name"] == "timbrooks/instruct-pix2pix":
        args["pipeline_type"] = StableDiffusionInstructPix2PixPipeline
        args.pop("height", None)
        args.pop("width", None)

        # this model defaults to 100, which we don't want
        if "num_inference_steps" not in args:
            args["num_inference_steps"] = 50

    elif args["model_name"] == "stabilityai/stable-diffusion-x4-upscaler":
        args["pipeline_type"] = StableDiffusionUpscalePipeline
        # this model will reject these two args
        args.pop("height", None)
        args.pop("width", None)

    elif (
        args["model_name"] == "stabilityai/stable-diffusion-2-inpainting"
        or args["model_name"] == "runwayml/stable-diffusion-inpainting"
    ):
        args["pipeline_type"] = StableDiffusionInpaintPipeline

    elif args["model_name"] == "stabilityai/stable-diffusion-2-depth":
        args["pipeline_type"] = StableDiffusionDepth2ImgPipeline
        args.pop("height", None)
        args.pop("width", None)

    # having an image signals to use the img2img workflow for SD 1.5
    elif "image" in args:
        args["pipeline_type"] = StableDiffusionImg2ImgPipeline
        # this model will reject these two args
        args.pop("height", None)
        args.pop("width", None)

    if "prompt" in args:
        args["prompt"] = clean_prompt(args["prompt"])

    if "negative_prompt" in args:
        args["negative_prompt"] = clean_prompt(args["negative_prompt"])

    return args


def clean_prompt(str):
    encoded = unquote(str).encode("utf8", "ignore")
    decoded = encoded.decode("utf8", "ignore")
    cleaned = decoded.replace('"', "").replace("'", "").strip()

    return cleaned


def download_image(url):
    image = Image.open(requests.get(url, allow_redirects=True, stream=True).raw)  # type: ignore
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    return image


def get_image(uri, size):
    head = requests.head(uri, allow_redirects=True)
    content_length = head.headers.pop("Content-Length", 0)
    content_type = head.headers.pop("Content-Type", "")

    if not content_type.startswith("image"):
        raise Exception(
            f"Input does not appear to be an image.\nContent type was {content_type}."
        )

    # to protect worker nodes, no external images over 3 MiB
    if int(content_length) > 1048576 * 3:
        raise Exception(
            f"Input image too large.\nMax size is {1048576 * 3} bytes.\nImage was {content_length}."
        )

    image = download_image(uri)

    # if we have a desired size and the image is alrger than it, scale the image down
    if size != None and (image.height > size[0] or image.width > size[1]):
        image.thumbnail(size, Image.Resampling.LANCZOS)

    elif image.height > max_size or image.width > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    return image
