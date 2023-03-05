from urllib.parse import unquote
import requests
from PIL import Image, ImageOps
from .diffusion.diffusion_func import diffusion_callback
from .captioning.caption_image import caption_callback

from .video.vid2vid import model_video_callback
from .type_helpers import get_type


max_size = 1024


def format_args(job):
    args = job.copy()

    workflow = args.pop("workflow", None)
    if workflow == "img2txt":
        return format_img2txt_args(args)

    if workflow == "vid2vid":
        return format_vid2vid_args(args)

    return format_stable_diffusion_args(args)


def format_vid2vid_args(args):
    if "prompt" in args:
        args["prompt"] = clean_prompt(args["prompt"])

    return model_video_callback, args


def format_img2txt_args(args):
    if "start_image_uri" in args:
        args["image"] = get_image(args.pop("start_image_uri"), None)

    if "prompt" in args:
        args["prompt"] = clean_prompt(args["prompt"])

    return caption_callback, args


def format_stable_diffusion_args(args):
    # this is where all of the input arguments are rationalized and model specific
    size = None
    if "height" in args and "width" in args:
        size = (args["height"], args["width"])
        if size[0] > max_size or size[1] > max_size:
            raise Exception(
                f"The max image size is (1024, 1024); got ({size[0]}, {size[1]})."
            )

    parameters = args.pop("parameters", {})
    if "start_image_uri" in args:
        args.pop("height", None)
        args.pop("width", None)

        args["image"] = get_image(args.pop("start_image_uri"), size)
        # if there is an input image and pipeline type is not specified then default it to img2img
        if "pipeline_type" not in parameters:
            parameters["pipeline_type"] = "StableDiffusionImg2ImgPipeline"

    if "mask_image_uri" in args:
        args.pop("height", None)
        args.pop("width", None)

        args["mask_image"] = get_image(args.pop("mask_image_uri"), size)

    args["pipeline_type"] = get_type(
        "diffusers", parameters.pop("pipeline_type", "StableDiffusionSAGPipeline")
    )
    args["scheduler_type"] = get_type(
        "diffusers", parameters.pop("scheduler_type", "DPMSolverMultistepScheduler")
    )

    # default num_inference_steps if not set
    if "num_inference_steps" not in args:
        args["num_inference_steps"] = 30

    # some pipelines don't like it when they get size arguments
    if (
        args["model_name"] == "stabilityai/stable-diffusion-x4-upscaler"
        or args["model_name"] == "stabilityai/stable-diffusion-2-depth"
        or args["model_name"] == "timbrooks/instruct-pix2pix"
    ):
        args.pop("height", None)
        args.pop("width", None)

    if "prompt" in args:
        args["prompt"] = clean_prompt(args["prompt"])

    if "negative_prompt" in args:
        args["negative_prompt"] = clean_prompt(args["negative_prompt"])

    return diffusion_callback, args


def clean_prompt(str):
    encoded = unquote(str).encode("utf8", "ignore")
    decoded = encoded.decode("utf8", "ignore")
    cleaned = decoded.replace('"', "").replace("'", "").strip()

    return cleaned


def download_image(url):
    image = Image.open(requests.get(url, allow_redirects=True, stream=True).raw)
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
