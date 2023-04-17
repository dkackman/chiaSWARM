import requests
from PIL import Image, ImageOps
from .diffusion.diffusion_func import diffusion_callback
from .video.tx2vid import txt2vid_diffusion_callback
from .captioning.caption_image import caption_callback
from .toolbox.stitch import stitch_callback
from .video.pix2pix import model_video_callback
from .type_helpers import get_type


max_size = 1024


def format_args(job):
    args = job.copy()

    workflow = args.pop("workflow", None)
    if workflow == "stitch":
        return stitch_callback, args

    if workflow == "img2txt":
        return format_img2txt_args(args)

    if workflow == "vid2vid":
        return model_video_callback, args

    if workflow == "txt2vid":
        return format_txt2vid_args(args)

    return format_stable_diffusion_args(args)


def format_txt2vid_args(args):
    parameters = args.pop("parameters", {})
    if "prompt" not in args:
        args["prompt"] = ""

    if "num_inference_steps" not in args:
        args["num_inference_steps"] = 25

    args.pop("num_images_per_prompt", None)

    args["pipeline_type"] = get_type(
        "diffusers", parameters.pop("pipeline_type", "DiffusionPipeline")
    )
    args["scheduler_type"] = get_type(
        "diffusers", parameters.pop("scheduler_type", "DPMSolverMultistepScheduler")
    )

    return txt2vid_diffusion_callback, args


def format_img2txt_args(args):
    if "start_image_uri" in args:
        args["image"] = get_image(args.pop("start_image_uri"), None)

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
    if "prompt" not in args:
        args["prompt"] = ""

    # these gets passed through to the diffusion callback
    args["supports_xformers"] = parameters.get("supports_xformers", True)
    args["upscale"] = parameters.get("upscale", False)

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

    if "num_inference_steps" not in args:
        # default num_inference_steps if not set - some pipelines have high default values
        args["num_inference_steps"] = 30

    args["pipeline_type"] = get_type(
        "diffusers", parameters.pop("pipeline_type", "DiffusionPipeline")
    )
    args["scheduler_type"] = get_type(
        "diffusers", parameters.pop("scheduler_type", "DPMSolverMultistepScheduler")
    )

    for arg in parameters.get("unsupported_pipeline_arguments", []):
        args.pop(arg, None)

    return diffusion_callback, args


def download_image(url):
    image = Image.open(requests.get(url, allow_redirects=True, stream=True).raw)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def get_image(uri, size):
    head = requests.head(uri, allow_redirects=True)  # type: ignore
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
