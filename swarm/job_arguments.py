import requests
from PIL import Image, ImageOps
from .diffusion.diffusion_func import diffusion_callback
from .video.tx2vid import txt2vid_diffusion_callback
from .captioning.caption_image import caption_callback
from .toolbox.stitch import stitch_callback
from .video.pix2pix import model_video_callback
from .audio.audioldm import txt2audio_diffusion_callback
from .audio.bark import bark_diffusion_callback
from .diffusion.diffusion_func_if import diffusion_if_callback
from .diffusion.kandinsky import kandinsky_callback
from .type_helpers import get_type
from .controlnet.input_processor import preprocess_image, resize_for_condition_image

max_size = 1024


def format_args(job):
    args = job.copy()

    workflow = args.pop("workflow", None)
    if workflow == "txt2audio":
        if args["model_name"] == "suno/bark":
            return bark_diffusion_callback, args

        return format_txt2audio_args(args)

    if workflow == "stitch":
        return stitch_callback, args

    if workflow == "img2txt":
        return format_img2txt_args(args)

    if workflow == "vid2vid":
        return model_video_callback, args

    if workflow == "txt2vid":
        return format_txt2vid_args(args)

    if args["model_name"].startswith("DeepFloyd/"):
        return diffusion_if_callback, args

    if args["model_name"].startswith("kandinsky-"):
        return format_kandinsky_args(args)

    return format_stable_diffusion_args(args)


def format_txt2audio_args(args):
    parameters = args.pop("parameters", {})
    if "prompt" not in args:
        args["prompt"] = ""

    if "num_inference_steps" not in args:
        args["num_inference_steps"] = 25

    args["pipeline_type"] = get_type(
        "diffusers", parameters.pop("pipeline_type", "AudioLDMPipeline")
    )
    args["scheduler_type"] = get_type(
        "diffusers", parameters.pop("scheduler_type", "DPMSolverMultistepScheduler")
    )

    for arg in parameters.get("unsupported_pipeline_arguments", []):
        args.pop(arg, None)

    return txt2audio_diffusion_callback, args


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


def format_kandinsky_args(args):
    size = None
    if "height" in args and "width" in args:
        size = (args["height"], args["width"])
        if size[0] > max_size or size[1] > max_size:
            raise Exception(
                f"The max image size is (1024, 1024); got ({size[0]}, {size[1]})."
            )

    if "num_inference_steps" not in args:
        args["num_inference_steps"] = 100

    if "start_image_uri" in args:
        args.pop("height", None)
        args.pop("width", None)

        args["image"] = get_image(args.pop("start_image_uri"), size)
        args["pipeline_type"] = get_type("diffusers", "KandinskyImg2ImgPipeline")

    # if there is start_image_uri2 we are interpolating
    if "start_image_uri2" in args:
        args["image2"] = get_image(args.pop("start_image_uri2"), size)
        args["pipeline_type"] = get_type("diffusers", "KandinskyPipeline")

    parameters = args.pop("parameters", {})
    if "model_name_prior" in parameters:
        args["model_name_prior"] = parameters["model_name_prior"]

    args.pop("revision", None)

    return kandinsky_callback, args


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

        controlnet = parameters.get("controlnet", None)
        image, control_image = get_image(args.pop("start_image_uri"), size, controlnet)
        args["image"] = image

        if controlnet is not None:
            args["control_image"] = control_image
            parameters["pipeline_type"] = "StableDiffusionControlNetImg2ImgPipeline"
            args.pop("strength", None)
            args["controlnet_model_name"] = controlnet.get(
                "controlnet_model_name", "lllyasviel/control_v11p_sd15_canny"
            )
            args["save_preprocessed_input"] = controlnet.get("preprocess", False)
            args["controlnet_conditioning_scale"] = controlnet.get(
                "controlnet_conditioning_scale", 1.0
            )

            # StableDiffusionControlNetPipeline does not accept strength
            args.pop("strength", None)

        elif "pipeline_type" not in parameters:
            parameters["pipeline_type"] = "StableDiffusionImg2ImgPipeline"

        if args["model_name"] == "timbrooks/instruct-pix2pix":
            # pix2pix models use image_guidance_scale instead of strength
            # image_guidance_scale has a range of 1-5 instead 0-1
            args["image_guidance_scale"] = args.pop("strength", 0.6) * 5

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


def get_image(uri, size, controlnet=None):
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

    # if we have a desired size and the image is larger than it, scale the image down
    if size != None and (image.height > size[0] or image.width > size[1]):
        image.thumbnail(size, Image.Resampling.LANCZOS)

    elif image.height > max_size or image.width > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # return the image and the pre-processed image if controlnet is specified
    if controlnet != None:
        # preprocess means generate the control image from the input image
        if controlnet.get("preprocess", False):
            return image, preprocess_image(image, controlnet)

        # base the resolution of of size - defaulting to 768
        W, H = size if size is not None else (768, 768)
        resolution = max(H, W)

        # user specified control image - go get it
        if isNotBlank(controlnet.get("control_image_uri", None)):
            control_image, _ = get_image(controlnet.get("control_image_uri"), size)
            return resize_for_condition_image(
                image, resolution
            ), resize_for_condition_image(control_image, resolution)

        # control image and input image are the same
        return resize_for_condition_image(
            image, resolution
        ), resize_for_condition_image(image, resolution)

    return image, None


def isNotBlank(myString):
    return bool(myString and myString.strip())
