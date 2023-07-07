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
from .controlnet.input_processor import scale_to_size
from .external_resources import get_image, get_control_image, max_size, download_images


async def format_args(job):
    args = job.copy()

    workflow = args.pop("workflow", None)
    if workflow == "txt2audio":
        if args["model_name"] == "suno/bark":
            return bark_diffusion_callback, args

        return format_txt2audio_args(args)

    if workflow == "stitch":
        return await format_stitch_args(args)

    if workflow == "img2txt":
        return format_img2txt_args(args)

    if workflow == "vid2vid":
        return model_video_callback, args

    if workflow == "txt2vid":
        return format_txt2vid_args(args)

    if args["model_name"].startswith("DeepFloyd/"):
        return diffusion_if_callback, args

    if args["model_name"].startswith("kandinsky-"):
        return await format_kandinsky_args(args)

    return await format_stable_diffusion_args(args, workflow)


async def format_stitch_args(args):
    # download all of the component images
    jobs = args["jobs"]
    image_urls = [job["resultUri"] for job in jobs]
    args["images"] = await download_images(image_urls)

    return stitch_callback, args


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


async def format_img2txt_args(args):
    if "start_image_uri" in args:
        args["image"] = await get_image(args.pop("start_image_uri"), None)

    return caption_callback, args


async def format_kandinsky_args(args):
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

        args["image"] = await get_image(args.pop("start_image_uri"), size)
        args["pipeline_type"] = get_type("diffusers", "KandinskyImg2ImgPipeline")

    # if there is start_image_uri2 we are interpolating
    if "start_image_uri2" in args:
        args["image2"] = await get_image(args.pop("start_image_uri2"), size)
        args["pipeline_type"] = get_type("diffusers", "KandinskyPipeline")

    parameters = args.pop("parameters", {})
    if "model_name_prior" in parameters:
        args["model_name_prior"] = parameters["model_name_prior"]

    args.pop("revision", None)

    return kandinsky_callback, args


async def format_stable_diffusion_args(args, workflow):
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

    if workflow == "img2img" or "start_image_uri" in args:
        start_image = await get_image(args.pop("start_image_uri"), size)

        controlnet = parameters.get("controlnet", None)
        if controlnet is not None:
            control_image = await get_control_image(start_image, controlnet, size)
            start_image = (
                control_image
                if start_image is None
                else scale_to_size(start_image, control_image.size)
            )

            args["control_image"] = control_image
            parameters["pipeline_type"] = "StableDiffusionControlNetImg2ImgPipeline"
            args["controlnet_model_name"] = controlnet.get(
                "controlnet_model_name", "lllyasviel/control_v11p_sd15_canny"
            )
            args["save_preprocessed_input"] = controlnet.get("preprocess", False)
            args["controlnet_conditioning_scale"] = float(
                controlnet.get("controlnet_conditioning_scale", 1.0)
            )

        elif "pipeline_type" not in parameters:
            parameters["pipeline_type"] = "StableDiffusionImg2ImgPipeline"
            args.pop("height", None)
            args.pop("width", None)

        if args["model_name"] == "timbrooks/instruct-pix2pix":
            # pix2pix models use image_guidance_scale instead of strength
            # image_guidance_scale has a range of 1-5 instead 0-1
            args["image_guidance_scale"] = args.pop("strength", 0.6) * 5

        if start_image is None:
            raise ValueError("Workflow requires an input image. None provided")

        args["image"] = start_image

    if "mask_image_uri" in args:
        args.pop("height", None)
        args.pop("width", None)

        args["mask_image"] = await get_image(args.pop("mask_image_uri"), size)

    if "num_inference_steps" not in args:
        # default num_inference_steps if not set - some pipelines have high default values
        args["num_inference_steps"] = 30

    args["pipeline_type"] = get_type(
        "diffusers", parameters.pop("pipeline_type", "DiffusionPipeline")
    )
    args["scheduler_type"] = get_type(
        "diffusers", parameters.pop("scheduler_type", "DPMSolverMultistepScheduler")
    )

    # remove any unsupported args
    for arg in parameters.pop("unsupported_pipeline_arguments", []):
        args.pop(arg, None)

    # now pass any remaining special args to the pipeine
    for key, value in parameters.items():
        args[key] = value

    return diffusion_callback, args
