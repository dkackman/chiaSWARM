from .diffusion.diffusion_func import diffusion_callback
from .video.tx2vid import txt2vid_diffusion_callback
from .captioning.caption_image import caption_callback
from .toolbox.stitch import stitch_callback
from .video.pix2pix import model_video_callback
from .audio.audioldm import txt2audio_diffusion_callback
from .audio.bark import bark_diffusion_callback
from .diffusion.diffusion_func_if import diffusion_if_callback
from .type_helpers import get_type, load_type_from_full_name
from .pre_processors.controlnet import scale_to_size
from .pre_processors.depth_estimator import make_hint
from .pre_processors.image_utils import resize_square
from .external_resources import get_image, get_control_image, max_size, download_images
from .loras import Loras


async def format_args(job, settings, device_identifier):
    args = prepare_args(job, settings)

    workflow = args.pop("workflow", None)
    if workflow == "txt2audio":
        if args["model_name"] == "suno/bark":
            return bark_diffusion_callback, args

        return format_txt2audio_args(args)

    if workflow == "stitch":
        return await format_stitch_args(args)

    if workflow == "img2txt":
        return await format_img2txt_args(args)

    if workflow == "vid2vid":
        return model_video_callback, args

    if workflow == "txt2vid":
        return format_txt2vid_args(args)

    if args["model_name"].startswith("DeepFloyd/"):
        return diffusion_if_callback, args

    return await format_stable_diffusion_args(args, workflow, device_identifier)


def prepare_args(job, settings):
    # ares shared by more than 1 workflow
    args = job.copy()
    if "lora" in args:
        loras = Loras(settings.lora_root_dir)
        args["lora"] = loras.resolve_lora(args["lora"])

    return args


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


async def format_stable_diffusion_args(args, workflow, device_identifier):
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

        controlnet = parameters.pop("controlnet", None)
        if controlnet is not None:
            if "pipeline_type" not in parameters:
                parameters["pipeline_type"] = "StableDiffusionControlNetImg2ImgPipeline"

            control_image = await get_control_image(
                start_image, controlnet, size, device_identifier
            )

            # the sdxl controlnet pipeline does not accept a control_image
            if parameters["pipeline_type"] == "StableDiffusionXLControlNetPipeline":
                start_image = control_image
            else:
                args["control_image"] = control_image
                if start_image is None:
                    start_image = control_image
                else:
                    start_image = scale_to_size(start_image, control_image.size)

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

        if (
            args["model_name"] == "timbrooks/instruct-pix2pix"
            or args["model_name"] == "diffusers/sdxl-instructpix2pix-768"
        ):
            # pix2pix models use image_guidance_scale instead of strength
            # image_guidance_scale has a range of 1-5 instead 0-1
            args["image_guidance_scale"] = args.pop("strength", 0.6) * 5

        if start_image is None:
            raise ValueError("Workflow requires an input image. None provided")

        # These two models need the size set to the size of the input image or the error out
        if (
            args["model_name"] == "diffusers/sdxl-instructpix2pix-768"
            or args["model_name"]
            == "kandinsky-community/kandinsky-2-2-controlnet-depth"
        ):
            start_image = resize_square(start_image).resize((768, 768))
            args["height"] = start_image.height
            args["width"] = start_image.width

        # further kandinsky controlnet uses "hint" instead of "image"
        if args["model_name"] == "kandinsky-community/kandinsky-2-2-controlnet-depth":
            args["hint"] = make_hint(start_image).to(device_identifier)
        else:
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

    if "pipeline_prior_type" in parameters:
        args["pipeline_prior_type"] = get_type(
            "diffusers",
            parameters.pop("pipeline_prior_type", "KandinskyV22PriorPipeline"),
        )

    if "prior_timesteps" in parameters:
        args["prior_timesteps"] = load_type_from_full_name(
            parameters.pop("prior_timesteps")
        )

    # set defaults if the model specifies them
    default_height = parameters.pop("default_height", None)
    default_width = parameters.pop("default_width", None)
    if default_height is not None and "height" not in args:
        args["height"] = default_height
    if default_width is not None and "width" not in args:
        args["width"] = parameters.pop("default_width")

    # remove any unsupported args
    for arg in parameters.pop("unsupported_pipeline_arguments", []):
        args.pop(arg, None)

    # now pass any remaining special args to the pipeline
    for key, value in parameters.items():
        args[key] = value

    return diffusion_callback, args
