from .diffusion.diffusion_func import diffusion_callback
from .video.tx2vid import txt2vid_diffusion_callback
from .captioning.caption_image import caption_callback
from .toolbox.stitch import stitch_callback
from .video.pix2pix import model_video_callback
from .audio.audioldm import txt2audio_diffusion_callback
from .audio.bark import bark_diffusion_callback
from .diffusion.diffusion_func_if import diffusion_if_callback
from .type_helpers import get_type, load_type_from_full_name
from .pre_processors.controlnet import preprocess_image
from .pre_processors.depth_estimator import make_hint
from .pre_processors.image_utils import resize_square, center_crop_resize
from .external_resources import (
    get_image,
    get_qrcode_image,
    max_size,
    download_images,
    is_not_blank,
)
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
        args["num_inference_steps"] = 20

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

    if "prompt" not in args:
        args["prompt"] = ""

    parameters = args.pop("parameters", {})

    if workflow == "img2img":
        await format_img2img_args(args, parameters, size, device_identifier)

    elif workflow == "inpaint" or "mask_image_uri" in args:
        await format_inpaint_args(args, parameters, size, device_identifier)

    elif workflow == "txt2img":
        await format_txt2img_args(args, parameters, size, device_identifier)

    if "num_inference_steps" not in args:
        # default num_inference_steps if not set - some pipelines have high default values
        args["num_inference_steps"] = 30

    if "pipeline_prior_type" in parameters:
        args["pipeline_prior_type"] = get_type(
            "diffusers",
            parameters.pop("pipeline_prior_type", "KandinskyV22PriorPipeline"),
        )

    if "prior_timesteps" in parameters:
        args["prior_timesteps"] = load_type_from_full_name(
            parameters.pop("prior_timesteps")
        )

    args["pipeline_type"] = get_type(
        "diffusers", parameters.pop("pipeline_type", "DiffusionPipeline")
    )
    args["scheduler_type"] = get_type(
        "diffusers", parameters.pop("scheduler_type", "DPMSolverMultistepScheduler")
    )

    # set defaults if the model specifies them
    default_height = parameters.pop("default_height", None)
    default_width = parameters.pop("default_width", None)
    if default_height is not None and "height" not in args:
        args["height"] = default_height
    if default_width is not None and "width" not in args:
        args["width"] = default_width

    # remove any unsupported args
    for arg in parameters.pop("unsupported_pipeline_arguments", []):
        args.pop(arg, None)

    # now pass any remaining special args to the pipeline
    for key, value in parameters.items():
        args[key] = value

    return diffusion_callback, args


async def format_inpaint_args(args, parameters, size, device_identifier):
    # inpaint inherits img2img setup since it has a start image
    await format_img2img_args(args, parameters, device_identifier)
    args["mask_image"] = await get_image(args.pop("mask_image_uri"), size)
    args.pop("height", None)
    args.pop("width", None)

    if "controlnet" in parameters:
        if "pipeline_type" not in parameters:
            # TODO don't rely on large_model flag for this
            if parameters.get("large_model", False):
                parameters[
                    "pipeline_type"
                ] = "StableDiffusionXLControlNetInpaintPipeline"
            else:
                parameters["pipeline_type"] = "StableDiffusionControlNetInpaintPipeline"

        await format_controlnet_args(args, parameters, None, size, device_identifier)

    else:
        if "pipeline_type" not in parameters:
            # TODO don't rely on large_model flag for this
            if parameters.get("large_model", False):
                parameters["pipeline_type"] = "StableDiffusionXLInpaintPipeline"
            else:
                parameters["pipeline_type"] = "StableDiffusionInpaintPipeline"


async def format_txt2img_args(args, parameters, size, device_identifier):
    if "controlnet" in parameters:
        if "pipeline_type" not in parameters:
            # TODO don't rely on large_model flag for this
            if parameters.get("large_model", False):
                parameters["pipeline_type"] = "StableDiffusionXLControlNetPipeline"
            else:
                parameters["pipeline_type"] = "StableDiffusionControlNetPipeline"

        await format_controlnet_args(args, parameters, None, size, device_identifier)


async def format_img2img_args(args, parameters, size, device_identifier):
    start_image = await get_image(args.pop("start_image_uri"), size)

    if size is None and start_image is not None:
        size = start_image.size

    if "controlnet" in parameters:
        await format_controlnet_args(
            args, parameters, start_image, size, device_identifier
        )
        if "pipeline_type" not in parameters:
            # TODO don't rely on large_model flag for this
            if parameters.get("large_model", False):
                parameters[
                    "pipeline_type"
                ] = "StableDiffusionXLControlNetImg2ImgPipeline"
            else:
                parameters["pipeline_type"] = "StableDiffusionControlNetImg2ImgPipeline"

    elif "pipeline_type" not in parameters:
        # TODO don't rely on large_model flag for this
        if parameters.get("large_model", False):
            parameters["pipeline_type"] = "StableDiffusionXLImg2ImgPipeline"
        else:
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

    # These two models need the size set to the size of the input image or they error out
    if (
        args["model_name"] == "diffusers/sdxl-instructpix2pix-768"
        or args["model_name"] == "kandinsky-community/kandinsky-2-2-controlnet-depth"
    ):
        start_image = resize_square(start_image).resize((768, 768))
        args["height"] = start_image.height
        args["width"] = start_image.width

    # if there is a control image, resize it to the size of the start image to match it
    if "control_image" in args:
        start_image = center_crop_resize(start_image, args["control_image"].size)

    args["image"] = start_image


async def format_controlnet_args(
    args, parameters, start_image, size, device_identifier
):
    controlnet = parameters.pop("controlnet")
    control_image = await get_image(controlnet.get("control_image_uri", None), size)
    args["save_preprocessed_input"] = True

    # if we have a qr code, use it to get the control image even if a control image was provided
    if is_not_blank(controlnet.get("qr_code_contents", None)):
        control_image = await get_qrcode_image(controlnet["qr_code_contents"], size)
        if start_image is None:
            start_image = control_image

    # if a preprocessor is specified derive the control_image from start image if present
    elif start_image is not None and is_not_blank(controlnet.get("preprocessor", None)):
        control_image = preprocess_image(
            start_image, controlnet["preprocessor"], device_identifier
        )

    # there's no start image but there is a control image - preprocess that if needed
    elif control_image is not None and is_not_blank(
        controlnet.get("preprocessor", None)
    ):
        control_image = preprocess_image(
            control_image, controlnet["preprocessor"], device_identifier
        )

    # finally just use the start_image as the control image
    elif control_image is None:
        control_image = start_image

    # we've tried every way we know to get a control image from the inputs but no joy
    if control_image is None:
        raise ValueError("Controlnet specified but no control image provided")

    controlnet_parameters = controlnet.get("parameters", {})

    args["controlnet_model_type"] = get_type(
        "diffusers",
        controlnet_parameters.get("controlnet_model_type", "ControlNetModel"),
    )
    if "controlnet_prepipeline_type" in controlnet_parameters:
        args["controlnet_prepipeline_type"] = get_type(
            "diffusers", controlnet_parameters["controlnet_prepipeline_type"]
        )
    args["controlnet_model_name"] = controlnet.get(
        "controlnet_model_name", "lllyasviel/control_v11p_sd15_canny"
    )
    args["controlnet_conditioning_scale"] = float(
        controlnet.get("controlnet_conditioning_scale", 1.0)
    )
    args["control_guidance_start"] = float(
        controlnet.get("control_guidance_start", 0.0)
    )
    args["control_guidance_end"] = float(controlnet.get("control_guidance_end", 1.0))

    # kandinsky controlnet uses "hint" instead of "image"
    if args["model_name"] == "kandinsky-community/kandinsky-2-2-controlnet-depth":
        args["hint"] = make_hint(control_image).to(device_identifier)
    # in this case we are not doing an img2img controlnet
    elif (
        parameters.get("pipeline_type", None) == "StableDiffusionControlNetPipeline"
        or parameters.get("pipeline_type", None)
        == "StableDiffusionXLControlNetPipeline"
    ):
        args["image"] = control_image
    else:
        args["control_image"] = control_image
