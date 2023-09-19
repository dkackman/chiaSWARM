import torch
from diffusers import DiffusionPipeline
from ..type_helpers import has_method
from ..post_processors.upscale import upscale_image


def prior_pipeline(args, device_identifier):
    pipeline_prior_type = args.pop("pipeline_prior_type", None)
    if pipeline_prior_type is not None:
        prompt = args.pop("prompt", "")
        negative_prompt = args.pop("negative_prompt", "")
        generator = args["generator"]

        pipe_prior = pipeline_prior_type.from_pretrained(
            args.pop("prior_model_name"), torch_dtype=torch.float16
        ).to(device_identifier)

        if args.pop("split_embeds", False):
            img = args["image"]
            strength = args.get("strength", 0.6)
            image_embeds = pipe_prior(
                prompt=prompt, image=img, strength=strength, generator=generator
            ).image_embeds
            negative_image_embeds = pipe_prior(
                prompt=negative_prompt, image=img, strength=1, generator=generator
            ).negative_image_embeds

        else:
            image_embeds, negative_image_embeds = pipe_prior(
                # prompt arguments are consumed by the prior pipeline
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator,
            ).to_tuple()

        args["image_embeds"] = image_embeds
        args["negative_image_embeds"] = negative_image_embeds


def refiner_pipeline(refiner, images, device_identifier, preserve_vram, args):
    if refiner is not None:
        refiner_pipeline = DiffusionPipeline.from_pretrained(
            refiner["model_name"],
            variant=refiner.get("variant", None),
            revision=refiner.get("revision", "main"),
            torch_dtype=torch.float16,
            use_safetensors=refiner.get("use_safetensors", True),
        ).to(device_identifier)

        if preserve_vram and has_method(refiner_pipeline, "enable_model_cpu_offload"):
            refiner_pipeline.enable_model_cpu_offload()

        return refiner_pipeline(
            image=images,
            prompt=args.get("prompt", ""),
            negative_prompt=args.get("negative_prompt", None),
            generator=args["generator"],
        ).images

    return images


def upscale_pipeline(upscale, images, device_identifier, args):
    if upscale:
        return upscale_image(
            images,
            device_identifier,
            args.get("prompt", ""),
            args.get("negative_prompt", None),
            args.get("num_images_per_prompt", 1),
            args["generator"],
            True, # always preserve vram fro upscaling
        )

    return images
