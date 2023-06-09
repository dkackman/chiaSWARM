from diffusers import DiffusionPipeline
import torch
from ..output_processor import OutputProcessor


def kandinsky_callback(device_identifier, model_name, **kwargs):
    pipeline_type = kwargs.pop("pipeline_type", DiffusionPipeline)
    pipeline_prior_type = kwargs.pop("pipeline_prior_type", DiffusionPipeline)
    model_name_prior = kwargs.pop(
        "model_name_prior", "kandinsky-community/kandinsky-2-1-prior"
    )

    guidance_scale = kwargs.get(
        "guidance_scale", 1.0
    )  # both pipelines need this so dont pop it

    output_processor = OutputProcessor(
        kwargs.pop("outputs", ["primary"]),
        kwargs.pop("content_type", "image/jpeg"),
    )

    pipe_prior = pipeline_prior_type.from_pretrained(
        model_name_prior, torch_dtype=torch.float16
    )
    pipe_prior.to(device_identifier)

    prompt = kwargs.pop("prompt", "")
    negative_prompt = kwargs.pop("negative_prompt", "")

    generator = kwargs["generator"]

    if "image" in kwargs and "image2" in kwargs:
        images_texts = [prompt, kwargs.pop("image"), kwargs.pop("image2")]
        weights = [0.2, 0.3, 0.5]
        image_embeds, negative_image_embeds = pipe_prior.interpolate(
            images_texts, weights
        )
        prompt = ""
    else:
        image_embeds, negative_image_embeds = pipe_prior(
            prompt, negative_prompt, guidance_scale=guidance_scale, generator=generator
        ).to_tuple()

    pipe = pipeline_type.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to(device_identifier)

    images = pipe(
        prompt,
        image_embeds=image_embeds,
        negative_image_embeds=negative_image_embeds,
        **kwargs,
    ).images

    output_processor.add_outputs(images)
    return (output_processor.get_results(), {})
