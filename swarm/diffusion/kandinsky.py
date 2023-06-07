from diffusers import KandinskyPipeline, KandinskyPriorPipeline
import torch
from ..output_processor import OutputProcessor


def kandinsky_callback(device_identifier, model_name, **kwargs):
    pipeline_type = kwargs.pop("pipeline_type", KandinskyPipeline)

    output_processor = OutputProcessor(
        kwargs.pop("outputs", ["primary"]),
        kwargs.pop("content_type", "image/jpeg"),
    )

    pipe_prior = KandinskyPriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16
    )
    pipe_prior.to(device_identifier)

    prompt = kwargs.pop("prompt", "")
    negative_prompt = kwargs.pop("negative_prompt", "")

    generator = kwargs["generator"]

    if "image" in kwargs and "image2" in kwargs:
        images_texts = [prompt, kwargs.pop("image"), kwargs.pop("image2")]
        weights = [0.2, 0.3, 0.5]
        image_embeds, negative_image_embeds = pipe_prior.interpolate(images_texts, weights)
        prompt = ""
    else:
        image_embeds = pipe_prior(
            prompt,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=generator,
            negative_prompt=negative_prompt,
        ).images
        negative_image_embeds = pipe_prior(
            negative_prompt,
            guidance_scale=1.0,
            num_inference_steps=25,
            generator=generator,
            negative_prompt=negative_prompt,
        ).images

    pipe = pipeline_type.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to(device_identifier)

    images = pipe(
        prompt,
        image_embeds=image_embeds,
        negative_image_embeds=negative_image_embeds,
        **kwargs,
    ).images
    images[0].save("./cheeseburger_monster.png")

    output_processor.add_outputs(images)
    return (output_processor.get_results(), {})
