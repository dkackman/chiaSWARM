import torch


def process_prior_pipeline(args, device_identifier):
    pipeline_prior_type = args.pop("pipeline_prior_type", None)
    if pipeline_prior_type is not None:
        prompt = args.pop("prompt", None)
        negative_prompt = args.pop("negative_prompt", None)
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
