import torch


def process_prior_pipeline(args, device_identifier):
    pipeline_prior_type = args.pop("pipeline_prior_type", None)
    if pipeline_prior_type is not None:
        pipe_prior = pipeline_prior_type.from_pretrained(
            args.pop("prior_model_name"), torch_dtype=torch.float16
        ).to(device_identifier)

        if args.pop("split_embeds", False):
            print("")
            # image_embeds = pipe_prior(prompt=prompt, image=img, strength=0.85, generator=generator)
            # negative_image_embeds = pipe_prior(prompt=negative_prior_prompt, image=img, strength=1, generator=generator)

        else:
            image_embeds, negative_image_embeds = pipe_prior(
                # prompt arguments are consumed by the prior pipeline
                prompt=args.pop("prompt", None),
                negative_prompt=args.pop("negative_prompt", None),
                generator=args["generator"],
            ).to_tuple()
        args["image_embeds"] = image_embeds
        args["negative_image_embeds"] = negative_image_embeds
