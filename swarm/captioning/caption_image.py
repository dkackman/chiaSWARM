import torch
from ..diffusion.output_processor import make_text_result


def get_type(type_name):
    module = __import__("transformers")
    return getattr(module, type_name)


def caption_callback(device_id, model_name, **kwargs):
    pipeline_config = {}
    results = {}
    try:
        print("Image captioning...")
        processor_type = get_type(kwargs.pop("processor_type", "BlipProcessor"))
        model_type = get_type(kwargs.pop("model_type", "BlipForConditionalGeneration"))
        processor = processor_type.from_pretrained(model_name)  # type: ignore
        model = model_type.from_pretrained(  # type: ignore
            model_name, torch_dtype=torch.float16
        ).to(  # type: ignore
            f"cuda:{device_id}"
        )  # type: ignore

        image = kwargs["image"]

        # unconditional image captioning
        inputs = processor(image, return_tensors="pt").to(f"cuda:{device_id}", torch.float16)  # type: ignore

        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        results["primary"] = make_text_result(caption)
        pipeline_config["caption"] = caption
        return results, pipeline_config

    except Exception as e:
        print(e)
        pipeline_config["error"] = str(e)
        results["primary"] = make_text_result(str(e))

        return results, pipeline_config
