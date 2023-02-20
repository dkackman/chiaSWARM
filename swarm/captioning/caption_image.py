import torch
from ..diffusion.output_processor import make_text_result
from ..type_helpers import get_type


def caption_callback(device_id, model_name, **kwargs):
    pipeline_config = {}
    results = {}
    try:
        print("Image captioning...")
        model_params = kwargs.pop("parameters")
        processor_type = get_type("transformers", model_params["processor_type"])
        model_type = get_type("transformers", model_params["model_type"])
        processor = processor_type.from_pretrained(model_name)  # type: ignore
        model = model_type.from_pretrained(  # type: ignore
            model_name, torch_dtype=torch.float16
        ).to(  # type: ignore
            f"cuda:{device_id}"
        )  # type: ignore

        image = kwargs["image"]

        if "prompt" in kwargs and len(kwargs["prompt"]) > 0:
            # conditional image captioning and VQA
            inputs = processor(image, kwargs["prompt"], return_tensors="pt")
        else:
            # unconditional image captioning
            inputs = processor(image, return_tensors="pt")

        inputs = inputs.to(f"cuda:{device_id}", torch.float16)  # type: ignore
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
