import torch
from diffusers import DiffusionPipeline
from ..type_helpers import has_method

def run_pipeline(pipeline_definition, device_identifier, previous_result = None):
    configuration = pipeline_definition.get("configuration", {})
    from_pretrained_arguments = pipeline_definition.get("from_pretrained_arguments", {})
    arguments = pipeline_definition.get("arguments", {})

    pipeline_type = configuration.pop("pipeline_type", DiffusionPipeline)
    model_name = from_pretrained_arguments.pop("model_name", None)    
    pipeline = pipeline_type.from_pretrained(model_name, **from_pretrained_arguments)

    if (configuration.pop("set_unet_memory_format", False)) and hasattr(pipeline, 'unet'):
        pipeline.unet.to(memory_format=torch.channels_last)
    if (configuration.pop("enable_vae_slicing", False)) and has_method(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if (configuration.pop("enable_vae_tiling", False)) and has_method(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()

    offload = configuration.get("offload", None)
    if offload == "full" and has_method(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
    elif offload == "sequential" and has_method(pipeline, "enable_sequential_cpu_offload"):
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline = pipeline.to(device_identifier)

    seed = configuration["seed"] if "seed" in configuration else torch.seed()
    arguments["generator"] = torch.Generator(device_identifier).manual_seed(seed)
    output = pipeline(**arguments)
    
    return output.images[0] if hasattr(output, "images") else output.image_embeddings[0]