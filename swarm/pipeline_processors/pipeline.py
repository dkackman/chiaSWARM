import torch
from diffusers import DiffusionPipeline
from ..type_helpers import has_method
from ..pre_processors.depth_estimator import make_hint

def run_pipeline(pipeline_definition, device_identifier, intermediate_results = {}):
    configuration = pipeline_definition.get("configuration", {})

    # first see if it's a preprocessor - these are always intermediate results
    if "preprocessor" in configuration:
        if configuration["preprocessor"] == "depth_estimator":
            intermediate_result_name = configuration["intermediate_result_name"]
            intermediate_results[intermediate_result_name] = make_hint(configuration["image"], device_identifier)
            return None    
        
        raise Exception(f"Unknown preprocessor {configuration['preprocessor']}")
    
    # now it has to be a pipeline
    from_pretrained_arguments = pipeline_definition.get("from_pretrained_arguments", None)
    arguments = pipeline_definition.get("arguments", {})

    pipeline_type = configuration.pop("pipeline_type", None)
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

    # if there are intermediate requested results, add them to the arguments
    intermediate_result_names = arguments.pop("intermediate_result_names", {})
    for name in intermediate_result_names:
        arguments[name] = intermediate_results[name]

    output = pipeline(**arguments)
    
    # the presence of this configuration key indicates that the output should be
    # stored as an intermediate result, not returned as an output
    if "intermediate_result_names" in configuration:
        for i, name in enumerate(configuration["intermediate_result_names"]):
            intermediate_results[name] = output[name]
        return None

    return output.images[0] if hasattr(output, "images") else output.image_embeddings[0]