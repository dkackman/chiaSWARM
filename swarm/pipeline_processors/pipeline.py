import torch
from ..type_helpers import has_method
from ..pre_processors.controlnet import preprocess_image

def run_pipeline(pipeline_definition, device_identifier, intermediate_results = {}):
    configuration, from_pretrained_arguments = validate_pipeline(pipeline_definition)

    # first see if it's a preprocessor - these are always intermediate results
    for preprocessor in pipeline_definition.get("preprocessors", []) :          
        preprocessed_image = preprocess_image(preprocessor["image"], preprocessor["name"], device_identifier)
        intermediate_result_name = preprocessor["intermediate_result_name"]
        intermediate_results[intermediate_result_name] = preprocessed_image
    
    if from_pretrained_arguments is None:
        raise Exception("from_pretrained_arguments is required for a pipeline")
    
    controlnet = pipeline_definition.get("controlnet", None)
    if controlnet is not None:
        print("Loading controlnet")
        controlnet_configuration, controlnet_from_pretrained_arguments = validate_pipeline(controlnet)
        if controlnet_from_pretrained_arguments is None:
            raise Exception("from_pretrained_arguments is required for a controlnet pipeline")   
             
        controlnet_pipeline = load_and_configure_pipeline(controlnet_configuration, controlnet_from_pretrained_arguments, device_identifier)
        from_pretrained_arguments["controlnet"] = controlnet_pipeline
   
    pipeline = load_and_configure_pipeline(configuration, from_pretrained_arguments, device_identifier)

    # load loras and fuse them
    loras = pipeline_definition.get("loras", [])        
    default_lora_scale = [0.7] * len(loras) # default to equally distributing lora weights
    for lora in loras:
        lora_name = lora.pop("lora_name", None)
        lora_scale = lora.pop("lora_scale", default_lora_scale)
        pipeline.load_lora_weights(lora_name, **lora)
        pipeline.fuse_lora(lora_scale=lora_scale)

    # run the pipeline
    arguments = pipeline_definition.get("arguments", {})

    seed = configuration["seed"] if "seed" in configuration else torch.seed()
    arguments["generator"] = torch.Generator(device_identifier).manual_seed(seed)

    # if there are intermediate results requested, add them to the arguments
    intermediate_result_names = arguments.pop("intermediate_result_names", {})
    for k, v in intermediate_result_names.items():
        arguments[k] = intermediate_results[v]

    output = pipeline(**arguments)
    
    # the presence of this configuration key indicates that the output should be
    # stored as an intermediate result, not returned as an output
    if "intermediate_result_names" in configuration:
        for i, name in enumerate(configuration["intermediate_result_names"]):
            intermediate_results[name] = output[name]
        return None

    return output.images[0] if hasattr(output, "images") else output.image_embeddings[0]


def validate_pipeline(pipeline_definition):
    configuration = pipeline_definition.get("configuration", None)
    if configuration is None:
        raise Exception("configuration is required for a pipeline")
    
    from_pretrained_arguments = pipeline_definition.get("from_pretrained_arguments", None)
    
    return configuration, from_pretrained_arguments
    

def load_and_configure_pipeline(configuration, from_pretrained_arguments, device_identifier):
    # load the pipeline
    pipeline_type = configuration.pop("pipeline_type", None)
    model_name = from_pretrained_arguments.pop("model_name", None)    
    pipeline = pipeline_type.from_pretrained(model_name, **from_pretrained_arguments)
            
    # configure the pipeline
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

    return pipeline
