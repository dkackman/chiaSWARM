import torch
from diffusers import BitsAndBytesConfig
from ..toolbox.type_helpers import has_method
from ..pre_processors.controlnet import preprocess_image


def run_pipeline(pipeline_definition, device_identifier, intermediate_results = {}):
    configuration, from_pretrained_arguments = validate_definition(pipeline_definition)

    # run all the prerpocessors first
    for preprocessor in pipeline_definition.get("preprocessors", []) :          
        preprocessed_image = preprocess_image(preprocessor["image"], preprocessor["name"], device_identifier)
        intermediate_result = preprocessor["capture_intermediate_result"]
        intermediate_results[intermediate_result] = preprocessed_image
    
    # then load the controlnet if specified
    controlnet = load_and_configure_component(pipeline_definition, "controlnet", device_identifier)
    if controlnet is not None:
        from_pretrained_arguments["controlnet"] = controlnet

    # then load the transformer if specified
    transformer = load_and_configure_component(pipeline_definition, "transformer", device_identifier)
    if transformer is not None:
        from_pretrained_arguments["transformer"] = transformer

    # load and configure the pipeline
    pipeline = load_and_configure_pipeline(configuration, from_pretrained_arguments, device_identifier)

    # load loras and fuse them into the pipeline
    loras = pipeline_definition.get("loras", [])        
    for lora in loras:
        default_lora_scale = 0.7 / len(loras) # default to equally distributing lora weights
        lora_name = lora.pop("lora_name", None)
        print(f"Loading lora {lora_name}...")
        lora_scale = lora.pop("lora_scale", default_lora_scale)
        pipeline.load_lora_weights(lora_name, **lora)
        pipeline.fuse_lora(lora_scale=lora_scale)

    # create a generator that will be used by each iteration if they don't set their own seed
    seed = configuration["seed"] if "seed" in configuration else torch.seed()
    default_generator = torch.Generator(device_identifier).manual_seed(seed)
    results = []

    # prepare and run pipeline iterations
    for iteration in pipeline_definition.get("iterations", []):
        arguments = iteration.get("arguments", {})

        # each iteration can use its own seed
        if "seed" in iteration:
            arguments["generator"] = torch.Generator(device_identifier).manual_seed(iteration["seed"])
        else:
            arguments["generator"] = default_generator

        # if there are intermediate results requested, add them to the iteration
        intermediate_result_names = iteration.get("intermediate_results", {})
        for k, v in intermediate_result_names.items():
            arguments[k] = intermediate_results[v]

        # run the pipeline
        output = pipeline(**arguments)
        result = get_result(output)
        results.append(result)
        #
        # the presence of this key indicates that the output should be
        # stored as an intermediate result, not returned as an output
        #
        # NOTE - the capture key can be used to diferentiate between different
        #        iterations of the same pipeline. It is not required.
        #
        if "capture_intermediate_results" in iteration:
            intermediate_result_names = iteration["capture_intermediate_results"]
            capture_key = iteration.get("capture_key", "")
            for k, v in intermediate_result_names.items():
                # output can have different shapes, so we need to check if the key is present
                if v in output:
                    intermediate_results[k + capture_key] = output[v]
                else:
                    intermediate_results[k + capture_key] = result

    return results


def load_and_configure_component(parent_definition, component_name, device_identifier):
    # then load the transformer if specified
    component_definition = parent_definition.get(component_name, None)
    if component_definition is not None:
        print(f"Loading {component_name}")
        component_configuration, component_from_pretrained_arguments = validate_definition(component_definition)
        return load_and_configure_pipeline(component_configuration, component_from_pretrained_arguments, device_identifier)

    return None


def get_result(output):
    if hasattr(output, "images"):
        return output.images[0]
    
    if hasattr(output, "image_embeds"):
        return output.image_embeds[0]

    if hasattr(output, "image_embeddings"):
        return output.image_embeddings[0]

    if hasattr(output, "frames"):
        return output.frames[0]
    
    if hasattr(output, "audios"):
        return output.audios[0].T.float().cpu().numpy()
    
    return None


def validate_definition(pipeline_definition):
    configuration = pipeline_definition.get("configuration", None)
    if configuration is None:
        raise Exception("configuration is required for a pipeline")
    
    from_pretrained_arguments = pipeline_definition.get("from_pretrained_arguments", None)
    if from_pretrained_arguments is None:
        raise Exception("from_pretrained_arguments is required for a pipeline")
    
    return configuration, from_pretrained_arguments
    

def load_and_configure_pipeline(configuration, from_pretrained_arguments, device_identifier):
    bits_and_bytes_config = configuration.get("bits_and_bytes_configuration", None)
    if bits_and_bytes_config is not None:
        print("Loading bits and bytes config")
        from_pretrained_arguments["quantization_config"] = BitsAndBytesConfig(**bits_and_bytes_config)

    # load the pipeline
    pipeline_type = configuration.get("pipeline_type", None)
    model_name = from_pretrained_arguments.pop("model_name", None)  
    print(f"Loading pipeline {model_name}...")

    pipeline = pipeline_type.from_pretrained(model_name, **from_pretrained_arguments)
            
    # configure the pipeline
    if (configuration.get("set_unet_memory_format", False)) and hasattr(pipeline, 'unet'):
        pipeline.unet.to(memory_format=torch.channels_last)
    if (configuration.get("enable_vae_slicing", False)) and has_method(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if (configuration.get("enable_vae_tiling", False)) and has_method(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()

    offload = configuration.get("offload", None)
    if offload == "full" and has_method(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
    elif offload == "sequential" and has_method(pipeline, "enable_sequential_cpu_offload"):
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline = pipeline.to(device_identifier)

    vae = configuration.get("vae", {})
    if vae.get("enable_slicing", False):
        pipeline.vae.enable_slicing()

    if vae.get("enable_tiling", False):
        pipeline.vae.enable_tiling()

    return pipeline
