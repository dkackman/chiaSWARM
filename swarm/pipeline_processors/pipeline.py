import torch
from diffusers import BitsAndBytesConfig
from ..pre_processors.controlnet import preprocess_image
from .result import Result

def run_pipeline(pipeline_definition, device_identifier, intermediate_results, shared_components):
    configuration, from_pretrained_arguments = validate_definition(pipeline_definition)

    # run all the prerpocessors first
    for preprocessor in pipeline_definition.get("preprocessors", []) :          
        preprocessed_image = preprocess_image(preprocessor["image"], preprocessor["name"], device_identifier)
        intermediate_result = preprocessor["capture_intermediate_result"]
        intermediate_results[intermediate_result] = preprocessed_image
    
    # grab any previously shared components and put them into from_pretrained_arguments
    # if a pipeline asks for both a shared component and specifies a configruation for 
    # that component, the configuration will take precedence
    for reused_component_name in pipeline_definition.get("reused_components", []):
        from_pretrained_arguments[reused_component_name] = shared_components[reused_component_name]

    # load the controlnet if specified
    controlnet = load_and_configure_component(pipeline_definition, "controlnet", device_identifier)
    if controlnet is not None:
        from_pretrained_arguments["controlnet"] = controlnet

    # load the transformer if specified
    transformer = load_and_configure_component(pipeline_definition, "transformer", device_identifier)
    if transformer is not None:
        from_pretrained_arguments["transformer"] = transformer

    # load and configure the pipeline
    pipeline = load_and_configure_pipeline(configuration, from_pretrained_arguments, device_identifier)

    # load and configure any custom scheduler
    load_and_configure_scheduler(pipeline_definition.get("scheduler", None), pipeline)
    
    # store any shared pipeline components for future use by other pipelines
    for shared_component_name in pipeline_definition.get("shared_components", []):
        shared_components[shared_component_name] = getattr(pipeline, shared_component_name)

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
        result = Result(pipeline(**arguments), iteration.get("result_properties", {"content_type": "image/png"}))
        results.append(result)
        #
        # the presence of this key indicates that the output should be
        # stored as an intermediate result, not returned as an output
        #
        # NOTE - the capture key can be used to differentiate between different
        #        iterations of the same pipeline. It is not required.
        #
        if "capture_intermediate_results" in iteration:
            intermediate_result_names = iteration["capture_intermediate_results"]
            capture_key = iteration.get("capture_key", "")
            for k, v in intermediate_result_names.items():
                raw_result = result.get_raw_result()
                # output can have different shapes, so we need to check if the key is present
                # if it is, capture that property of the result, otherwise just capture the result itself
                intermediate_results[k + capture_key] = raw_result.get(v, result.get_output())

    return results


def load_and_configure_scheduler(scheduler_definition, pipeline):
    if scheduler_definition is not None:
        print(f"Loading scheduler")
        scheduler_configuration = scheduler_definition.get("configuration", None)
        if scheduler_configuration is None:
            raise Exception("configuration is required for a scheduler")
        
        scheduler_type = scheduler_configuration.get("scheduler_type", None)
        pipeline.scheduler = scheduler_type.from_config(pipeline.scheduler.config, **scheduler_configuration)


def load_and_configure_component(parent_definition, component_name, device_identifier):
    component_definition = parent_definition.get(component_name, None)
    if component_definition is not None:
        print(f"Loading {component_name}")
        component_configuration, component_from_pretrained_arguments = validate_definition(component_definition)
        return load_and_configure_pipeline(component_configuration, component_from_pretrained_arguments, device_identifier)

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
    bits_and_bytes_configuration = configuration.get("bits_and_bytes_configuration", None)
    if bits_and_bytes_configuration is not None:
        print("Loading bits and bytes configuration")
        from_pretrained_arguments["quantization_config"] = BitsAndBytesConfig(**bits_and_bytes_configuration)

    # load the pipeline
    pipeline_type = configuration.get("pipeline_type", None)
    model_name = from_pretrained_arguments.pop("model_name", None)  
    print(f"Loading pipeline {model_name}...")

    pipeline = pipeline_type.from_pretrained(model_name, **from_pretrained_arguments)
            
    # configure the pipeline
    if (configuration.get("set_unet_memory_format", False)):
        pipeline.unet.to(memory_format=torch.channels_last)
    if (configuration.get("enable_vae_slicing", False)):
        pipeline.enable_vae_slicing()
    if (configuration.get("enable_vae_tiling", False)):
        pipeline.enable_vae_tiling()

    offload = configuration.get("offload", None)
    if offload == "full":
        pipeline.enable_model_cpu_offload()
    elif offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline = pipeline.to(device_identifier)

    vae = configuration.get("vae", {})
    if vae.get("enable_slicing", False):
        pipeline.vae.enable_slicing()

    if vae.get("enable_tiling", False):
        pipeline.vae.enable_tiling()

    unet = configuration.get("unet", {})
    if unet.get("enable_forward_chunking", False):
        pipeline.unet.enable_forward_chunking()

    return pipeline
