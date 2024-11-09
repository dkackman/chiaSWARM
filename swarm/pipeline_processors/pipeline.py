from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from ..type_helpers import has_method
import torch

def run_pipeline(pipeline_definition, device_identifier, previous_result = None):

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

    prompt = "A cat holding a sign that says hello world"
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
    ).images[0]
    image.save("flux-schnell.png")

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

    if (configuration.pop("enable_model_cpu_offload", False)) and has_method(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
        pipeline = pipeline.to(device_identifier)
    elif (configuration.pop("enable_sequential_cpu_offload", False)) and has_method(pipeline, "enable_sequential_cpu_offload"):
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline = pipeline.to(device_identifier)

    seed = configuration["seed"] if "seed" in configuration else torch.seed()

    arguments["generator"] = torch.Generator(device_identifier).manual_seed(seed)
    prompt = arguments.pop("prompt", None)
    output = pipeline(prompt,     
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0))
    
    return {}