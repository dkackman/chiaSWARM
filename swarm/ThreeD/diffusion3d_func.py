import tempfile
import torch
from diffusers import ShapEPipeline, ShapEImg2ImgPipeline
from diffusers.utils import export_to_gif
import shutil
import PIL
from typing import List
from diffusers.utils import export_to_gif, load_image

def diffusion3d_callback(device_identifier, model_name, **kwargs):
    ckpt_id = "openai/shap-e-img2img"
    pipe = ShapEImg2ImgPipeline.from_pretrained(ckpt_id).to("cuda")

    img_url = "https://pbs.twimg.com/profile_images/1646578911390707715/CgtZln01_400x400.jpg"
    image = load_image(img_url)

    generator = torch.Generator(device="cuda").manual_seed(0)
    batch_size = 4
    guidance_scale = 3.0

    images = pipe(
        image, 
        num_images_per_prompt=batch_size, 
        generator=generator, 
        guidance_scale=guidance_scale,
        num_inference_steps=64, 
        frame_size=128, 
        output_type="pil"
    ).images[0]

    gif_path = export_to_gif(images, "corgi_sampled_3d.gif")
    shutil.copy(gif_path, "./3d.gif")

    # pipe = ShapEPipeline.from_pretrained(model_name).to(device_identifier)
    # kwargs.pop("outputs", ["primary"])

    # images = pipe(**kwargs).images[0]

    # gif_path = export_to_gif(images, "shark_3d.gif")
    # shutil.copy(gif_path, "./3d.gif")

def export_to_gif(image: List[PIL.Image.Image], output_gif_path: str = None) -> str:
    if output_gif_path is None:
        output_gif_path = tempfile.NamedTemporaryFile(suffix=".gif").name

    image[0].save(
        output_gif_path,
        save_all=True,
        append_images=image[1:],
        optimize=False,
        duration=100,
        loop=0,
    )
    return output_gif_path