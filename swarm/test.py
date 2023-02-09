from .generator import do_work
from .worker import startup
import asyncio
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from .job_arguments import download_image
from .pez import pez

test_job = {
    "id": "__test__",
    "model_name": "stabilityai/stable-diffusion-2-1",
    "prompt": "spoons",
    "num_inference_steps": 10,
    # "outputs": ["primary", "inference_video", "inference_image_strip"],
    "outputs": ["primary", "inference_image_strip"],
    # "outputs": ["primary"],
}


async def test_pex():
    pez(
        "http://www.trbimg.com/img-537a9d05/turbine/la-trb-at-indy-course-vintage-race-cars-20140519"
    )


async def pix2pix():
    model_id = "timbrooks/instruct-pix2pix"

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(  # type: ignore
        "cuda"
    )

    url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
    image = download_image(url)

    prompt = "make the mountains snowy"
    pipe = pipe(
        prompt,
        image=image,
        num_inference_steps=20,
        image_guidance_scale=1.5,
        guidance_scale=7,
    )  # type: ignore
    pipe.images[0].save("snowy_mountains.png")  # type: ignore


async def run_test():
    await startup()
    try:
        result = await do_work(test_job)

        if "error" in result["pipeline_config"]:
            print(result["pipeline_config"]["error"])
        else:
            print("ok")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    asyncio.run(test_pex())
