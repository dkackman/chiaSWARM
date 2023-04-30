from .generator import do_work
from .worker import startup
import asyncio
from . import __version__
from .gpu.device import Device

test_job = {
    "id": "__test__",
    "model_name": "stabilityai/stable-diffusion-2-1",
    "prompt": "spoons",
    "num_inference_steps": 10,
    "outputs": ["primary", "inference_image_strip"],
}

txt2audio_job = {
    "id": "__test__",
    "model_name": "cvssp/audioldm",
    "workflow": "txt2audio",
    "prompt": "Techno music with a strong, upbeat tempo and high melodic riffs",
    "num_inference_steps": 10,
    "outputs": ["primary"],
}

vid2vid_job = {
    "id": "__test__",
    "model_name": "timbrooks/instruct-pix2pix",
    "prompt": "make it sunny",
    "negative_prompt": "ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, blurry, bad anatomy, bad proportions, cloned face, disfigured, out of frame, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, long neck",
    "num_inference_steps": 10,
    "workflow": "vid2vid",
    "video_uri": "https://nftfactory.blob.core.windows.net/images/Pexels%20Videos%202795750.mp4",
    "outputs": ["primary"],
}

bark_job = {
    "id": "__test__",
    "model_name": "suno/bark",
    "prompt": "Hola, mi nombre es Pepe'. Y, eh, y me gusta pepe' coin. [laughs] Pero me gustaría que apostara si tuviera más marmotas",
    "workflow": "txt2audio",
    "outputs": ["primary"],
}


async def run_test(job):
    await startup()
    try:
        result = await do_work(job, Device(0))

        if "error" in result["pipeline_config"]:
            print(result["pipeline_config"]["error"])
        else:
            print("ok")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    asyncio.run(run_test(bark_job))
