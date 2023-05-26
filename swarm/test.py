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

txt2vidJob = {
    "id": "__test__",
    "model_name": "timbrooks/instruct-pix2pix",
    "prompt": "dogs dancing",
    "negative_prompt": "ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, blurry, bad anatomy, bad proportions, cloned face, disfigured, out of frame, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, long neck",
    "num_inference_steps": 10,
    "workflow": "txt2vid",
    "outputs": ["primary"],
}

bark_job = {
    "id": "__test__",
    "model_name": "suno/bark",
    "prompt": "Hola, mi nombre es Pepe'. Y, eh, y me gusta pepe' coin. [laughs] Pero me gustaría que apostara si tuviera más marmotas",
    "workflow": "txt2audio",
    "outputs": ["primary"],
}

if_job = {
    "id": "__test__",
    "model_name": "DeepFloyd/IF-II-L-v1.0",
    "prompt": 'a photo of a green frog wearing blue sunglasses standing in front of the eiffel tower holding a sign that says "i shill chia"',
    "workflow": "txt2img",
    "outputs": ["primary"],
}

kandinsky_job = {
    "id": "__test__",
    "model_name": "kandinsky-community/kandinsky-2-1",
    "prompt": 'a photo of a green frog wearing blue sunglasses standing in front of the eiffel tower holding a sign that says "i shill chia"',
    "negative_prompt": "low quality, bad quality",
    "workflow": "txt2img",
    "num_inference_steps": 100,
    "guidance_scale": 4.0,
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
    asyncio.run(run_test(kandinsky_job))
