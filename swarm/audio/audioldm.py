import torch
from diffusers import AudioLDMPipeline, DPMSolverMultistepScheduler
import scipy
import numpy as np
import tempfile
import pathlib
from io import BytesIO
from ..output_processor import make_result
from pydub import AudioSegment


def txt2audio_diffusion_callback(device_identifier, model_name, **kwargs):
    scheduler_type = kwargs.pop("scheduler_type", DPMSolverMultistepScheduler)
    pipeline_type = kwargs.pop("pipeline_type", AudioLDMPipeline)
    kwargs["num_inference_steps"] = kwargs.pop("num_inference_steps", 20)
    kwargs["audio_length_in_s"] = kwargs.pop("audio_length_in_s", 10)
    content_type = kwargs.pop("content_type", "audio/mpeg")
    kwargs.pop("outputs", ["primary"])
    pipeline = pipeline_type.from_pretrained(model_name, torch_dtype=torch.float16)
    pipeline.scheduler = scheduler_type.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device_identifier)

    with tempfile.TemporaryDirectory() as tmpdirname:
        audio = pipeline(**kwargs).audios[0]

        temp_wav_file = pathlib.Path(tmpdirname).joinpath("music.wav").__str__()
        scipy.io.wavfile.write(temp_wav_file, rate=16000, data=audio.astype(np.float32))
        audio_data = AudioSegment.from_file(temp_wav_file, format="wav")

        temp_mp3_file = pathlib.Path(tmpdirname).joinpath("music.mp3").__str__()
        audio_data.export(temp_mp3_file, format="mp3")
        with open(temp_mp3_file, "rb") as audio_file:
            buffer = BytesIO(audio_file.read())

    results = {"primary": make_result(buffer, None, content_type)}
    return (results, pipeline.config)  # type: ignore
