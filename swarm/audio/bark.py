from bark import SAMPLE_RATE, generate_audio, preload_models
import pathlib
import tempfile
from io import BytesIO
from ..output_processor import make_result
from pydub import AudioSegment
import scipy
import numpy as np


def bark_diffusion_callback(device_identifier, model_name, **kwargs):
    content_type = kwargs.pop("content_type", "audio/mpeg")
    outputs = kwargs.pop("outputs", ["primary"])

    # download and load all models
    preload_models(
        text_use_gpu=True, coarse_use_gpu=True, fine_use_gpu=True, codec_use_gpu=True
    )
    prompt = kwargs["prompt"]

    audio_array = generate_audio(prompt)

    with tempfile.TemporaryDirectory() as tmpdirname:
        audio = audio_array

        temp_wav_file = pathlib.Path(tmpdirname).joinpath(f"music.wav").__str__()
        scipy.io.wavfile.write(
            temp_wav_file, rate=SAMPLE_RATE, data=audio.astype(np.float32)
        )
        audio_data = AudioSegment.from_file(temp_wav_file, format="wav")

        temp_mp3_file = pathlib.Path(tmpdirname).joinpath(f"music.mp3").__str__()
        audio_data.export(temp_mp3_file, format="mp3")
        with open(temp_mp3_file, "rb") as audio_file:
            buffer = BytesIO(audio_file.read())

    results = {}
    results["primary"] = make_result(buffer, None, content_type)

    return (results, {})  # type: ignore
