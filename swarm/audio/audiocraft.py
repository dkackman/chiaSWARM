from bark import SAMPLE_RATE, generate_audio, preload_models
import pathlib
import tempfile
from io import BytesIO
from ..output_processor import make_result
from pydub import AudioSegment
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


def audiocraft_diffusion_callback(device_identifier, model_name, **kwargs):
    content_type = kwargs.pop("content_type", "audio/mpeg")
    kwargs.pop("outputs", ["primary"])

    model = MusicGen.get_pretrained(model_name.split("/")[-1])
    model.set_generation_params(kwargs.pop("duration", 8))
    descriptions = [kwargs.pop("prompt", "")]
    wav = model.generate(descriptions)

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_wav_file = pathlib.Path(tmpdirname).joinpath("music.wav").__str__()
        audio_write(
            tempfile,
            wav.cpu(),
            model.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )

        audio_data = AudioSegment.from_file(temp_wav_file, format="wav")
        temp_mp3_file = pathlib.Path(tmpdirname).joinpath("music.mp3").__str__()
        audio_data.export(temp_mp3_file, format="mp3")
        audio_data.export("audiocraft", format="mp3")
        with open(temp_mp3_file, "rb") as audio_file:
            _buffer = BytesIO(audio_file.read())

    results = {"primary": make_result(_buffer, None, content_type)}
    return (results, {})  # type: ignore
