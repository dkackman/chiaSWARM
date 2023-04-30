FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update

# silence tzdaata
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y tzdata
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get install -y libgl1 libglib2.0-0 gcc git python3.10 ffmpeg

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN python -m pip install --upgrade pip
RUN python -m pip install wheel setuptools

RUN pip install torch torchvision torchaudio
RUN pip install diffusers[torch] transformers accelerate scipy ftfy safetensors moviepy opencv-python xformers
RUN pip install aiohttp concurrent-log-handler pydub
RUN pip install git+https://github.com/suno-ai/bark.git@main

WORKDIR /sdaas
COPY ./ /sdaas

# this will be mounted as a bind point so the image can use the host's model files
RUN mkdir /root/.cache/huggingface/

ENV SDAAS_ROOT=/sdaas/

# these are the configurable settings for the worker
ENV SDAAS_TOKEN=
ENV SDAAS_URI=https://chiaswarm.ai
ENV SDAAS_WORKERNAME=dock_worker
CMD ["python", "-m", "swarm.worker"]


# docker build -t dkackman/chiaswarm .
# docker run -it --gpus all --mount type=bind,src=C:\Users\don\.cache\huggingface,target=/root/.cache/huggingface/ --env SDAAS_TOKEN=<YOUR TOKEN> dkackman/chiaswarm /bin/bash
# docker run --gpus all --mount type=bind,src=C:\Users\don\.cache\huggingface,target=/root/.cache/huggingface/ --env SDAAS_TOKEN=<YOUR TOKEN> dkackman/chiaswarm
