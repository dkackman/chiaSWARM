FROM python:3.10.6-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 gcc git

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /sdaas

RUN python3 -m pip install --upgrade pip
RUN python -m pip install wheel setuptools

RUN pip install torch torchvision torchaudio
RUN pip install diffusers[torch] accelerate scipy ftfy concurrent-log-handler safetensors xformers==0.0.16rc425 triton moviepy
RUN pip install -U git+https://github.com/huggingface/transformers.git

COPY ./ /sdaas

# this will be mounted as a bind point so the image can use the host's model files
RUN mkdir /root/.cache/huggingface/

ENV SDAAS_ROOT=/sdaas/
ENV SDAAS_TOKEN=
ENV SDAAS_URI=https://chiaswarm.ai
ENV SDAAS_WORKERNAME=dock_worker
CMD ["python", "-m", "swarm.worker"]


# docker build -t dkackman/chiaswarm .
# docker run -it --gpus all --mount type=bind,src=C:\Users\don\.cache\huggingface,target=/root/.cache/huggingface/ --env HUGGINGFACE_TOKEN=<YOUR TOKEN> dkackman/chiaswarm /bin/bash
# docker run --gpus all --mount type=bind,src=C:\Users\don\.cache\huggingface,target=/root/.cache/huggingface/ --env HUGGINGFACE_TOKEN=<YOUR TOKEN> dkackman/chiaswarm
