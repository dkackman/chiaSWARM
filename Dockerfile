FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04
WORKDIR /sdaas

# preapre the OS
RUN apt-get update
RUN apt-get -y install nvtop python3 python3-pip

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install wheel setuptools
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install diffusers[torch] transformers accelerate scipy ftfy concurrent-log-handler safetensors

COPY ./ /sdaas

# this will be mounted as a bind point so the image can use the hosts model files
RUN mkdir /root/.cache/huggingface/

ENV SDAAS_ROOT=/sdaas/
ENV SDAAS_TOKEN=
ENV SDAAS_URI=http://localhost:9511
ENV HUGGINGFACE_TOKEN=

# CMD ["conda", "run", "-n", "fing", "python", "-m", "generator.worker"]


# docker build -t chiaswarm .
# docker run --gpus all --env HUGGINGFACE_TOKEN=<YOUR TOKEN> dkackman/fing
