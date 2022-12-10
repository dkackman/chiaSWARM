FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
WORKDIR /sdaas

RUN apt-get update
RUN apt-get -y install wget
# preapre the OS
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update
RUN apt-get -y install nvtop python3 python3-pip cuda

RUN sudo apt-get -y install cuda

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install wheel setuptools
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install diffusers[torch] transformers accelerate scipy ftfy concurrent-log-handler safetensors

COPY ./ /sdaas

# this will be mounted as a bind point so the image can use the hosts model files
RUN mkdir /root/.cache/huggingface/

ENV SDAAS_ROOT=/sdaas/
ENV SDAAS_TOKEN=
ENV SDAAS_URI=http://fing.kackman.net:9511
ENV SDAAS_WORKERNAME=dock_worker
ENV HUGGINGFACE_TOKEN=
CMD ["python3", "-m", "swarm.worker"]


# docker build -t dkackman/chiaswarm .
# docker run --gpus all --env HUGGINGFACE_TOKEN=<YOUR TOKEN> dkackman/fing
