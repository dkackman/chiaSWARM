FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04
WORKDIR /fing

# preapre the OS
RUN apt-get update
RUN apt-get -y install curl nvtop 

# prepare miniconda
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p /miniconda
RUN /miniconda/bin/conda init bash
RUN /miniconda/bin/conda update conda
RUN /miniconda/bin/conda update -n base -c defaults conda

RUN /miniconda/bin/conda install pytorch torchvision torchaudio pytorch-cuda=11.6 xformers \
transformers accelerate scipy ftfy diffusers[torch] concurrent-log-handler \
-c conda-forge -c pytorch -c nvidia -c xformers/label/dev

RUN find /miniconda/ -follow -type f -name '*.a' -delete && \
    find /miniconda/ -follow -type f -name '*.js.map' -delete
RUN /miniconda/bin/conda clean -afy

COPY ./src /fing/

# fing worker runtime environment
RUN mkdir /root/.cache/huggingface/

ENV FING_ROOT=/sdaas/
ENV SDAAS_TOKEN=
ENV SDAAS_URI=http://localhost:9511
ENV HUGGINGFACE_TOKEN=

#CMD ["conda", "run", "-n", "fing", "python", "-m", "generator.worker"]


# docker build -t dkackman/fing --build-arg HUGGINGFACE_TOKEN=<YOUR_TOKEN> .
# docker run --gpus all --env HUGGINGFACE_TOKEN=<YOUR TOKEN> dkackman/fing

# docker run -it -v "/home/don/.cache/huggingface:/root/.cache/huggingface/" --gpus all --env HUGGINGFACE_TOKEN=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 --env SDAAS_TOKEN=6aca390c-f14f-4b7f-9557-b45e9ee891dc --env SDAAS_URI=http://fing.kackman.net:9511 dkackman/fing