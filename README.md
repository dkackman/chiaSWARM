# chiaSWARM

Distributed GPU compute

## Stabel Diffusion

Alpha: https://chiaswarm-dev.azurewebsites.net/

## Worker Node

### Resources

- [Install nvidia drivers ubuntu](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [NVIDIA Docker Overview](https://hub.docker.com/r/nvidia/cuda) - if you're going to use docker

### Prepare the Environment

#### Ubuntu 22.10

```bash
# install linux nvidia drivers
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall # reboot after
sudo apt install nvtop # this is just handy to ahve around

# install miniconda
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh
conda update conda
```

#### Windows

```powershell
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o Miniconda3-latest-Windows-x86_64.exe
./Miniconda3-latest-Windows-x86_64.exe
```

### Install Dependencies

```bash
# create environment
# this is optional but create an environment now if desired
conda create --name swarm python==3.10.4
conda activate swarm

# install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 xformers \
transformers accelerate scipy ftfy diffusers[torch] concurrent-log-handler \
-c conda-forge -c pytorch -c nvidia -c xformers/label/dev
```

### Get Code and Run

```bash
git clone https://github.com/dkackman/chiaSWARM.git
cd chiaSWARM/src
conda activate swarm # if you created an environment
python -m swarm.initialize # only needed once - will take a long time if you've never used higging face models
python -m swarm.worker
```

If you see an error about `torch` not being available, leave and re-enter the environment and try again.

```bash
conda deactivate
conda activate fing
```

## Docker

### still a work in progress

First install the [NVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on the docker host.

The docker image needs a bind mount for the huggingface model cache. Make sure the host machine has the cached models (typically in `~/.cahce/huggingface`).

```bash
docker build -t dkackman/fing . -f worker.Dockerfile
docker run -it -v "/home/YOUR_USERNAME/.cache/huggingface:/root/.cache/huggingface/" \
    --gpus all \
    --env HUGGINGFACE_TOKEN=YOUR TOKEN \
    --env SDAAS_TOKEN=YOUR TOKEN \
    --env SDAAS_URI=http://fing.kackman.net:9511 \
    dkackman/fing
```
