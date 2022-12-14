# chiaSWARM

Distributed GPU compute or "All these GPUs are idle now. Let's use em for something other than PoW!"

## Introduction

The chiaSWARM is a distributed network of GPU nodes, that run AI and ML workloads on behalf of users that may not have the requisite hardware.

Soon(tm) nodes will be able to earn _swarmBUCKS_ on the [chia blockchain](https://www.chia.net/) which can also be bought and sold.

_This is NOT Proof of Work on chia._

## Workloads

### Stable Diffusion

The first supported workload is various type of stable diffusion image generation and manipulation.

Open an issue to gain acess and give it a try on the alpha network at <https://chiaswarm-dev.azurewebsites.net/>!

## Roadmap

- &check; Networking and core protocal
- &check; Basic stable diffusion workloads (txt2image, img2img, various models)
- &check; Image upscale, inpainting, and stable diffusion 2.1
- More stable diffusion workloads (other interesting models & ongoing version bumps)
- swarmBUCKS integration (earn, buy, sell, spend)
- GPT workloads
- Whatever else catches our fancy

Suggestions, issues and puill requests welcome.

## Becoming the SWARM

In order to run a swarm node, you need a [CUDA](https://nvidia.custhelp.com/app/answers/detail/a_id/2132/~/what-is-cuda%3F) capable NVIDIA GPU; 30XX or better recommended. 

While we test the network, nodes are by invite only. To participate please contact mailto:admin@chiaswarm.ai.

From the repo root run `sh install.sh` on linux or `install.ps1` on windows.

These scripts will create a virtual environment and install the swarm to it. To run the swarm worker:

### Linux

```bash
sh install.sh
. ./activate
python -m swarm.initialize # only needed once
python -m swarm.worker
```

### Windows

```powershell
.\install.ps1
venv\Scripts\activate
python -m swarm.initialize # only needed once
python -m swarm.worker
```

The `swarm.initialize` command will ask for your [huggingface token](https://huggingface.co/docs/hub/security-tokens), your swarm access token and the swarm uri.

The current swarm uri is <https://chiaswarm-dev.azurewebsites.net>.

It will also download all of the needed machine learning models which will take quite some time. It only needs to be run once.

## Docker

### still a work in progress

First install the [NVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on the docker host.

The docker image needs a bind mount for the huggingface model cache. Make sure the host machine has the cached models (typically in `~/.cache/huggingface`).

```bash
docker build -t chiaSWARM . -f Dockerfile
docker run -it -v "/home/YOUR_USERNAME/.cache/huggingface:/root/.cache/huggingface/" \
    --gpus all \
    --env HUGGINGFACE_TOKEN=YOUR TOKEN \
    --env SDAAS_TOKEN=YOUR TOKEN \
    --env SDAAS_URI=https://chiaswarm-dev.azurewebsites.net \
    chiaSWARM
```
