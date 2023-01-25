# chiaSWARM

[![CodeQL](https://github.com/dkackman/chiaSWARM/actions/workflows/codeql.yml/badge.svg)](https://github.com/dkackman/chiaSWARM/actions/workflows/codeql.yml)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Distributed GPU compute or "All these GPUs are idle now. Let's use em for something other than PoW!"

## Introduction

The chiaSWARM is a distributed network of GPU nodes, that run AI and ML workloads on behalf of users that may not have the requisite hardware.

Soon(tm) nodes will be able to earn _swarm_ on the [chia blockchain](https://www.chia.net/) which can also be bought and sold.

_This is NOT Proof of Work on chia._

## Workloads

### Stable Diffusion

The first supported workload is various type of stable diffusion image generation and manipulation.

Open an issue to gain acess and give it a try on [the beta network](https://chiaswarm.ai/)!

## Roadmap

- &check; Networking and core protocal
- &check; Basic stable diffusion workloads (txt2image, img2img, various models)
- &check; Image upscale, inpainting, and stable diffusion 2.1
- More stable diffusion workloads (other interesting models & ongoing version bumps)
- _swarm integration (earn, buy, sell, spend)
- GPT workloads
- Whatever else catches our fancy

Suggestions, issues and pull requests welcome.

## Becoming the SWARM

In order to be a swarm node, you need a [CUDA](https://nvidia.custhelp.com/app/answers/detail/a_id/2132/~/what-is-cuda%3F) capable NVIDIA GPU; 30XX or better recommended.

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

If, when running the powershell script you see an erro about not being able to run scripts, you will need to enable powershell script execution with an [execution policy](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.3).

```powershell
# this part needs to be run as an adminstrator
Set-ExecutionPolicy Unrestricted
```

```powershell
.\install.ps1
venv\Scripts\activate
python -m swarm.initialize # only needed once
python -m swarm.worker
```

The `swarm.initialize` command will ask for your [huggingface token](https://huggingface.co/docs/hub/security-tokens), your swarm access token and the swarm uri.

The current swarm uri is <https://chiaswarm.ai>.

It will also download all of the needed machine learning models which will take quite some time. It only needs to be run once.

## Docker

The docker image needs a bind mount for the huggingface model cache. Set the `src` in the example below property to `~/.cache/huggingface` or `C:\Users\don\.cache\huggingface`.

```bash
docker pull dkackman/chiaswarm

# only needs to be run once. files will be cached on the host
docker run \
    --mount "src=/home/YOUR_USERNAME/.cache/huggingface,target=/root/.cache/huggingface/" \
    dkackman/chiaswarm \
    python -m swarm.initialize --silent

# starts the swarm worker 
docker run \
    --mount "src=/home/YOUR_USERNAME/.cache/huggingface,target=/root/.cache/huggingface/" \
    --gpus all \
    --env SDAAS_TOKEN=YOUR TOKEN \
    dkackman/chiaswarm
```
