# chiaSWARM

[![CodeQL](https://github.com/dkackman/chiaSWARM/actions/workflows/codeql.yml/badge.svg)](https://github.com/dkackman/chiaSWARM/actions/workflows/codeql.yml)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Distributed GPU compute or "All these GPUs are idle now. Let's use em for something other than PoW!"

## Introduction

The chiaSWARM is a distributed network of GPU nodes, that run AI and ML workloads on behalf of users that may not have the requisite hardware.

GPU nodes are paid in [XCH](https://www.chia.net/).

_This is NOT Proof of Work on chia._

## Workloads

### Stable Diffusion

The first supported workload is various type of stable diffusion image generation and manipulation.

Open an issue to gain access and give it a try on [the swarm network](https://chiaswarm.ai/)!

## Roadmap

- &check; Networking and core protocol
- &check; Basic stable diffusion workloads (txt2image, img2img, various models)
- &check; Image upscale, inpainting, and stable diffusion 2.1
- &check; Docker
- &check; More stable diffusion workloads (other interesting models & ongoing version bumps)
  - See the current [list of supported models](https://chiaswarm.ai/about)
- &check; XCH integration
- GPT workloads
- REAL ESRGAN image upscale and face fixing
- Whatever else catches our fancy

Suggestions, issues and pull requests welcome.

## Becoming the SWARM

In order to be a swarm node, you need a [CUDA](https://nvidia.custhelp.com/app/answers/detail/a_id/2132/~/what-is-cuda%3F) capable NVIDIA GPU with at least 8GB of VRAM; 30XX or better recommended.

Follow [the installation instructions](https://github.com/dkackman/chiaSWARM/wiki/Installation) to get started.
