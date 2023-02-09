from .open_clip import *
from .optim_utils import *
import argparse
from PIL import Image
from .job_arguments import download_image


def pez(image_path):
    # load the target image
    image = download_image(image_path)

    # load args
    args = argparse.Namespace()
    args.__dict__.update(
        read_json(
            "C:\\Users\\don\\src\\github\\dkackman\\chiaSWARM\\swarm\\pez_config.json"
        )
    )

    # load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrain, device=device
    )

    # You may modify the hyperparamters
    args.prompt_len = 8  # number of tokens for the learned prompt

    # optimize prompt
    learned_prompt = optimize_prompt(
        model, preprocess, args, device, target_images=[image]
    )
    print(learned_prompt)
