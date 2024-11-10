import torch
import numpy as np
from transformers import pipeline
from torchvision import transforms
from transformers import pipeline


def make_hint_tensor(image, device_identifier):
    depth_estimator = pipeline("depth-estimation", device=device_identifier)

    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    hint = detected_map.permute(2, 0, 1)
    return hint.unsqueeze(0).half().to(device_identifier)


def make_hint_image(image, device_identifier):
    hint = make_hint_tensor(image, device_identifier)
    # Convert the tensor to a Pillow image
    to_pil = transforms.ToPILImage()
    return to_pil(hint[0].cpu())
