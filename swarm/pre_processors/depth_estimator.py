import torch
import numpy as np

from transformers import pipeline

def make_hint(image):
    depth_estimator = pipeline("depth-estimation")

    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    hint = detected_map.permute(2, 0, 1)
    return hint.unsqueeze(0).half()