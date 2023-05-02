import cv2
import numpy as np
from PIL import Image
from controlnet_aux import MLSDdetector


def pre_process_image(image, controlnet):
    if controlnet.get("type") == "canny":
        return image_to_canny(image, controlnet)

    if controlnet.get("type") == "mlsd":
        return image_to_mlsd(image)


def image_to_canny(image, controlnet):
    image = np.array(image)

    image = cv2.Canny(
        image,
        controlnet.get("low_threshold", 100),
        controlnet.get("high_threshold", 200),
    )
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def image_to_mlsd(image):
    processor = MLSDdetector.from_pretrained("lllyasviel/ControlNet")

    return processor(image)
