import copy
from ..toolbox.type_helpers import load_type_from_name
from diffusers.utils import load_image

def prepare_args(input_args):
    if input_args is None:
        return {}

    args = copy.deepcopy(input_args)

    process_args(args)

    return args


def process_args(d):    
    if isinstance(d, dict):
        for k, v in d.items():
                if k.endswith("_image") or k == "image":
                    d[k] = process_image(v)    
                elif isinstance(v, dict):
                    process_args(v)
                elif isinstance(v, list):
                    for item in v:
                        process_args(item)
                elif (k.endswith("_type") or k.endswith("_dtype")) and k != "content_type":
                    d[k] = load_type_from_name(v)
                elif (k.startswith("{") and k.endswith("}")):
                    d[k] = load_type_from_name(v)                    
            
    elif isinstance(d, list):
        for item in d:
            process_args(item)

def process_image(image):
    # escape indicator for intermediate result references
    if isinstance(image, str):
        return image.strip("{}")
    
    img = load_image(image["location"])
    if "size" in image:
        img = img.resize((image["size"]["height"], image["size"]["width"]))

    return img