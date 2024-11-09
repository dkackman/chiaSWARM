from ..external_resources import (
    get_image,
    get_qrcode_image,
    max_size,
    download_images,
    is_not_blank,
)
from ..type_helpers import load_type_from_name
from ..loras import Loras

def prepare_args(args):
    if args is None:
        return {}
    
    process_args(args)

    return args

def process_args(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                process_args(v)
            elif isinstance(v, list):
                for item in v:
                    process_args(item)
            elif (k.endswith("_type") or k.endswith("_dtype")) and k != "content_type":
                d[k] = load_type_from_name(v)
    elif isinstance(d, list):
        for item in d:
            process_args(item)