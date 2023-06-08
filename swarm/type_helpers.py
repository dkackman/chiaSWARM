import platform
import sys
import torch

is_windows = any(platform.win32_ver())
is_3_11 = sys.version_info >= (3, 11)

def get_type(module_name, type_name):
    module = __import__(module_name)
    return getattr(module, type_name)


def has_method(o, name):
    return callable(getattr(o, name, None))

# torch compile not supported on all platforms

#leaving disabled for now
run_compile = False # hasattr(torch.nn.functional, "scaled_dot_product_attention") and not (is_windows or is_3_11)