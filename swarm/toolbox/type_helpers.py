import platform
import sys
import importlib

is_windows = any(platform.win32_ver())
is_3_11 = sys.version_info >= (3, 11)


def get_type(module_name, type_name):
    module = __import__(module_name)
    return getattr(module, type_name)

def load_type_from_name(type_name):
    if "." in type_name:
        return load_type_from_full_name(type_name)

    return get_type("diffusers", type_name)


def load_type_from_full_name(full_name):
    # Split the full name into module path and object name
    module_path, object_name = full_name.rsplit(".", 1)

    # Dynamically import the module
    module = importlib.import_module(module_path)

    # Get the object from the module
    return getattr(module, object_name)


def has_method(o, name):
    return callable(getattr(o, name, None))


# torch compile not supported on all platforms

# leaving disabled for now
run_compile = False  # hasattr(torch.nn.functional, "scaled_dot_product_attention") and not (is_windows or is_3_11)
