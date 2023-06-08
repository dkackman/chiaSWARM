import platform
import sys

is_windows = any(platform.win32_ver())
is_3_11 = sys.version_info >= (3, 11)

def get_type(module_name, type_name):
    module = __import__(module_name)
    return getattr(module, type_name)


def has_method(o, name):
    return callable(getattr(o, name, None))

# torch compile not supported on window or 3.11
run_compile = not (is_windows or is_3_11)