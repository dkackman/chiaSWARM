def get_type(module_name, type_name):
    module = __import__(module_name)
    return getattr(module, type_name)


def has_method(o, name):
    return callable(getattr(o, name, None))
