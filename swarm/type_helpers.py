def get_type(module_name, type_name):
    module = __import__(module_name)
    return getattr(module, type_name)
