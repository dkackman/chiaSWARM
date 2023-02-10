def get_transformers_type(type_name):
    module = __import__("transformers")
    return getattr(module, type_name)
