import os


class Loras:
    def __init__(self, root_dir):
        self.root_dir = os.path.expanduser(root_dir)

    def resolve_lora_path(self, lora):
        # if it is of format 'publisher/lora-name' then
        # it's a hugging face model and we don't need to
        # resolve it
        if "/" in lora:
            return lora

        # otherwise it is a local lora and we need to
        # resolve it to a fully qualified path
        return os.path.join(self.root_dir, lora)
