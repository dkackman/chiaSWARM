import os


class Loras:
    def __init__(self, root_dir):
        self.root_dir = os.path.expanduser(root_dir)

    def resolve_lora(self, lora):
        # if it is of format 'publisher/lora-name' then
        # it's a hugging face model and we don't need to
        # resolve it
        parts = lora.split("/")
        if len(parts) == 1:
            # otherwise it is a local lora and we need to
            # resolve it to a fully qualified path
            return {
                "lora": os.path.join(self.root_dir, lora),
                "weight_name": None,
                "subfolder": None,
            }
        
        # just the model repo publisher/repo
        if len(parts) == 2:
            return {"lora": lora, "weight_name": None, "subfolder": None}
        
        # publisher/repo/file_name
        if len(parts) == 3:
            return {"lora": "/".join(parts[0:2]), "subfolder": None, "weight_name": parts[-1]}
        
        # publisher/repo/subfolder(s)/file_name
        return {"lora": "/".join(parts[0:2]), "subfolder": "/".join(parts[parts[2:-1]]), "weight_name": parts[-1]}
