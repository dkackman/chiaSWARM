from huggingface_hub import model_info, ModelFilter, HfApi

hub = HfApi()
filter = ModelFilter(tags=["diffusers", "textual_inversion"])
inversions = hub.list_models(filter=filter, search="sd-concepts-library")
print(len(inversions))

# LoRA weights ~3 MB
model_path = "sd-concepts-library/buhu-art-style"

info = model_info(model_path)
model_base = info.cardData["base_model"]
print(model_base)  # CompVis/stable-diffusion-v1-4
