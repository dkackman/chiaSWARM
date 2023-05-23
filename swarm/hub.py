from huggingface_hub import model_info, ModelFilter, HfApi

hub = HfApi()
inversions = hub.list_models(
    filter=ModelFilter(
        tags=[
            "diffusers",
            "textual_inversion"]),
    search="sd-concepts-library")
print(len(inversions))

# LoRA weights ~3 MB
model_path = f"sd-concepts-library/buhu-art-style"

info = model_info(model_path)
model_base = info.cardData["base_model"]
print(model_base)  # CompVis/stable-diffusion-v1-4
