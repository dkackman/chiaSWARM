from transformers import AutoTokenizer, AutoModelForCausalLM


def load(model_name, device):
    print(f"Loading {model_name} to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    print("done.")
    return tokenizer, model


def chat(tokenizer, model, device):
    print("Starting chat...")

    prompt = "Hello, my dog is cute"
    tokenized_items = tokenizer(prompt, return_tensors="pt").to(device)
    logits = model.generate(**tokenized_items)
    output = tokenizer.decode(logits[0], skip_special_tokens=True)

    print(output)
