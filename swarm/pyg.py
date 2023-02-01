from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-6b")

model = AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-6b")
model = model.to("cuda")

prompt = "Hello, my dog is cute"
tokenized_items = tokenizer(prompt, return_tensors="pt").to("cuda")
logits = model.generate(**tokenized_items)
output = tokenizer.decode(logits[0], skip_special_tokens=True)

print(output)
