import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-v0.1"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",  # optional for large models
    attn_implementation="flash_attention_2"
)

# Prompt
prompt = "My favourite condiment is"

# Tokenize and move input to model.device
model_inputs = tokenizer(prompt, return_tensors="pt")
model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}  # 🚨 fix

# Generate and decode
generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(output)

