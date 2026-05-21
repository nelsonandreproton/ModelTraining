from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

# Carrega o modelo treinado
model = GPT2LMHeadModel.from_pretrained("./my_pretrained_model")
tokenizer = AutoTokenizer.from_pretrained("./my_pretrained_model")
model.eval()

# Testa geração de texto
prompt = "<|user|> Qual é a capital"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"Prompt:    {prompt}")
print(f"Gerado:    {generated}")