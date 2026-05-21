from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Carrega modelo base + adaptadores LoRA
tokenizer = AutoTokenizer.from_pretrained("./my_lora_model")
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base_model, "./my_lora_model")
model.eval()

prompts = [
    "<|user|> Qual é a capital de Portugal? <|assistant|>",
    "<|user|> Quanto é 2 + 2? <|assistant|>",
    "<|user|> O que é um LLM? <|assistant|>",
]

print("=== RESPOSTAS DO MODELO FINE-TUNADO ===\n")
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.3,   # ← penaliza repetições
    no_repeat_ngram_size=4,   # ← proíbe repetir sequências de 4 tokens
)
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"Prompt:   {prompt}")
    print(f"Resposta: {resposta}")
    print("-" * 60)