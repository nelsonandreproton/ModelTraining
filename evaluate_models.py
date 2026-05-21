import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

textos = [
    "<|im_start|>user\nQual é a capital de Portugal?<|im_end|>\n<|im_start|>assistant\nA capital de Portugal é Lisboa.<|im_end|>",
    "<|im_start|>user\nQuanto é 2 + 2?<|im_end|>\n<|im_start|>assistant\n2 + 2 é igual a 4.<|im_end|>",
    "<|im_start|>user\nO que é um LLM?<|im_end|>\n<|im_start|>assistant\nUm LLM é um modelo de linguagem treinado em grandes quantidades de texto.<|im_end|>",
]

def calcular_perplexidade(model, tokenizer, textos):
    model.eval()
    losses = []

    for texto in textos:
        inputs = tokenizer(
            texto,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)
            loss = outputs.loss.item()
            losses.append(loss)

    loss_media = sum(losses) / len(losses)
    perplexidade = math.exp(loss_media)
    return loss_media, perplexidade

# ── Modelo 1: Qwen2.5-1.5B base (sem fine-tuning) ─────
print("A carregar Qwen2.5-1.5B base...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
modelo_base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

loss1, ppl1 = calcular_perplexidade(modelo_base, tokenizer, textos)
print(f"\nQwen2.5-1.5B Base:")
print(f"   Loss media:    {loss1:.4f}")
print(f"   Perplexidade:  {ppl1:.2f}")

# ── Modelo 2: Qwen2.5-1.5B + LoRA fine-tunado ─────────
print("\nA carregar Qwen2.5-1.5B + LoRA...")
modelo_lora = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
modelo_lora = PeftModel.from_pretrained(modelo_lora, "./my_lora_model")

loss2, ppl2 = calcular_perplexidade(modelo_lora, tokenizer, textos)
print(f"\nQwen2.5-1.5B + LoRA Fine-tunado:")
print(f"   Loss media:    {loss2:.4f}")
print(f"   Perplexidade:  {ppl2:.2f}")

# ── Comparação final ───────────────────────────────────
print("\n" + "="*50)
print("COMPARACAO FINAL")
print("="*50)
modelos = [
    ("Qwen2.5-1.5B Base", loss1, ppl1),
    ("Qwen2.5-1.5B + LoRA", loss2, ppl2),
]
modelos_sorted = sorted(modelos, key=lambda x: x[2])

for nome, loss, ppl in modelos_sorted:
    print(f"{nome:25s} | Loss: {loss:.4f} | Perplexidade: {ppl:.2f}")