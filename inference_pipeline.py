from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Carrega modelo ─────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("./my_lora_model")
tokenizer.padding_side = "right"
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "./my_lora_model")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device="cpu",
)

def perguntar(pergunta: str) -> str:
    prompt = f"<|im_start|>user\n{pergunta}<|im_end|>\n<|im_start|>assistant\n"
    resultado = generator(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.eos_token_id,
    )
    resposta = resultado[0]["generated_text"]
    resposta = resposta.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    return resposta

# ── Testa ──────────────────────────────────────────────
perguntas = [
    "Qual é a capital de Portugal?",
    "Quanto é 2 + 2?",
    "O que é um LLM?",
]

print("=== PIPELINE DE INFERÊNCIA ===\n")
for pergunta in perguntas:
    resposta = perguntar(pergunta)
    print(f"❓ {pergunta}")
    print(f"💬 {resposta}")
    print("-" * 50)