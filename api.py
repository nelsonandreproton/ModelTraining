from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import uvicorn

# ── App ────────────────────────────────────────────────
app = FastAPI(
    title="LLM API",
    description="API de inferência do modelo fine-tunado",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Carrega modelo no arranque ─────────────────────────
print("A carregar modelo...")
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
print("Modelo carregado!")

# ── Schemas ────────────────────────────────────────────
class PerguntaRequest(BaseModel):
    pergunta: str
    max_tokens: int = 50
    temperature: float = 0.7

class RespostaResponse(BaseModel):
    pergunta: str
    resposta: str

# ── Endpoints ──────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "online", "modelo": "Qwen2.5-1.5B + LoRA"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/perguntar", response_model=RespostaResponse)
def perguntar(request: PerguntaRequest):
    prompt = f"<|im_start|>user\n{request.pergunta}<|im_end|>\n<|im_start|>assistant\n"

    resultado = generator(
        prompt,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=request.temperature,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.eos_token_id,
    )

    resposta = resultado[0]["generated_text"]
    resposta = resposta.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()

    return RespostaResponse(
        pergunta=request.pergunta,
        resposta=resposta,
    )

# ── Arranque ───────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)