import os
import gradio as gr
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

MODEL_REPO = os.environ.get("MODEL_REPO", "nelsondiasandre/qwen25-1.5b-pt-qa-lora")
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"A carregar modelo: {MODEL_REPO}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = "right"
model = AutoPeftModelForCausalLM.from_pretrained(MODEL_REPO)
model.eval()
print("Modelo carregado!")


def responder(pergunta: str, historico: list, max_tokens: int, temperature: float) -> str:
    prompt = f"<|im_start|>user\n{pergunta}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_tokens),
        do_sample=True,
        temperature=float(temperature),
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    resposta = generated.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    return resposta


with gr.Blocks(title="Qwen2.5-1.5B PT-QA") as demo:
    gr.Markdown(
        """
        # Qwen2.5-1.5B — Perguntas & Respostas em Português
        Modelo fine-tunado com LoRA sobre 500 pares Q&A em Português (PT-PT).
        Categorias: história, ciência, cultura, tecnologia, IA/ML e mais.
        """
    )

    chatbot = gr.Chatbot(label="Conversa", height=450, type="messages")

    with gr.Row():
        txt = gr.Textbox(
            placeholder="Escreve a tua pergunta em português...",
            show_label=False,
            scale=8,
        )
        btn_enviar = gr.Button("Enviar", variant="primary", scale=1)
        btn_limpar = gr.Button("Limpar", scale=1)

    with gr.Accordion("Parâmetros", open=False):
        max_tokens = gr.Slider(20, 200, value=80, step=10, label="Máximo de tokens")
        temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperatura")

    gr.Examples(
        examples=[
            "Qual é a capital de Portugal?",
            "Quem foi Fernando Pessoa?",
            "Explica o que é aprendizagem automática.",
            "Qual é o rio mais longo de Portugal?",
            "O que é a inteligência artificial?",
        ],
        inputs=txt,
    )

    def enviar(mensagem: str, historico: list, max_tok: int, temp: float):
        if not mensagem.strip():
            return historico, ""
        resposta = responder(mensagem, historico, max_tok, temp)
        historico = historico + [
            {"role": "user", "content": mensagem},
            {"role": "assistant", "content": resposta},
        ]
        return historico, ""

    btn_enviar.click(
        enviar,
        inputs=[txt, chatbot, max_tokens, temperature],
        outputs=[chatbot, txt],
    )
    txt.submit(
        enviar,
        inputs=[txt, chatbot, max_tokens, temperature],
        outputs=[chatbot, txt],
    )
    btn_limpar.click(lambda: ([], ""), outputs=[chatbot, txt])

if __name__ == "__main__":
    demo.launch()
