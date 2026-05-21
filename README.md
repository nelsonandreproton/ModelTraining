# ModelTraining

Pipeline completo de fine-tuning e inferência de LLMs em português, usando Qwen2.5-1.5B-Instruct + LoRA.

## O que faz

- Fine-tuning eficiente do Qwen2.5-1.5B-Instruct com LoRA (PEFT) em 500 pares Q&A em português
- Pré-treino de um tiny GPT-2 do zero (fins educativos)
- API REST (FastAPI) para inferência
- Chat web para interagir com o modelo

## Stack

- Python 3.11+
- PyTorch + HuggingFace Transformers
- PEFT (LoRA) + TRL (SFTTrainer)
- FastAPI + Uvicorn

## Setup

```bash
# Activar ambiente virtual
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

# Verificar ambiente
python test_setup.py
```

## Pipeline

Executar pela seguinte ordem:

```bash
# 1. Dados
python create_dataset.py       # cria 500 pares Q&A → my_dataset/
python explore_dataset.py      # aplica template + split 80/20 → my_dataset_processed/
python tokenize_dataset.py     # tokeniza (max 512 tokens) → my_dataset_tokenized/

# 2. Treino
python finetune_lora.py        # LoRA fine-tuning → my_lora_model/
python pretrain.py             # pré-treino do zero → my_pretrained_model/ (opcional)

# 3. Avaliação
python evaluate_models.py      # compara perplexidade dos 2 modelos

# 4. Inferência
python inference_pipeline.py   # teste directo na linha de comandos
python api.py                  # API REST em http://localhost:8000

# 5. Chat web
# Abrir chat.html no browser com api.py a correr
```

## API

```bash
python api.py
```

**Endpoint:** `POST /perguntar`

```json
// Request
{ "pergunta": "Qual é a capital de Portugal?", "max_tokens": 50, "temperature": 0.7 }

// Response
{ "pergunta": "Qual é a capital de Portugal?", "resposta": "A capital de Portugal é Lisboa." }
```

Outros endpoints: `GET /` (status), `GET /health`

## Configuração LoRA

| Parâmetro | Valor | Notas |
|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct (1.54B) | pré-treinado Alibaba |
| Rank (r) | 8 | equilíbrio capacidade/tamanho |
| Alpha | 16 | escala = alpha/r = 2 |
| Target modules | q_proj, k_proj, v_proj, o_proj | camadas de atenção |
| Epochs | 20 | com cosine LR scheduler |
| Learning rate | 2e-4 | típico para LoRA |
| Max length | 512 tokens | |
| Batch size | 2 | CPU constraint |
| Gradient accum. | 4 | batch efectivo = 8 |
| Parâmetros treináveis | ~2.18M (0.14%) | apenas os adaptadores LoRA |

## Resultados

| Modelo | Loss | Perplexidade |
|---|---|---|
| Qwen2.5-1.5B + LoRA (fine-tuned) | 0.6199 | **1.86** |
| Qwen2.5-1.5B Base | 3.1232 | 22.72 |

Melhoria de **12×** em relação ao modelo base.

## Publicar no HuggingFace Hub

Após o treino, podem ser publicados dois repositórios:

| Tipo | Repo | Conteúdo |
|---|---|---|
| Model | `username/qwen25-1.5b-pt-qa-lora` | Adaptador LoRA (~tens MB, não os 3GB do modelo base) |
| Dataset | `username/portuguese-qa-instruct-500` | 500 pares Q&A PT-PT (train/test) |

### Passos

1. Criar conta em [huggingface.co](https://huggingface.co) e gerar token em Settings → Access Tokens
2. Adicionar ao `.env`:
   ```
   HF_TOKEN=hf_your_token_here
   ```
3. Preencher os placeholders nos cards:
   - `hub/model_card.md` — substituir `PLACEHOLDER_USERNAME`, `PLACEHOLDER_PERPLEXITY`, `PLACEHOLDER_LOSS`
   - `hub/dataset_card.md` — substituir `PLACEHOLDER_USERNAME`
4. Fazer o upload:
   ```bash
   python upload_to_hub.py --username nelsondiasandre --model --dataset
   # ou começar privado:
   python upload_to_hub.py --username nelsondiasandre --model --dataset --private
   ```

### O que fica em cada repo

**Model repo** (`hub/model_card.md` → `README.md` no Hub):
- `adapter_config.json` — hiperparâmetros LoRA
- `adapter_model.safetensors` — pesos do adaptador
- Ficheiros do tokenizer
- `README.md` com model card completo

**Dataset repo** (`hub/dataset_card.md` → `README.md` no Hub):
- 500 exemplos com colunas `instruction`, `response`, `text`
- Split train (400) / test (100)
- `README.md` com dataset card completo

### Carregar o modelo publicado

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("nelsondiasandre/qwen25-1.5b-pt-qa-lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

prompt = "<|im_start|>user\nQual é a capital de Portugal?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

## Documentação visual

Abrir `overview.html` no browser para visualização interactiva do pipeline, configurações e melhorias possíveis.
