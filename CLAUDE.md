# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Educational project demonstrating LLM fine-tuning with three approaches:
1. **LoRA fine-tuning** of base GPT-2 using PEFT adapters (parameter-efficient)
2. **Pretraining from scratch** of a custom tiny GPT-2 (~3M params)
3. **FastAPI inference server** serving the LoRA model

All training data and code comments are in Portuguese (PT-PT).

## Setup

All dependencies are installed in `./venv/`. Activate it before running anything:

```bash
# Windows
.\venv\Scripts\activate

# Verify environment
python test_setup.py
```

Key dependencies: `torch`, `transformers`, `datasets`, `peft`, `trl`, `fastapi`, `uvicorn`.

**Note:** `requirements.txt` is empty — dependencies exist only in the venv.

## Data Pipeline (must run in order)

```bash
python create_dataset.py        # Creates raw dataset → my_dataset/
python explore_dataset.py       # Applies instruction template → my_dataset_processed/
python tokenize_dataset.py      # Tokenizes (max 128 tokens) → my_dataset_tokenized/
```

Instruction format: `<|user|> {instruction} <|assistant|> {response}`

## Training

These can run independently after the data pipeline:

```bash
python finetune_lora.py         # Fine-tunes GPT-2 with LoRA → my_lora_model/
python pretrain.py              # Trains tiny GPT-2 from scratch → my_pretrained_model/
```

Both run on **CPU only** (`use_cpu=True`). Training saves epoch checkpoints to `lora_output/` and `model_output/`.

## Evaluation & Inference

```bash
python evaluate_models.py       # Compares all 3 models by perplexity
python inference_pipeline.py    # Interactive Q&A via HuggingFace pipeline
python test_lora.py             # Direct generation test (LoRA model)
python test_pretrained.py       # Direct generation test (pretrained model)
```

## API Server

```bash
python api.py                   # FastAPI on http://0.0.0.0:8000
```

Endpoints:
- `GET /` — status
- `GET /health` — health check
- `POST /perguntar` — `{"pergunta": str, "max_tokens": int, "temperature": float}` → `{"pergunta": str, "resposta": str}`

## Architecture

```
create_dataset → explore_dataset → tokenize_dataset
                                         │
                              ┌──────────┴──────────┐
                         finetune_lora.py       pretrain.py
                              │                      │
                         my_lora_model/    my_pretrained_model/
                              │
                     ┌────────┴────────┐
                 api.py          inference_pipeline.py
```

### Model Comparison

| | LoRA | Pretrained from scratch |
|---|---|---|
| Base | GPT-2 (full weights) | Custom tiny GPT-2 |
| Params trained | ~adapters only | ~3M full |
| Output | Adapter weights (lightweight) | Full model weights |

## Training Hyperparameters

- **LoRA**: rank=8, alpha=16, dropout=0.05, LR=2e-4, 10 epochs, target modules `c_attn`/`c_proj`
- **Pretrain**: 128 embedding dim, 4 layers, 4 heads, LR=5e-4, 10 epochs

## No pytest configuration exists

Manual test scripts (`test_*.py`) cover inference validation only. There is no automated test suite.
