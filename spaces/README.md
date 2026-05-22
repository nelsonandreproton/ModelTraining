---
title: Qwen2.5-1.5B PT-QA
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# Qwen2.5-1.5B — Perguntas & Respostas em Português

Demo interativa do modelo [`nelsondiasandre/qwen25-1.5b-pt-qa-lora`](https://huggingface.co/nelsondiasandre/qwen25-1.5b-pt-qa-lora).

## Sobre o modelo

- **Base:** Qwen/Qwen2.5-1.5B-Instruct (1.54B parâmetros)
- **Fine-tuning:** LoRA (rank=8, alpha=16) — apenas 2.18M parâmetros treináveis (0.14%)
- **Dataset:** 500 pares Q&A em Português PT-PT, 20+ categorias
- **Perplexidade:** 1.86 (vs 22.72 base — melhoria de 12×)

## Como usar

Escreve uma pergunta em português e carrega **Enviar** ou prime Enter.
Ajusta os parâmetros de geração no acordeão **Parâmetros**.
