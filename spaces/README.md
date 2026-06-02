---
title: Qwen2.5-1.5B PT-QA
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.33.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# Qwen2.5-1.5B — Perguntas & Respostas em Português

Demo interativa do modelo [`nelsondiasandre/qwen25-1.5b-pt-qa-lora`](https://huggingface.co/nelsondiasandre/qwen25-1.5b-pt-qa-lora).

## Sobre o modelo

- **Base:** Qwen/Qwen2.5-1.5B-Instruct (1.54B parâmetros)
- **Fine-tuning:** LoRA (rank=16, alpha=32) — ~4.36M parâmetros treináveis (0.28%)
- **Dataset:** 5005 pares Q&A em Português PT-PT, 20+ categorias
- **Treino:** epoch-2 (early-stopping, teto 10 épocas) — o ponto antes do overfit
- **Avaliação (judge cego LoRA vs base, set isolado):** LoRA vence **76%** das vezes; fluência Δ+1.12; perplexidade isolada 12.95 → 7.26; contaminação PT-BR fortemente reduzida

## Como usar

Escreve uma pergunta em português e carrega **Enviar** ou prime Enter.
Ajusta os parâmetros de geração no acordeão **Parâmetros**.
