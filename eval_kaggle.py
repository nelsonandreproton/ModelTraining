# Avaliação GPU (Kaggle) — Qwen2.5-1.5B Base vs LoRA Fine-tunado
# Versão GPU do eval_compare.py, SEM o LLM-as-Judge.
#
# Porquê esta versão:
#   eval_compare.py corre em CPU local (sem .to("cuda")) — perplexidade sobre o
#   conjunto de validação + 75 gerações × 2 modelos demora ~1h. Em GPU é ~5 min.
#   O LLM-as-Judge (Camada 5) depende do LM Studio em localhost:1234 — inacessível
#   a partir do Kaggle — por isso fica de fora. Em vez disso este script GRAVA todas
#   as gerações em eval_generations.json; o judge corre localmente sobre esse ficheiro,
#   sem reinferência (o judge são só chamadas à API, não inferência local pesada).
#
# O que produz:
#   - eval_report_kaggle.txt : perplexidade + ROUGE-L + BERTScore + contaminação PT-BR
#   - eval_generations.json  : todas as gerações (Base/LoRA, verbose/conciso) p/ judge local
#
# Fluxo recomendado:
#   1. Kaggle: corre este script após o fine-tune (modelo ainda quente na sessão)
#   2. Download eval_generations.json do output do Kaggle
#   3. Local: corre o judge sobre o JSON (ver bloco no fim deste ficheiro)

# ── Pin a UM único T4 — tem de ser a primeira coisa do processo ────────────────
# CUDA_VISIBLE_DEVICES só faz efeito ANTES da primeira chamada CUDA. Qualquer
# import (torch/accelerate) ou probe inicializa o contexto CUDA com os GPUs
# visíveis nesse momento — depois disto a env var é no-op. Por isso fica no topo
# absoluto. (Mesma lição do finetune_lora_kaggle.py.)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 1 T4 chega para 2× 1.5B + BERT
os.environ["PYTORCH_ALLOC_CONF"]   = "expandable_segments:True"

import torch
print("visible GPUs:", torch.cuda.device_count(), flush=True)  # tem de imprimir 1
_free, _total = torch.cuda.mem_get_info()
print(f"GPU free: {_free/1e9:.2f} / {_total/1e9:.2f} GB", flush=True)

# ── Stack coesa (mesmos pins do fine-tune) ─────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.46.1",
                "peft==0.13.2",
                "accelerate==1.1.1",
                "datasets==3.1.0",
                "bert-score==0.3.13"], check=True)
print("deps instalados — se houver ImportError abaixo, Run > Restart & Run All uma vez", flush=True)

import json
import math
from datetime import datetime, UTC
from pathlib import Path

from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from kaggle_secrets import UserSecretsClient
hf_token = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token

# ── Config ──────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_REPO_ID  = "nelsondiasandre/qwen25-1.5b-pt-qa-lora"  # adapter no Hub
DATASET_NAME  = "nelsondiasandre/portuguese-qa-instruct-raw"

DEVICE = "cuda:0"
N_VALIDACAO        = 50    # amostra do split de validação (contaminado — baixa fiabilidade)
MAX_TOKENS_VERBOSE = 200
MAX_TOKENS_CONCISO = 80
RANDOM_SEED        = 42

REPORT_PATH      = Path("/kaggle/working/eval_report_kaggle.txt")
GENERATIONS_PATH = Path("/kaggle/working/eval_generations.json")

# ── Conjunto de teste ISOLADO (idêntico a eval_compare.py) ─────────────────────
TESTE_ISOLADO = [
    {"instruction": "Qual é o ponto mais ocidental da Europa Continental?",
     "response": "O Cabo da Roca, em Portugal, é o ponto mais ocidental da Europa Continental."},
    {"instruction": "Qual é a região autónoma mais oriental de Portugal?",
     "response": "A região autónoma mais oriental de Portugal é os Açores, especialmente a ilha das Flores, no grupo ocidental."},
    {"instruction": "Quem foi D. Sebastião e o que foi o Sebastianismo?",
     "response": "D. Sebastião foi o rei de Portugal que desapareceu na Batalha de Alcácer Quibir em 1578. O Sebastianismo é a crença mítica no seu regresso para salvar Portugal numa hora de necessidade."},
    {"instruction": "O que foi o Marquês de Pombal conhecido por fazer após o terramoto de 1755?",
     "response": "O Marquês de Pombal coordenou a reconstrução de Lisboa após o terramoto de 1755, reformou o ensino com a expulsão dos jesuítas e modernizou a economia portuguesa."},
    {"instruction": "O que é a teoria das cordas na física?",
     "response": "A teoria das cordas é uma teoria física que propõe que as partículas subatômicas são manifestações de minúsculas cordas de energia vibratória, tentando unificar a mecânica quântica com a relatividade geral."},
    {"instruction": "O que é a computação quântica?",
     "response": "A computação quântica é um modelo de computação que usa fenómenos da mecânica quântica como superposição e entrelaçamento para processar informação de forma radicalmente mais eficiente em certos problemas."},
    {"instruction": "Qual é a diferença entre 'mau' e 'mal' em português?",
     "response": "'Mau' é um adjetivo que qualifica substantivos ('um mau resultado'), enquanto 'mal' é um advérbio que modifica verbos ou adjetivos ('correu mal', 'mal feito')."},
    {"instruction": "O que é o infinitivo pessoal em português?",
     "response": "O infinitivo pessoal é uma forma verbal exclusiva do português que conjuga o infinitivo com sujeito próprio: 'para eu falar', 'para tu falares', 'para eles falarem'."},
    {"instruction": "O que é o Museu Nacional do Azulejo em Lisboa?",
     "response": "O Museu Nacional do Azulejo, instalado no Convento da Madre de Deus em Lisboa, é dedicado à arte do azulejo, com peças desde o século XV até à atualidade."},
    {"instruction": "Quem foi Sophia de Mello Breyner Andresen?",
     "response": "Sophia de Mello Breyner Andresen foi uma das maiores poetisas portuguesas do século XX, vencedora do Prémio Camões em 1999, conhecida pela clareza da linguagem e pelos temas da luz e do mar."},
    {"instruction": "O que é o arroz de sarrabulho?",
     "response": "O arroz de sarrabulho é um prato típico do Minho, feito com arroz cozido em sangue de porco e carnes variadas, muito presente nos arraiais populares."},
    {"instruction": "O que é a cataplana?",
     "response": "A cataplana é tanto um utensílio de cozinha típico do Algarve (em cobre com fecho hermético) como o prato confeccionado nele, geralmente com peixe, mariscos, legumes e especiarias."},
    {"instruction": "O que é o fine-tuning de um modelo de linguagem?",
     "response": "O fine-tuning é o processo de continuar o treino de um modelo pré-treinado com dados específicos de um domínio, adaptando-o a uma tarefa particular sem treinar do zero."},
    {"instruction": "O que é o RAG (Retrieval-Augmented Generation)?",
     "response": "RAG é uma técnica de IA que combina geração de linguagem com recuperação de documentos: o modelo pesquisa informação relevante numa base de conhecimento antes de gerar a resposta, reduzindo alucinações."},
    {"instruction": "Quais são os poderes do Primeiro-Ministro em Portugal?",
     "response": "O Primeiro-Ministro em Portugal dirige o Governo, coordena e orienta a ação dos ministros, representa o poder executivo e responde politicamente perante a Assembleia da República."},
    {"instruction": "O que é o Tribunal Constitucional em Portugal?",
     "response": "O Tribunal Constitucional é o órgão jurisdicional português com competência específica para apreciar a inconstitucionalidade de normas e para fiscalizar a constitucionalidade de leis."},
    {"instruction": "O que é o AVC (Acidente Vascular Cerebral)?",
     "response": "O AVC é uma interrupção súbita do fluxo sanguíneo ao cérebro, por obstrução (isquémico) ou rotura de vaso (hemorrágico), podendo causar paralisia, afasia e outros défices neurológicos."},
    {"instruction": "O que é a vacinação e como funciona a imunidade de grupo?",
     "response": "A vacinação estimula o sistema imunitário a produzir anticorpos sem causar doença. A imunidade de grupo ocorre quando uma percentagem elevada da população está imune, protegendo também os não vacinados."},
    {"instruction": "O que é o Campeonato do Mundo de Futebol e quando o Portugal ganhou?",
     "response": "O Campeonato do Mundo de Futebol é a competição máxima de seleções nacionais, organizado pela FIFA de quatro em quatro anos. Portugal nunca ganhou o Mundial, mas foi terceiro em 1966 com Eusébio."},
    {"instruction": "Quem é Rosa Mota?",
     "response": "Rosa Mota é uma maratonista portuguesa, tricampeã europeia e campeã olímpica nos Jogos de Seul em 1988, considerada uma das melhores maratonistas da história."},
    {"instruction": "O que é a Convenção de Ramsar?",
     "response": "A Convenção de Ramsar é um tratado internacional de 1971 para a conservação e uso sustentável das zonas húmidas, especialmente como habitat de aves aquáticas."},
    {"instruction": "O que é o teorema de Bayes?",
     "response": "O teorema de Bayes é uma fórmula da teoria da probabilidade que descreve como atualizar a probabilidade de uma hipótese com base em nova evidência, fundamental em estatística e inteligência artificial."},
    {"instruction": "O que é 'O Livro do Desassossego' de Bernardo Soares?",
     "response": "'O Livro do Desassossego' é uma obra de Fernando Pessoa escrita sob o semi-heterónimo Bernardo Soares, um diário fragmentário de meditações sobre a vida, a identidade e a escrita."},
    {"instruction": "O que é o utilitarismo?",
     "response": "O utilitarismo é uma teoria ética que defende que a ação moralmente correta é aquela que maximiza a felicidade ou bem-estar total — 'o maior bem para o maior número', associada a Bentham e Mill."},
    {"instruction": "O que significa a expressão portuguesa 'à balda'?",
     "response": "'À balda' é uma expressão informal portuguesa que significa sem esforço, de forma descuidada ou sem trabalhar, equivalente a 'ao molho e fé'."},
]

# ── Marcadores PT-BR (idêntico a eval_compare.py) ──────────────────────────────
PT_BR_MARKERS = [
    "você", "vocês", "gerenciar", "gerenciamento", "planilha", "bilhões",
    "ônibus", "realizado", "utilizado", "também é", "também são",
    "através de", "ao redor", "em termos de", "no entanto,",
    "aqui estão", "alguns pontos", "pontos importantes", "pontos-chave",
]


def contar_contaminacao_ptbr(respostas):
    texto = " ".join(r.lower() for r in respostas)
    contagens = {m: texto.count(m) for m in PT_BR_MARKERS}
    return sum(contagens.values()), {k: v for k, v in contagens.items() if v > 0}


# ── Métricas (GPU para perplexidade/geração; CPU para ROUGE) ───────────────────

def calcular_perplexidade(model, tokenizer, textos):
    model.eval()
    losses = []
    for texto in textos:
        inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
            losses.append(out.loss.item())
    media = sum(losses) / len(losses)
    return media, math.exp(media)


def gerar_resposta(model, tokenizer, instrucao, max_new_tokens):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instrucao}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


def rouge_l(hipotese, referencia):
    h, r = hipotese.lower().split(), referencia.lower().split()
    if not h or not r:
        return 0.0
    m, n = len(h), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if h[i-1] == r[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p, rc = lcs / m, lcs / n
    return 0.0 if p + rc == 0 else 2 * p * rc / (p + rc)


def bertscore_aproximado(hipoteses, referencias):
    try:
        from bert_score import score as bs_score
        _, _, F1 = bs_score(hipoteses, referencias, lang="pt", verbose=False)
        return F1.tolist()
    except ImportError:
        print("[AVISO] bert_score indisponível — a usar ROUGE-L como substituto.")
        return [rouge_l(h, r) for h, r in zip(hipoteses, referencias)]


def gerar_par(modelo_base, modelo_lora, tokenizer, exemplos, max_new_tokens, label):
    perg, refs, rb, rl = [], [], [], []
    for i, ex in enumerate(exemplos):
        print(f"  [{label}] {i+1}/{len(exemplos)} (max={max_new_tokens})", flush=True)
        perg.append(ex["instruction"])
        refs.append(ex["response"])
        rb.append(gerar_resposta(modelo_base, tokenizer, ex["instruction"], max_new_tokens))
        rl.append(gerar_resposta(modelo_lora, tokenizer, ex["instruction"], max_new_tokens))
    return perg, refs, rb, rl


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import random
    random.seed(RANDOM_SEED)
    linhas = []

    def log(t=""):
        print(t, flush=True)
        linhas.append(t)

    log("=" * 72)
    log("AVALIAÇÃO GPU (Kaggle) — Qwen2.5-1.5B Base vs LoRA (sem judge)")
    log(f"Data: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    log("=" * 72)

    # ── Dataset ────────────────────────────────────────────────────────────────
    # NOTA: estes 50 exemplos saem do MESMO train_test_split(test_size=50, seed=42)
    # que o finetune_lora_kaggle.py usa para o isolated_test — logo NÃO foram vistos
    # em treino (são held-out, não contaminados). Servem só de amostra interna do
    # dataset para perplexidade de referência; o sinal fiável vem do TESTE_ISOLADO
    # (perguntas escritas à mão, nunca no dataset). Mantém-se baixa prioridade.
    log("\nA carregar dataset...")
    raw = load_dataset(DATASET_NAME, token=hf_token)["train"]
    full = raw.train_test_split(test_size=50, seed=RANDOM_SEED)
    amostra_ds = full["test"]  # 50 held-out do dataset (mesmo split do fine-tune)
    log(f"Amostra do dataset (held-out): {len(amostra_ds)} | Teste isolado: {len(TESTE_ISOLADO)}")

    def fmt(ex):
        return (f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
                f"<|im_start|>assistant\n{ex['response']}<|im_end|>")

    # ── Modelos em GPU ───────────────────────────────────────────────────────────
    log("\nA carregar tokenizer e modelos em GPU...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log("  → Base")
    modelo_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, token=hf_token, torch_dtype=torch.float16, device_map=DEVICE)
    log("  → LoRA (adapter do Hub)")
    modelo_lora = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, token=hf_token, torch_dtype=torch.float16, device_map=DEVICE)
    modelo_lora = PeftModel.from_pretrained(modelo_lora, LORA_REPO_ID, token=hf_token)

    # ── CAMADA 1: Perplexidade ─────────────────────────────────────────────────
    log("\n" + "-" * 72)
    log("CAMADA 1 — Perplexidade")
    log("-" * 72)

    textos_val = [fmt(amostra_ds[i]) for i in range(min(N_VALIDACAO, len(amostra_ds)))]
    lb_v, pb_v = calcular_perplexidade(modelo_base, tokenizer, textos_val)
    ll_v, pl_v = calcular_perplexidade(modelo_lora, tokenizer, textos_val)
    log(f"\n  1a. Amostra do dataset ({len(textos_val)} held-out) — referência [baixa prioridade]")
    log(f"      Base {pb_v:8.2f} | LoRA {pl_v:8.2f} | Δ {pb_v - pl_v:+.2f}")

    textos_iso = [fmt(ex) for ex in TESTE_ISOLADO]
    lb_i, pb_i = calcular_perplexidade(modelo_base, tokenizer, textos_iso)
    ll_i, pl_i = calcular_perplexidade(modelo_lora, tokenizer, textos_iso)
    log(f"\n  1b. Isolado ({len(textos_iso)}) — nunca visto [✓ fiável]")
    log(f"      Base {pb_i:8.2f} | LoRA {pl_i:8.2f} | Δ {pb_i - pl_i:+.2f}")

    # ── CAMADA 2: Geração (3 passagens) ─────────────────────────────────────────
    log("\n" + "-" * 72)
    log("CAMADA 2 — Geração")
    log("-" * 72)
    val_list = [dict(amostra_ds[i]) for i in range(min(N_VALIDACAO, len(amostra_ds)))]

    log("\n  2a. Verbose — amostra do dataset")
    pv, rv, rbv, rlv = gerar_par(modelo_base, modelo_lora, tokenizer, val_list, MAX_TOKENS_VERBOSE, "val-verbose")
    log("\n  2b. Verbose — isolado")
    pi, ri, rbi, rli = gerar_par(modelo_base, modelo_lora, tokenizer, TESTE_ISOLADO, MAX_TOKENS_VERBOSE, "iso-verbose")
    log("\n  2c. Conciso — isolado")
    _, _, rbic, rlic = gerar_par(modelo_base, modelo_lora, tokenizer, TESTE_ISOLADO, MAX_TOKENS_CONCISO, "iso-conciso")

    # ── CAMADA 3: Contaminação PT-BR ────────────────────────────────────────────
    log("\n" + "-" * 72)
    log("CAMADA 3 — Contaminação PT-BR (nº ocorrências)")
    log("-" * 72)
    for label, rb, rl in [("Validação verbose", rbv, rlv),
                          ("Isolado verbose", rbi, rli),
                          ("Isolado conciso", rbic, rlic)]:
        tb, _ = contar_contaminacao_ptbr(rb)
        tl, _ = contar_contaminacao_ptbr(rl)
        log(f"  {label:<20} Base {tb:3d} | LoRA {tl:3d}")

    # ── CAMADA 4: ROUGE-L + BERTScore ───────────────────────────────────────────
    log("\n" + "-" * 72)
    log("CAMADA 4 — ROUGE-L + BERTScore")
    log("-" * 72)
    for label, refs, rb, rl in [("Validação verbose", rv, rbv, rlv),
                                ("Isolado verbose", ri, rbi, rli),
                                ("Isolado conciso", ri, rbic, rlic)]:
        rlb = sum(rouge_l(h, r) for h, r in zip(rb, refs)) / len(rb)
        rll = sum(rouge_l(h, r) for h, r in zip(rl, refs)) / len(rl)
        bsb = sum(bertscore_aproximado(rb, refs)) / len(rb)
        bsl = sum(bertscore_aproximado(rl, refs)) / len(rl)
        log(f"\n  {label}:")
        log(f"    ROUGE-L     Base {rlb:.4f} | LoRA {rll:.4f} | Δ {rll - rlb:+.4f}")
        log(f"    BERTScore   Base {bsb:.4f} | LoRA {bsl:.4f} | Δ {bsl - bsb:+.4f}")

    # ── Gravar gerações para judge local ────────────────────────────────────────
    geracoes = {
        "meta": {
            "data": datetime.now(UTC).isoformat(),
            "base_model": BASE_MODEL_ID,
            "lora_repo": LORA_REPO_ID,
            "dataset": DATASET_NAME,
            "perplexidade": {
                "amostra_dataset": {"base": pb_v, "lora": pl_v},
                "isolado":         {"base": pb_i, "lora": pl_i},
            },
        },
        "isolado_verbose": [
            {"pergunta": q, "referencia": ref, "base": b, "lora": l}
            for q, ref, b, l in zip(pi, ri, rbi, rli)
        ],
        "isolado_conciso": [
            {"pergunta": q, "referencia": ref, "base": b, "lora": l}
            for q, ref, b, l in zip(pi, ri, rbic, rlic)
        ],
    }
    GENERATIONS_PATH.write_text(json.dumps(geracoes, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text("\n".join(linhas), encoding="utf-8")
    log(f"\nRelatório: {REPORT_PATH}")
    log(f"Gerações p/ judge local: {GENERATIONS_PATH}")
    log("\nPróximo passo: download eval_generations.json → corre judge_local.py em casa (LM Studio).")


if __name__ == "__main__":
    main()
