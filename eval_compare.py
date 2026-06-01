"""
eval_compare.py — Avaliação comparativa: Qwen2.5-1.5B Base vs LoRA Fine-tunado

Camadas de avaliação:
  1. Perplexidade — dois conjuntos:
       (a) dataset["test"] — conjunto de VALIDAÇÃO (usado em treino via load_best_model_at_end)
       (b) conjunto de teste ISOLADO — 25 perguntas novas, nunca vistas durante treino
  2. ROUGE-L + BERTScore — subconjunto do conjunto isolado (N_ISOLADO exemplos)
  3. Contaminação PT-BR — contagem de marcadores PT-BR por modelo
  4. LLM-as-Judge — julgamento cego A/B via LM Studio:
       (a) max_new_tokens=200 (verboso) — para comparação com v1
       (b) max_new_tokens=80  (conciso) — elimina viés de comprimento
     Prompt corrigido: penaliza PT-BR, verbosidade e erros factuais explicitamente.
     Veredito `melhor` separado das pontuações por dimensão.

Configuração do LM Studio:
  - Iniciar LM Studio e carregar o modelo Qwen2.5-7B-Instruct
  - Activar o servidor local (por defeito: http://localhost:1234/v1)
  - Ajustar JUDGE_MODEL abaixo se necessário
"""

import json
import math
import random
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# Windows console defaults to cp1252, which cannot encode the box-drawing chars
# (─ │ • etc.) used in the report. Force UTF-8 on stdout so print() does not
# UnicodeEncodeError. The report file is already written with encoding="utf-8".
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from datasets import load_from_disk
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuração ──────────────────────────────────────────────────────────────

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./my_lora_model"
DATASET_PATH = "./my_dataset_processed"

N_VALIDACAO = 25       # exemplos do conjunto de validação (contaminado) para geração
N_ISOLADO = 25         # exemplos do conjunto isolado para geração + judge
RANDOM_SEED = 42

# max_new_tokens para as duas passagens de geração
MAX_TOKENS_VERBOSE = 200   # passagem verbosa (comparável com v1)
MAX_TOKENS_CONCISO = 80    # passagem concisa (elimina viés de comprimento)

# LM Studio — servidor local OpenAI-compatível
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
JUDGE_MODEL = "qwen2.5-7b-instruct"  # nome exacto como aparece no LM Studio

REPORT_PATH = Path("eval_report.txt")

# ── Conjunto de teste ISOLADO ─────────────────────────────────────────────────
# 25 perguntas PT-PT NUNCA presentes em create_dataset.py
# Cobertura: temas do dataset original mas perguntas diferentes + temas novos

TESTE_ISOLADO = [
    # Geografia
    {"instruction": "Qual é o ponto mais ocidental da Europa Continental?",
     "response": "O Cabo da Roca, em Portugal, é o ponto mais ocidental da Europa Continental."},
    {"instruction": "Qual é a região autónoma mais oriental de Portugal?",
     "response": "A região autónoma mais oriental de Portugal é os Açores, especialmente a ilha das Flores, no grupo ocidental."},
    # História
    {"instruction": "Quem foi D. Sebastião e o que foi o Sebastianismo?",
     "response": "D. Sebastião foi o rei de Portugal que desapareceu na Batalha de Alcácer Quibir em 1578. O Sebastianismo é a crença mítica no seu regresso para salvar Portugal numa hora de necessidade."},
    {"instruction": "O que foi o Marquês de Pombal conhecido por fazer após o terramoto de 1755?",
     "response": "O Marquês de Pombal coordenou a reconstrução de Lisboa após o terramoto de 1755, reformou o ensino com a expulsão dos jesuítas e modernizou a economia portuguesa."},
    # Ciência
    {"instruction": "O que é a teoria das cordas na física?",
     "response": "A teoria das cordas é uma teoria física que propõe que as partículas subatômicas são manifestações de minúsculas cordas de energia vibratória, tentando unificar a mecânica quântica com a relatividade geral."},
    {"instruction": "O que é a computação quântica?",
     "response": "A computação quântica é um modelo de computação que usa fenómenos da mecânica quântica como superposição e entrelaçamento para processar informação de forma radicalmente mais eficiente em certos problemas."},
    # Língua Portuguesa
    {"instruction": "Qual é a diferença entre 'mau' e 'mal' em português?",
     "response": "'Mau' é um adjetivo que qualifica substantivos ('um mau resultado'), enquanto 'mal' é um advérbio que modifica verbos ou adjetivos ('correu mal', 'mal feito')."},
    {"instruction": "O que é o infinitivo pessoal em português?",
     "response": "O infinitivo pessoal é uma forma verbal exclusiva do português que conjuga o infinitivo com sujeito próprio: 'para eu falar', 'para tu falares', 'para eles falarem'."},
    # Cultura
    {"instruction": "O que é o Museu Nacional do Azulejo em Lisboa?",
     "response": "O Museu Nacional do Azulejo, instalado no Convento da Madre de Deus em Lisboa, é dedicado à arte do azulejo, com peças desde o século XV até à atualidade."},
    {"instruction": "Quem foi Sophia de Mello Breyner Andresen?",
     "response": "Sophia de Mello Breyner Andresen foi uma das maiores poetisas portuguesas do século XX, vencedora do Prémio Camões em 1999, conhecida pela clareza da linguagem e pelos temas da luz e do mar."},
    # Gastronomia
    {"instruction": "O que é o arroz de sarrabulho?",
     "response": "O arroz de sarrabulho é um prato típico do Minho, feito com arroz cozido em sangue de porco e carnes variadas, muito presente nos arraiais populares."},
    {"instruction": "O que é a cataplana?",
     "response": "A cataplana é tanto um utensílio de cozinha típico do Algarve (em cobre com fecho hermético) como o prato confeccionado nele, geralmente com peixe, mariscos, legumes e especiarias."},
    # IA e Tecnologia
    {"instruction": "O que é o fine-tuning de um modelo de linguagem?",
     "response": "O fine-tuning é o processo de continuar o treino de um modelo pré-treinado com dados específicos de um domínio, adaptando-o a uma tarefa particular sem treinar do zero."},
    {"instruction": "O que é o RAG (Retrieval-Augmented Generation)?",
     "response": "RAG é uma técnica de IA que combina geração de linguagem com recuperação de documentos: o modelo pesquisa informação relevante numa base de conhecimento antes de gerar a resposta, reduzindo alucinações."},
    # Política
    {"instruction": "Quais são os poderes do Primeiro-Ministro em Portugal?",
     "response": "O Primeiro-Ministro em Portugal dirige o Governo, coordena e orienta a ação dos ministros, representa o poder executivo e responde politicamente perante a Assembleia da República."},
    {"instruction": "O que é o Tribunal Constitucional em Portugal?",
     "response": "O Tribunal Constitucional é o órgão jurisdicional português com competência específica para apreciar a inconstitucionalidade de normas e para fiscalizar a constitucionalidade de leis."},
    # Saúde
    {"instruction": "O que é o AVC (Acidente Vascular Cerebral)?",
     "response": "O AVC é uma interrupção súbita do fluxo sanguíneo ao cérebro, por obstrução (isquémico) ou rotura de vaso (hemorrágico), podendo causar paralisia, afasia e outros défices neurológicos."},
    {"instruction": "O que é a vacinação e como funciona a imunidade de grupo?",
     "response": "A vacinação estimula o sistema imunitário a produzir anticorpos sem causar doença. A imunidade de grupo ocorre quando uma percentagem elevada da população está imune, protegendo também os não vacinados."},
    # Desporto
    {"instruction": "O que é o Campeonato do Mundo de Futebol e quando o Portugal ganhou?",
     "response": "O Campeonato do Mundo de Futebol é a competição máxima de seleções nacionais, organizado pela FIFA de quatro em quatro anos. Portugal nunca ganhou o Mundial, mas foi terceiro em 1966 com Eusébio."},
    {"instruction": "Quem é Rosa Mota?",
     "response": "Rosa Mota é uma maratonista portuguesa, tricampeã europeia e campeã olímpica nos Jogos de Seul em 1988, considerada uma das melhores maratonistas da história."},
    # Ambiente
    {"instruction": "O que é a Convenção de Ramsar?",
     "response": "A Convenção de Ramsar é um tratado internacional de 1971 para a conservação e uso sustentável das zonas húmidas, especialmente como habitat de aves aquáticas."},
    # Matemática
    {"instruction": "O que é o teorema de Bayes?",
     "response": "O teorema de Bayes é uma fórmula da teoria da probabilidade que descreve como atualizar a probabilidade de uma hipótese com base em nova evidência, fundamental em estatística e inteligência artificial."},
    # Literatura
    {"instruction": "O que é 'O Livro do Desassossego' de Bernardo Soares?",
     "response": "'O Livro do Desassossego' é uma obra de Fernando Pessoa escrita sob o semi-heterónimo Bernardo Soares, um diário fragmentário de meditações sobre a vida, a identidade e a escrita."},
    # Filosofia
    {"instruction": "O que é o utilitarismo?",
     "response": "O utilitarismo é uma teoria ética que defende que a ação moralmente correta é aquela que maximiza a felicidade ou bem-estar total — 'o maior bem para o maior número', associada a Bentham e Mill."},
    # Expressões
    {"instruction": "O que significa a expressão portuguesa 'à balda'?",
     "response": "'À balda' é uma expressão informal portuguesa que significa sem esforço, de forma descuidada ou sem trabalhar, equivalente a 'ao molho e fé'."},
]

# ── Marcadores PT-BR ──────────────────────────────────────────────────────────

PT_BR_MARKERS = [
    "você", "vocês", "gerenciar", "gerenciamento", "planilha", "bilhões",
    "ônibus", "realizado", "utilizado", "também é", "também são",
    "através de", "ao redor", "em termos de", "no entanto,",
    "aqui estão", "alguns pontos", "pontos importantes", "pontos-chave",
]

def contar_contaminacao_ptbr(respostas: list[str]) -> dict[str, int | float]:
    texto_total = " ".join(r.lower() for r in respostas)
    contagens = {m: texto_total.count(m) for m in PT_BR_MARKERS}
    total = sum(contagens.values())
    ocorrencias = {k: v for k, v in contagens.items() if v > 0}
    return {"total": total, "por_marcador": ocorrencias}


# ── Utilitários ───────────────────────────────────────────────────────────────

def calcular_perplexidade(model, tokenizer, textos: list[str]) -> tuple[float, float]:
    model.eval()
    losses = []
    for texto in textos:
        inputs = tokenizer(
            texto,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)
            losses.append(outputs.loss.item())
    loss_media = sum(losses) / len(losses)
    return loss_media, math.exp(loss_media)


def gerar_resposta(model, tokenizer, instrucao: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": instrucao}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    novo = output_ids[0][input_len:]
    return tokenizer.decode(novo, skip_special_tokens=True).strip()


def rouge_l(hipotese: str, referencia: str) -> float:
    h = hipotese.lower().split()
    r = referencia.lower().split()
    if not h or not r:
        return 0.0
    m, n = len(h), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if h[i-1] == r[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / m if m else 0.0
    recall = lcs / n if n else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bertscore_aproximado(hipoteses: list[str], referencias: list[str]) -> list[float]:
    try:
        from bert_score import score as bs_score
        _, _, F1 = bs_score(hipoteses, referencias, lang="pt", verbose=False)
        return F1.tolist()
    except ImportError:
        print("[AVISO] bert_score não instalado — a usar ROUGE-L como substituto.")
        return [rouge_l(h, r) for h, r in zip(hipoteses, referencias)]


# ── LLM-as-Judge (prompt corrigido) ──────────────────────────────────────────

JUDGE_SYSTEM = textwrap.dedent("""\
    És um avaliador rigoroso de respostas em Português Europeu (PT-PT).
    Vais receber uma pergunta e duas respostas (Resposta A e Resposta B).

    REGRAS OBRIGATÓRIAS:
    1. Respostas mais longas NÃO são necessariamente melhores. Avalia a qualidade,
       não a quantidade. Uma resposta concisa e correcta é superior a uma longa com erros.
    2. Se uma resposta contém um erro factual, a sua pontuação de 'correcção' deve ser
       ≤2, mesmo que o resto seja bem escrito.
    3. Se uma resposta usa Português do Brasil (PT-BR) em vez de Português Europeu (PT-PT)
       — marcadores: 'você', 'gerenciar', 'planilha', 'bilhões', 'também é', 'ao redor',
       'aqui estão', 'alguns pontos importantes' — a sua pontuação de 'fluência' deve ser ≤2.
    4. O campo 'melhor' deve reflectir a qualidade factual e linguística, não o comprimento.
       Se ambas as respostas têm erros mas uma tem menos, escolhe a menos errada.

    Avalia cada resposta (escala 1-5):
      - correcção: factualmente correcto (erros factuais → ≤2)
      - fluência: gramática e naturalidade em PT-PT (PT-BR → ≤2)
      - completude: cobre os aspectos essenciais da pergunta (sem recompensar padding)

    Responde EXCLUSIVAMENTE em JSON com este formato exacto:
    {
      "A": {"correcção": <1-5>, "fluência": <1-5>, "completude": <1-5>},
      "B": {"correcção": <1-5>, "fluência": <1-5>, "completude": <1-5>},
      "melhor": "<A|B|empate>",
      "justificação": "<1 frase focada em factos e PT-PT, não em comprimento>"
    }
""")


def chamar_judge(
    client: OpenAI,
    pergunta: str,
    resp_a: str,
    resp_b: str,
) -> dict | None:
    prompt_user = (
        f"Pergunta: {pergunta}\n\n"
        f"Resposta A:\n{resp_a}\n\n"
        f"Resposta B:\n{resp_b}"
    )
    for tentativa in range(2):
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt_user},
                ],
                temperature=0.0,
                max_tokens=350,
            )
            raw = completion.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            if tentativa == 0:
                print(f"   [judge] retentativa após erro: {e}")
            else:
                print(f"   [judge] falhou após 2 tentativas: {e}")
    return None


# ── Geração + métricas para um conjunto ──────────────────────────────────────

def correr_geracao(
    modelo_base,
    modelo_lora,
    tokenizer,
    exemplos: list[dict],
    max_new_tokens: int,
    label: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Gera respostas Base e LoRA para todos os exemplos. Devolve (perguntas, refs, resps_base, resps_lora)."""
    perguntas, referencias, resps_base, resps_lora = [], [], [], []
    for i, ex in enumerate(exemplos):
        print(f"  [{label}] Gerando [{i+1}/{len(exemplos)}] (max={max_new_tokens}): {ex['instruction'][:55]}...")
        perguntas.append(ex["instruction"])
        referencias.append(ex["response"])
        resps_base.append(gerar_resposta(modelo_base, tokenizer, ex["instruction"], max_new_tokens))
        resps_lora.append(gerar_resposta(modelo_lora, tokenizer, ex["instruction"], max_new_tokens))
    return perguntas, referencias, resps_base, resps_lora


def correr_judge(
    client: OpenAI,
    perguntas: list[str],
    resps_base: list[str],
    resps_lora: list[str],
    seed_offset: int = 0,
) -> tuple[dict, dict, dict, list[str]]:
    """Corre o judge sobre os pares. Devolve (pontos_base, pontos_lora, contagem_melhor, detalhes)."""
    rng = random.Random(RANDOM_SEED + seed_offset)
    dims = ["correcção", "fluência", "completude"]
    pontos_base: dict[str, list[int]] = {d: [] for d in dims}
    pontos_lora: dict[str, list[int]] = {d: [] for d in dims}
    contagem = {"base": 0, "lora": 0, "empate": 0, "erro": 0}
    detalhes: list[str] = []

    for i, (pergunta, r_base, r_lora) in enumerate(zip(perguntas, resps_base, resps_lora)):
        print(f"  Judge [{i+1}/{len(perguntas)}]: {pergunta[:55]}...")
        invertido = rng.random() < 0.5
        if invertido:
            resp_a, resp_b = r_lora, r_base
            label_a, label_b = "lora", "base"
        else:
            resp_a, resp_b = r_base, r_lora
            label_a, label_b = "base", "lora"

        resultado = chamar_judge(client, pergunta, resp_a, resp_b)
        if resultado is None:
            contagem["erro"] += 1
            detalhes.append(f"\n[{i+1}] {pergunta}\n  judge: ERRO")
            continue

        for d in dims:
            va = resultado.get("A", {}).get(d, 3)
            vb = resultado.get("B", {}).get(d, 3)
            if invertido:
                pontos_lora[d].append(va)
                pontos_base[d].append(vb)
            else:
                pontos_base[d].append(va)
                pontos_lora[d].append(vb)

        melhor_ab = resultado.get("melhor", "empate")
        melhor_real = (label_a if melhor_ab == "A" else label_b if melhor_ab == "B" else "empate")
        contagem[melhor_real] += 1

        # Recuperar pontos já gravados
        pb_corr = pontos_base["correcção"][-1] if pontos_base["correcção"] else "?"
        pb_flu  = pontos_base["fluência"][-1]  if pontos_base["fluência"]  else "?"
        pb_comp = pontos_base["completude"][-1] if pontos_base["completude"] else "?"
        pl_corr = pontos_lora["correcção"][-1]  if pontos_lora["correcção"]  else "?"
        pl_flu  = pontos_lora["fluência"][-1]   if pontos_lora["fluência"]   else "?"
        pl_comp = pontos_lora["completude"][-1]  if pontos_lora["completude"]  else "?"

        just = resultado.get("justificação", "")
        detalhes.append(
            f"\n[{i+1}] {pergunta}\n"
            f"  Base: corr={pb_corr} flu={pb_flu} comp={pb_comp}\n"
            f"  LoRA: corr={pl_corr} flu={pl_flu} comp={pl_comp}\n"
            f"  Melhor: {melhor_real} — {just}"
        )

    return pontos_base, pontos_lora, contagem, detalhes


def media_dim(pontos: dict[str, list[int]]) -> dict[str, float]:
    return {k: sum(v) / len(v) if v else 0.0 for k, v in pontos.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    linhas: list[str] = []

    def log(t: str = ""):
        print(t)
        linhas.append(t)

    log("=" * 72)
    log("AVALIAÇÃO COMPARATIVA v2 — Qwen2.5-1.5B Base vs LoRA Fine-tunado")
    log(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("─" * 72)
    log("Melhorias v2:")
    log("  • Conjunto de teste ISOLADO (25 perguntas nunca vistas em treino)")
    log("  • Detetor de contaminação PT-BR")
    log("  • Passagem concisa (max_new_tokens=80) para eliminar viés de comprimento")
    log("  • Prompt do judge corrigido: penaliza PT-BR, erros factuais, verbosidade")
    log("  • Veredito `melhor` separado das pontuações por dimensão")
    log("=" * 72)

    # ── Carregar dataset e modelos ────────────────────────────────────────────
    log("\nA carregar dataset...")
    ds = load_from_disk(DATASET_PATH)
    validacao = ds["test"]
    log(f"Conjunto de validação (contaminado): {len(validacao)} exemplos")
    log(f"Conjunto de teste isolado: {len(TESTE_ISOLADO)} exemplos")

    indices_val = random.sample(range(len(validacao)), min(N_VALIDACAO, len(validacao)))
    subconjunto_val = [validacao[i] for i in indices_val]

    log("\nA carregar tokenizer e modelos (pode demorar)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log("  → Qwen2.5-1.5B Base")
    modelo_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)

    log("  → Qwen2.5-1.5B + LoRA")
    modelo_lora = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)
    modelo_lora = PeftModel.from_pretrained(modelo_lora, LORA_PATH)

    # ── CAMADA 1: Perplexidade ────────────────────────────────────────────────
    log("\n" + "─" * 72)
    log("CAMADA 1 — Perplexidade")
    log("─" * 72)

    # 1a: conjunto de validação (contaminado)
    # Amostra de N_VALIDACAO exemplos (não os 1001 todos): este número está
    # marcado como baixa fiabilidade (split usado em treino), por isso não vale
    # a pena pagar 1001 passagens de perplexidade em CPU. Reutiliza o mesmo
    # subconjunto_val já usado na geração.
    textos_val = [ex["text"] for ex in subconjunto_val]
    log(f"\n  1a. Conjunto de VALIDAÇÃO (amostra {len(textos_val)} de {len(validacao)}) — contaminado por treino")
    log("  A calcular Base...")
    loss_base_val, ppl_base_val = calcular_perplexidade(modelo_base, tokenizer, textos_val)
    log("  A calcular LoRA...")
    loss_lora_val, ppl_lora_val = calcular_perplexidade(modelo_lora, tokenizer, textos_val)

    log(f"\n  {'Modelo':<30} {'Loss':>8} {'Perplexidade':>14}")
    log(f"  {'-'*54}")
    log(f"  {'Qwen2.5-1.5B Base':<30} {loss_base_val:>8.4f} {ppl_base_val:>14.2f}")
    log(f"  {'Qwen2.5-1.5B + LoRA':<30} {loss_lora_val:>8.4f} {ppl_lora_val:>14.2f}")
    log(f"  Δ (Base − LoRA): {ppl_base_val - ppl_lora_val:+.2f}  [⚠ baixa fiabilidade — split usado em treino]")

    # 1b: conjunto isolado
    def formatar_texto_isolado(ex: dict) -> str:
        return (
            f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['response']}<|im_end|>"
        )

    textos_isolado = [formatar_texto_isolado(ex) for ex in TESTE_ISOLADO]
    log(f"\n  1b. Conjunto ISOLADO ({len(textos_isolado)} exemplos) — nunca visto em treino")
    log("  A calcular Base...")
    loss_base_iso, ppl_base_iso = calcular_perplexidade(modelo_base, tokenizer, textos_isolado)
    log("  A calcular LoRA...")
    loss_lora_iso, ppl_lora_iso = calcular_perplexidade(modelo_lora, tokenizer, textos_isolado)

    log(f"\n  {'Modelo':<30} {'Loss':>8} {'Perplexidade':>14}")
    log(f"  {'-'*54}")
    log(f"  {'Qwen2.5-1.5B Base':<30} {loss_base_iso:>8.4f} {ppl_base_iso:>14.2f}")
    log(f"  {'Qwen2.5-1.5B + LoRA':<30} {loss_lora_iso:>8.4f} {ppl_lora_iso:>14.2f}")
    delta_iso = ppl_base_iso - ppl_lora_iso
    log(f"  Δ (Base − LoRA): {delta_iso:+.2f}  [✓ fiável — conjunto isolado]")

    # ── CAMADA 2: Geração — duas passagens ───────────────────────────────────
    log("\n" + "─" * 72)
    log("CAMADA 2 — Geração de respostas")
    log("─" * 72)

    log(f"\n  2a. Passagem VERBOSA (max_new_tokens={MAX_TOKENS_VERBOSE}) — conjunto validação")
    p_val, r_val, rb_val, rl_val = correr_geracao(
        modelo_base, modelo_lora, tokenizer, subconjunto_val,
        MAX_TOKENS_VERBOSE, "val-verbose"
    )

    log(f"\n  2b. Passagem VERBOSA (max_new_tokens={MAX_TOKENS_VERBOSE}) — conjunto isolado")
    p_iso, r_iso, rb_iso, rl_iso = correr_geracao(
        modelo_base, modelo_lora, tokenizer, TESTE_ISOLADO,
        MAX_TOKENS_VERBOSE, "iso-verbose"
    )

    log(f"\n  2c. Passagem CONCISA (max_new_tokens={MAX_TOKENS_CONCISO}) — conjunto isolado")
    _, _, rb_iso_c, rl_iso_c = correr_geracao(
        modelo_base, modelo_lora, tokenizer, TESTE_ISOLADO,
        MAX_TOKENS_CONCISO, "iso-conciso"
    )

    # ── CAMADA 3: Contaminação PT-BR ──────────────────────────────────────────
    log("\n" + "─" * 72)
    log("CAMADA 3 — Contaminação PT-BR")
    log("─" * 72)

    for label, resps_b, resps_l, conjunto in [
        ("Validação (verbose)", rb_val, rl_val, "validação"),
        ("Isolado (verbose)", rb_iso, rl_iso, "isolado"),
        ("Isolado (conciso)", rb_iso_c, rl_iso_c, "isolado"),
    ]:
        cb = contar_contaminacao_ptbr(resps_b)
        cl = contar_contaminacao_ptbr(resps_l)
        log(f"\n  {label}:")
        log(f"    Base — {cb['total']} ocorrências PT-BR | LoRA — {cl['total']} ocorrências PT-BR")
        if cb["por_marcador"]:
            top_b = sorted(cb["por_marcador"].items(), key=lambda x: -x[1])[:5]
            log(f"    Base top-5: {', '.join(f'{k}={v}' for k, v in top_b)}")
        if cl["por_marcador"]:
            top_l = sorted(cl["por_marcador"].items(), key=lambda x: -x[1])[:5]
            log(f"    LoRA top-5: {', '.join(f'{k}={v}' for k, v in top_l)}")

    # ── CAMADA 4: ROUGE-L + BERTScore ────────────────────────────────────────
    log("\n" + "─" * 72)
    log("CAMADA 4 — ROUGE-L + BERTScore")
    log("─" * 72)

    resultados_metricas = []
    for label, pergs, refs, resps_b, resps_l in [
        ("Validação (verbose)", p_val, r_val, rb_val, rl_val),
        ("Isolado (verbose)", p_iso, r_iso, rb_iso, rl_iso),
        ("Isolado (conciso)", p_iso, r_iso, rb_iso_c, rl_iso_c),
    ]:
        rl_b = [rouge_l(h, r) for h, r in zip(resps_b, refs)]
        rl_l = [rouge_l(h, r) for h, r in zip(resps_l, refs)]
        m_rl_b = sum(rl_b) / len(rl_b)
        m_rl_l = sum(rl_l) / len(rl_l)

        log(f"\n  A calcular BERTScore [{label}]...")
        bs_b = bertscore_aproximado(resps_b, refs)
        bs_l = bertscore_aproximado(resps_l, refs)
        m_bs_b = sum(bs_b) / len(bs_b)
        m_bs_l = sum(bs_l) / len(bs_l)

        resultados_metricas.append((label, m_rl_b, m_rl_l, m_bs_b, m_bs_l))
        log(f"\n  {label}:")
        log(f"  {'Métrica':<20} {'Base':>10} {'LoRA':>10} {'Δ':>12}")
        log(f"  {'-'*54}")
        log(f"  {'ROUGE-L':<20} {m_rl_b:>10.4f} {m_rl_l:>10.4f} {m_rl_l - m_rl_b:>+12.4f}")
        log(f"  {'BERTScore F1':<20} {m_bs_b:>10.4f} {m_bs_l:>10.4f} {m_bs_l - m_bs_b:>+12.4f}")

    # ── CAMADA 5: LLM-as-Judge ────────────────────────────────────────────────
    log("\n" + "─" * 72)
    log(f"CAMADA 5 — LLM-as-Judge ({JUDGE_MODEL}) — prompt v2 corrigido")
    log("─" * 72)

    client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio")

    resultados_judge = {}

    for label, pergs, resps_b, resps_l, seed_off in [
        ("Isolado (verbose)", p_iso, rb_iso, rl_iso, 0),
        ("Isolado (conciso)", p_iso, rb_iso_c, rl_iso_c, 100),
    ]:
        log(f"\n  Judge — {label}")
        pb, pl, cont, dets = correr_judge(client, pergs, resps_b, resps_l, seed_off)
        med_b = media_dim(pb)
        med_l = media_dim(pl)
        resultados_judge[label] = (med_b, med_l, cont, dets)

        total_v = cont["base"] + cont["lora"] + cont["empate"]
        log(f"\n  {'Dimensão':<18} {'Base':>8} {'LoRA':>8} {'Δ':>8}  [score 1-5]")
        log(f"  {'-'*46}")
        for d in ["correcção", "fluência", "completude"]:
            log(f"  {d:<18} {med_b[d]:>8.2f} {med_l[d]:>8.2f} {med_l[d]-med_b[d]:>+8.2f}")

        log(f"\n  Veredito `melhor` (independente dos scores):")
        log(f"    Base={cont['base']}  LoRA={cont['lora']}  Empate={cont['empate']}  Erros={cont['erro']}")
        if total_v:
            log(f"    Base {cont['base']/total_v:.0%} | LoRA {cont['lora']/total_v:.0%} | Empate {cont['empate']/total_v:.0%}")
        log(f"  [⚠ Nota: `melhor` pode diferir dos scores — reflecte qualidade global percepcionada]")

    # ── Resumo Final ──────────────────────────────────────────────────────────
    log("\n" + "=" * 72)
    log("RESUMO FINAL")
    log("=" * 72)

    log(f"\n  Perplexidade:")
    log(f"  {'Conjunto':<32} {'Base':>8} {'LoRA':>8} {'Δ':>10}  Fiabilidade")
    log(f"  {'-'*70}")
    log(f"  {'Validação (contaminado)':<32} {ppl_base_val:>8.2f} {ppl_lora_val:>8.2f} {ppl_lora_val-ppl_base_val:>+10.2f}  ⚠ baixa")
    log(f"  {'Isolado (nunca visto)':<32} {ppl_base_iso:>8.2f} {ppl_lora_iso:>8.2f} {ppl_lora_iso-ppl_base_iso:>+10.2f}  ✓ alta")

    log(f"\n  ROUGE-L / BERTScore:")
    log(f"  {'Conjunto':<30} {'RL-Base':>9} {'RL-LoRA':>9} {'ΔRL':>8}  {'BS-Base':>9} {'BS-LoRA':>9} {'ΔBS':>8}")
    log(f"  {'-'*86}")
    for label, mrlb, mrll, mbsb, mbsl in resultados_metricas:
        log(f"  {label:<30} {mrlb:>9.4f} {mrll:>9.4f} {mrll-mrlb:>+8.4f}  {mbsb:>9.4f} {mbsl:>9.4f} {mbsl-mbsb:>+8.4f}")

    log(f"\n  Judge (scores médios, 1-5):")
    log(f"  {'Conjunto':<30} {'corr-B':>8} {'corr-L':>8} {'flu-B':>7} {'flu-L':>7} {'comp-B':>7} {'comp-L':>7}  Veredito")
    log(f"  {'-'*90}")
    for label, (med_b, med_l, cont, _) in resultados_judge.items():
        tv = cont["base"] + cont["lora"] + cont["empate"]
        verd = f"LoRA {cont['lora']/tv:.0%}" if tv else "—"
        log(
            f"  {label:<30} {med_b['correcção']:>8.2f} {med_l['correcção']:>8.2f}"
            f" {med_b['fluência']:>7.2f} {med_l['fluência']:>7.2f}"
            f" {med_b['completude']:>7.2f} {med_l['completude']:>7.2f}  {verd}"
        )

    # ── Respostas lado a lado (conjunto isolado, passagem verbosa) ────────────
    log("\n" + "=" * 72)
    log("RESPOSTAS LADO A LADO — Conjunto Isolado (verbose)")
    log("=" * 72)
    for i, (pergunta, ref, rb, rl) in enumerate(zip(p_iso, r_iso, rb_iso, rl_iso)):
        log(f"\n[{i+1}] {pergunta}")
        log(f"  Ref:  {ref}")
        log(f"  Base: {rb}")
        log(f"  LoRA: {rl}")

    log("\n" + "=" * 72)
    log("RESPOSTAS LADO A LADO — Conjunto Isolado (conciso)")
    log("=" * 72)
    for i, (pergunta, ref, rb, rl) in enumerate(zip(p_iso, r_iso, rb_iso_c, rl_iso_c)):
        log(f"\n[{i+1}] {pergunta}")
        log(f"  Ref:  {ref}")
        log(f"  Base: {rb}")
        log(f"  LoRA: {rl}")

    # ── Detalhes do judge ─────────────────────────────────────────────────────
    for label, (_, _, _, dets) in resultados_judge.items():
        log(f"\n{'=' * 72}")
        log(f"DETALHES DO JUDGE — {label}")
        log("=" * 72)
        for d in dets:
            log(d)

    REPORT_PATH.write_text("\n".join(linhas), encoding="utf-8")
    print(f"\nRelatório guardado em: {REPORT_PATH}")


if __name__ == "__main__":
    main()
