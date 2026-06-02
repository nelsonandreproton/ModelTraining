# LLM-as-Judge local — consome eval_generations.json produzido por eval_kaggle.py
#
# Porquê separado: a geração (pesada, GPU) corre no Kaggle; o judge são só chamadas
# à API do LM Studio (localhost:1234), sem inferência local pesada. Assim avalia-se
# o modelo treinado no Kaggle sem reinferir nada em casa.
#
# Pré-requisitos:
#   1. LM Studio aberto, Qwen2.5-7B-Instruct carregado, servidor em localhost:1234
#   2. eval_generations.json no diretório (download do output do Kaggle)
#
# Uso: python judge_local.py

import json
import random
import re
import sys
import textwrap
from pathlib import Path

from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 → UTF-8 p/ box chars

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
JUDGE_MODEL        = "qwen2.5-7b-instruct"
GENERATIONS_PATH   = Path("eval_generations.json")
REPORT_PATH        = Path("judge_report.txt")
RANDOM_SEED        = 42

JUDGE_SYSTEM = textwrap.dedent("""\
    És um avaliador rigoroso de respostas em Português Europeu (PT-PT).
    Vais receber uma pergunta e duas respostas (Resposta A e Resposta B).

    REGRAS OBRIGATÓRIAS:
    1. Respostas mais longas NÃO são necessariamente melhores. Avalia qualidade,
       não quantidade. Uma resposta concisa e correcta é superior a uma longa com erros.
    2. Se uma resposta contém um erro factual, a 'correcção' deve ser ≤2.
    3. Se uma resposta usa Português do Brasil (PT-BR) em vez de PT-PT
       — marcadores: 'você', 'gerenciar', 'planilha', 'bilhões', 'também é', 'ao redor',
       'aqui estão', 'alguns pontos importantes' — a 'fluência' deve ser ≤2.
    4. O campo 'melhor' reflecte qualidade factual e linguística, não comprimento.

    Avalia cada resposta (escala 1-5):
      - correcção: factualmente correcto (erros → ≤2)
      - fluência: gramática e naturalidade PT-PT (PT-BR → ≤2)
      - completude: cobre o essencial (sem recompensar padding)

    Responde EXCLUSIVAMENTE em JSON:
    {
      "A": {"correcção": <1-5>, "fluência": <1-5>, "completude": <1-5>},
      "B": {"correcção": <1-5>, "fluência": <1-5>, "completude": <1-5>},
      "melhor": "<A|B|empate>",
      "justificação": "<1 frase focada em factos e PT-PT>"
    }
""")


def chamar_judge(client, pergunta, resp_a, resp_b):
    prompt = f"Pergunta: {pergunta}\n\nResposta A:\n{resp_a}\n\nResposta B:\n{resp_b}"
    for tentativa in range(2):
        try:
            comp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=350,
            )
            m = re.search(r"\{.*\}", comp.choices[0].message.content.strip(), re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception as e:
            print(f"   [judge] {'retry' if tentativa == 0 else 'falhou'}: {e}")
    return None


def correr_judge(client, pares, seed_offset, log):
    rng = random.Random(RANDOM_SEED + seed_offset)
    dims = ["correcção", "fluência", "completude"]
    pb = {d: [] for d in dims}
    pl = {d: [] for d in dims}
    cont = {"base": 0, "lora": 0, "empate": 0, "erro": 0}

    for i, par in enumerate(pares):
        print(f"  Judge [{i+1}/{len(pares)}]: {par['pergunta'][:55]}...", flush=True)
        invertido = rng.random() < 0.5
        # cego: A/B baralhado para o judge não saber qual é o modelo treinado
        resp_a, resp_b = (par["lora"], par["base"]) if invertido else (par["base"], par["lora"])
        la, lb = ("lora", "base") if invertido else ("base", "lora")

        res = chamar_judge(client, par["pergunta"], resp_a, resp_b)
        if res is None:
            cont["erro"] += 1
            continue
        for d in dims:
            va = res.get("A", {}).get(d, 3)
            vb = res.get("B", {}).get(d, 3)
            if invertido:
                pl[d].append(va); pb[d].append(vb)
            else:
                pb[d].append(va); pl[d].append(vb)
        melhor = res.get("melhor", "empate")
        real = la if melhor == "A" else lb if melhor == "B" else "empate"
        cont[real] += 1

    media = lambda p: {k: (sum(v) / len(v) if v else 0.0) for k, v in p.items()}
    return media(pb), media(pl), cont


def main():
    if not GENERATIONS_PATH.exists():
        sys.exit(f"ERRO: {GENERATIONS_PATH} não encontrado. Faz download do output do Kaggle primeiro.")

    dados = json.loads(GENERATIONS_PATH.read_text(encoding="utf-8"))
    linhas = []

    def log(t=""):
        print(t, flush=True)
        linhas.append(t)

    log("=" * 72)
    log("LLM-as-JUDGE LOCAL — sobre gerações do Kaggle")
    log(f"Modelo treinado: {dados['meta']['lora_repo']}")
    log(f"Gerado em: {dados['meta']['data']}")
    log("=" * 72)

    ppl = dados["meta"]["perplexidade"]
    log(f"\nPerplexidade (do Kaggle):")
    log(f"  Amostra DS Base {ppl['amostra_dataset']['base']:.2f} | LoRA {ppl['amostra_dataset']['lora']:.2f}  [held-out, baixa prioridade]")
    log(f"  Isolado    Base {ppl['isolado']['base']:.2f} | LoRA {ppl['isolado']['lora']:.2f}  [✓ fiável]")

    client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio")

    for chave, seed_off in [("isolado_verbose", 0), ("isolado_conciso", 100)]:
        log(f"\n{'-' * 72}")
        log(f"Judge — {chave}")
        log("-" * 72)
        mb, ml, cont = correr_judge(client, dados[chave], seed_off, log)
        log(f"\n  {'Dimensão':<14} {'Base':>6} {'LoRA':>6} {'Δ':>7}")
        for d in ["correcção", "fluência", "completude"]:
            log(f"  {d:<14} {mb[d]:>6.2f} {ml[d]:>6.2f} {ml[d]-mb[d]:>+7.2f}")
        total = cont["base"] + cont["lora"] + cont["empate"]
        log(f"\n  Veredito: Base={cont['base']} LoRA={cont['lora']} Empate={cont['empate']} Erros={cont['erro']}")
        if total:
            log(f"  LoRA vence {cont['lora']/total:.0%} | Base {cont['base']/total:.0%} | Empate {cont['empate']/total:.0%}")

    REPORT_PATH.write_text("\n".join(linhas), encoding="utf-8")
    log(f"\nRelatório guardado: {REPORT_PATH}")


if __name__ == "__main__":
    main()
