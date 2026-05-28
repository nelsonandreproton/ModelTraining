# check_dataset_quality.py
# Measures quality of a generated Q&A pairs JSON file.
# Runs 5 checks: filter pass-rate, template distribution, instruction diversity,
# length distribution, and PT-BR contamination.
#
# Usage:
#   python check_dataset_quality.py generated_pairs.json
#   python check_dataset_quality.py generated_pairs.json --sample 30

import argparse
import io
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

# Force UTF-8 stdout on Windows (default is cp1252 which can't encode box-drawing chars)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Reuse validators from generate_dataset.py ────────────────────────────────

_PLACEHOLDER_RE = re.compile(
    r"\(pergunta real|"
    r"\(item real|"
    r"\(afirma.{1,4}o (real|controversa real)|"
    r"\(facto real|"
    r"\(conceito real|"
    r"\(resposta real|"
    r"\(defini.{1,4}o real|"
    r"\(an.lise (com|nuan)|"
    r"\[item\]|"
    r"\[explica.{1,4}o\]|"
    r"\[A\]\s*(e|em|vs)\s*\[B\]|"
    r"\[Pergunta sobre Portugal|"
    r"\[Resposta factual|"
    r"\[Afirma.{1,4}o errada|"
    r"\[Compara.{1,4}o|"
    r"\[Explica.{1,4}o com detalhes|"
    r"\[Explica.{1,4}o causal|"
    r"\[Defini.{1,4}o\]|"
    r"\[exemplo real portugu|"
    r"\[An.lise nuan|"
    r"\[An.lise do impacto|"
    r"\[correc.{1,4}o com factos|"
    r"\[processo portugu|"
    r"\[itens portugu|"
    r"\[facto portugu|"
    r"\[afirma.{1,4}o sobre Portugal|"
    r"afirmar que \[|"
    r"razão \[patrim|"
    r"razão \[|"
    # Generic bracket placeholder: [Title Case phrase] at start of response or standalone
    r"^\s*\[[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][^\]]{5,}\]\s*[.\n]",
    re.IGNORECASE | re.MULTILINE,
)

_OFFTRACK_RE = re.compile(r"^[^\n]{5,200}\?$")

_PORTUGAL_RE = re.compile(
    r"portugu[eê]|portugal|lisboa|porto\b|algarve|alentejo|bragan[çc]a|"
    r"coimbra|fado\b|descobrimentos|sal[ao]zar|rep[úu]blica|av[ií]s|"
    r"lusit[aâ]n|lu[sí]ofon|atl[aâ]ntico|ib[eé]r|"
    r"afonso henriques|d\. afonso|d\. jo[aã]o|d\. manuel|d\. dinis|d\. sebasti|"
    r"d\. carlos|d\. maria|d\. pedro|d\. fil[ií]p|"
    r"diogo c[aã]o|vasco da gama|bartolomeu dias|cabral|camões|pessoa\b|saramago|"
    r"salazar|pombal|eça de queir|egas moniz|humberto delgado|mário soares|"
    r"aljubarrota|alcácer quibir|tordesilhas|sagres|"
    r"guimarães|braga\b|évora|sintra|óbidos|tomar\b|batalha\b|alcobaça|jer[oó]nimos|"
    r"belém\b|madeira\b|a[çc]ores\b|douro\b|tejo\b|minho\b|guadiana|"
    r"peneda.ger[eê]s|ria formosa|serra da estrela|"
    r"snc\b|sns\b|rtp\b|pide\b|mfa\b|prec\b|cea\b|cplp\b|nato\b|cee\b|"
    r"reconquista|inquisição|estado novo|primeira república|"
    r"25 de abril|5 de outubro|10 de junho|1 de dezembro",
    re.IGNORECASE,
)

_PTBR_RE = re.compile(
    r"\bvocê\b|\bvocês\b|\bônibus\b|\bcelular\b|\btime\b|\blegal\b|"
    r"\bcadastrar\b|\bdeletar\b|\bplanilha\b|\bbilhões\b|\btrem\b|"
    r"\bbanheiro\b|\bsobrenome\b|\bpedestre\b",
    re.IGNORECASE,
)

# Template type keywords — inferred from instruction phrasing
_TEMPLATE_PATTERNS = [
    ("correction",  re.compile(r"(está errad|é falso|incorret|mito|engano|verdade que|afirma.{1,4}o corret|"
                               r"é corre[ct]o (afirmar|dizer)|é certo (afirmar|dizer)|"
                               r"simplista|equívoco|não é totalmente|corret[ao]\?)", re.I)),
    ("comparison",  re.compile(r"(diferen[çc]a|compar|versus|vs\.?|semelhan[çc]a|contrast)", re.I)),
    ("howto",       re.compile(r"^como (se |)(faz|funciona|era|é feito|surgiu|se tornou|se desenvolveu|"
                               r"realiza|ocorre|funciona)", re.I)),
    ("why",         re.compile(r"^(por que|porquê|qual (a )?raz[aã]o|o que levou|o que causou)", re.I)),
    ("enumeration", re.compile(r"(principais|lista|enumera|quais (são|foram)|cite|mencione|exemplos de)", re.I)),
    ("definition",  re.compile(r"(o que significa|defin|conceito de|em que consiste|o que é o\b)", re.I)),
    ("debate",      re.compile(r"(argumento|defenda|critics|favor ou contra|concordas|opinion|perspetiva)", re.I)),
    ("contextual",  re.compile(r"(no contexto|na época|durante o século|nesse período|à luz d)", re.I)),
    ("factual",     re.compile(r"^(o que (é|foi|são)|quem (foi|é)|quando (foi|ocorreu)|onde (fica|ficava)|"
                               r"qual (é|foi|era)|qual o papel|qual a import|qual o impacto|"
                               r"qual foi o|quais (são|foram)|o que caracteriza)", re.I)),
]


def classify_template(instruction: str) -> str:
    for name, pattern in _TEMPLATE_PATTERNS:
        if pattern.search(instruction):
            return name
    return "other"


def prefix_key(instruction: str, n: int = 8) -> str:
    words = instruction.lower().split()
    return " ".join(words[:n])


def percentile(sorted_values: list, p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * p / 100)
    return sorted_values[min(idx, len(sorted_values) - 1)]


def bar(value: int, total: int, width: int = 30) -> str:
    filled = int(width * value / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset quality checker")
    parser.add_argument("file", help="Path to generated_pairs.json")
    parser.add_argument("--sample", type=int, default=20, metavar="N",
                        help="Number of random pairs to print for manual review (default: 20)")
    parser.add_argument("--dedup-prefix", type=int, default=0, metavar="N",
                        help="Keep at most N pairs per unique 8-word prefix (0 = no dedup, default: 0)")
    parser.add_argument("--out", metavar="FILE",
                        help="Write deduplicated pairs to this file (requires --dedup-prefix)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("ERROR: expected a JSON array at the top level", file=sys.stderr)
        sys.exit(1)

    total = len(data)
    print(f"\n{'='*60}")
    print(f"  Dataset Quality Report — {path.name}")
    print(f"  Total pairs in file: {total}")
    print(f"{'='*60}\n")

    # ── Check 1: Filter pass-rate ─────────────────────────────────────────────
    seen: set[str] = set()
    passed = []
    reject_placeholder = 0
    reject_offtrack    = 0
    reject_tooshort    = 0
    reject_notportugal = 0
    reject_duplicate   = 0
    reject_empty       = 0

    for p in data:
        if not isinstance(p, dict):
            reject_empty += 1
            continue
        instr = p.get("instruction", "").strip()
        resp  = p.get("response",    "").strip()
        if not instr or not resp:
            reject_empty += 1
            continue
        if len(instr) < 25 or len(resp) < 40:
            reject_tooshort += 1
            continue
        if instr in seen:
            reject_duplicate += 1
            continue
        if _PLACEHOLDER_RE.search(instr) or _PLACEHOLDER_RE.search(resp):
            reject_placeholder += 1
            continue
        if _OFFTRACK_RE.match(resp):
            reject_offtrack += 1
            continue
        if not _PORTUGAL_RE.search(instr + " " + resp):
            reject_notportugal += 1
            continue
        seen.add(instr)
        passed.append(p)

    n_passed = len(passed)
    n_rejected = total - n_passed
    pass_rate = 100 * n_passed / total if total else 0

    print("── 1. Filter Pass-Rate ──────────────────────────────────────")
    print(f"  Passed  : {n_passed:>5}  ({pass_rate:.1f}%)")
    print(f"  Rejected: {n_rejected:>5}  ({100-pass_rate:.1f}%)")
    print(f"  {bar(n_passed, total)}  {pass_rate:.0f}%")
    print()
    if n_rejected:
        print("  Rejection breakdown:")
        for label, count in [
            ("placeholder echo", reject_placeholder),
            ("off-topic (no PT)", reject_notportugal),
            ("too short",         reject_tooshort),
            ("response=question", reject_offtrack),
            ("duplicate",         reject_duplicate),
            ("empty/malformed",   reject_empty),
        ]:
            if count:
                pct = 100 * count / n_rejected
                print(f"    {label:<22} {count:>4}  ({pct:.0f}% of rejects)")
    print()

    if not passed:
        print("No pairs passed filters. Cannot continue.")
        sys.exit(0)

    # ── Check 2: Template distribution ───────────────────────────────────────
    print("── 2. Template Distribution (inferred) ──────────────────────")
    template_counts = Counter(classify_template(p["instruction"]) for p in passed)
    for tmpl, count in sorted(template_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / n_passed
        print(f"  {tmpl:<14} {count:>4}  {bar(count, n_passed, 20)}  {pct:.1f}%")
    print()

    # ── Check 3: Instruction diversity ───────────────────────────────────────
    print("── 3. Instruction Diversity ──────────────────────────────────")
    prefixes = [prefix_key(p["instruction"]) for p in passed]
    prefix_counts = Counter(prefixes)
    unique_prefixes = len(prefix_counts)
    top_repeated = prefix_counts.most_common(5)

    diversity_pct = 100 * unique_prefixes / n_passed
    print(f"  Unique 8-word prefixes: {unique_prefixes} / {n_passed}  ({diversity_pct:.1f}%)")
    print(f"  {bar(unique_prefixes, n_passed)}  {diversity_pct:.0f}% unique")

    repeated = [(k, v) for k, v in top_repeated if v > 1]
    if repeated:
        print(f"\n  Most repeated openings:")
        for prefix, count in repeated:
            print(f"    [{count}x] \"{prefix}...\"")
    else:
        print("  No repeated openings — good diversity.")
    print()

    # ── Check 4: Length distribution ─────────────────────────────────────────
    print("── 4. Length Distribution ────────────────────────────────────")
    instr_lens = sorted(len(p["instruction"]) for p in passed)
    resp_lens  = sorted(len(p["response"])    for p in passed)

    print(f"  {'':12}  {'min':>6}  {'p25':>6}  {'median':>6}  {'p75':>6}  {'max':>6}")
    print(f"  {'instruction':12}  "
          f"{instr_lens[0]:>6}  "
          f"{percentile(instr_lens,25):>6.0f}  "
          f"{percentile(instr_lens,50):>6.0f}  "
          f"{percentile(instr_lens,75):>6.0f}  "
          f"{instr_lens[-1]:>6}")
    print(f"  {'response':12}  "
          f"{resp_lens[0]:>6}  "
          f"{percentile(resp_lens,25):>6.0f}  "
          f"{percentile(resp_lens,50):>6.0f}  "
          f"{percentile(resp_lens,75):>6.0f}  "
          f"{resp_lens[-1]:>6}")

    # Flag truncation: responses at max_new_tokens boundary (~1500 chars * ~4 bytes/token ≈ 6000 chars)
    truncated = sum(1 for l in resp_lens if l >= 5500)
    if truncated:
        print(f"\n  WARNING: {truncated} responses >= 5500 chars — may be truncated at MAX_NEW_TOKENS")
    print()

    # ── Check 5: PT-BR contamination ─────────────────────────────────────────
    print("── 5. PT-BR Contamination ────────────────────────────────────")
    ptbr_hits = []
    for p in passed:
        text = p["instruction"] + " " + p["response"]
        m = _PTBR_RE.search(text)
        if m:
            ptbr_hits.append((m.group(), p["instruction"][:80]))

    if ptbr_hits:
        print(f"  WARNING: {len(ptbr_hits)} pairs contain PT-BR markers ({100*len(ptbr_hits)/n_passed:.1f}%)")
        for marker, instr in ptbr_hits[:5]:
            print(f"    marker='{marker}'  instr=\"{instr}...\"")
        if len(ptbr_hits) > 5:
            print(f"    ... and {len(ptbr_hits)-5} more")
    else:
        print(f"  No PT-BR markers found in {n_passed} passing pairs.")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("── Summary ───────────────────────────────────────────────────")
    grades = []
    if pass_rate >= 85:
        grades.append(f"  Pass-rate {pass_rate:.0f}% >= 85%  ✓")
    else:
        grades.append(f"  Pass-rate {pass_rate:.0f}% < 85%   ✗  (investigate rejections)")

    if diversity_pct >= 90:
        grades.append(f"  Diversity {diversity_pct:.0f}% >= 90%  ✓")
    else:
        grades.append(f"  Diversity {diversity_pct:.0f}% < 90%   ✗  (model recycling openings)")

    dominant = template_counts.most_common(1)[0]
    dom_pct = 100 * dominant[1] / n_passed
    if dom_pct <= 40:
        grades.append(f"  Template balance: '{dominant[0]}' dominates at {dom_pct:.0f}%  ✓")
    else:
        grades.append(f"  Template balance: '{dominant[0]}' dominates at {dom_pct:.0f}%  ✗  (one template over-represented)")

    if not ptbr_hits:
        grades.append("  PT-BR: clean  ✓")
    else:
        grades.append(f"  PT-BR: {len(ptbr_hits)} hits  ✗")

    for g in grades:
        print(g)
    print()

    # ── Prefix dedup (optional) ───────────────────────────────────────────────
    deduped = passed
    if args.dedup_prefix > 0:
        print(f"── Prefix Dedup (keep ≤ {args.dedup_prefix} per 8-word prefix) ────────────")
        prefix_seen: Counter = Counter()
        deduped = []
        dropped = 0
        for p in passed:
            key = prefix_key(p["instruction"])
            if prefix_seen[key] < args.dedup_prefix:
                deduped.append(p)
                prefix_seen[key] += 1
            else:
                dropped += 1
        print(f"  Before: {n_passed}  After: {len(deduped)}  Dropped: {dropped}")
        if args.out:
            out_path = Path(args.out)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(deduped, f, ensure_ascii=False, indent=2)
            print(f"  Written to: {out_path}")
        else:
            print(f"  Add --out <file> to save the deduplicated result.")
        print()

    # ── Manual spot-check ─────────────────────────────────────────────────────
    n_sample = min(args.sample, len(deduped))
    if n_sample > 0:
        print(f"── Manual Spot-Check ({n_sample} random pairs) ───────────────────")
        sample = random.sample(deduped, n_sample)
        for i, p in enumerate(sample, 1):
            instr = p["instruction"]
            resp  = p["response"]
            print(f"\n  [{i}/{n_sample}] ({classify_template(instr)})")
            print(f"  Q: {instr[:120]}{'...' if len(instr)>120 else ''}")
            print(f"  A: {resp[:200]}{'...' if len(resp)>200 else ''}")
        print()

    print("=" * 60)
    print(f"  Usable pairs: {n_passed} / {total}")
    if args.dedup_prefix > 0:
        print(f"  After prefix dedup: {len(deduped)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
