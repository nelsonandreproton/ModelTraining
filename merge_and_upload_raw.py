# merge_and_upload_raw.py
# Merge new generated_pairs JSON files into the live HF "raw" Q&A dataset,
# fixing PT-BR spelling contamination and deduplicating by instruction.
#
# WHY THIS EXISTS:
#   upload_to_hub.py --raw-pairs REPLACES the whole HF dataset with one file.
#   To GROW the dataset (1001 → 3000 → 5000) we must merge: pull what's live,
#   add the new pairs, clean, dedupe, then push the union back.
#
# USAGE:
#   # Dry run — clean + merge + dedupe locally, write merged_raw_preview.json, push NOTHING:
#   python merge_and_upload_raw.py --new generated_pairs_2000.json --username nelsondiasandre
#
#   # Real upload (only after the dry-run counts look right):
#   python merge_and_upload_raw.py --new generated_pairs_2000.json --username nelsondiasandre --push
#
#   # Merge several new files at once:
#   python merge_and_upload_raw.py --new a.json --new b.json --username nelsondiasandre --push

import argparse
import json
import os
import re
import sys
import pathlib

from dotenv import load_dotenv

load_dotenv()

# ── PT-BR → PT-PT spelling fix ────────────────────────────────────────────────
# CLOSED set of words where Brazilian Portuguese keeps the circumflex (ô/ê) but
# European Portuguese uses the acute accent (ó/é). These are UNAMBIGUOUS — the
# ô/ê form is simply never correct in PT-PT — so a literal replace is safe here.
# A blanket ô→ó replace would be WRONG (it would corrupt "português", "têm",
# "influência", "independência", which are all correct PT-PT circumflex words).
# Explicit, complete word→word map for the PT-BR/PT-PT accent divergence: BR keeps
# the circumflex (ô/ê) where PT-PT uses the acute (ó/é). This list was built by
# extracting the COMPLETE set of unique ô/ê words across the whole dataset and
# classifying each as BR-contaminant vs legitimate PT-PT — so the fix-set and the
# verify-set are the SAME ground truth. CRITICAL: do NOT add the -ês family
# (português, inglês, francês, mês, três) or vê/lê/crê — those are correct PT-PT.
_PTBR_FIXES = {
    # -ómico / -ómica
    "econômico": "económico", "econômica": "económica",
    "econômicos": "económicos", "econômicas": "económicas",
    "socioeconômico": "socioeconómico", "socioeconômica": "socioeconómica",
    "socioeconômicos": "socioeconómicos", "socioeconômicas": "socioeconómicas",
    "astronômico": "astronómico", "astronômica": "astronómica",
    "astronômicos": "astronómicos", "astronômicas": "astronómicas",
    "gastronômico": "gastronómico", "gastronômica": "gastronómica",
    "gastronômicos": "gastronómicos", "gastronômicas": "gastronómicas",
    "harmônico": "harmónico", "harmônica": "harmónica",
    "harmônicos": "harmónicos", "harmônicas": "harmónicas",
    # -ónico / -ónica
    "arquitetônico": "arquitetónico", "arquitetônica": "arquitetónica",
    "arquitetônicos": "arquitetónicos", "arquitetônicas": "arquitetónicas",
    "eletrônico": "eletrónico", "eletrônica": "eletrónica",
    "eletrônicos": "eletrónicos", "eletrônicas": "eletrónicas",
    "irônico": "irónico", "irônica": "irónica",
    "irônicos": "irónicos", "irônicas": "irónicas",
    "icônico": "icónico", "icônica": "icónica",
    "icônicos": "icónicos", "icônicas": "icónicas",
    "crônico": "crónico", "crônica": "crónica",
    "crônicos": "crónicos", "crônicas": "crónicas",
    "tônico": "tónico", "tônica": "tónica",
    "tônicos": "tónicos", "tônicas": "tónicas",
    "monotônico": "monotónico", "monotônica": "monotónica",
    # -ónomo / -ónoma
    "astrônomo": "astrónomo", "astrônoma": "astrónoma",
    "astrônomos": "astrónomos", "astrônomas": "astrónomas",
    "autônomo": "autónomo", "autônoma": "autónoma",
    "autônomos": "autónomos", "autônomas": "autónomas",
    # -ómeno
    "fenômeno": "fenómeno", "fenômenos": "fenómenos",
    # -ónimo
    "sinônimo": "sinónimo", "sinônimos": "sinónimos",
    "antônimo": "antónimo", "antônimos": "antónimos",
    "heterônimo": "heterónimo", "heterônimos": "heterónimos",
    "heterônima": "heterónima", "heterônimas": "heterónimas",
    # -ónia / -ónio (ô → ó)
    "cerimônia": "cerimónia", "cerimônias": "cerimónias",
    "patrimônio": "património", "patrimônios": "patrimónios",
    "colônia": "colónia", "colônias": "colónias",
    # -émio / -émico (ê → é)
    "prêmio": "prémio", "prêmios": "prémios",
    "acadêmico": "académico", "acadêmica": "académica",
    "acadêmicos": "académicos", "acadêmicas": "académicas",
    # ê → é
    "gênero": "género", "gêneros": "géneros",
    # PT-PT drops the circumflex entirely on these
    "vôo": "voo", "vôos": "voos",
    "metrô": "metro",
}
# Case-insensitive whole-word replacement, preserving the original capitalization.
_PTBR_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _PTBR_FIXES) + r")\b", re.IGNORECASE)


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement.capitalize()
    return replacement


def fix_ptbr(text: str) -> tuple[str, int]:
    """Return (corrected_text, num_fixes)."""
    count = 0

    def _sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        orig = m.group(0)
        return _preserve_case(orig, _PTBR_FIXES[orig.lower()])

    return _PTBR_RE.sub(_sub, text), count


def clean_pair(p: dict) -> tuple[dict, int]:
    instr, n1 = fix_ptbr(p["instruction"])
    resp, n2 = fix_ptbr(p["response"])
    return {"instruction": instr, "response": resp}, n1 + n2


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Merge new pairs into the live HF raw dataset.")
parser.add_argument("--new", action="append", required=True, metavar="FILE",
                    help="New generated_pairs JSON file (repeatable).")
parser.add_argument("--username", required=True, help="HuggingFace username.")
parser.add_argument("--repo", default="portuguese-qa-instruct-raw",
                    help="Raw dataset repo name (default: portuguese-qa-instruct-raw).")
parser.add_argument("--push", action="store_true",
                    help="Actually push to the Hub. Without it, this is a DRY RUN.")
parser.add_argument("--preview", default="merged_raw_preview.json",
                    help="Where to write the merged set for inspection (default: merged_raw_preview.json).")
args = parser.parse_args()

repo_id = f"{args.username}/{args.repo}"

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN not found in .env")
    sys.exit(1)

# ── Pull what is currently live on the Hub ────────────────────────────────────
existing: list[dict] = []
try:
    from datasets import load_dataset
    print(f"Loading live dataset {repo_id} ...")
    live = load_dataset(repo_id, token=HF_TOKEN)
    for split in live.values():
        for ex in split:
            if "instruction" in ex and "response" in ex:
                existing.append({"instruction": ex["instruction"], "response": ex["response"]})
    print(f"  live rows: {len(existing)}")
except Exception as e:
    print(f"  could not load live dataset ({e}). Treating live set as empty.")

# ── Load new files ────────────────────────────────────────────────────────────
new_pairs: list[dict] = []
for f in args.new:
    src = pathlib.Path(f)
    if not src.exists():
        print(f"Error: file not found: {src}")
        sys.exit(1)
    data = json.load(open(src, encoding="utf-8"))
    new_pairs.extend(p for p in data if isinstance(p, dict) and "instruction" in p and "response" in p)
    print(f"  loaded {len(data)} from {src.name}")

# ── Clean + merge + dedupe ────────────────────────────────────────────────────
seen: set[str] = set()
merged: list[dict] = []
total_fixes = 0
dupes = 0

for p in existing + new_pairs:          # existing first → its phrasing wins on a tie
    cleaned, nf = clean_pair(p)
    total_fixes += nf
    key = cleaned["instruction"].strip()
    if key in seen:
        dupes += 1
        continue
    seen.add(key)
    merged.append(cleaned)

print("\n-- Merge summary -----------------------------")
print(f"  live:            {len(existing)}")
print(f"  new (raw):       {len(new_pairs)}")
print(f"  PT-BR fixes:     {total_fixes}")
print(f"  duplicates cut:  {dupes}")
print(f"  MERGED UNIQUE:   {len(merged)}")
print("----------------------------------------------")

# Always write the preview so the result is inspectable.
with open(args.preview, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
print(f"Preview written: {args.preview}")

if not args.push:
    print("\nDRY RUN — nothing pushed. Re-run with --push once the counts look right.")
    sys.exit(0)

# ── Push the merged union back ────────────────────────────────────────────────
from huggingface_hub import login, HfApi
from datasets import Dataset

login(token=HF_TOKEN)
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

ds = Dataset.from_list(merged)
ds.push_to_hub(repo_id, token=HF_TOKEN, set_default=True)
print(f"\nPushed {len(ds)} rows -> https://huggingface.co/datasets/{repo_id}")
print(f"  columns: {ds.column_names}")
