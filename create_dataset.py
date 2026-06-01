# create_dataset.py
# Downloads the Portuguese Q&A dataset from the HuggingFace Hub and saves it
# locally as an Arrow dataset (./my_dataset) for the rest of the pipeline.
#
# This used to hold ~620 Q&A pairs hardcoded in the file. The dataset now lives
# on the Hub (nelsondiasandre/portuguese-qa-instruct-raw, 5000+ pairs) as the
# single source of truth — grown via generate_dataset_kaggle_vllm.py +
# merge_and_upload_raw.py. Edit HF_DATASET below to point at a different repo.
#
# Run: python create_dataset.py   (needs HF_TOKEN in .env)

import json
import os

from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

HF_DATASET = "nelsondiasandre/portuguese-qa-instruct-raw"

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("Error: HF_TOKEN not found in .env")

print(f"Downloading {HF_DATASET} from the HuggingFace Hub...")
hub = load_dataset(HF_DATASET, token=HF_TOKEN)

# The raw dataset has a single split ("train"); flatten it to one dataset.
dataset = hub["train"] if "train" in hub else next(iter(hub.values()))

# Keep only the two columns the pipeline expects (drop any extras defensively).
keep = [c for c in ("instruction", "response") if c in dataset.column_names]
dataset = dataset.select_columns(keep)

print(f"Número de exemplos: {len(dataset)}")
print(f"Colunas: {dataset.column_names}")
print("\nPrimeiro exemplo:")
print(json.dumps(dataset[0], ensure_ascii=False, indent=2))
print("\nÚltimo exemplo:")
print(json.dumps(dataset[-1], ensure_ascii=False, indent=2))

# Guarda em disco (formato Arrow - padrão HF)
dataset.save_to_disk("./my_dataset")
print(f"\nDataset guardado em ./my_dataset ({len(dataset)} exemplos)")
