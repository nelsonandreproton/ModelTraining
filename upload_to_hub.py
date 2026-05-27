import os
import sys
import argparse
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from datasets import load_from_disk

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN not found in environment. Add it to .env")
    sys.exit(1)

login(token=HF_TOKEN)
api = HfApi()

parser = argparse.ArgumentParser(description="Upload model and/or dataset to HuggingFace Hub")
parser.add_argument("--username", required=True, help="Your HuggingFace username")
parser.add_argument("--model", action="store_true", help="Upload LoRA adapter")
parser.add_argument("--dataset", action="store_true", help="Upload processed dataset")
parser.add_argument("--private", action="store_true", help="Create private repos")
parser.add_argument(
    "--model-repo",
    default="qwen25-1.5b-pt-qa-lora",
    help="Model repo name (default: qwen25-1.5b-pt-qa-lora)",
)
parser.add_argument(
    "--dataset-repo",
    default="portuguese-qa-instruct-620",
    help="Dataset repo name (default: portuguese-qa-instruct-620)",
)
args = parser.parse_args()

if not args.model and not args.dataset:
    print("Nothing to upload. Use --model and/or --dataset.")
    parser.print_help()
    sys.exit(1)

model_repo_id = f"{args.username}/{args.model_repo}"
dataset_repo_id = f"{args.username}/{args.dataset_repo}"

# ── Model upload ───────────────────────────────────────
if args.model:
    print(f"\nUploading LoRA adapter to {model_repo_id}...")

    api.create_repo(
        repo_id=model_repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    # Upload adapter files
    api.upload_folder(
        folder_path="./my_lora_model",
        repo_id=model_repo_id,
        repo_type="model",
    )

    # Upload model card (overwrites auto-generated README)
    api.upload_file(
        path_or_fileobj="./hub/model_card.md",
        path_in_repo="README.md",
        repo_id=model_repo_id,
        repo_type="model",
    )

    print(f"Model uploaded: https://huggingface.co/{model_repo_id}")

# ── Dataset upload ─────────────────────────────────────
if args.dataset:
    print(f"\nUploading dataset to {dataset_repo_id}...")

    api.create_repo(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    ds = load_from_disk("./my_dataset_processed")
    ds.push_to_hub(dataset_repo_id, token=HF_TOKEN)

    # Upload dataset card (overwrites auto-generated README)
    api.upload_file(
        path_or_fileobj="./hub/dataset_card.md",
        path_in_repo="README.md",
        repo_id=dataset_repo_id,
        repo_type="dataset",
    )

    print(f"Dataset uploaded: https://huggingface.co/datasets/{dataset_repo_id}")

print("\nDone.")
