# Fine-tuning Qwen2.5-1.5B-Instruct with LoRA — Kaggle GPU version
# Dataset: nelsondiasandre/portuguese-qa-instruct-raw (PT-PT only, 5000+ pairs)
# Upgrades vs local version: r=16, bf16=True, early stopping, GPU

# ── Install a COHESIVE training stack (Kaggle ships an incompatible mix) ──────
# Do NOT use `-U` here: upgrading each package independently produced a half-
# upgraded transformers missing `divide_to_patches`, which trl's import chain
# needs → ImportError. Pin a set released together. transformers 4.46.x is the
# same line proven to install cleanly on this Kaggle image; trl 0.12.x is the
# matching SFTConfig/processing_class API; peft 0.13.x + accelerate 1.1.x pair
# with them. These satisfy this script's modern-but-stable TRL usage.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.46.1",
                "trl==0.12.1",
                "peft==0.13.2",
                "accelerate==1.1.1",
                "datasets==3.1.0"], check=True)
print("deps installed — if you see an import error below, do Run > Restart & Run All once", flush=True)

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

# ── HuggingFace token from Kaggle secret ─────────────────────────────────────
# Add via: Kaggle notebook → Add-ons → Secrets → name: HF_TOKEN
from kaggle_secrets import UserSecretsClient
hf_token = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # single T4 — DataParallel on 2×T4 duplicates
                                           # model+grads on GPU0, causing OOM at backward pass

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME    = "nelsondiasandre/portuguese-qa-instruct-raw"
HF_REPO_ID      = "nelsondiasandre/qwen25-1.5b-pt-qa-lora"
OUTPUT_DIR      = "/kaggle/working/lora_output"
FINAL_MODEL_DIR = "/kaggle/working/my_lora_model"

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
raw = load_dataset(DATASET_NAME, token=hf_token)

# Build a true held-out test set (50 examples) before any train/eval split.
# This avoids the load_best_model_at_end leakage: those 50 examples never
# influence checkpoint selection.
full = raw["train"].train_test_split(test_size=50, seed=42)
isolated_test = full["test"]   # 50 examples — never used during training
remaining     = full["train"]  # rest goes into train/eval

split = remaining.train_test_split(test_size=0.15, seed=42)
train_ds = split["train"]
eval_ds  = split["test"]

print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)} | Isolated test: {len(isolated_test)}")

# ── Format into chat template ─────────────────────────────────────────────────
# Raw dataset has instruction+response columns; SFTTrainer needs a single text
# field with the chat template applied.
def format_example(ex):
    ex["text"] = (
        f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{ex['response']}<|im_end|>"
    )
    return ex

train_ds     = train_ds.map(format_example)
eval_ds      = eval_ds.map(format_example)
isolated_test = isolated_test.map(format_example)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Model ─────────────────────────────────────────────────────────────────────
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=hf_token,
    torch_dtype=torch.float16,   # load directly in fp16 (~3GB vs ~6GB for fp32)
    device_map="cuda:0",         # pin to GPU 0 — avoids DataParallel across both T4s
)

# ── LoRA config (r=16, alpha=32 — upgraded from r=8/alpha=16) ─────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Training config ───────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,              # 5005 examples × 5 ≈ 3000 steps; was 20 (set for ~500 examples — overkill + overfit risk at 10× data). Early stopping still guards.
    per_device_train_batch_size=2,   # T4 has 15GB; batch 4 OOMs with 1.5B model
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch = 2×4 = 8, same as before
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,                       # T4 = Turing arch, no bf16 support (needs Ampere+)
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    max_seq_length=512,              # trl 0.12 name (was max_length in newer trl)
    dataset_text_field="text",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    use_cpu=False,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Starting training...")
trainer.train()

# ── Save adapter ──────────────────────────────────────────────────────────────
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"Adapter saved to {FINAL_MODEL_DIR}")

# ── Push to HuggingFace Hub ───────────────────────────────────────────────────
print(f"Pushing to HuggingFace Hub: {HF_REPO_ID}")
model.push_to_hub(HF_REPO_ID, token=hf_token)
tokenizer.push_to_hub(HF_REPO_ID, token=hf_token)
print("Done. Adapter available at https://huggingface.co/" + HF_REPO_ID)

# ── Quick sanity check on isolated test set ───────────────────────────────────
print("\n=== Sanity check on isolated test set (first 3 examples) ===")
model.eval()
for ex in isolated_test.select(range(3)):
    prompt = f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {ex['instruction']}")
    print(f"Expected: {ex['response'][:80]}...")
    print(f"Got:      {response[:80]}...")
