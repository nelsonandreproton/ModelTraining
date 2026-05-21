from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk

# ── Tokenizer ──────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ── Arquitetura do modelo (tiny GPT-2) ─────────────────
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,  # tamanho do vocabulário
    n_embd=128,       # dimensão dos embeddings
    n_layer=4,        # número de camadas transformer
    n_head=4,         # número de attention heads
    n_positions=128,  # contexto máximo (tokens)
)

model = GPT2LMHeadModel(config)

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Modelo criado com {total_params:,} parâmetros")

# ── Dataset ────────────────────────────────────────────
dataset = load_from_disk("./my_dataset_tokenized")

# ── Data Collator ──────────────────────────────────────
# Cria automaticamente os labels para language modeling
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # False = causal LM (GPT), True = masked LM (BERT)
)

# ── Argumentos de treino ───────────────────────────────
args = TrainingArguments(
    output_dir="./model_output",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    learning_rate=5e-4,
    load_best_model_at_end=True,
    report_to="none",  # desativa wandb/tensorboard
)

# ── Trainer ────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collator,
)

print("🚀 A iniciar treino...\n")
trainer.train()

print("\n✅ Treino concluído!")
trainer.save_model("./my_pretrained_model")
tokenizer.save_pretrained("./my_pretrained_model")
print("✅ Modelo guardado em ./my_pretrained_model")