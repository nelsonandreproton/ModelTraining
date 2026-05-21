from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Parâmetros totais: {sum(p.numel() for p in model.parameters()):,}")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_from_disk("./my_dataset_processed")

sft_config = SFTConfig(
    output_dir="./lora_output",
    num_train_epochs=20,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    max_length=512,
    dataset_text_field="text",
    report_to="none",
    load_best_model_at_end=True,
    use_cpu=True,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

print("\nA iniciar fine-tuning Qwen2.5-1.5B + LoRA...\n")
trainer.train()

print("\nFine-tuning concluido!")

model.save_pretrained("./my_lora_model")
tokenizer.save_pretrained("./my_lora_model")
print("Modelo LoRA guardado em ./my_lora_model")
