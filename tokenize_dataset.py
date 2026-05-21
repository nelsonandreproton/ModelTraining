from transformers import AutoTokenizer
from datasets import load_from_disk

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_from_disk("./my_dataset_processed")

print(f"Splits disponíveis: {list(dataset.keys())}")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,      # corta se passar o limite
        max_length=512,       # limite de tokens por exemplo
        padding="max_length", # padding para ficarem todos do mesmo tamanho
    )

tokenized = dataset.map(tokenize, batched=True)

print("\n=== DATASET TOKENIZADO ===")
print(f"Colunas: {tokenized['train'].column_names}")
print(f"\nPrimeiro exemplo (train):")
print(f"  text:          {tokenized['train'][0]['text']}")
print(f"  input_ids:     {tokenized['train'][0]['input_ids'][:20]}... ({len(tokenized['train'][0]['input_ids'])} tokens)")
print(f"  attention_mask:{tokenized['train'][0]['attention_mask'][:20]}...")

tokenized.save_to_disk("./my_dataset_tokenized")
print("\nDataset tokenizado guardado em ./my_dataset_tokenized")