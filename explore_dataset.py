from datasets import load_from_disk, DatasetDict
from sklearn.model_selection import train_test_split

# Carrega o dataset guardado
dataset = load_from_disk("./my_dataset")

print("=== INFO ===")
print(f"Exemplos: {len(dataset)}")
print(f"Colunas: {dataset.column_names}")
print(f"Features: {dataset.features}")

# Divide em treino/validação (80/20) - prática padrão
split = dataset.train_test_split(test_size=0.2, seed=42)

print("\n=== SPLIT ===")
print(f"Treino: {len(split['train'])} exemplos")
print(f"Validação: {len(split['test'])} exemplos")

def format_instruction(example):
    example["text"] = (
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['response']}<|im_end|>"
    )
    return example

split = split.map(format_instruction)

print("\n=== EXEMPLOS FORMATADOS ===")
for i, example in enumerate(split["train"]):
    print(f"\n[{i+1}] {example['text']}")

# Guarda o dataset processado
split.save_to_disk("./my_dataset_processed")
print("\nDataset processado guardado em ./my_dataset_processed")