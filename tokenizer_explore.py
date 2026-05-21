from transformers import AutoTokenizer

# Carrega o tokenizer do GPT-2 (modelo que vamos usar)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Textos de teste
textos = [
    "Olá mundo!",
    "A capital de Portugal é Lisboa.",
    "tokenização é fascinante",
    "<|user|> Quanto é 2 + 2? <|assistant|> 4.",
]

print("=== TOKENIZAÇÃO ===\n")
for texto in textos:
    tokens_ids = tokenizer.encode(texto)
    tokens_str = tokenizer.convert_ids_to_tokens(tokens_ids)
    decoded = tokenizer.decode(tokens_ids)

    print(f"Texto:    {texto}")
    print(f"Tokens:   {tokens_str}")
    print(f"IDs:      {tokens_ids}")
    print(f"Decoded:  {decoded}")
    print(f"Nº tokens: {len(tokens_ids)}")
    print("-" * 60)