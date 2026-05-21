---
license: cc-by-4.0
language:
  - pt
pretty_name: "Portuguese Q&A Instruction Dataset (500 pairs)"
size_categories:
  - n<1K
task_categories:
  - question-answering
  - text-generation
tags:
  - portuguese
  - pt-pt
  - instruction-tuning
  - qa-pairs
  - question-answering
dataset_info:
  features:
    - name: instruction
      dtype: string
    - name: response
      dtype: string
    - name: text
      dtype: string
  splits:
    - name: train
      num_examples: 400
    - name: test
      num_examples: 100
---

# Portuguese Q&A Instruction Dataset (500 pairs)

500 Portuguese (PT-PT) question-answer pairs formatted for instruction fine-tuning of language models.

## Dataset Structure

Each example has three columns:

| Column | Description | Example |
|---|---|---|
| `instruction` | The question in Portuguese | `"Qual e a capital de Portugal?"` |
| `response` | The answer in Portuguese | `"A capital de Portugal e Lisboa."` |
| `text` | Pre-formatted instruction template (see below) | `"<\|im_start\|>user\n..."` |

### Instruction template

The `text` column uses the Qwen ChatML format:

```
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

## Dataset Creation

Questions and answers were written in PT-PT (European Portuguese) across 20+ categories:

| Category | Examples |
|---|---|
| Geography | Capitals, rivers, mountains, regions |
| Portuguese history | Kings, discoveries, historical events |
| World history | Wars, empires, revolutions |
| Science | Biology, physics, chemistry, astronomy |
| Mathematics | Arithmetic, geometry, basic concepts |
| Portuguese literature | Camoes, Pessoa, Eca de Queiroz |
| World literature | Shakespeare, Cervantes, Dante |
| Technology | Computing, internet, AI concepts |
| Sports | Football, Olympics, Portuguese athletes |
| Culture | Fado, gastronomy, traditions |
| Politics | EU, democracy, Portuguese government |
| Economy | Concepts, Portugal's economy |
| Philosophy | Socrates, Plato, Aristotle |
| Medicine | Anatomy, diseases, treatments |
| Art | Painters, movements, Portuguese art |
| Environment | Climate, ecosystems, conservation |
| Religion | Christianity, Islam, Buddhism |
| Languages | Etymology, linguistics |
| Music | Genres, instruments, musicians |
| Cinema | Directors, films, history |

## Splits

| Split | Examples |
|---|---|
| train | 400 (80%) |
| test | 100 (20%) |

Split was performed with `seed=42`.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("nelsondiasandre/portuguese-qa-instruct-500")

# Access a training example
print(ds["train"][0])
# {
#   "instruction": "Qual e a capital de Portugal?",
#   "response": "A capital de Portugal e Lisboa.",
#   "text": "<|im_start|>user\nQual e a capital de Portugal?<|im_end|>\n<|im_start|>assistant\nA capital de Portugal e Lisboa.<|im_end|>"
# }
```

## Intended Use

Supervised fine-tuning (SFT) of language models for Portuguese question answering, particularly with instruction-following models like Qwen2.5.

See the [model card](https://huggingface.co/nelsondiasandre/qwen25-1.5b-pt-qa-lora) for the LoRA adapter trained on this dataset.

## Limitations

- 500 examples is a small dataset — models trained on it may overfit or lack coverage
- PT-PT only (European Portuguese) — not validated for Brazilian Portuguese
- Answers are intentionally short and direct — not suitable for training long-form generation

## License

CC-BY-4.0
