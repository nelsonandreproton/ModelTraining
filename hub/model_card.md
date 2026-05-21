---
license: apache-2.0
language:
  - pt
library_name: peft
base_model: Qwen/Qwen2.5-1.5B-Instruct
pipeline_tag: text-generation
tags:
  - peft
  - lora
  - portuguese
  - question-answering
  - instruction-tuning
  - qwen2.5
  - pt-pt
datasets:
  - nelsondiasandre/portuguese-qa-instruct-500
model-index:
  - name: qwen25-1.5b-pt-qa-lora
    results:
      - task:
          type: text-generation
        dataset:
          name: portuguese-qa-instruct-500
          type: nelsondiasandre/portuguese-qa-instruct-500
        metrics:
          - type: perplexity
            name: Perplexity (eval set, 100 examples)
            value: 1.86
widget:
  - text: "<|im_start|>user\nQual e a capital de Portugal?<|im_end|>\n<|im_start|>assistant\n"
    example_title: "Capital de Portugal"
  - text: "<|im_start|>user\nO que e a fotossintese?<|im_end|>\n<|im_start|>assistant\n"
    example_title: "Ciencias"
  - text: "<|im_start|>user\nQuem escreveu Os Lusiadas?<|im_end|>\n<|im_start|>assistant\n"
    example_title: "Literatura Portuguesa"
---

# Qwen2.5-1.5B PT-PT Q&A LoRA

LoRA adapter fine-tuned on [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) for Portuguese question answering.

## Usage

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("nelsondiasandre/qwen25-1.5b-pt-qa-lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

def perguntar(pergunta: str) -> str:
    prompt = f"<|im_start|>user\n{pergunta}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
    )
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=False)
    resposta = resposta.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    return resposta

print(perguntar("Qual e a capital de Portugal?"))
```

## Chat Template

This model uses the **Qwen ChatML format**:

```
<|im_start|>user
{pergunta}<|im_end|>
<|im_start|>assistant
{resposta}<|im_end|>
```

Do **not** use Qwen's native `apply_chat_template()` — it produces a slightly different format than what this adapter was trained on. Use the prompt string directly as shown above.

## Training Details

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Epochs | 20 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.1 |
| Batch size | 2 (effective 8 with grad accum 4) |
| Max sequence length | 512 tokens |
| Training hardware | CPU |
| Training dataset | 500 PT-PT instruction pairs (400 train / 100 eval) |

## Training Data

500 Portuguese (PT-PT) question-answer pairs across 20+ categories:
- Geography and capitals
- History (Portuguese and world)
- Science and biology
- Mathematics
- Literature (including Portuguese classics)
- Culture, gastronomy, and traditions
- Technology and computing
- Sports (including football)

See the [dataset card](https://huggingface.co/datasets/nelsondiasandre/portuguese-qa-instruct-500) for details.

## Evaluation

| Metric | Value |
|---|---|
| Eval loss (final epoch) | 0.6199 |
| Perplexity (eval set) | 1.86 |

## Limitations

- Small training set (500 examples) — may not generalise well to topics outside training data
- Trained on CPU only — no GPU-optimised quantisation applied
- Portuguese (PT-PT) only — not validated for Brazilian Portuguese
- Short answers expected — trained on concise Q&A format, not long-form generation

## License

Apache 2.0 (inherited from Qwen2.5-1.5B-Instruct base model).
