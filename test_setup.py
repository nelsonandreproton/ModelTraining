import torch
import transformers

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CPU threads: {torch.get_num_threads()}")
print("✅ Tudo OK!")