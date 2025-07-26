import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
    print("GPU Memory Allocated (MB):", torch.cuda.memory_allocated(0) / 1024**2)
    print("GPU Memory Cached (MB):", torch.cuda.memory_reserved(0) / 1024**2)
else:
    print("GPU is NOT available. Running on CPU.")
