import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_mem / 1e9:.1f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")
else:
    print("No CUDA GPU detected - will use CPU")
