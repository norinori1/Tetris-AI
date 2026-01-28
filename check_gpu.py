"""Check GPU/CUDA availability"""
import torch

print("=" * 60)
print("GPU/CUDA Status Check")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    print(f"\nCurrent device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is NOT available. Using CPU for training.")
    print("\nTo enable GPU support:")
    print("1. Install NVIDIA GPU drivers")
    print("2. Install CUDA Toolkit")
    print("3. Install PyTorch with CUDA support:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("=" * 60)

# Test GPU computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice for training: {device}")

# Create a simple tensor and move it to device
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)
z = torch.matmul(x, y)
print(f"Test computation successful on {device}")
print(f"Result tensor device: {z.device}")
