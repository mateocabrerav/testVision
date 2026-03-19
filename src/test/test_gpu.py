"""Quick test to verify GPU acceleration is working."""

import torch

print("="*60)
print("GPU ACCELERATION TEST")
print("="*60)

# Check PyTorch
print(f"\n📦 PyTorch version: {torch.__version__}")

# Check CUDA
if torch.cuda.is_available():
    print(f"✅ CUDA is available!")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Quick benchmark
    print(f"\n⚡ Running quick benchmark...")
    import time
    
    # CPU test
    x = torch.randn(1000, 1000)
    start = time.time()
    for _ in range(100):
        y = x @ x
    cpu_time = time.time() - start
    print(f"   CPU: {cpu_time:.3f}s")
    
    # GPU test
    x = torch.randn(1000, 1000).cuda()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y = x @ x
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"   GPU: {gpu_time:.3f}s")
    print(f"   Speedup: {cpu_time/gpu_time:.1f}x faster! 🚀")
    
else:
    print(f"❌ CUDA is NOT available")
    print(f"   PyTorch is using CPU only")
    print(f"\n💡 To enable GPU:")
    print(f"   pip uninstall torch torchvision")
    print(f"   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "="*60)
