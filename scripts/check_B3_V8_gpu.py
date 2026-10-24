"""Check real CUDA support in the SAME Python venv used for MAPPO."""
import sys
try:
    import torch
except Exception as e:
    print('PYTHON:',sys.executable)
    raise SystemExit(f'PyTorch import FAILED: {e}')
print('PYTHON:',sys.executable)
print('TORCH:',torch.__version__)
print('TORCH CUDA BUILD:',torch.version.cuda)
print('CUDA AVAILABLE:',torch.cuda.is_available())
print('GPU COUNT:',torch.cuda.device_count() if torch.cuda.is_available() else 0)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props=torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} | VRAM={props.total_memory/1024**3:.2f} GB')
    # Actually calculate with GPU tensors, not just check installed CUDA metadata.
    x=torch.rand((256,256),device='cuda:0')
    y=x@x
    torch.cuda.synchronize()
    print('CUDA MATMUL PASS:',y.device,tuple(y.shape))
else:
    print('NO CUDA: Do not start 5-hour GPU training. NVIDIA CUDA GPU/driver and CUDA-enabled torch required.')
