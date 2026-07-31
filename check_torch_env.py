import importlib.util
import sys

print(sys.executable)
spec = importlib.util.find_spec("torch")
print("has_torch", spec is not None)
if spec is not None:
    import torch

    print("torch", torch.__version__)
    print("torch_cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
