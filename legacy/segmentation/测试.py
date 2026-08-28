import torch
print(f"CUDA是否可用：{torch.cuda.is_available()}")
print(f"识别到的GPU型号：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未识别到GPU'}")