import torch
print(torch.cuda.current_device())  # ควรเป็น 1
print(torch.cuda.get_device_name(torch.cuda.current_device()))
