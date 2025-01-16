import torch

x = torch.arange(48).reshape(2, 2, 3, 4)
print(x)
print(x.permute(0, 2, 3, 1))
print(x.reshape(2, 3, 4, 2))