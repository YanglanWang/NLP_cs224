import torch
x = torch.randn(4, 2,3)
y=x.view(4,-1)

print(x)
print(y)
