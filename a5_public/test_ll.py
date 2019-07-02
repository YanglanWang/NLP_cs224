import torch

a=torch.rand(4,5)
print(str(a)+'\n')
b=torch.max(a,dim=1)
print(str(b[0])+'\n')
print(str(b[1]))