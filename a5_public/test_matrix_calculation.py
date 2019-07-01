import numpy as np
import torch.nn as nn
import torch


a=np.array([[0.3138, 0.2737, 0.4125],
        [0.3388, 0.2479, 0.4134],
        [0.2521, 0.3545, 0.3934],
        [0.2362, 0.4206, 0.3431],
        [0.2830, 0.3113, 0.4057]])
x_proj=np.array([[0.0000, 0.0000, 0.0228],
        [0.0000, 0.0000, 0.0774],
        [0.0000, 0.0000, 0.0000],
        [0.2466, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000]])
x_con_out=np.array([[0.5870, 0.1604, 0.2206],
        [0.7860, 0.1820, 0.4263],
        [0.4508, 0.4506, 0.0056],
        [0.0864, 0.9202, 0.7248],
        [0.4675, 0.2579, 0.0752]])

x_highway=a*x_proj+(np.ones(a.shape)-a)*x_con_out
print(x_highway)

# x=np.array([[0.5870, 0.1604, 0.2206],
#         [0.7860, 0.1820, 0.4263],
#         [0.4508, 0.4506, 0.0056],
#         [0.0864, 0.9202, 0.7248],
#         [0.4675, 0.2579, 0.0752]])
# b=np.array([-0.1410, -0.0763,  0.1917])
# out=np.matmul(x,a.T)+b




# predict highway is  torch.Size([5, 3])
# tensor([[0.4028, 0.1165, 0.1390],
#         [0.5197, 0.1369, 0.2821],
#         [0.3372, 0.2909, 0.0034],
#         [0.1242, 0.5331, 0.4761],
#         [0.3352, 0.1776, 0.0447]], grad_fn=<AddBackward0>)
#
# the intermediate values and weights are as followed:
#
# X_conv_outtensor([[0.5870, 0.1604, 0.2206],
#         [0.7860, 0.1820, 0.4263],
#         [0.4508, 0.4506, 0.0056],
#         [0.0864, 0.9202, 0.7248],
#         [0.4675, 0.2579, 0.0752]])
#
# W_proj_tmpParameter containing:
# tensor([[-0.4560,  0.5322,  0.1965],
#         [ 0.4592,  0.1093, -0.5460],
#         [-0.1371, -0.5174,  0.4522]], requires_grad=True)  b_projParameter containing:
# tensor([-0.3462, -0.3437,  0.0865], requires_grad=True)
#
# X_proj:torch.Size([5, 3])  tensor([[0.0000, 0.0000, 0.0228],
#         [0.0000, 0.0000, 0.0774],
#         [0.0000, 0.0000, 0.0000],
#         [0.2466, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000]], grad_fn=<ClampBackward>)
#
# W_gateParameter containing:
# tensor([[ 0.0588, -0.5042,  0.2507],
#         [-0.3574,  0.5324, -0.3087],
#         [-0.0021, -0.1900, -0.0848]], requires_grad=True)  b_gateParameter containing:
# tensor([-0.1410, -0.0763,  0.1917], requires_grad=True)
#
# X_gate_tmp:torch.Size([5, 3])  tensor([[-0.1321, -0.2689,  0.1413],
#         [-0.0797, -0.3920,  0.1194],
#         [-0.3403,  0.0007,  0.1047],
#         [-0.4182,  0.1589, -0.0448],
#         [-0.2247, -0.1293,  0.1354]], grad_fn=<AddmmBackward>)
#
# X_gate:torch.Size([5, 3])  tensor([[0.3138, 0.2737, 0.4125],
#         [0.3388, 0.2479, 0.4134],
#         [0.2521, 0.3545, 0.3934],
#         [0.2362, 0.4206, 0.3431],
#         [0.2830, 0.3113, 0.4057]], grad_fn=<SoftmaxBackward>)





# m = nn.Linear(2, 4)
# input = torch.randn(3, 2)
# output = m(input)
# print(str(input)+'\n')
# print(str(m.weight)+'\n')
# print(str(m.bias)+'\n')
# print(str(output)+'\n')
#
# input2=input.numpy()
# weight2=m.weight.detach().numpy()
# bias2=m.bias.detach().numpy()
# out2=np.matmul(input2,weight2.T)+bias2
# print(out2)