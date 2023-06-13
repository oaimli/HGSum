import torch
import torch.nn as nn

input1 = torch.randn(3, 128)
input2 = torch.randn(3, 128)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)
# print(output)
# print(torch.tensor(input1))

x = []
x.append(torch.tensor([1, 2]))
# x.append([2, 3])
print(torch.tensor(x))