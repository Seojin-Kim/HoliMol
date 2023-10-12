import torch
a = torch.Tensor([1,2,3,4]).unsqueeze(0)

print(a.shape)
print(a.transpose(0,1).shape)
print(a + a.transpose(0,1))


