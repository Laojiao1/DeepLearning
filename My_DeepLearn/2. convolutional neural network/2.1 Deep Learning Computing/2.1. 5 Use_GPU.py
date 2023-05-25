import torch
from torch import nn

print(torch.device('cpu'))
print(torch.device('cuda'))
print(torch.device('cuda:1'))

# 可以查询可用gpu的数量
print(torch.cuda.device_count())

# 定义了两个方便的函数， 这两个函数允许在不存在所需所有GPU的情况下运行代码。
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回搜友可用的GPU，如果没有GPU，则返回cpu"""
    devices = [torch.device(f'cuda:{i}')
              for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu())
print(try_gpu(10))
print(try_all_gpus())

# 查询张量所在的设备。 默认情况下，张量是在CPU上创建的。
X = torch.tensor([1, 2, 3])
print(X.device)

# 将变量存储在GPU上
# 在创建张量时指定存储设备
Y = torch.ones(2, 3, device=try_gpu())
print(Y.device)

# 复制
Z = X.cuda(0)
# Z = X.cuda(1)
print(X)
print(Z)

# 现在数据在同一个GPU上（Z和X都在），我们可以将它们相加。
print(X + Z)

# 假设变量Z已经存在于第二个GPU上。 如果我们还是调用Z.cuda(1) 它将返回Z，而不会复制并分配新内存。
print(Z.cuda(1) is Z)

# 类似地，神经网络模型可以指定设备
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(X))

# 确认模型参数存储在同一个GPU上
print(net[0].weight.data.device)