import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
"""
    利用d2l中自带的函数构造数据集
    特征(features)是在标准正态分布中随机采样的，并与真实权重相乘和加上真实偏差。
    标签(labels)是特征加上一个随机噪声项，并与真实权重相乘和加上真实偏差。
    """
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 1. 调用框架中现有的API来读取数据
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

# 2. 使用框架来预定义好层
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

# 3. 初始化模型参数
"""访问网络中的第一层，"""
net[0].weight.data.normal_(0, 0.01) # 对权重张量的数据部分进行正态分布初始化
net[0].bias.data.fill_(0) # 获取偏差张量的数据部分，并将其所有元素赋值为0

# 4. 计算均方误差
loss = nn.MSELoss()
"""使用随机梯度下降，将模型参数传递给优化器,并设置学习速率为0.03"""
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 5. 训练
num_epoch = 3
for epoch in range(num_epoch):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()

    l = loss(net(features), labels)
    print(f'epoch: {epoch + 1}, loss: {l:f}')

# 最后看一下与真实值的误差
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)