# 多层感知机的从零开始实现
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 实现一个具有单隐藏层的多层感知机，包含256个隐藏单元
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True))
b1 = nn.parameter(torch.zeros(num_hiddens, requires_grad=True))

W2 = nn.parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))
b2 = nn.parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

# 实现ReLU激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 实现模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return relu(H @ W2 + b2)

loss = nn.CrossEntropyLoss(reduction='none')

# 训练
num_epochs = 10
lr = 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

d2l.plt.show()


