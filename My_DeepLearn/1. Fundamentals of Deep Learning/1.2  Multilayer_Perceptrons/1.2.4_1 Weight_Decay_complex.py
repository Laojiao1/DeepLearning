# 用一个例子展现权重衰减（从零开始实现）
import torch
from torch import nn
from d2l import torch as d2l

# 生成数据
"""
    选择标签是关于输入的线性函数。 
    标签同时被均值为0，标准差为0.01高斯噪声破坏。 
    为了使过拟合的效果更加明显，我们可以将问题的维数增加到d = 200
    并使用一个只包含20个样本的小训练集。
"""
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# 定义函数来随机化初始参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 定义L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

# 定义训练代码实现
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())

# 忽略正则化直接训练，这里训练误差有了减少，但测试误差没有减少， 这意味着出现了严重的过拟合。
# train(lambd=0)

# 使用权重衰减，注意，在这里训练误差增大，但测试误差减小。 这正是我们期望从正则化中得到的效果。
train(lambd=3)
d2l.plt.show()