import torch
import torch.nn.functional as F
from torch import nn

# 构造一个没有任何参数的自定义层。要从其输入中减去均值
class CenteredLayer(nn.Module):
    def __init__(self):
        super(CenteredLayer, self).__init__()

    def forward(self, X):
        return X - X.mean()

# 提供一些数据，验证是否能工作
layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

# 可以将层作为组件合并到更复杂的模型中
net = nn.Sequential(
    nn.Linear(8, 128),
    CenteredLayer()
)
Y = net(torch.rand(4, 8))
print(Y.mean())

# 自定义版本的全连接层。该层需要两个参数，一个用于表示权重，另一个用于表示偏置项。
# 使用修正线性单元作为激活函数
# 该层需要输入参数：in_units 和 units，分别表示输入数和输出数。
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.zeros(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
print(linear.weight)

# 使用自定义层直接执行前向传播计算
print(linear(torch.rand(2, 5)))

# 使用自定义层构建模型
net = nn.Sequential (
    MyLinear(64, 8),
    MyLinear(8, 1)
)
print(net(torch.rand(2, 64)))