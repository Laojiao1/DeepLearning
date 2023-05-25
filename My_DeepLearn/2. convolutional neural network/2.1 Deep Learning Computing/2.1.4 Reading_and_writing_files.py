import torch
from torch import nn
from torch.nn import functional as F

# 保存数据
x = torch.arange(4)
torch.save(x, "./data/x-file")

# 将数据读回内存
x = torch.load("./data/x-file")
print(x)

# 存储一个张量列表，然后读回内存
y = torch.zeros(4)
torch.save([x, y], "./data/x-files")

x2, y2 = torch.load("./data/x-files")
print(x2, "\n", y2)

# 写入或读取从字符串映射到张量的字典
# 当我们要读取或写入模型中的所有权重时，这很方便
mydict = {'x': x, 'y': y}
torch.save(mydict, "./data/mydict")

mydict2 = torch.load("./data/mydict")
print(mydict2)

# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# 接下来将模型的参数保存在一个叫做“mlp.params”的文件中
torch.save(net.state_dict(), "./data/mlp.params")

# 为了恢复模型，我们实例化了原始多层感知机模型的一个备份。
# 这里我们不需要随机初始化模型参数，而是直接读取文件中存储的参数。
clone = MLP()
clone.load_state_dict(torch.load("./data/mlp.params"))
print(clone.eval())

Y_clone = clone(X)
print(Y_clone == Y)