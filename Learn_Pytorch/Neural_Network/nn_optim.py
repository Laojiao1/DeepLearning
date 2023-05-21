# 优化器
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)

class Laojiao(nn.Module):
    def __init__(self):
        super(Laojiao, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

laojiao = Laojiao()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(laojiao.parameters(), lr=0.01) # 定义优化器

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = laojiao(imgs)
        result_loss = loss(output, targets)

        # 将优化器中每一个网络的梯度清零
        optim.zero_grad()
        # 调用损失函数的反向传播求出每个节点的梯度
        result_loss.backward()
        # 对模型参数进行调优
        optim.step()

        running_loss = running_loss + result_loss
    print(running_loss)

