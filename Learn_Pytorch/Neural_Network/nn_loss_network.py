# 使用损失函数与反向传播, 计算实际输出与目标之间的差距，为我们更新数据提供一定的反向传播
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
for data in dataloader:
    imgs, targets = data
    output = laojiao(imgs)
    result_loss = loss(output, targets)
    print(result_loss)
    result_loss.backward()
