# 5. 非线性激活(小于等于0的变成0, 大于0的保留原始数据)
# 作用：给网络中引入一些非线性特征，非线性特征越多的话才能训练出符合特征或者一些曲线的模型，减少泛化性
import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


input = torch.tensor([[1, -0.5],
                      [-1, 3]])

# input = torch.reshape(input, (-1, 1, 2, 2))
# print(input.shape)

class Laojiao(nn.Module):
    def __init__(self):
        super(Laojiao, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

laojiao = Laojiao()
# output = laojiao(input)
# print(output)

writer = SummaryWriter("logs_relu")

step = 1
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = laojiao(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()

