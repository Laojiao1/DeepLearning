import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


# 方式1
model = torch.load("vgg16_method1.pth")
print(model)

# 方式2
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
print(vgg16)

# 陷阱
# 如果采用方式一的话，一定要让这个程序能够访问到你定义的模型（可以不用复制，直接引入也可以）
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

model = torch.load("laojiao_method1.pth")
print(model)