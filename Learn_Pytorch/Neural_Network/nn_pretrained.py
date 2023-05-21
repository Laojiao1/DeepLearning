# 现有网络模型的使用和修改
import torchvision.datasets

# train_data = torchvision.datasets.ImageNet("./data", split='train', download=True, transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights='DEFAULT')
print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())

# 加入pytorch提供的一些现有的网络模型
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# 对现有的网络模型进行修改
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
