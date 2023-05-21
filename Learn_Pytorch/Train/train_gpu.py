# 使用GPU进行训练，可以更快
import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
import time

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

device = torch.device("cuda")

# 获取数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            # 卷积
            nn.Conv2d(3, 32, 5, 1, 2),
            # 最大池化
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            # 展平
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = Model()

# 利用GPU训练
if torch.cuda.is_available():
    model = model.cuda()

# model.to(device)

# 创建损失函数
loss_function = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_function = loss_function.cuda()
# loss_function.to(device)

# 优化器
# 本次实例调用一下随机梯度下降
learn_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./logs_train")

start_time = time.time()
for i in range(epoch):
    print("-----------第{}轮训练开始-----------".format(i+1))

    # 训练开始
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        # imgs = imgs.to(device)
        # targets = targets.to(device)
        output = model(imgs)
        loss = loss_function(output, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("本次训练花费的时间：{}".format(end_time - start_time))
            print("训练次数：{}, loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            # imgs = imgs.to(device)
            # targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)

    total_test_step = total_test_step + 1

    torch.save(model, "./model_train/model_{}.pth".format(i))
    # torch.save(model.state_dict(), "./model_train/model_{}.path".format(i))
    print("模型已保存")

writer.close()

