import torchvision.datasets

# 准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
test_Loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片以及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
step = 0
for data in test_Loader:
    imgs, targets = data
    writer.add_images("test_data", imgs, step)
    step = step + 1

writer.close()

