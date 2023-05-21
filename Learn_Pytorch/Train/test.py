# 完整的模型验证套路
import torch
import torchvision.transforms
from PIL import Image
from model import *

# 打开目标图片
image_path = "./imgs/dog.png"
image = Image.open(image_path)
print(image)

# Png图片格式有四个通道，有额外的一个透明度通道，使用此代码保留其颜色通道
image = image.convert('RGB')

# 对其进行剪裁和转换为totensor类型
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

# 下载训练模型
# 如果使用gpu训练的模型，需要加上'map_location=torch.device('cpu')'
model = torch.load("./model_train/model_2.pth", map_location=torch.device('cpu'))
# 在网络训练中是需要batch-size的，所以需要对image进行reshape一下
image = torch.reshape(image, (1, 3, 32, 32))
# 把模型转换为测试类型(可写可不写, 最好写上)
model.eval()
# 这一步可以节约内存和性能(可写可不写, 最好写上)
with torch.no_grad():
    output = model(image)

# 此次没有预测成功是因为训练的模型还不够到位
print(output.argmax(1).item())
