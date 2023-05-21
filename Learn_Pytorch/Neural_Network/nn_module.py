# 1. 神经网络的基本骨架 nn.Module

import torch
from torch import nn


class My_Module(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input + 1
        return output

my_module = My_Module()
x = torch.tensor(1.0)
output = my_module(x)
print(output)

