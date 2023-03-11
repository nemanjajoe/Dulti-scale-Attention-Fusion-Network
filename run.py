import torch

from thop import profile,clever_format
from trainer import Trainer_synapse
from config import *

net = Trainer_synapse(DAFNet_synapse_version1_cls9_v17)
# net.test()
# def print_model_size():
#     x = torch.randn((2,1,224,224)).to(net.device)
#     flops,params = profile(model=net.model, inputs=(x,))
#     flops,params = clever_format([flops,params], "%.3f")
#     print(flops,params)

# if __name__ == '__main__':
#     x = torch.randn((2,1,224,224)).to(net.device)
#     y = net.model(x)
#     print(x.shape, y.shape)
#     print_model_size()

