import torch

from thop import profile,clever_format
from trainner import Trainer_synapse
from config import DAFNet_synapse_version1_cls9_v2 as hyper2
from config import DAFNet_synapse_version1_cls9_v3 as hyper3
from config import DAFNet_synapse_version1_cls9_v4 as hyper4
from config import DAFNet_synapse_version1_cls9_v5 as hyper5
from config import DAFNet_synapse_version1_cls9_v6 as hyper6

dataset_path = "D:\\Projects\\datasets\\Synapse_npy"
net = Trainer_synapse(dataset_path,hyper6)

def print_model_size():
    x = torch.randn((2,1,224,224)).to(net.device)
    flops,params = profile(model=net.model, inputs=(x,))
    flops,params = clever_format([flops,params], "%.3f")
    print(flops,params)

if __name__ == '__main__':
    x = torch.randn((2,1,224,224)).to(net.device)
    y = net.model(x)
    print(x.shape, y.shape)
    print_model_size()

