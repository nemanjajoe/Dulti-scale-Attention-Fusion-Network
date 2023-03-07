import torch
from thop import profile,clever_format
from trainner import Trainer_synapse
from config import DAFNet_synapse as hyper1
from config import DAFNet_synapse2 as hyper2 
from config import DAFNet_synapse_version1_cls9_v2 as hyper3 
from config import DAFNet_synapse_version1_cls9_v3 as hyper4
from config import DAFNet_synapse_version2_cls9_v1 as hyper5
from config import DAFNet_synapse_version2_cls9_v2 as hyper6
from config import DAFNet_synapse_version2_cls9_v3 as hyper7
from config import DAFNet_synapse_version2_cls9_v4 as hyper8

net = Trainer_synapse("D:\\Projects\\datasets\\Synapse_npy",hyper8)

# if __name__ == '__main__':
#     x = torch.randn((2,1,224,224)).to(net.hyper['device'])
#     flops,params = profile(model=net.model,inputs=(x,))
#     flops,params = clever_format([flops,params], "%.3f")
#     print(flops,params)


