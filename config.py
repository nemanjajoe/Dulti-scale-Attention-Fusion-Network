import torch
import os

from networks import DAFN
from utils import SynapseLoss


NCLS=9
DAFNet_synapse_version1_cls9_v17= {
  'describe'  : "optimize training procedure", 
  'save_path' : "D:\\Projects\\results\\DAFNet\\version1_cls_9_v17",
  'dataset_path': "D:\\Projects\\datasets\\Synapse_data",
  'device'    : 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'epoch_num' : 400,
  'num_classes': NCLS,
  'pretrained_params' : None,
  'model'     : DAFN,
  'model_args': {'img_size':224, 'dim_in':1, 'dim_out':NCLS, 'embed_dim':64,'depth':[1,2,4,2],
                 'split_size':[1,2,2,7], 'num_heads':[2,4,8,16],'qkv_bias':False, 'qk_scale':None,
                 },
  'criterion' : SynapseLoss,
  'criterion_args': {'n_classes' : NCLS, 'alpha':0.4, 'beta':0.6}, # number of classes should equal to the number of out channels in model_args
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
  'scheduler_args':{'T_max':90000},
  'train_loader_args' : {'batch_size':32, 'shuffle':True, 'num_workers':6, 'pin_memory':True, 'drop_last':True},
  'test_loader_args': {'batch_size':1, 'shuffle':False, 'num_workers':1},
  'eval_frequncy':20,
  'n_gpu': 1,
  'grad_clipping': False,
}

