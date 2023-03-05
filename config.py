import torch
import os

from networks import DAFN
from utils import DiceLoss

NCLS = 8
DAFNet_synapse= {
  'save_path' : 'D:\\Projects\\results\\DAFNet',
  'device'    : 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'epoch_num' : 240,
  'num_classes': NCLS,
  'pretrained_params' : None,
  'model'     : DAFN,
  'model_args': {'img_size':224, 'dim_in':1, 'dim_out':NCLS},
  'criterion' : DiceLoss,
  'criterion_args': {'n_classes' : NCLS}, # number of classes should equal to the number of out channels in model_args
  'optimizer' : torch.optim.Adam,
  'optimizer_args': {'lr' : 1e-4, 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':64, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':64, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy':5,
  'flags':{
    'save_history':False
  }
}


NCLS = 9
DAFNet_synapse2= {
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9',
  'device'    : 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'epoch_num' : 240,
  'num_classes': NCLS,
  'pretrained_params' : None,
  'model'     : DAFN,
  'model_args': {'img_size':224, 'dim_in':1, 'dim_out':9, 'embed_dim':32,
                 'split_size':[1,2,2,7], 'num_heads':[2,4,8,16],'qkv_bias':False, 'qk_scale':None,
                 },
  'criterion' : DiceLoss,
  'criterion_args': {'n_classes' : NCLS}, # number of classes should equal to the number of out channels in model_args
  'optimizer' : torch.optim.Adam,
  'optimizer_args': {'lr' : 1e-4, 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':64, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':64, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy':5,
  'flags':{
    'save_history':False
  }
}


NCLS = 9
DAFNet_synapse_version1_cls9_v2= {
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9_v2',
  'device'    : 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'epoch_num' : 240,
  'num_classes': NCLS,
  'pretrained_params' : None,
  'model'     : DAFN,
  'model_args': {'img_size':224, 'dim_in':1, 'dim_out':9, 'embed_dim':16,
                 'split_size':[1,2,2,7], 'num_heads':[2,4,8,16],'qkv_bias':False, 'qk_scale':None,
                 },
  'criterion' : DiceLoss,
  'criterion_args': {'n_classes' : NCLS}, # number of classes should equal to the number of out channels in model_args
  'optimizer' : torch.optim.Adam,
  'optimizer_args': {'lr' : 1e-4, 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':64, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':64, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy':5,
  'flags':{
    'save_history':True
  }
}

NCLS = 9
DAFNet_synapse_version1_cls9_v3= {
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9_v3',
  'device'    : 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'epoch_num' : 240,
  'num_classes': NCLS,
  'pretrained_params' : None,
  'model'     : DAFN,
  'model_args': {'img_size':224, 'dim_in':1, 'dim_out':9, 'embed_dim':16,
                 'split_size':[1,2,2,7], 'num_heads':[2,4,8,16],'qkv_bias':False, 'qk_scale':None,
                 },
  'criterion' : DiceLoss,
  'criterion_args': {'n_classes' : NCLS}, # number of classes should equal to the number of out channels in model_args
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':64, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':64, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy':5,
  'flags':{
    'save_history':True
  }
}


