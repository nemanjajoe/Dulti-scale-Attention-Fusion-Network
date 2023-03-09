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

NCLS = 9
DAFNet_synapse_version1_cls9_v4= {
  'describe'  : "MLP Mixer token mixer: nn.Linear -> nn.Conv1d; model size: 106M -> 66M", 
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9_v4',
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


NCLS = 9
DAFNet_synapse_version1_cls9_v5= {
  'describe'  : "embed dim: 32 -> 16", 
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9_v5',
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

NCLS = 9
DAFNet_synapse_version1_cls9_v6= {
  'describe'  : "Down Sample ->Shift Patch Merging", 
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9_v6',
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


NCLS = 9
DAFNet_synapse_version1_cls9_v7= {
  'describe'  : "Shift Patch Merging: shift -1 -> 0", 
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9_v7',
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
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy':5,
  'flags':{
    'save_history':True
  }
}

NCLS = 9
DAFNet_synapse_version1_cls9_v8= {
  'describe'  : "Shift Patch Merging: shift 0 -> shift -1 +1", 
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9_v8',
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
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy':5,
  'flags':{
    'save_history':True
  }
}

NCLS = 9
DAFNet_synapse_version1_cls9_v8_repeat= {
  'describe'  : "Shift Patch Merging: shift 0 -> shift -1 +1", 
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9_v8_repeat',
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
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy':5,
  'flags':{
    'save_history':True
  }
}


NCLS = 9
DAFNet_synapse_version1_cls9_v9= {
  'describe'  : "Normal Conv Downsample VS Symmetric Fusion Conv Downsample", 
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9_v9',
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
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy':5,
  'flags':{
    'save_history':True
  }
}


NCLS = 9
DAFNet_synapse_version1_cls9_v9_repeat= {
  'describe'  : "Normal Conv Downsample VS Symmetric Fusion Conv Downsample", 
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9_v9_repeat',
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
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy':5,
  'flags':{
    'save_history':True
  }
}

DAFNet_synapse_version1_cls9_v10= {
  'describe'  : "symmetric fusion conv : double intermidiate channel", 
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9_v10',
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
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy':5,
  'flags':{
    'save_history':True
  }
}

DAFNet_synapse_version1_cls9_v11= {
  'describe'  : "multi dual scale attention fusion stage", 
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9_v11',
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
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy':5,
  'flags':{
    'save_history':True
  }
}

DAFNet_synapse_version1_cls9_v12= {
  'describe'  : "remove residual connection in DA transformer", 
  'save_path' : 'D:\\Projects\\results\\DAFNet\\version1_cls_9_v12',
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
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':32, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy':5,
  'flags':{
    'save_history':True
  }
}
