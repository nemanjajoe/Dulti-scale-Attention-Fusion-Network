import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MlpMixer(nn.Module):
    def __init__(self,seq_len, dim) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = nn.Linear(seq_len,seq_len)
        self.act1 = nn.GELU()
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mixer = nn.Linear(dim,dim)
        # self.act2 = nn.GELU()
    
    def forward(self,x):
        """
            x: B L C
        """
        t = self.norm1(x).transpose(-1,-2)
        x = self.token_mixer(t).transpose(-1,-2) + x
        x = self.channel_mixer(self.norm2(x)) + x

        return x

class EfficentLePE(nn.Module):
    def __init__(self, dim, res, idx, split_size=7, num_heads=8, qk_scale=None) -> None:
        super().__init__()
        self.res = res
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        if idx == -1:
            H_sp, W_sp = self.res, self.res
        elif idx == 0:
            H_sp, W_sp = self.res, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.res, self.split_size
        else:
            print("ERROR MODE : ",idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1,groups=dim)

    def im2cswin(self,x):
        B,L,C = x.shape
        H = W = int(np.sqrt(L))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x
    
    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe
    
    def forward(self,qkv):
        q,k,v = qkv[0],qkv[1],qkv[2]
        B,L,C = q.shape
        H = W = self.res
        assert(L == H*W)
        k = self.im2cswin(k)
        v = self.im2cswin(v)
        q, lepe = self.get_lepe(q,self.get_v)
        att = k.transpose(-2,-1)@v * self.scale
        att = nn.functional.softmax(att,dim=-1,dtype=att.dtype)

        x = q@att +lepe
        x = x.transpose(1,2).reshape(-1,self.H_sp*self.W_sp,C)

        x = windows2img(x,self.H_sp,self.W_sp,H,W).view(B,-1,C)
        return x

class CSWinBlock(nn.Module):
    def __init__(self, dim, res, num_heads, split_size,
                 qkv_bias=False, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,last_stage=False) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_res = res
        self.split_size = split_size
        self.to_qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.norm = norm_layer(dim)

        if self.patch_res == split_size:
            last_stage = True
        
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2

        if last_stage:
            self.attns = nn.ModuleList([
                EfficentLePE(dim, res, -1, split_size, num_heads, qk_scale)
            for i in range(self.branch_num)
            ])
        else:
            self.attns = nn.ModuleList([
                EfficentLePE(dim, res, i, split_size, num_heads, qk_scale)
            for i in range(self.branch_num)
            ])

    def forward(self,x):
        """
        Args:
            x: B H*W C
        Returns:
            x: B H*W C
        """
        H = W = self.patch_res
        B,L,C = x.shape
        assert(H*W == L)
        x = self.norm(x)

        qkv = self.to_qkv(x).reshape(B,-1,3,C).permute(2,0,1,3)
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:,:,:,:C//2])
            x2 = self.attns[1](qkv[:,:,:,C//2:])
            att = torch.cat([x1,x2],dim=2)
        else:
            att = self.attns[0](qkv)
        
        return att

class DualBlock(nn.Module):
    def __init__(self,dim, res, split_size_h,split_size_l,
                 num_heads_h, num_heads_l, qkv_bias=False, qk_scale=None, act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,last_stage=False) -> None:
        super().__init__()
        self.block_h = CSWinBlock(dim,res,num_heads_h,split_size_h,qkv_bias,qk_scale,act_layer,norm_layer,last_stage)
        self.mixer_h = MlpMixer(res*res,dim)
        self.block_l = CSWinBlock(dim,res,num_heads_l,split_size_l,qkv_bias,qk_scale,act_layer,norm_layer,last_stage)
        self.mixer_l = MlpMixer(res*res,dim)
        self.last_stage = last_stage
        self.res = res
        if last_stage:
            self.merge = nn.Sequential(
                Rearrange("b (h w) c -> b c h w", h=res,w=res),
                nn.Conv2d(2*dim,dim,3,1,2),
                Rearrange("b c h w -> b (h w) c"),
                nn.LayerNorm(dim)
            )
    def forward(self,x_h, x_l):
        """
        Args:
            x_h: B L C (high resolution features)
            x_l: B L C (low resolution features)
        Returns:
            x_h: B L C
            x_l: B L C 
        or 
            x : B L C (if last stage)
        """
        assert(self.res*self.res == x_h.shape[1]) # H*W == L
        assert(self.res*self.res == x_l.shape[1]) # H*W == L
        t_h = self.block_h(x_h)
        t_l = self.block_l(x_l)
        m_h = self.mixer_h(t_h + x_l)
        m_l = self.mixer_l(t_l + x_h)
        x_h = m_h + t_l
        x_l = m_l + t_h
        if self.last_stage:
            x = torch.cat([x_h,x_l], dim=-1)
            x = self.merge(x)
            return x
        return x_h, x_l

class DownSample(nn.Module):
    def __init__(self,dim_in,dim_out=None, norm_layer=nn.LayerNorm,act_layer=nn.GELU) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out or dim_in
        self.down_conv = nn.Conv2d(self.dim_in,self.dim_out,7,2,3)
        self.norm = norm_layer(self.dim_out)
        self.conv1 = nn.Conv2d(self.dim_out, 2*self.dim_out,1)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(2*self.dim_out,self.dim_out,1)

    def forward(self,att,x):
        """
        Args:
            att: B H*W C
            x  : B H*W C
        Returns:
            x  : B (H/2)(W/2) 4C
        """
        assert(att.shape == x.shape)
        B,L,C = att.shape
        H = W = int(np.sqrt(L))
        x = torch.cat([att,x],dim=-1)
        x = rearrange(x,"b (h w) c -> b c h w", h=H,w=W)
        x = self.down_conv(x)
        t = self.norm(x)
        t = self.conv1(t)
        t = self.act(t)
        x = self.conv2(t) + x
        
        return rearrange(x,"b c h w -> b (h w) c")
        
class UpSample(nn.Module):
    def __init__(self) -> None:
        super().__init__()


