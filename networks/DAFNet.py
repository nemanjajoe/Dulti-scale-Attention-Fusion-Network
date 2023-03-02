import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from .blocks import DownSample,UpSample,DualBlock
from thop import profile, clever_format


class EncoderFusionStage(nn.Module):
    def __init__(self, dim_in, dim_out, res, split_size_h, split_size_l,
                 num_heads_h, num_heads_l, qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, last_stage=False) -> None:
        super().__init__()
        self.last_stage = last_stage
        self.conv_h = DownSample(res,dim_in=dim_in,dim_out=dim_out,norm_layer=norm_layer,act_layer=act_layer)
        self.conv_l = DownSample(res,dim_in=dim_in,dim_out=dim_out,norm_layer=norm_layer,act_layer=act_layer)
        res = res//2
        self.dual_fusion = DualBlock(dim_out,res,split_size_h,split_size_l,num_heads_h,num_heads_l,
                                     qkv_bias,qk_scale,act_layer,norm_layer,last_stage)
        self.img2token = Rearrange("b c h w -> b (h w) c")
        self.token2img = Rearrange("b (h w) c -> b c h w", h=res,w=res)

    def forward(self,x_h,x_l):
        """
        Args:
            x_h,x_l: B C H W
        Returns:
            att_h,att_l: B C H W
        """
        x_h = self.conv_h(x_h)
        x_l = self.conv_l(x_l)
        
        if self.last_stage:
            att = self.dual_fusion(self.img2token(x_h),self.img2token(x_l))
            att_h = att_l = self.token2img(att)
            return att_h,att_l,x_l
        else:
            att_h,att_l = self.dual_fusion(self.img2token(x_h),self.img2token(x_l))
        
        att_h = self.token2img(att_h) + x_h
        att_l = self.token2img(att_l) + x_l

        return att_h,att_l,x_l

class Encoder(nn.Module):
    def __init__(self, img_size=224, dim_in=1, embed_dim=32,
                 split_size=[1,2,7], num_heads=[2,4,8],qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.conv_embed = DownSample(img_size,dim_in=dim_in, dim_out=embed_dim, norm_layer=norm_layer,act_layer=act_layer)
        res = img_size//2
        dim = embed_dim
        self.stages = []
        assert(len(split_size) == len(num_heads))
        depth = len(split_size)
        for i in range(depth):
            last_stage = False
            # if i == depth - 1:
            #     last_stage=True
            
            s_h = split_size[depth-i-1]
            s_l = split_size[i]
            n_h = num_heads[depth-i-1]
            n_l = num_heads[i]


            stage = EncoderFusionStage(dim,2*dim,res,s_h,s_l,n_h,n_l,qkv_bias,qk_scale,
                                       act_layer,norm_layer,last_stage)
            res = res//2
            dim = dim*2
            self.stages.append(stage)

        self.stages = nn.ModuleList(self.stages)
        self.final_res = res
        self.final_dim = dim
    
    def forward(self,img):
        """
        Args:
            img: B C H W
        Returns:
            skips [(att,x_l)...]:
                att,x_l: B C H W  
        """
        skips = []
        x = self.conv_embed(img)
        skips.append(x)
        att_h,att_l,x = self.stages[0](x,x)
        skips.append((att_h+att_l,x)) # (att_h + att_l)/2 ? 

        for stage in self.stages[1:-1]:
            att_h,att_l,x = stage(att_h,att_l)
            skips.append((att_h+att_l,x))
        
        att_h,att_l,x = self.stages[-1](att_h,att_l)
        skips.append((att_h,x))

        return tuple(skips)

class DecoderFusion(nn.Module):
    def __init__(self, dim_in, dim_out, res, split_size_h, split_size_l,
                 num_heads_h, num_heads_l, qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        # print(dim_in, res)
        self.dual_fusion = DualBlock(dim_in,res,split_size_h,split_size_l,num_heads_h,num_heads_l,
                                     qkv_bias,qk_scale,act_layer,norm_layer,False)
        self.img2token = Rearrange("b c h w -> b (h w) c")
        self.token2img = Rearrange("b (h w) c -> b c h w", h=res,w=res)
        self.deconv = UpSample(res,4*dim_in,dim_out,norm_layer,act_layer)
    
    def forward(self,att,x_l,x):
        """
        Args:
            att: B C H W
            x_l: B C H W
            x  : B C H W
        Returns:
            x  : B C H W
        """
        t_h = self.img2token(att)
        t_l = self.img2token(x)
        t_h,t_l = self.dual_fusion(t_h,t_l)
        t_h = self.token2img(t_h)
        t_l = self.token2img(t_l) # B C H W
        x = torch.cat([x_l,t_h,t_l,x],dim=1) # cat in channel
        x = self.deconv(x)

        return x

class Decoder(nn.Module):
    def __init__(self,res, dim_in, dim_out,
                 split_size=[7,2,1], num_heads=[8,4,2],qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        res = res
        dim = dim_in
        self.stages = []
        assert(len(split_size) == len(num_heads))
        depth = len(split_size)
        for i in range(depth):
            s_h = split_size[depth-i-1]
            s_l = split_size[i]
            n_h = num_heads[depth-i-1]
            n_l = num_heads[i]

            stage = DecoderFusion(dim,dim//2,res,s_h,s_l,n_h,n_l,qkv_bias,qk_scale,
                                       act_layer,norm_layer)
            res = res*2
            dim = dim//2
            self.stages.append(stage)
        # self.stages.reverse()
        self.stages = nn.ModuleList(self.stages)
        
        self.deconv_embed = UpSample(res,2*dim,dim_out,norm_layer,act_layer)

    def forward(self,skips):
        l = len(skips)
        for i in range(l - 1):
            att,x_l = skips[-i-1]
            if i == 0:
                x = x_l
            
            # print(att.shape,x_l.shape,x.shape)
            # exit(0)
            x = self.stages[i](att,x_l,x)
        
        x = torch.cat([x,skips[0]],dim=1) # B 2C H W
        x = self.deconv_embed(x)

        return x

class DAFN(nn.Module):
    def __init__(self, img_size=224, dim_in=1, dim_out=9, embed_dim=32,
                 split_size=[1,2,7], num_heads=[2,4,8],qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.encoder = Encoder(img_size,dim_in,embed_dim,split_size,num_heads,
                               qkv_bias,qk_scale,act_layer,norm_layer)
        res = self.encoder.final_res
        dim = self.encoder.final_dim
        split_size.reverse()
        num_heads.reverse()
        self.decoder = Decoder(res,dim,dim_out,split_size,num_heads,
                               qkv_bias,qk_scale,act_layer,norm_layer)
        
    def forward(self,img):
        return self.decoder(self.encoder(img))


def main():
    x = torch.randn((2,1,224,224))
    model =  DAFN()
    flops,params = profile(model,(x,))
    flops,params = clever_format([flops,params],'%.3f')
    
    print("flops per sample: ",flops)
    print("model parameters: ",params)

if __name__ == '__main__':
    main()

