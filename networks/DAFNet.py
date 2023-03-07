import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from .basic_blocks import DualBlock,DualFusionBlock,DualFusionBlockMerge
from thop import profile, clever_format


class EncoderFusionStage(nn.Module):
    def __init__(self, dim_in, res, split_size_h, split_size_l,
                 num_heads_h, num_heads_l,attn_drop=0.,drop_path=0.,
                 qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, last_stage=False) -> None:
        super().__init__()
        self.last_stage = last_stage
        self.down_conv = nn.Sequential(
            nn.Conv2d(dim_in,4*dim_in,7,2,3),
            nn.LayerNorm([4*dim_in, res//2, res//2])
        )
        # print(dim_in)
        self.conv_h = nn.Conv2d(2*dim_in,dim_in,3,1,1)
        self.conv_l = nn.Conv2d(2*dim_in,dim_in,3,1,1)
        self.res = res
        res = res//2
        self.dual_fusion = DualBlock(dim_in,res,split_size_h,split_size_l,num_heads_h,num_heads_l,
                                     attn_drop,drop_path,qkv_bias,qk_scale,act_layer,norm_layer,last_stage)
        self.img2token = Rearrange("b c h w -> b (h w) c")
        self.token2img = Rearrange("b (h w) c -> b c h w", h=res,w=res)
        
    def forward(self,x):
        """
        Args:
            x: B C H W
        Returns:
            att: B C H W 
        """
        # assert(H == W == self.res)
        x = self.down_conv(x)
        B,C,H,W = x.shape
        # print(x[:,:C//2,:,:].shape)
        # exit(0)
        x_h = self.img2token(self.conv_h(x[:,:C//2,:,:]))
        x_l = self.img2token(self.conv_l(x[:,C//2:,:,:]))
        
        if self.last_stage:
            att = self.dual_fusion(x_h,x_l)
            att_h = att_l = att
            return att_h,att_l,x_l
        else:
            att_h,att_l = self.dual_fusion(x_h,x_l)
        
        att_h = att_h + x_h
        att_l = att_l + x_l

        return self.token2img(torch.cat([att_h,att_l], dim=-1))

class Encoder(nn.Module):
    def __init__(self, img_size=224, dim_in=1, embed_dim=32,
                 split_size=[1,2,7], num_heads=[2,4,8],qkv_bias=False, qk_scale=None,
                 attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        res = img_size//2
        self.conv_embed = nn.Sequential(
            nn.Conv2d(dim_in,embed_dim,7,2,3),
            nn.LayerNorm([embed_dim,res,res])
        )
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

            stage = EncoderFusionStage(dim,res,s_h,s_l,n_h,n_l,attn_drop,drop_path,
                                       qkv_bias,qk_scale,act_layer,norm_layer,last_stage)
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
            skips : 
        """
        skips = []
        x = self.conv_embed(img)
        # print(x.shape)
        # exit(0)
        # skips.append(x)
        # att_h,att_l,x = self.stages[0](x,x)
        # skips.append((att_h+att_l,x)) # (att_h + att_l)/2 ? 

        for stage in self.stages:
            x = stage(x)
            # print(x.shape)
            skips.append(rearrange(x, "b c h w -> b (h w) c"))
        
        # att_h,att_l,x = self.stages[-1](att_h,att_l)
        # skips.append((att_h,x))

        # print(len(skips))
        return tuple(skips)

# class DecoderFusion(nn.Module):
#     def __init__(self, dim_in, dim_out, res, split_size_h, split_size_l,
#                  num_heads_h, num_heads_l, qkv_bias=False, qk_scale=None,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
#         super().__init__()
#         # print(dim_in, res)
#         self.dual_fusion = DualBlock(dim_in,res,split_size_h,split_size_l,num_heads_h,num_heads_l,
#                                      qkv_bias,qk_scale,act_layer,norm_layer,False)
#         self.img2token = Rearrange("b c h w -> b (h w) c")
#         self.token2img = Rearrange("b (h w) c -> b c h w", h=res,w=res)
#         self.deconv = UpSample(res,4*dim_in,dim_out,norm_layer,act_layer)
    
#     def forward(self,att,x_l,x):
#         """
#         Args:
#             att: B C H W
#             x_l: B C H W
#             x  : B C H W
#         Returns:
#             x  : B C H W
#         """
#         t_h = self.img2token(att)
#         t_l = self.img2token(x)
#         t_h,t_l = self.dual_fusion(t_h,t_l)
#         t_h = self.token2img(t_h)
#         t_l = self.token2img(t_l) # B C H W
#         x = torch.cat([x_l,t_h,t_l,x],dim=1) # cat in channel
#         x = self.deconv(x)

#         return x

# class Decoder(nn.Module):
#     def __init__(self,res, dim_in, dim_out,
#                  split_size=[7,2,1], num_heads=[8,4,2],qkv_bias=False, qk_scale=None,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
#         super().__init__()
#         res = res
#         dim = dim_in
#         self.stages = []
#         assert(len(split_size) == len(num_heads))
#         depth = len(split_size)
#         for i in range(depth):
#             s_h = split_size[depth-i-1]
#             s_l = split_size[i]
#             n_h = num_heads[depth-i-1]
#             n_l = num_heads[i]

#             stage = DecoderFusion(dim,dim//2,res,s_h,s_l,n_h,n_l,qkv_bias,qk_scale,
#                                        act_layer,norm_layer)
#             res = res*2
#             dim = dim//2
#             self.stages.append(stage)
#         # self.stages.reverse()
#         self.stages = nn.ModuleList(self.stages)
        
#         self.deconv_embed = UpSample(res,dim,dim_out,norm_layer,act_layer)

#     def forward(self,skips):
#         l = len(skips)
#         for i in range(l - 1):
#             att,x_l = skips[-i-1]
#             if i == 0:
#                 x = x_l
            
#             # print(att.shape,x_l.shape,x.shape)
#             # exit(0)
#             x = self.stages[i](att,x_l,x)
        
#         # x = torch.cat([x,skips[0]],dim=1) # B 2C H W
#         x = self.deconv_embed(x)

#         return x
class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H = W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x


class Decoder(nn.Module):
    def __init__(self,res, dim_in, dim_out,attn_drop=0.,drop_path=0.,
                 split_size=[1,2,2,7], num_heads=[2,4,8,16],qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        res = res
        dim = dim_in # 8C
        self.fusion_merge1 = DualFusionBlockMerge(dim,res,2,7,8,16,attn_drop,drop_path,
                                                  qkv_bias,qk_scale,act_layer,norm_layer)
        res = res*2
        dim = dim//2 # 4C
        self.fusion_merge3 = DualFusionBlockMerge(dim,res,2,4,4,8,attn_drop,drop_path,
                                                  qkv_bias,qk_scale,act_layer,norm_layer)
        self.fusion_merge4 = DualFusionBlockMerge(dim,res,2,4,4,8,attn_drop,drop_path,
                                                  qkv_bias,qk_scale,act_layer,norm_layer)
        res = res*2
        dim = dim//2 # 2C
        self.fusion_merge2 = DualFusionBlockMerge(dim,res,1,2,2,4,attn_drop,drop_path,
                                                  qkv_bias,qk_scale,act_layer,norm_layer)
        self.fusion_merge5 = DualFusionBlockMerge(dim,res,1,2,2,4,attn_drop,drop_path,
                                                  qkv_bias,qk_scale,act_layer,norm_layer)
        self.fusion_merge6 = DualFusionBlockMerge(dim,res,1,2,2,4,attn_drop,drop_path,
                                                  qkv_bias,qk_scale,act_layer,norm_layer)
        res = res*2
        dim = dim//2
        self.linear_pro = FinalPatchExpand_X4(res,dim)
        res = res*4
        self.conv = nn.Sequential(
            Rearrange("b (h w) c -> b c h w", h=res, w=res),
            nn.Conv2d(dim,dim_out,3,1,1)
        )

    def forward(self,skips):
        """
        Args:
            x : B L C
        Returns:
            x : B C H W
        """
        # for x in skips:
        #     print(x.shape)
        # exit(0)
        x1 = self.fusion_merge1(skips[-1], skips[-2])
        x2 = self.fusion_merge2(skips[-3], skips[-4])
        x3 = self.fusion_merge3(x1, skips[-3])
        x4 = self.fusion_merge4(x1,x3)
        x5 = self.fusion_merge5(x3,x2)
        x = self.fusion_merge6(x4,x5)
        # print(x.shape)
        # exit(0)
        x = self.linear_pro(x)
        # print(x.shape)
        # exit(0)
        x = self.conv(x)
        return x
        

        
class DAFN(nn.Module):
    def __init__(self, img_size=224, dim_in=1, dim_out=9, embed_dim=32,attn_drop=0.,drop_path=0.,
                 split_size=[1,2,2,7], num_heads=[2,4,8,16],qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.encoder = Encoder(img_size,dim_in,embed_dim,split_size,num_heads,
                               qkv_bias,qk_scale,attn_drop,drop_path, act_layer,norm_layer)
        res = self.encoder.final_res
        dim = self.encoder.final_dim
        split_size.reverse()
        num_heads.reverse()
        self.decoder = Decoder(res,dim,dim_out,attn_drop,drop_path,split_size,num_heads,
                               qkv_bias,qk_scale,act_layer,norm_layer)
        
    def forward(self,img):
        return self.decoder(self.encoder(img))


def main():
    x = torch.randn((2,1,224,224))
    model =  DAFN()
    y = model(x)
    # print(x.shape,y.shape)
    flops,params = profile(model,(x,))
    flops,params = clever_format([flops,params],'%.3f')
    
    print("flops per sample: ",flops)
    print("model parameters: ",params)

if __name__ == '__main__':
    main()

