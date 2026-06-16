import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft as fft



class FFCM(nn.Module):
    def __init__(self, in_channels,out_channels, mid_channels=48):
        super().__init__()
        self.p_conv = PConv(in_channels, mid_channels)
        self.split = lambda x: torch.chunk(x, 2, dim=1)  
        
        # 3×3 分支
        self.branch3x3 = nn.Sequential(
            DConv(mid_channels//2, kernel_size=3),
            nn.GELU(),
            PConv(mid_channels//2, mid_channels//2)
        )
        
        # 5×5 分支
        self.branch5x5 = nn.Sequential(
            DConv(mid_channels//2, kernel_size=5),
            nn.GELU(),
            PConv(mid_channels//2, mid_channels//2)
        )
        
        # 频域融合
        self.freq_fusion = FrequencyFusion(mid_channels)
        self.out_conv = PConv(mid_channels, out_channels)
    
    def forward(self, x):
       
        x = self.p_conv(x)  
        
        x3, x5 = self.split(x)  
        x3 = self.branch3x3(x3)
        x5 = self.branch5x5(x5)
        x = torch.cat([x3, x5], dim=1)  # 分支融合 (B, C, H, W)
        
        x_freq = self.freq_fusion(x)   # 频域处理后的空域特征
        x = x + x_freq 

        x = self.out_conv(x)
        return x



class DDI2D(nn.Module):

    def __init__(self, dim, patch_size=(2, 2), mlp_ratio=2.0):

        super().__init__()
        self.dim = dim
        self.patch_size = patch_size  # (ph, pw) 二维Patch大小
        ph, pw = patch_size
        hidden_dim = int(dim * mlp_ratio)
        

        self.channel_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x  
        ph, pw = self.patch_size
        

        assert H % ph == 0 and W % pw == 0, 
        num_patches_h = H // ph  
        num_patches_w = W // pw  
        
        x_patched = x.reshape(B, C, num_patches_h, ph, num_patches_w, pw)
        x_patched = x_patched.permute(0, 2, 4, 3, 5, 1).contiguous()
        
        Np = num_patches_h * num_patches_w

        
        

        x_patched = x_mlp2.reshape(B, num_patches_h, num_patches_w, ph, pw, C)
        x_merged = x_patched.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        
        scale = self.channel_scale(x_merged)  
        x_scaled = x_merged * scale      
        

        out = x_scaled + residual
        return out

class Att_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.att = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.Sigmoid())

    def forward(self, x):
        att = self.att(x)
        x = x * att
        return x


class Fuse_block(nn.Module):
    def __init__(self, in_channels): 
        super().__init__()
      
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels//2, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):  
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x



class ResidualSaliencyFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
       
        self.residual_gen = nn.Sequential(
            nn.Conv2d(channels+1, channels, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        
    def forward(self, x, saliency):
       
        concat = torch.cat([x, saliency], dim=1)
        residual = self.residual_gen(concat)
        
        fused = x + residual
        return fused

class Fuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = 8
        self.ir_embed = nn.Sequential(nn.Conv2d(1, self.channel, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.vi_embed = nn.Sequential(nn.Conv2d(1, self.channel, kernel_size=3, stride=1, padding=1), nn.ReLU())
        
        self.ffcm1=FFCM(in_channels=self.channel, out_channels=self.channel)
        self.ffcm2=FFCM(in_channels=self.channel, out_channels=self.channel)
        self.residual_fusion_ir = ResidualSaliencyFusion(self.channel)
        # self.ddi=DDI2D(self.channel)
        # self.eca1=ECA(self.channel)
        # SELF.eca2=ECA(self.channel)
        # self.ddi2=DDI2D(self.channel)
        self.deca=DECA(self.channel)
        self.fus_block = Fuse_block(self.channel *3)
    
    def forward(self, ir, vi, mask):
        ir = self.ir_embed(ir)
        vi = self.vi_embed(vi)
        # ir=self.eca1(ir)
        # vi=self.eca2(vi)
        # t = ir+vi
        # sali_ir = self.residual_fusion_ir(ir,mask)
        t=self.deca(ir,vi)
        fre_ir=self.ffcm1(ir)
        fre_ir = self.residual_fusion_ir(fre_ir,mask)
        fre_vi=self.ffcm2(vi)
        # t = self.ddi(t)
        # vi = self.ddi2(vi)
        fus = torch.cat([fre_ir,fre_vi,t], dim=1)
        fus = self.fus_block(fus)
        fus = (fus - torch.min(fus)) / (torch.max(fus) - torch.min(fus))

        return fus