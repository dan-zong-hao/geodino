import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import cv2
import torch.autograd as autograd
# from MedSAM.models.sam import sam_model_registry
from torchvision.transforms import v2
from transformers import AutoImageProcessor, AutoModel
import cfg as cfg
import matplotlib.pyplot as plt
import math
from lora_utils_test import apply_dual_lora_to_vit_encoder
from types import MethodType
try:
    from dinov3.layers.attention import SelfAttention
except ImportError:
    # 如果找不到路径，为了代码不报错，定义一个伪类（实际运行时请修正路径）
    SelfAttention = nn.Module

class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out

class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class WF_single(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF_single, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.post_conv(x)
        return x
    
class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y
    
class SEFusion(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SEFusion, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out


class FeatureRefinementHead_single(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # weights = nn.ReLU()(self.weights)
        # fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        # x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x
      
class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x

class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder_single(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder_single, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF_single(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF_single(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead_single(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x)
        x = self.b3(x)

        x = self.p2(x)
        x = self.b2(x)

        x = self.p1(x)
        
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)
        
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def draw_features(feature, savename=''):
    H = W = 256
    visualize = F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=False)
    visualize = visualize.detach().cpu().numpy()
    visualize = np.mean(visualize, axis=1).reshape(H, W)
    visualize = (((visualize - np.min(visualize)) / (np.max(visualize) - np.min(visualize))) * 255).astype(np.uint8)
    # fvis = np.fft.fft2(visualize)
    # fshift = np.fft.fftshift(fvis)
    # fshift = 20*np.log(np.abs(fshift))
    savedir = savename
    visualize = cv2.applyColorMap(visualize, cv2.COLORMAP_JET)
    cv2.imwrite(savedir, visualize)

# ==========================================
# 1. 几何先验生成器
# ==========================================
class GeometryPriorGenerator(nn.Module):
    """
    Test15 改进版 GeoPrior:
    1. 继承 Test14 的 MaxPool (Tree) + Roughness (Road) 特征
    2. 新增 Head Specialization: 前 N 个 Head 不加 Bias，保留 DINOv3 原生能力
    3. 新增 Dynamic Gating: Alpha 系数由 RGB 图像内容动态生成，而非静态参数
    """
    def __init__(self, input_size=256, patch_size=14, num_heads=16, 
                 frozen_heads=8): # 新增参数: 冻结前8个Head
        super().__init__()
        self.H_grid = input_size // patch_size
        self.W_grid = input_size // patch_size
        self.num_heads = num_heads
        self.frozen_heads = frozen_heads
        self.active_heads = num_heads - frozen_heads
        
        assert self.active_heads > 0, "必须至少有一个 Active Head"

        # --- A. 空间距离 (保持不变) ---
        y_idx = torch.arange(self.H_grid, dtype=torch.float32)
        x_idx = torch.arange(self.W_grid, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_idx, x_idx, indexing='ij')
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1) 
        spatial_dist = torch.cdist(coords, coords, p=1).unsqueeze(0).unsqueeze(0)
        self.register_buffer("spatial_dist", spatial_dist)

        # --- B. 动态门控网络 (Dynamic Gating Network) ---
        # 输入: RGB 图像 (B, 3, H, W)
        # 输出: (B, active_heads, 3) -> 对应 spatial, depth, rough 三个系数
        self.gating_net = nn.Sequential(
            # 快速下采样提取全局语义
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3), # -> 64x64
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=4, padding=1), # -> 16x16
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), # -> (B, 64, 1, 1) 全局池化
            nn.Flatten(),
            nn.Linear(64, self.active_heads * 3) # 输出每个 Active Head 的 3 个系数
        )
        
        # 初始化 Gating Net，使其初始输出接近 0 (Zero Init 策略的延续)
        # 最后一层 Linear 权重初始化为极小值
        nn.init.constant_(self.gating_net[-1].weight, 0.0)
        nn.init.constant_(self.gating_net[-1].bias, -3.0) # Sigmoid(-3) ≈ 0.04，初始给予很小的关注

    def forward(self, dsm_map, rgb_map):
        """
        dsm_map: [B, 1, H, W]
        rgb_map: [B, 3, H, W]  <-- 新增输入，用于生成动态权重
        """
        B = dsm_map.shape[0]
        
        # ==========================================
        # 1. 动态生成 Alpha 系数 (Dynamic Gating)
        # ==========================================
        # gating_logits: [B, active_heads * 3]
        gating_logits = self.gating_net(rgb_map) 
        
        # Reshape: [B, active_heads, 3, 1, 1]
        alphas = torch.sigmoid(gating_logits).view(B, self.active_heads, 3, 1, 1)
        
        # 分离系数 (不再是全局 Parameter，而是 Sample-dependent 的变量)
        alpha_spatial = alphas[:, :, 0] # [B, active_heads, 1, 1]
        alpha_depth   = alphas[:, :, 1]
        alpha_rough   = alphas[:, :, 2]

        # ==========================================
        # 2. DSM 特征提取 (Test14 逻辑: Max + Avg + Std)
        # ==========================================
        dsm_avg = F.adaptive_avg_pool2d(dsm_map, (self.H_grid, self.W_grid))
        dsm_max = F.adaptive_max_pool2d(dsm_map, (self.H_grid, self.W_grid))
        
        dsm_sq = dsm_map ** 2
        avg_sq = F.adaptive_avg_pool2d(dsm_sq, (self.H_grid, self.W_grid))
        dsm_std = torch.sqrt(torch.clamp(avg_sq - dsm_avg ** 2, min=1e-6))

        h_tokens = dsm_max.flatten(2).transpose(1, 2)
        r_tokens = dsm_std.flatten(2).transpose(1, 2)

        # ==========================================
        # 3. 计算 Bias 并融合
        # ==========================================
        depth_dist = torch.abs(h_tokens - h_tokens.transpose(1, 2)).unsqueeze(1) # [B, 1, N, N]
        rough_dist = torch.abs(r_tokens - r_tokens.transpose(1, 2)).unsqueeze(1)
        
        # 注意: spatial_dist 是 [1, 1, N, N], 但 alpha 是 [B, heads, 1, 1]
        # 广播机制会自动处理
        active_bias = - (alpha_spatial * self.spatial_dist + 
                         alpha_depth   * depth_dist +
                         alpha_rough   * rough_dist)
        
        # ==========================================
        # 4. Head Specialization (填充冻结的 Head)
        # ==========================================
        if self.frozen_heads > 0:
            # 构造全 0 的 Bias 给冻结的 Head
            # Shape: [B, frozen_heads, N, N]
            zeros = torch.zeros(B, self.frozen_heads, active_bias.shape[-2], active_bias.shape[-1], 
                                device=active_bias.device, dtype=active_bias.dtype)
            
            # 拼接: [B, 16, N, N]
            # 前 8 个是 0 (纯语义)，后 8 个是 Active Bias (几何感知)
            full_bias = torch.cat([zeros, active_bias], dim=1)
        else:
            full_bias = active_bias
            
        return full_bias

# ==========================================
# 2. Geometry Wrapper (Attention 包装器)
# ==========================================

class GeometryAwareAttention(nn.Module):
    def __init__(self, original_attn_layer):
        super().__init__()
        self.original_layer = original_attn_layer

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        geo_bias = getattr(self, 'geometry_bias', None)
        
        # [Test13 修复]: 显式过滤极小值，防止全0 Tensor 破坏 Flash Attention 优化
        if geo_bias is not None and geo_bias.abs().max() < 1e-6:
            geo_bias = None

        if geo_bias is not None:
            B, seq_len, _ = hidden_states.shape
            num_patches = geo_bias.shape[-1]
            
            # 处理 CLS/Register tokens
            if seq_len > num_patches:
                num_special_tokens = seq_len - num_patches
                if geo_bias.device != hidden_states.device:
                    geo_bias = geo_bias.to(hidden_states.device)
                geo_bias = F.pad(geo_bias, (num_special_tokens, 0, num_special_tokens, 0), value=0.0)
            
            if geo_bias.device != hidden_states.device:
                geo_bias = geo_bias.to(hidden_states.device)
            if geo_bias.dtype != hidden_states.dtype:
                geo_bias = geo_bias.to(dtype=hidden_states.dtype)

            # 注入 Bias
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    dtype_min = torch.finfo(hidden_states.dtype).min
                    base_mask = torch.zeros_like(geo_bias)
                    base_mask.masked_fill_(~attention_mask, dtype_min)
                    attention_mask = base_mask + geo_bias
                else:
                    attention_mask = attention_mask + geo_bias
            else:
                attention_mask = geo_bias

        return self.original_layer(
            hidden_states, 
            attention_mask=attention_mask, 
            head_mask=head_mask, 
            output_attentions=output_attentions, 
            **kwargs
        )
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)

def replace_attention_with_geometry_aware(model):
    modules_to_replace = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if ("Attention" in child.__class__.__name__ 
                and hasattr(child, "q_proj") 
                and not isinstance(child, GeometryAwareAttention)):
                modules_to_replace.append((name, module, child_name, child))

    print(f"🔍 扫描完成，共发现 {len(modules_to_replace)} 个目标 Attention 层。")
    count = 0
    for name, parent, child_name, child in modules_to_replace:
        if isinstance(getattr(parent, child_name), GeometryAwareAttention):
            continue
        wrapped_layer = GeometryAwareAttention(child)
        setattr(parent, child_name, wrapped_layer)
        count += 1
    print(f"✅ 成功替换了 {count} 个 Attention 层。")


# ==========================================
# 3. UNetFormer 主类
# ==========================================
class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6,
                 dinov3_model_name="/root/autodl-tmp/modelscope/hub/facebook/dinov3-vitl16-pretrain-sat493m",
                 freeze_backbone=True,
                 use_lora=True,
                 lora_rank=8, lora_alpha=16, lora_dropout=0.0,
                 lora_last_k=8,
                 lora_targets=('q_proj', 'k_proj', 'v_proj', 'o_proj')
                ):
        super().__init__()
        # args = cfg.parse_args() # 确保 cfg 已定义

        # 1) 加载 Backbone
        self.image_encoder = AutoModel.from_pretrained(dinov3_model_name, trust_remote_code=True)

        # 2) 优先应用 LoRA (关键顺序修改)
        # 必须在 Wrapping 之前应用 LoRA，以确保 LoRA 能找到原始的 Linear 层
        if use_lora:
            total_layers = 24  # ViT-L/16
            lora_layers = list(range(total_layers - lora_last_k, total_layers))
            apply_dual_lora_to_vit_encoder(
                self.image_encoder,
                layer_ids=lora_layers,
                targets=lora_targets,
                r=lora_rank, alpha=lora_alpha, dropout=lora_dropout
            )
            freeze_backbone = False # LoRA 模式下自动解冻 LoRA 参数

        # 3) 替换 Attention 为几何感知 Wrapper
        # 此时 Attention 内部可能已经是 DualLoRALinear 了，Wrapper 会将其视为黑盒包裹起来，不影响
        replace_attention_with_geometry_aware(self.image_encoder)

        # 4) 初始化几何先验生成器
        embed_dim = self.image_encoder.embed_dim if hasattr(self.image_encoder, "embed_dim") else 1024
        num_heads = self.image_encoder.num_heads if hasattr(self.image_encoder, "num_heads") else 16
        patch_size = self.image_encoder.patch_size if hasattr(self.image_encoder, "patch_size") else 16
        
        self.geo_gen = GeometryPriorGenerator(
            input_size=256, 
            patch_size=patch_size, 
            num_heads=num_heads
        )

        # 5) 冻结控制
        if freeze_backbone:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        
        # 确保几何先验参数可训练
        for p in self.geo_gen.parameters():
            p.requires_grad = True

        self.transform = self.make_transform(256)

        # 后续 FPN / Decoder 定义 (保持不变)
        dinov3_out = 1024
        out_c = 256
        encoder_channels = (out_c, out_c, out_c, out_c)
        self.layer_ids = [5, 11, 17, 23]
        
        self.proj_shared = nn.ModuleList([nn.Linear(dinov3_out, out_c) for _ in range(4)])
        self.bn_rgb = nn.ModuleList([nn.BatchNorm1d(out_c) for _ in range(4)])
        self.bn_dsm = nn.ModuleList([nn.BatchNorm1d(out_c) for _ in range(4)])

        self.fpn1x = nn.Sequential(nn.ConvTranspose2d(out_c, out_c, 2, 2), Norm2d(out_c), nn.GELU(), nn.ConvTranspose2d(out_c, out_c, 2, 2))
        self.fpn2x = nn.Sequential(nn.ConvTranspose2d(out_c, out_c, 2, 2))
        self.fpn3x = nn.Identity()
        self.fpn4x = nn.MaxPool2d(2, 2)

        self.fpn1y = nn.Sequential(nn.ConvTranspose2d(out_c, out_c, 2, 2), nn.BatchNorm2d(out_c), nn.GELU(), nn.ConvTranspose2d(out_c, out_c, 2, 2))
        self.fpn2y = nn.Sequential(nn.ConvTranspose2d(out_c, out_c, 2, 2))
        self.fpn3y = nn.Identity()
        self.fpn4y = nn.MaxPool2d(2, 2)

        self.fusion1 = SEFusion(out_c)
        self.fusion2 = SEFusion(out_c)
        self.fusion3 = SEFusion(out_c)
        self.fusion4 = SEFusion(out_c)
        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    @staticmethod
    def make_transform(resize_size: int = 256):
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
        ])

    def _strip_special_tokens(self, tokens):
        # 保持你原来的 strip 逻辑
        B, T, C = tokens.shape
        for k in (1, 5):
            N = T - k
            r = int(math.isqrt(N))
            if r * r == N: return tokens[:, k:, :], r
        for k in range(0, 10):
            N = T - k
            if N <= 0: break
            r = int(math.isqrt(N))
            if r * r == N: return tokens[:, k:, :], r
        raise ValueError(f"Cannot infer tokens: {tokens.shape}")

    def _tokens_to_map_with_bn(self, toks, proj, bn):
        toks, _ = self._strip_special_tokens(toks)
        x = proj(toks)
        B, N, C = x.shape
        x = bn(x.reshape(-1, C)).reshape(B, N, C)
        H = int(math.isqrt(N))
        return x.reshape(B, H, H, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x, y, mode='Train'):
        # 1. 预处理输入
        # DSM: (B, H, W) -> (B, 3, H, W) -> Transform
        y_in = self.transform(torch.unsqueeze(y, dim=1).repeat(1, 3, 1, 1))
        # RGB: Transform
        x = self.transform(x)

        # 2. 生成几何 Bias (仅供 RGB 分支使用)
        # y.unsqueeze(1) 变成 (B, 1, H, W) 以适配 AdaptiveAvgPool
        geo_bias = self.geo_gen(y.unsqueeze(1), x) 
        
        # 获取所有被包裹的 Attention 层
        att_layers = [m for m in self.image_encoder.modules() if isinstance(m, GeometryAwareAttention)]

        # ==========================================
        # [Pass 1] RGB 分支 (启用几何先验)
        # ==========================================
        # 1.1 挂载 Bias
        for m in att_layers:
            m.geometry_bias = geo_bias
        
        try:
            # 1.2 设置模态 & 前向传播
            if hasattr(self.image_encoder, 'set_modality'): 
                self.image_encoder.set_modality('rgb')
            
            # 此时 Attention 会读取 geo_bias 并注入计算
            out_x = self.image_encoder(x, output_hidden_states=True)
            
        finally:
            # 1.3 立即清理 Bias (关键步骤!)
            # 无论 RGB pass 是否成功，必须确保 Bias 被移除，以免污染后续操作
            for m in att_layers:
                m.geometry_bias = None

        # ==========================================
        # [Pass 2] DSM 分支 (保持原样，不使用几何先验)
        # ==========================================
        # 此时 m.geometry_bias 已经是 None 了
        # GeometryAwareAttention 会直接透传调用原始层，相当于没有任何修改
        if hasattr(self.image_encoder, 'set_modality'): 
            self.image_encoder.set_modality('dsm')
            
        out_y = self.image_encoder(y_in, output_hidden_states=True)

        # ==========================================
        # [后续] FPN, Fusion, Decoder (保持不变)
        # ==========================================
        feats_x, feats_y = [], []
        for i, lid in enumerate(self.layer_ids):
            fx = self._tokens_to_map_with_bn(out_x.hidden_states[lid], self.proj_shared[i], self.bn_rgb[i])
            fy = self._tokens_to_map_with_bn(out_y.hidden_states[lid], self.proj_shared[i], self.bn_dsm[i])
            feats_x.append(fx)
            feats_y.append(fy)

        res1x = self.fpn1x(feats_x[0])
        res2x = self.fpn2x(feats_x[1])
        res3x = self.fpn3x(feats_x[2])
        res4x = self.fpn4x(feats_x[3])

        res1y = self.fpn1y(feats_y[0])
        res2y = self.fpn2y(feats_y[1])
        res3y = self.fpn3y(feats_y[2])
        res4y = self.fpn4y(feats_y[3])

        align_loss = torch.tensor(0.0, device=x.device)
        # if hasattr(self, 'local_infonce'):
        #     for fx, fy in zip(feats_x, feats_y):
        #         align_loss += self.local_infonce(fx, fy)
        #     align_loss /= len(feats_x)

        res1 = self.fusion1(res1x, res1y)
        res2 = self.fusion2(res2x, res2y)
        res3 = self.fusion3(res3x, res3y)
        res4 = self.fusion4(res4x, res4y)

        out = self.decoder(res1, res2, res3, res4, x.size(-2), x.size(-1))
        
        return out, align_loss

