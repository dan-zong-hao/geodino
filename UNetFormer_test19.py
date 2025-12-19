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

class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6,
                 dinov3_model_name="/root/autodl-tmp/modelscope/hub/facebook/dinov3-vitl16-pretrain-sat493m",
                 freeze_backbone=True,
                 use_lora=True,        # 新增：是否启用 LoRA
                 lora_rank=8, lora_alpha=16, lora_dropout=0.0,
                 lora_last_k=8,        # 新增：对最后 k 层注入 LoRA，ViT-L 一共 24 层→默认最后 8 层
                 lora_targets=('q_proj', 'k_proj', 'v_proj', 'o_proj'),  # 默认只在注意力的 Q/V 上做 LoRA
                 use_MMST=False
                ):
        super().__init__()
        args = cfg.parse_args()

        # 1) backbone & 预处理
        self.use_MMST = use_MMST
        self.image_encoder = AutoModel.from_pretrained(dinov3_model_name)
        if use_lora:
            total_layers = 24  # ViT-L/16
            lora_layers = list(range(total_layers - lora_last_k, total_layers))  # 例如 [16..23]
            apply_dual_lora_to_vit_encoder(
                self.image_encoder,
                layer_ids=lora_layers,
                targets=lora_targets,
                r=lora_rank, alpha=lora_alpha, dropout=lora_dropout
            )
            # 覆盖 freeze_backbone：LoRA 会自动只训练 lora_A/B
            freeze_backbone = False
        self.transform = self.make_transform(256)

        dinov3_out = 1024
        out_c = 256
        encoder_channels = (out_c, out_c, out_c, out_c)

        # 2) 线性投影（按层），把 token 维 1024 → 256
        #    选更分散的层位：ViT-L/16 共24层(0..23)，这里用 [2,6,12,23]
        self.layer_ids = [5, 11, 17, 23]
        # self.projections = nn.ModuleList([nn.Linear(dinov3_out, out_c) for _ in range(4)])
        # 2 套线性投影（RGB / DSM 各自独立）
        # self.proj_rgb = nn.ModuleList([nn.Linear(dinov3_out, out_c) for _ in range(4)])
        # self.proj_dsm = nn.ModuleList([nn.Linear(dinov3_out, out_c) for _ in range(4)])
        # 共享 Linear + 独立 BatchNorm
        self.proj_shared = nn.ModuleList([nn.Linear(dinov3_out, out_c) for _ in range(4)])
        self.bn_rgb = nn.ModuleList([nn.BatchNorm1d(out_c) for _ in range(4)])
        self.bn_dsm = nn.ModuleList([nn.BatchNorm1d(out_c) for _ in range(4)])


        # 3) 你原有的 FPN 分支（不改）
        self.fpn1x = nn.Sequential(
            nn.ConvTranspose2d(out_c, out_c, kernel_size=2, stride=2),  # 16->32
            Norm2d(out_c),
            nn.GELU(),
            nn.ConvTranspose2d(out_c, out_c, kernel_size=2, stride=2),  # 32->64
        )
        self.fpn2x = nn.Sequential(
            nn.ConvTranspose2d(out_c, out_c, kernel_size=2, stride=2),  # 16->32
        )
        self.fpn3x = nn.Identity()                                     # 16->16
        self.fpn4x = nn.MaxPool2d(kernel_size=2, stride=2)             # 16->8

        self.fpn1y = nn.Sequential(
            nn.ConvTranspose2d(out_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.ConvTranspose2d(out_c, out_c, kernel_size=2, stride=2),
        )
        self.fpn2y = nn.Sequential(
            nn.ConvTranspose2d(out_c, out_c, kernel_size=2, stride=2),
        )
        self.fpn3y = nn.Identity()
        self.fpn4y = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4) 融合 & 解码（保持不变）
        self.fusion1 = SEFusion(out_c)
        self.fusion2 = SEFusion(out_c)
        self.fusion3 = SEFusion(out_c)
        self.fusion4 = SEFusion(out_c)
        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

        self.decoder_rgb  = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)
        self.decoder_dsm  = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)


        # 5) 可选冻结
        if freeze_backbone:
            for _, p in self.image_encoder.named_parameters():
                p.requires_grad = False

    @staticmethod
    def make_transform(resize_size: int = 256):
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
        ])
    # def make_transform(resize_size: int = 256):
    #     to_tensor = v2.ToImage()
    #     resize = v2.Resize((resize_size, resize_size), antialias=True)
    #     to_float = v2.ToDtype(torch.float32, scale=True)
    #     normalize = v2.Normalize(
    #         mean=(0.485, 0.456, 0.406),
    #         std=(0.229, 0.224, 0.225),
    #     )
    #     return v2.Compose([to_tensor, resize, to_float, normalize])

    @staticmethod
    def _strip_special_tokens(tokens):
        """
        tokens: [B, T, C]，尝试去掉 1 或 5 个特殊 token（CLS [+ 4 register]）
        使得剩余 N 为完全平方数 -> 可 reshape 成 HxW
        """
        B, T, C = tokens.shape
        for k in (1, 5):
            N = T - k
            r = int(math.isqrt(N))
            if r * r == N:
                return tokens[:, k:, :], r
        for k in range(0, 10):
            N = T - k
            if N <= 0: break
            r = int(math.isqrt(N))
            if r * r == N:
                return tokens[:, k:, :], r
        raise ValueError(f"Cannot infer special token count from tokens shape {tokens.shape}")


    def _tokens_to_map_with_bn(self, toks, proj, bn):
        toks, _ = self._strip_special_tokens(toks)  # [B,N,C]
        x = proj(toks)                              # [B,N,256]
        # BN 是针对特征维做的，这里 reshape [B*N,256]
        B, N, C = x.shape
        x = bn(x.reshape(-1, C)).reshape(B, N, C)
        H = int(math.isqrt(N))
        x = x.reshape(B, H, H, C).permute(0, 3, 1, 2).contiguous()  # [B,256,H,W]
        return x

    @staticmethod
    def local_infonce(fx, fy, tau=0.07, num_neg=256):
        assert fx.dim() == 4 and fy.dim() == 4, f"{fx.shape=}, {fy.shape=}"
        B, C, H, W = fx.shape
        fx = F.normalize(fx.flatten(2).transpose(1,2), dim=-1)  # [B,HW,C]
        fy = F.normalize(fy.flatten(2).transpose(1,2), dim=-1)
        
        sim_matrix = torch.bmm(fx, fy.transpose(1,2)) / tau      # [B,HW,HW]
        pos = torch.diagonal(sim_matrix, dim1=1, dim2=2)         # [B,HW]
        # 负样本来自同图像的不同位置
        loss = -torch.log(torch.exp(pos) / torch.exp(sim_matrix).sum(-1))
        return loss.mean()
    # @staticmethod
    # def local_infonce(fx, fy, tau=0.07, num_neg=256):
    #     assert fx.dim() == 4 and fy.dim() == 4, f"{fx.shape=}, {fy.shape=}"
    #     B, C, H, W = fx.shape
    #     fx = F.normalize(fx.flatten(2).transpose(1, 2), dim=-1)  # [B,HW,C]
    #     fy = F.normalize(fy.flatten(2).transpose(1, 2), dim=-1)

    #     sim = torch.bmm(fx, fy.transpose(1, 2)) / tau            # [B,HW,HW]
    #     pos = sim.diagonal(dim1=1, dim2=2)                       # [B,HW]

    #     # 稳定版：-log( exp(pos)/sum exp(sim) ) == -(pos - logsumexp(sim))
    #     loss = -(pos - sim.logsumexp(dim=-1))                    # [B,HW]
    #     return loss.mean()

    def forward(self, x, y, mode='Train'):
        H_img, W_img = x.size()[-2:]
        # DSM 1ch -> 3ch
        y = torch.unsqueeze(y, dim=1).repeat(1, 3, 1, 1)

        x = self.transform(x)
        y = self.transform(y)

        # 取 hidden_states
        self.image_encoder.set_modality('rgb')
        out_x = self.image_encoder(x, output_hidden_states=True)
        self.image_encoder.set_modality('dsm')
        out_y = self.image_encoder(y, output_hidden_states=True)

        # 5) 四层：变成四个 [B,256,16,16] 的 feature map
        # feats_x = []
        # feats_y = []
        # for i, lid in enumerate(self.layer_ids):
        #     fx = self._tokens_to_map(out_x.hidden_states[lid], self.proj_rgb[i])
        #     fy = self._tokens_to_map(out_y.hidden_states[lid], self.proj_dsm[i])
        #     feats_x.append(fx)
        #     feats_y.append(fy)
        feats_x, feats_y = [], []
        for i, lid in enumerate(self.layer_ids):
            fx = self._tokens_to_map_with_bn(out_x.hidden_states[lid],
                                            self.proj_shared[i], self.bn_rgb[i])
            fy = self._tokens_to_map_with_bn(out_y.hidden_states[lid],
                                            self.proj_shared[i], self.bn_dsm[i])
            feats_x.append(fx)
            feats_y.append(fy)

        # 6) 走你原来的 FPN 分支：输入均为 [B,256,16,16]
        #    输出分别是 [64,64] / [32,32] / [16,16] / [8,8]
        res1x = self.fpn1x(feats_x[0])
        res2x = self.fpn2x(feats_x[1])
        res3x = self.fpn3x(feats_x[2])
        res4x = self.fpn4x(feats_x[3])

        res1y = self.fpn1y(feats_y[0])
        res2y = self.fpn2y(feats_y[1])
        res3y = self.fpn3y(feats_y[2])
        res4y = self.fpn4y(feats_y[3])

        # align_loss = 0.0
        # for fx, fy in zip(feats_x, feats_y):
        #     # 归一化后计算像素级余弦差异
        #     fx_norm = F.normalize(fx, dim=1)
        #     fy_norm = F.normalize(fy, dim=1)
        #     # 1 - cosine_similarity ∈ [0,2]
        #     cos_sim = torch.sum(fx_norm * fy_norm, dim=1, keepdim=True)
        #     align_loss += torch.mean(1.0 - cos_sim)
        # align_loss = align_loss / len(feats_x)  # 平均到4层
        align_loss = 0.0
        for fx, fy in zip(feats_x, feats_y):
            align_loss += self.local_infonce(fx, fy)
        align_loss /= len(feats_x)

        # 7) 融合（与你原先一致）
        res1 = self.fusion1(res1x, res1y)  # [B,256,64,64]
        res2 = self.fusion2(res2x, res2y)  # [B,256,32,32]
        res3 = self.fusion3(res3x, res3y)  # [B,256,16,16]
        res4 = self.fusion4(res4x, res4y)  # [B,256, 8, 8]

        # 8) 解码回原图大小
        out = self.decoder(res1, res2, res3, res4, H_img, W_img)
        if self.use_MMST and mode == "Train":
            aux_rgb  = self.decoder_rgb (res1x, res2x, res3x, res4x, H_img, W_img)
            aux_dsm  = self.decoder_dsm (res1y, res2y, res3y, res4y, H_img, W_img)
            return out, aux_rgb, aux_dsm, align_loss
        return out, align_loss
