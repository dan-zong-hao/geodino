# lora_utils.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 双模态 LoRA ----------
class DualLoRALinear(nn.Module):
    """
    同一主干权重共享, 但为 rgb / dsm 各自维护一套 LoRA(A,B)
    模态切换通过 self.modality 控制 ('rgb' or 'dsm')
    """
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # 冻结主干
        self.weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        self.bias = None
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data.clone(), requires_grad=False)

        # 两套 LoRA 参数
        def make_lora():
            A = nn.Parameter(torch.zeros(self.r, self.in_features))
            B = nn.Parameter(torch.zeros(self.out_features, self.r))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.zeros_(B)
            return A, B

        self.lora_A_rgb, self.lora_B_rgb = make_lora()
        self.lora_A_dsm, self.lora_B_dsm = make_lora()
        self.modality = "rgb"  # 默认使用 RGB 分支

    def set_modality(self, modality: str):
        assert modality in ("rgb", "dsm")
        self.modality = modality

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        x_d = self.dropout(x)
        if self.modality == "rgb":
            A, B = self.lora_A_rgb, self.lora_B_rgb
        else:
            A, B = self.lora_A_dsm, self.lora_B_dsm
        lora_update = F.linear(F.linear(x_d, A), B) * self.scaling
        return out + lora_update


# ---------- 工具函数 ----------
def _replace_linear_with_dual_lora(module: nn.Module, path: str, r=8, alpha=16, dropout=0.0):
    """将 module 下给定路径的 nn.Linear 替换为 DualLoRALinear"""
    parts = path.split('.')
    cur = module
    for p in parts[:-1]:
        cur = getattr(cur, p) if not p.isdigit() else cur[int(p)]
    last = parts[-1]
    base_linear = getattr(cur, last)
    assert isinstance(base_linear, nn.Linear), f"{path} 不是 nn.Linear"
    setattr(cur, last, DualLoRALinear(base_linear, r=r, alpha=alpha, dropout=dropout))


def apply_dual_lora_to_vit_encoder(encoder: nn.Module,
                                   layer_ids,
                                   targets=('q_proj', 'v_proj'),
                                   r=8, alpha=16, dropout=0.0):
    """
    对 ViT 的若干层注入双模态 LoRA
    - 与原 apply_lora_to_vit_encoder 类似，但替换成 DualLoRALinear
    - 注入后 encoder 将拥有 encoder.set_modality('rgb'/'dsm')
    """
    # 冻结主干
    for p in encoder.parameters():
        p.requires_grad = False

    # 注入双 LoRA
    for lid in layer_ids:
        for t in targets:
            path = f"layer.{lid}.attention.{t}"
            _replace_linear_with_dual_lora(encoder, path, r=r, alpha=alpha, dropout=dropout)

    # 仅解冻 LoRA 参数
    for n, p in encoder.named_parameters():
        if 'lora_A' in n or 'lora_B' in n:
            p.requires_grad = True

    # 模态切换接口
    def set_modality(modality: str):
        for m in encoder.modules():
            if isinstance(m, DualLoRALinear):
                m.set_modality(modality)
    encoder.set_modality = set_modality