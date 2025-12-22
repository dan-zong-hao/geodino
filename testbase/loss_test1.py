import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# -----------------------
# 1. Focal Loss (保持不变)
# -----------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: [B, C, H, W], targets: [B, H, W]
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha, ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        valid_mask = (targets != self.ignore_index).float()
        return (focal_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)


# -----------------------
# 2. Dice Loss (保持不变)
# -----------------------
class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_1h = F.one_hot(targets.clamp(0, num_classes-1), num_classes).permute(0, 3, 1, 2).float()
        
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
        probs = probs * valid_mask
        targets_1h = targets_1h * valid_mask

        inter = (probs * targets_1h).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets_1h.sum(dim=(2,3))
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()


# -----------------------
# 3. Boundary Loss (已注释弃用)
# -----------------------
# 原因：原实现中使用 argmax 导致梯度截断，无法反向传播。
# 如需使用边缘损失，建议使用 InverseForm 或直接对 Softmax 概率图做 Sobel 计算。
# 这里暂时保留类定义以便兼容旧代码引用，但在 CombinedLoss 中不再调用。
class SobelEdge(nn.Module):
    def __init__(self):
        super(SobelEdge, self).__init__()
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
        self.register_buffer('weight_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('weight_y', sobel_y.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        x = x.float()
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        grad = torch.sqrt(grad_x**2 + grad_y**2)
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)
        return grad

class BoundaryLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(BoundaryLoss, self).__init__()
        self.sobel = SobelEdge()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # 警告：此处的 argmax 导致梯度无法回传
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(1).float().unsqueeze(1) 
        targets = targets.float().unsqueeze(1)
        mask = (targets != self.ignore_index).float()

        edge_pred = self.sobel(preds) * mask
        edge_gt = self.sobel(targets) * mask

        return F.l1_loss(edge_pred, edge_gt)


# -----------------------
# 4. 组合损失 (CombinedLoss) - 修改版
# -----------------------
class CombinedLoss(nn.Module):
    def __init__(self, ignore_index=255, use_aux=False):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        # 移除 Boundary Loss
        # self.boundary = BoundaryLoss(ignore_index=ignore_index) 
        self.use_aux = use_aux

    def forward(self, outputs, targets):
        """
        outputs: Tensor [B, C, H, W] 或 dict {'main': Tensor, 'aux': Tensor}
        targets: Tensor [B, H, W]
        """
        loss_total = 0.0
        loss_dict = {}
        pred_main = outputs

        # 处理 dict 情况 (如果有辅助头)
        if isinstance(outputs, dict):
            pred_main = outputs.get('main')
            if self.use_aux and 'aux' in outputs:
                loss_aux = self.focal(outputs['aux'], targets) + self.dice(outputs['aux'], targets)
                loss_total += 0.4 * loss_aux
        
        # 计算主损失 (Focal + Dice)
        # 权重分配：0.5 Focal + 0.5 Dice (均衡配置)
        loss_focal = self.focal(pred_main, targets)
        loss_dice = self.dice(pred_main, targets)
        loss_dict['focal'] = loss_focal.item()
        loss_dict['dice'] = loss_dice.item()

        loss_main = 0.5 * loss_focal + 0.5 * loss_dice
        loss_total += loss_main

        return loss_total, loss_dict