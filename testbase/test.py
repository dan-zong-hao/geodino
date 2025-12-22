import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from typing import Tuple
import cv2
import numpy as np
import os
import glob
from pathlib import Path

# --- 配置区域 ---
INPUT_H = 256  # 目标高度 (输入图片大小)
INPUT_W = 256  # 目标宽度
STRIDE = 16    # 下采样倍率 (通常对应Transformer的Patch Size)

# 特征图尺寸 (16x16)
FEAT_H = INPUT_H // STRIDE
FEAT_W = INPUT_W // STRIDE

# 文件夹路径配置 (请修改为你实际的文件夹路径)
RGB_DIR = '/home/csf1/Documents/dataset/seg/Hunan/images_png'      # RGB文件夹路径
DEPTH_DIR = '/home/csf1/Documents/dataset/seg/Hunan/dsm_pngs'  # Depth文件夹路径
OUTPUT_DIR = './output_vis'                # 结果保存路径
# ----------------

class GeoPrior(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, initial_value=2, heads_range=6):
        super().__init__()
        # 生成角度编码
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value  
        self.heads_range = heads_range 
        self.num_heads = num_heads
        # 生成衰减参数
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        
    def generate_pos_decay(self, H: int, W: int):
        '''生成位置衰减掩码'''
        index_h = torch.arange(H).to(self.decay) 
        index_w = torch.arange(W).to(self.decay) 
        # indexing='ij' 确保网格方向正确
        grid = torch.meshgrid([index_h, index_w], indexing='ij') 
        grid = torch.stack(grid, dim=-1).reshape(H*W, 2) 
        mask = grid[:, None, :] - grid[None, :, :] 
        mask = (mask.abs()).sum(dim=-1)
        return mask
    
    def generate_2d_depth_decay(self, H: int, W: int, depth_grid):
        '''生成深度衰减掩码'''
        B,_,H_in,W_in = depth_grid.shape
        grid_d = depth_grid.reshape(B, H*W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None,:, :] 
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) 
        return mask_d
    
    def forward(self, slen: Tuple[int], depth_map, activate_recurrent=False, chunkwise_recurrent=False):
        # 将深度图下采样到特征图尺寸
        depth_map = F.interpolate(depth_map, size=slen, mode='bilinear', align_corners=False)
        depth_map = depth_map.float()
        
        index = torch.arange(slen[0]*slen[1]).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :]) 
        sin = sin.reshape(slen[0], slen[1], -1) 
        cos = torch.cos(index[:, None] * self.angle[None, :]) 
        cos = cos.reshape(slen[0], slen[1], -1) 
        
        mask_1 = self.generate_pos_decay(slen[0], slen[1]) 
        mask_d = self.generate_2d_depth_decay(slen[0], slen[1], depth_map)
        
        mask = mask_d 
        # 融合位置编码和深度编码
        mask_sum = (0.85*mask_1.cuda()+0.15*mask) * self.decay[:, None, None].cuda()
        retention_rel_pos = ((sin, cos), mask, mask_1, mask_sum)

        return retention_rel_pos

def fangda(mask, in_size, out_size):
    """
    将低分辨率的 Attention Mask 放大到原图尺寸
    """
    # 增加维度以使用 interpolate: (H, W) -> (1, 1, H, W)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    
    # 使用 nearest 插值保持像素块状，如果想要平滑效果可改用 bilinear
    new_mask = F.interpolate(mask, size=out_size, mode='nearest')
    return new_mask.squeeze()

def put_mask(image, mask, out_size, color_rgb=None, border_mask=False, color_temp='jet', num_c='', beta=2, fixed_num=None):
    """
    将 Attention Mask 叠加到 RGB 图像上
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    target_w, target_h = out_size[1], out_size[0]
    
    # 确保尺寸一致
    image = cv2.resize(image, dsize=(target_w, target_h), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, dsize=(target_w, target_h), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    
    color = np.zeros((1,1,3), dtype=np.uint8)
    if color_rgb is not None:
        color[0,0,2], color[0,0,1], color[0,0,0] = color_rgb
    else:
        color[0, 0, 2], color[0, 0, 1], color[0, 0, 0] = 120, 86, 87

    # 简单的归一化处理
    if fixed_num is not None:
        mask = ((1-mask/255))
    else:
        max_val = np.max(mask)
        if max_val == 0: max_val = 1
        mask = (1 - mask / max_val)

    # 使用 torchcam 库进行热力图叠加 (如无此库需自行实现 addWeighted)
    from torchcam.utils import overlay_mask
    result = overlay_mask(to_pil_image(image.astype(np.uint8)), to_pil_image(mask), colormap=color_temp, alpha=0.4)

    return np.array(result)

def process_single_image(rgb_path, depth_path, model, output_dir, file_id):
    """
    处理单对图片
    """
    # 读取图像
    img = cv2.imread(rgb_path)
    grid_d_original = cv2.imread(depth_path, 0) # 以灰度模式读取深度图
    
    if img is None:
        print(f"Error reading RGB: {rgb_path}")
        return
    if grid_d_original is None:
        print(f"Error reading Depth: {depth_path}")
        return

    # 1. Resize RGB 到输入尺寸 (256x256)
    img = cv2.resize(img, dsize=(INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    
    # 2. Resize Depth 到特征图尺寸 (16x16) 用于计算 Attention
    grid_d = cv2.resize(grid_d_original, dsize=(FEAT_W, FEAT_H), interpolation=cv2.INTER_LINEAR)
    
    # 转换为 Tensor 格式: (B, C, H, W)
    grid_d_tensor = torch.tensor(grid_d).reshape(1, 1, FEAT_H, FEAT_W).float().cuda()
    
    # 准备可视化用的 Depth 图像 (Resize 到 256x256 以便拼接)
    grid_d_vis = cv2.imread(depth_path)
    grid_d_vis = cv2.resize(grid_d_vis, dsize=(INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)

    # 模型前向计算
    with torch.no_grad():
        ((sin,cos), depth_map, mask_1, mask_sum) = model((FEAT_H, FEAT_W), grid_d_tensor)

    # 动态计算中心点索引 (防止硬编码越界)
    # 选取特征图正中心的点作为 Query Point
    center_idx = (FEAT_H // 2) * FEAT_W + (FEAT_W // 2)
    indices_to_show = [center_idx] 

    def normalize_mask(m):
        m_min, m_max = torch.min(m), torch.max(m)
        if m_max - m_min == 0: return m
        return 255 * (m - m_min) / (m_max - m_min)

    # 生成可视化结果
    for idx in indices_to_show:
        # 获取 Attention Masks
        temp_mask_d = depth_map[0,0,idx,:].reshape(FEAT_H, FEAT_W).cpu() # 深度相关 Attention
        temp_mask = mask_1[idx,:].reshape(FEAT_H, FEAT_W).cpu() # 位置相关 Attention
        
        # 归一化
        temp_mask_d = torch.nn.functional.normalize(temp_mask_d, p=2.0, dim=1, eps=1e-12)
        temp_mask_d = normalize_mask(temp_mask_d)
        temp_mask = normalize_mask(temp_mask)
        
        gama = 0.55
        
        # 定义放大参数
        in_sz = (FEAT_H, FEAT_W)
        out_sz = (INPUT_H, INPUT_W)
        
        # 放大 Mask 到 256x256
        mask_a0 = fangda(temp_mask, in_sz, out_sz) # 仅位置
        mask_a2 = fangda(temp_mask_d, in_sz, out_sz) # 仅深度
        mask_a3 = fangda(gama*temp_mask + (1-gama)*temp_mask_d, in_sz, out_sz) # 融合
        
        color_temp = 'jet_r'
        
        # 叠加 Mask 到 RGB 图像上
        a0 = put_mask(img, mask_a0, out_size=out_sz, color_temp=color_temp)
        a2 = put_mask(img, mask_a2, out_size=out_sz, color_temp=color_temp)
        a3 = put_mask(img, mask_a3, out_size=out_sz, color_temp=color_temp)
        
        # 拼接白色间隔条
        jiange = np.ones((INPUT_H, 10, 3), dtype=np.uint8) * 255
        
        # 拼接最终图像: RGB | Depth | Pos Attention | Depth Attention | Fused Attention
        image_concat = np.concatenate([img, grid_d_vis, jiange, a0, jiange, a2, jiange, a3], axis=1)
        
        save_name = f'{file_id}_vis.png'
        cv2.imwrite(os.path.join(output_dir, save_name), image_concat)
        print(f"Saved: {os.path.join(output_dir, save_name)}")

def main():
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    model = GeoPrior().cuda()
    model.eval()
    
    # 自动获取 RGB 文件夹下的所有图片
    extensions = ['*.jpg', '*.png', '*.jpeg']
    rgb_files = []
    for ext in extensions:
        rgb_files.extend(glob.glob(os.path.join(RGB_DIR, ext)))
    rgb_files = sorted(rgb_files)
    
    if len(rgb_files) == 0:
        print(f"Warning: No images found in {RGB_DIR}")
        return
    else:
        print(f"Found {len(rgb_files)} images. Starting processing...")
    
    for rgb_path in rgb_files:
        path_obj = Path(rgb_path)
        file_stem = path_obj.stem # 获取不带后缀的文件名 (例如 '1')
        file_ext = path_obj.suffix # 获取后缀 (例如 '.jpg')
        
        # 自动匹配 Depth 文件
        # 1. 尝试完全同名 (如 1.jpg -> 1.jpg)
        depth_path = os.path.join(DEPTH_DIR, file_stem + file_ext)
        
        # 2. 如果不存在，尝试 .png (常见的深度图格式)
        if not os.path.exists(depth_path):
             depth_path = os.path.join(DEPTH_DIR, file_stem + '.png')
             
        # 3. 如果还不存在，尝试 .jpg
        if not os.path.exists(depth_path):
             depth_path = os.path.join(DEPTH_DIR, file_stem + '.jpg')

        if os.path.exists(depth_path):
            process_single_image(rgb_path, depth_path, model, OUTPUT_DIR, file_stem)
        else:
            print(f"Skipping {file_stem}: Depth file not found.")

if __name__ == '__main__':
    main()