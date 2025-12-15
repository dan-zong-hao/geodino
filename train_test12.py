import numpy as np
from glob import glob
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
import time
import cv2
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils_1 import *
from torch.autograd import Variable
from IPython.display import clear_output
# from UNetFormer_DINO import UNetFormer2 as MFNet
from UNetFormer_test12 import UNetFormer as MFNet
from loss_test import CombinedLoss
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener
import math
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed):
    random.seed(seed)                 # 控制 ISPRS_dataset 里用到的 random.randint / random.random
    np.random.seed(seed)              # 以后如果有 numpy 随机也会受控
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN 相关：尽量走确定性实现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
SEED = 666
set_seed(SEED)
# 新增：TensorBoard 日志
writer = SummaryWriter(log_dir="./runs/unetformer_test12")

net = MFNet(num_classes=N_CLASSES, dinov3_model_name="/home/csf1/modelscope/models/facebook/dinov3-vitl16-pretrain-lvd1689m", lora_last_k=24).cuda()

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('All Params:   ', params)

params1 = 0
params2 = 0

# 统计整个图像编码器的参数数量
for name, param in net.image_encoder.named_parameters():
    params1 += param.nelement()

# 统计图像编码器中可训练参数的数量（LoRA参数）
for name, param in net.image_encoder.named_parameters():
    if param.requires_grad:
        params2 += param.nelement()

print('ImgEncoder:   ', params1)
print('Lora:         ', params2)
print('Others:       ', params - params1)

# for name, parms in net.named_parameters():
#     print('%-50s' % name, '%-30s' % str(parms.shape), '%-10s' % str(parms.nelement()))

# params = 0
# for name, param in net.sam.prompt_encoder.named_parameters():
#     params += param.nelement()
# print('prompt_encoder: ', params)

# params = 0
# for name, param in net.sam.mask_decoder.named_parameters():
#     params += param.nelement()
# print('mask_decoder: ', params)

# print(net)

print("training : ", len(train_ids))
print("testing : ", len(test_ids))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

# base_lr = 1e-3
# optimizer = torch.optim.AdamW(
#     net.parameters(),
#     lr=base_lr,
#     betas=(0.9, 0.999),
#     weight_decay=0.01
# )

# # Cosine + Warmup
# warmup_epochs = 5
# def warmup_lambda(epoch):
#     if epoch < warmup_epochs:
#         return float(epoch) / float(max(1, warmup_epochs))
#     return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
# # 训练脚本里（替换你现有的 optimizer 构造）
# def is_lora_param(n):  # 名字里包含 lora_A/lora_B 的就是 LoRA
#     return ('lora_A' in n) or ('lora_B' in n)

# lora_params, head_params = [], []
# for n, p in net.named_parameters():
#     if not p.requires_grad:
#         continue
#     if is_lora_param(n):
#         lora_params.append(p)
#     else:
#         head_params.append(p)

# # 建议 AdamW，LoRA 稍微更大的 lr
# optimizer = torch.optim.AdamW(
#     [
#         {'params': head_params, 'lr': 1e-4, 'weight_decay': 1e-2},
#         {'params': lora_params, 'lr': 5e-4, 'weight_decay': 0.0},  # LoRA 分支更大学习率，且通常不做 wd
#     ],
#     betas=(0.9, 0.999),
# )

# # 调度器建议：Poly 或 Cosine + Warmup（二选一）
from torch.optim.lr_scheduler import LambdaLR
# epochs = epochs  # 你已有
# warmup = 5
# def cosine_warmup(e):
#     if e < warmup:
#         return (e + 1) / warmup
#     progress = (e - warmup) / max(1, (epochs - warmup))
#     return 0.5 * (1 + math.cos(math.pi * progress))

# scheduler = LambdaLR(optimizer, lr_lambda=cosine_warmup)
# ======================================================
# 1. 参数分组
# ======================================================
def is_lora_param(n):
    return ('lora_A' in n) or ('lora_B' in n)

lora_params, head_params, norm_params = [], [], []

for n, p in net.named_parameters():
    if not p.requires_grad:
        continue
    if is_lora_param(n):
        lora_params.append(p)
    elif any(nd in n for nd in ['norm', 'bn', 'bias', 'ln', 'LayerNorm']):
        norm_params.append(p)
    else:
        head_params.append(p)

# ======================================================
# 2. 优化器
# ======================================================
optimizer = optim.AdamW(
    [
        {'params': head_params, 'lr': 1e-4, 'weight_decay': 1e-2},
        {'params': norm_params, 'lr': 1e-4, 'weight_decay': 0.0},   # norm层不做wd
        {'params': lora_params, 'lr': 3e-4, 'weight_decay': 0.0},   # LoRA略高lr，无正则
    ],
    betas=(0.9, 0.999)
)

# ======================================================
# 3. 调度器：Cosine + Warmup 改进版
# ======================================================
total_epochs = epochs
warmup_epochs = max(5, int(total_epochs * 0.1))  # 前10%热身

def cosine_warmup(epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    # 平滑尾部衰减
    return 0.5 * (1 + math.cos(math.pi * progress)) ** 1.2

scheduler = LambdaLR(optimizer, lr_lambda=cosine_warmup)


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    if DATASET == 'Hunan':
        eroded_labels = ((np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64')) for id in test_ids)
    else:
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)

    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                min = np.min(dsm)
                max = np.max(dsm)
                if DATASET == 'Hunan':
                    dsm = (dsm - min) / (max - min + 1e-8)
                else:
                    dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)

                # Do the inference
                outs, align_loss = net(image_patches, dsm_patches, mode='Test')
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
    
    if DATASET == 'Hunan':
        accuracy = metrics_loveda(np.concatenate([p.ravel() for p in all_preds]),
                        np.concatenate([p.ravel() for p in all_gts]).ravel())
    else:
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                        np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


from torch.amp import autocast, GradScaler
scaler = GradScaler()
max_grad_norm = 5.0
def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.00
    # criterion = CombinedLoss(ignore_index=255, use_aux=False).cuda()
    for e in range(1, epochs + 1):
        
        net.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output, align_loss = net(data, dsm, mode='Train')
                λ = 0.1
                # loss = criterion(output, target) + λ * align_loss
                # loss = criterion(output, target)
                loss = loss_calc(output, target, weights) + λ * align_loss
            # loss = CrossEntropy2d(output, target, weight=weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.data
            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

        if scheduler is not None:
            scheduler.step()

        if e % save_epoch == 0:
            train_time = time.time()
            print("Training time: {:.3f} seconds".format(train_time - start_time))
            # We validate with the largest possible stride for faster computing
            epoch_loss /= len(train_loader)
            net.eval()
            MIoU = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            # 写入 TensorBoard
            writer.add_scalar("Loss/train", epoch_loss, e)
            writer.add_scalar("mIoU/val", MIoU, e)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], e)
            test_time = time.time()
            
            print("Test time: {:.3f} seconds".format(test_time - train_time))
            if MIoU > MIoU_best:
                if DATASET == 'Vaihingen':
                    torch.save(net.state_dict(), './resultsv_test12/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                elif DATASET == 'Potsdam':
                    torch.save(net.state_dict(), './resultsp/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                elif DATASET == 'Hunan':
                    torch.save(net.state_dict(), './resultsh_sar_test2loraall/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                MIoU_best = MIoU
    writer.close()
    print('MIoU_best: ', MIoU_best)

if MODE == 'Train':
    train(net, optimizer, epochs, scheduler, weights=WEIGHTS, save_epoch=save_epoch)

elif MODE == 'Test':
    if DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('./resultsv_test2loraall/UNetformer_epoch41_0.840658898176093'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsv/inference_UNetFormer_{}_tile_{}.png'.format('huge', id_), img)

    elif DATASET == 'Potsdam':
        net.load_state_dict(torch.load('./resultsp/UNetformer_epoch38_0.8560702376572351'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsp/inference_UNetFormer_{}_tile_{}.png'.format('huge', id_), img)

    elif DATASET == 'Hunan':
        net.load_state_dict(torch.load('./resultsh/UNetformer_epoch23_0.5009448279555474'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=128)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsh/inference_UNetFormer_{}_tile_{}.png'.format('base', id_), img)