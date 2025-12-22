import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import itertools
from torchvision.utils import make_grid
from PIL import Image
from skimage import io
import os
import json

# Parameters
WINDOW_SIZE = (256, 256)  # Patch size
STRIDE = 32
IN_CHANNELS = 3
FOLDER = "/home/csf1/Documents/dataset/seg/"
BATCH_SIZE = 10

MODEL = 'UNetformer'
MODE = 'Train'
FTune = 'LoRA'

DATASET = 'VALID'   # <<< 使用 VALID
IF_SAM = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
N_CLASSES = len(LABELS)
WEIGHTS = torch.ones(N_CLASSES)
CACHE = True

# Default palette (Vaihingen)
palette = {
    0: (255, 255, 255),
    1: (0, 0, 255),
    2: (0, 255, 255),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (255, 0, 0),
    6: (0, 0, 0)
}
invert_palette = {v: k for k, v in palette.items()}

# ---------------- Dataset config ----------------
if DATASET == 'Vaihingen':
    train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
    test_ids = ['5', '21', '15', '30']
    Stride_Size = 32
    epochs = 50
    save_epoch = 1
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

elif DATASET == 'Potsdam':
    train_ids = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
                 '4_12', '6_8', '6_12', '6_7', '4_11']
    test_ids = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']
    Stride_Size = 128
    epochs = 50
    save_epoch = 1
    MAIN_FOLDER = FOLDER + 'Potsdam/'
    DATA_FOLDER = MAIN_FOLDER + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
    DSM_FOLDER = MAIN_FOLDER + '1_DSM_normalisation/dsm_potsdam_{}_normalized_lastools.jpg'
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'

elif DATASET == 'Hunan':
    train_ids = ['10434' ,'11524' ,'11607' ,'11724' ,'11854' ,'11856' ,'11919' ,'12152' ,'12350' ,'12563' ,'12669' ,'12813' ,'1302' ,'13258' ,'13383' ,'13524' ,'13565' ,'13932' ,'14477' ,'14694' ,'15001' ,'15023' ,'15201' ,'15230' ,'15548' ,'15603' ,'15686' ,'1599' ,'15998' ,'16090' ,'16217' ,'16541' ,'16582' ,'16703' ,'16709' ,'17092' ,'17269' ,'18186' ,'18950' ,'1899' ,'1906' ,'19098' ,'19680' ,'19915' ,'20175' ,'20386' ,'20561' ,'20734' ,'20752' ,'20759' ,'21562' ,'21565' ,'21738' ,'21820' ,'22000' ,'2232' ,'22547' ,'22729' ,'2431' ,'24718' ,'24733' ,'25069' ,'2584' ,'2617' ,'26622' ,'27163' ,'27308' ,'27312' ,'2791' ,'29393' ,'29886' ,'30560' ,'30719' ,'31175' ,'31188' ,'31586' ,'31797' ,'3817' ,'4529' ,'4530' ,'4889' ,'5223' ,'6213' ,'6597' ,'6600' ,'6768' ,'7329' ,'8232' ,'830' ,'8931' ,'944' ,'9956' ,'2087' ,'17721' ,'13990' ,'13622' ,'13563' ,'18009' ,'12148' ,'16888' ,'14758' ,'1773' ,'16516' ,'20408' ,'2070' ,'10062' ,'17637' ,'14942' ,'13931' ,'13410' ,'11959' ,'15150' ,'17582' ,'17820' ,'21545' ,'21563' ,'21592' ,'21922' ,'2255' ,'26628' ,'28533' ,'28801' ,'29621' ,'29796' ,'30482' ,'31302' ,'31355' ,'4171' ,'4887' ,'5994' ,'6167' ,'6777' ,'7421' ,'833' ,'9064' ,'9662' ,'14936' ,'15493' ,'2097' ,'25709' ,'30301' ,'18117' ,'11766' ,'3994' ,'5830' ,'14786' ,'1774' ,'16032' ,'2597' ,'18164' ,'8976' ,'2427' ,'418' ,'23961' ,'1165' ,'6383' ,'22906' ,'26032' ,'18371' ,'6156' ,'7167' ,'20736' ,'16880' ,'29145' ,'21211' ,'7473' ,'29172' ,'22077' ,'14755' ,'2428' ,'16922' ,'15144' ,'5232' ,'25777' ,'21736' ,'14290' ,'15275' ,'1025' ,'11173' ,'12040' ,'12779' ,'14126' ,'15695' ,'16214' ,'16577' ,'18079' ,'1930' ,'21804' ,'22154' ,'25699' ,'29675' ,'298' ,'31653' ,'5042' ,'637' ,'6581' ,'708' ,'7679' ,'1031' ,'11272' ,'14463' ,'16745' ,'12244' ,'1775' ,'1752' ,'16744' ,'17095' ,'20910' ,'13742' ,'16702' ,'13925' ,'17800' ,'17040' ,'2062' ,'16912' ,'19149' ,'11371' ,'21601' ,'21610' ,'21737' ,'21747' ,'21982' ,'23232' ,'2606' ,'27860' ,'28532' ,'28933' ,'29028' ,'29284' ,'29288' ,'30597' ,'3122' ,'31242' ,'4362' ,'6405' ,'6770' ,'726' ,'7330' ,'7331' ,'8234' ,'2402' ,'646' ,'13718' ,'12363' ,'13995' ,'13807' ,'1471' ,'27168' ,'18298' ,'16093' ,'15763' ,'12042' ,'29020' ,'8831' ,'11375' ,'23772' ,'12728' ,'13448' ,'27960' ,'14467' ,'14763' ,'19866' ,'13766' ,'24296' ,'1436' ,'5236' ,'28796' ,'10258' ,'28736' ,'2100' ,'1451' ,'12918' ,'14155' ,'15184' ,'19471' ,'21822' ,'22728' ,'22837' ,'2408' ,'3100' ,'5450' ,'6186' ,'7181' ,'9258' ,'11625' ,'11644' ,'12098' ,'12154' ,'12241' ,'13248' ,'13522' ,'13564' ,'1383' ,'13927' ,'14287' ,'14822' ,'15075' ,'15520' ,'15864' ,'16028' ,'1619' ,'16699' ,'16742' ,'16866' ,'17369' ,'17934' ,'18183' ,'18185' ,'18732' ,'1928' ,'19688' ,'20314' ,'20464' ,'20737' ,'21111' ,'21561' ,'21938' ,'22555' ,'23080' ,'23588' ,'23701' ,'2604' ,'27282' ,'27718' ,'28073' ,'28189' ,'29171' ,'29286' ,'29844' ,'30148' ,'30399' ,'30425' ,'30606' ,'31014' ,'31583' ,'31621' ,'4363' ,'5043' ,'5221' ,'5238' ,'6379' ,'6407' ,'6601' ,'9174' ,'2748' ,'29829' ,'10970' ,'20926' ,'6795' ,'24149' ,'18121' ,'20935' ,'942' ,'29391' ,'29638' ,'20054' ,'3161' ,'6772' ,'17933' ,'13535' ,'5412' ,'20599' ,'299' ,'19609' ,'452' ,'28191' ,'11659' ,'1450' ,'13019' ,'11838' ,'29892' ,'12151' ,'13933' ,'11568' ,'11233' ,'12153' ,'13433' ,'13436' ,'14105' ,'14169' ,'16724' ,'18895' ,'19853' ,'21981' ,'2246' ,'22907' ,'29654' ,'30669' ,'3534' ,'6794' ,'1057' ,'11271' ,'11603' ,'11678' ,'11957' ,'1262' ,'12863' ,'13109' ,'13299' ,'13415' ,'14603' ,'14873' ,'15398' ,'15596' ,'1605' ,'16089' ,'16530' ,'16870' ,'17799' ,'17943' ,'1932' ,'21740' ,'21967' ,'23342' ,'2396' ,'2397' ,'2429' ,'2542' ,'25778' ,'2618' ,'26776' ,'28038' ,'29104' ,'29567' ,'29733' ,'29779' ,'30276' ,'31708' ,'3936' ,'419' ,'5066' ,'7898' ,'20933' ,'15597' ,'18118' ,'16356' ,'14937' ,'10503' ,'13437' ,'14247' ,'21566' ,'22099' ,'22731' ,'27437' ,'28078' ,'29394' ,'29571' ,'30130' ,'6771' ,'940']
    test_ids = ['11767' ,'11816' ,'1239' ,'12626' ,'12815' ,'1290' ,'1303' ,'13254' ,'13257' ,'13515' ,'13765' ,'14108' ,'14293' ,'15426' ,'1625' ,'16373' ,'16750' ,'16890' ,'17039' ,'17055' ,'17107' ,'17455' ,'17821' ,'17980' ,'18958' ,'1908' ,'1923' ,'20028' ,'2106' ,'21385' ,'21423' ,'22312' ,'2256' ,'2265' ,'2442' ,'2603' ,'27264' ,'28965' ,'29287' ,'30275' ,'3280' ,'4017' ,'4166' ,'4886' ,'5215' ,'5233' ,'5410' ,'639' ,'7176' ,'11624']
    Stride_Size = 256
    epochs = 50
    save_epoch = 1
    MAIN_FOLDER = FOLDER + 'Hunan/'
    LABELS = ["cropland", "forest", "grassland", "wetland", "water", "unused land", "built-up area"]
    N_CLASSES = len(LABELS) # Number of classes
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    DATA_FOLDER = MAIN_FOLDER + 'images_png/{}.png'
    DSM_FOLDER = MAIN_FOLDER + 'dsm_pngs/{}.png'
    LABEL_FOLDER = MAIN_FOLDER + 'masks_png1/{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'masks_png1/{}.tif'
    palette = {
        0: (196, 90, 17),       # cropland
        1: (51, 129, 88),       # forest
        2: (177, 205, 61),      # grassland
        3: (228, 84, 96),       # wetland
        4: (91, 154, 214),      # water
        5: (225, 174, 110),     # unused land
        6: (239, 159, 2)}       # built-up area
    invert_palette = {v: k for k, v in palette.items()}

elif DATASET == 'VALID':
    MAIN_FOLDER = FOLDER + 'VALID/'
    Stride_Size = 256
    epochs = 80
    save_epoch = 1

    # split 文件：优先找 MAIN_FOLDER/splits/xxx，同时兼容 split/ 与根目录
    def _read_valid_split(fname):
        cand = [
            os.path.join(MAIN_FOLDER, "splits", fname),  # <<< 你的 tree 显示为 splits
            os.path.join(MAIN_FOLDER, "split", fname),   # 兼容其它版本命名
            os.path.join(MAIN_FOLDER, fname),            # 兼容放根目录
        ]
        for p in cand:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return [ln.strip() for ln in f if ln.strip() and (not ln.strip().startswith("#"))]
        raise FileNotFoundError(f"Cannot find split file: {fname}, tried: {cand}")

    train_ids1 = _read_valid_split("seg_train.txt")
    val_ids1   = _read_valid_split("seg_val.txt")
    train_ids = train_ids1 + val_ids1
    test_ids  = _read_valid_split("seg_test.txt")

    # categories.json -> palette / invert_palette
    with open(os.path.join(MAIN_FOLDER, "categories.json"), "r", encoding="utf-8") as f:
        cats = json.load(f)
    cats = sorted(cats, key=lambda x: int(x["id"]))
    LABELS = [c["category_name"] for c in cats]
    N_CLASSES = len(LABELS)
    WEIGHTS = torch.ones(N_CLASSES)
    palette = {int(c["id"]): tuple(c["color"]) for c in cats}
    invert_palette = {tuple(c["color"]): int(c["id"]) for c in cats}

else:
    raise ValueError(f"Unsupported DATASET={DATASET}")

print(MODEL + ', ' + MODE + ', ' + DATASET + ', IF_SAM: ' + str(IF_SAM) + ', WINDOW_SIZE: ', WINDOW_SIZE,
      ', BATCH_SIZE: ' + str(BATCH_SIZE), ', Stride_Size: ', str(Stride_Size),
      ', epochs: ' + str(epochs), ', save_epoch: ', str(save_epoch),)

# ---------------- Utils ----------------
def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    if arr_3d.ndim == 3 and arr_3d.shape[2] >= 4:
        arr_3d = arr_3d[:, :, :3]
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d

def save_img(tensor, name, normalize=True):
    if isinstance(tensor, torch.Tensor):
        t = tensor.detach().cpu()
    else:
        t = torch.tensor(tensor)

    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim == 3:
        t = t.unsqueeze(0)

    if normalize:
        if not t.is_floating_point():
            t = t.float()
        im = make_grid(t, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
        im = (im.numpy() * 255.).astype(np.uint8)
    else:
        # assume uint8 [0,255] image already
        if t.dtype != torch.uint8:
            t = t.clamp(0, 255).to(torch.uint8)
        # take first in batch
        im = t[0].permute(1, 2, 0).numpy()

    Image.fromarray(im).save(name + '.jpg')


def _valid_paths_from_label_json(root, json_name, json_dir="label"):
    """
    json_name: '00001.json' (from seg_train/val/test)
    label json example:
      {
        "id": "00001",
        "file_name": "images/airport/100/xxx.png",
        "segmentation": {"semantic_filename": "semantic/airport/100/yyy.png", ...}
      }

    depth is flat: root/depth/{id}.png
    """
    if not json_name.endswith(".json"):
        json_name = json_name + ".json"

    jpath = os.path.join(root, json_dir, json_name)
    with open(jpath, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # -------- depth (flat) --------
    # Prefer meta["id"] (e.g. "00001"), fallback to json filename stem
    sid = str(meta.get("id", os.path.splitext(os.path.basename(json_name))[0]))
    depth_path = os.path.join(root, "depth", f"{sid}.png")

    # -------- rgb path --------
    rgb_rel = meta["file_name"].replace("\\", "/")
    if not rgb_rel.startswith("/") and not rgb_rel.startswith("images/"):
        # Robust: if only "airport/100/xxx.png" is given
        rgb_rel = "images/" + rgb_rel
    rgb_path = rgb_rel if rgb_rel.startswith("/") else os.path.join(root, rgb_rel)

    # -------- semantic gt path (use json field!) --------
    seg = meta.get("segmentation", {})
    sem_rel = seg.get("semantic_filename", None)
    if sem_rel is None:
        # Fallback: derive by replacement (only if json missing field)
        sem_rel = rgb_rel.replace("images/", "semantic/", 1)
    sem_rel = sem_rel.replace("\\", "/")
    sem_path = sem_rel if sem_rel.startswith("/") else os.path.join(root, sem_rel)

    # -------- optional fallback for depth extension --------
    if (not os.path.isfile(depth_path)):
        alt = os.path.join(root, "depth", f"{sid}.tif")
        if os.path.isfile(alt):
            depth_path = alt

    return rgb_path, depth_path, sem_path


def get_random_pos(img, window_shape):
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

# ---------------- Dataset ----------------
class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()
        self.augmentation = augmentation
        self.cache = cache

        if DATASET == "VALID":
            self.data_files, self.dsm_files, self.label_files = [], [], []
            json_dir = "label"  # 也可以改成 "refine_label"
            for name in ids:
                rgb_p, dep_p, sem_p = _valid_paths_from_label_json(MAIN_FOLDER, name, json_dir=json_dir)
                self.data_files.append(rgb_p)
                self.dsm_files.append(dep_p)
                self.label_files.append(sem_p)
        else:
            self.data_files = [DATA_FOLDER.format(id) for id in ids]
            self.dsm_files = [DSM_FOLDER.format(id) for id in ids]
            self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError(f'{f} is not a file !')

        self.data_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        if DATASET in ['Potsdam', 'Vaihingen', 'VALID']:
            return BATCH_SIZE * 1000
        elif DATASET == 'Hunan':
            return BATCH_SIZE * 500
        else:
            return BATCH_SIZE * 1000

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
        return tuple(results)

    def __getitem__(self, i):
        random_idx = random.randint(0, len(self.data_files) - 1)

        # ---- RGB ----
        if random_idx in self.data_cache_:
            data = self.data_cache_[random_idx]
        else:
            if DATASET == 'Potsdam':
                img = io.imread(self.data_files[random_idx])[:, :, :3]
                data = img.transpose((2, 0, 1))
                data = (1 / 255) * np.asarray(data, dtype='float32')
            else:
                img = io.imread(self.data_files[random_idx])
                if img.ndim == 3 and img.shape[2] >= 4:
                    img = img[:, :, :3]
                data = (1 / 255) * np.asarray(img.transpose((2, 0, 1)), dtype='float32')

            if self.cache:
                self.data_cache_[random_idx] = data

        # ---- Depth/DSM ----
        if random_idx in self.dsm_cache_:
            dsm = self.dsm_cache_[random_idx]
        else:
            dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype='float32')

            # VALID depth: true depth = value/256
            if DATASET == "VALID":
                dsm = dsm / 256.0

            mn = float(np.min(dsm))
            mx = float(np.max(dsm))
            dsm = (dsm - mn) / (mx - mn + 1e-8)

            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        # ---- Label ----
        if random_idx in self.label_cache_:
            label = self.label_cache_[random_idx]
        else:
            if DATASET == 'Hunan':
                label = np.asarray(io.imread(self.label_files[random_idx]), dtype='int64')
            else:
                label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')

            if self.cache:
                self.label_cache_[random_idx] = label

        # ---- Patch ----
        if DATASET == 'Hunan':
            data_p, dsm_p, label_p = data, dsm, label
        else:
            x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
            data_p = data[:, x1:x2, y1:y2]
            dsm_p = dsm[x1:x2, y1:y2]
            label_p = label[x1:x2, y1:y2]

        # ---- Augment ----
        data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)

        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))

# ---------------- Loss ----------------
class CrossEntropy2d_ignore(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1)).cuda()
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

def loss_calc(pred, label, weights):
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d_ignore(ignore_label=255).cuda()
    loss = criterion(pred, label, weight=weights)
    return loss

# ---------------- Debug ----------------
if __name__ == "__main__":
    os.makedirs("./save", exist_ok=True)
    train_set = ISPRS_dataset(train_ids, cache=CACHE)

    for idx, (a, b, c) in enumerate(train_set):
        save_img(a, f"./save/{idx}_rgb")
        save_img(b, f"./save/{idx}_depth")
        # 如需可视化 label（id->color），可用：
        lbl_color = convert_to_color(c.numpy().astype(np.uint8))
        save_img(torch.from_numpy(lbl_color).permute(2,0,1), f"./save/{idx}_label", normalize=False)
        if idx >= 20:
            break
