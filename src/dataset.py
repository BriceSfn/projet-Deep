import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _to_input_tensor(pil_img, img_size):
    pil_img = pil_img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()


def _to_mask_tensor(mask_arr, img_size):
    mask = Image.fromarray((mask_arr * 255).astype(np.uint8))
    mask = mask.resize((img_size, img_size), Image.NEAREST)
    mask = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
    return torch.from_numpy(mask)[None]


class CocoBinaryDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=128, max_samples=None):
        from pycocotools.coco import COCO

        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.img_size = img_size

        ids = sorted(self.coco.getImgIds())
        ids = [i for i in ids if len(self.coco.getAnnIds(imgIds=i, iscrowd=None)) > 0]
        if max_samples is not None:
            ids = ids[:max_samples]
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, info["file_name"])
        image = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))

        return _to_input_tensor(image, self.img_size), _to_mask_tensor(mask, self.img_size)


class SyntheticShapes(Dataset):
    def __init__(self, n_samples=256, img_size=128, seed=0):
        self.n_samples = n_samples
        self.img_size = img_size
        self.seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 100000 + idx)
        s = self.img_size

        bg = rng.integers(0, 120, size=3)
        img = np.ones((s, s, 3), dtype=np.float32) * bg
        img += rng.normal(0, 10, size=(s, s, 3)).astype(np.float32)

        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        mask_pil = Image.fromarray(np.zeros((s, s), dtype=np.uint8))
        mdraw = ImageDraw.Draw(mask_pil)

        for _ in range(rng.integers(1, 4)):
            color = tuple(int(c) for c in rng.integers(140, 256, size=3))
            x0, y0 = rng.integers(0, s - s // 4, size=2)
            w, h = rng.integers(s // 6, s // 2, size=2)
            x1, y1 = min(x0 + w, s - 1), min(y0 + h, s - 1)
            if rng.random() < 0.5:
                draw.ellipse([x0, y0, x1, y1], fill=color)
                mdraw.ellipse([x0, y0, x1, y1], fill=1)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=color)
                mdraw.rectangle([x0, y0, x1, y1], fill=1)

        mask = np.asarray(mask_pil, dtype=np.float32)
        return _to_input_tensor(pil, s), _to_mask_tensor(mask, s)
