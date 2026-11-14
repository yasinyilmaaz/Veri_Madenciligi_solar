"""Deney 37: CNN kalite ağı için karşı-olgusal maske varyantları.

Her OOF adayı için aynı olasılık haritasından iki gerçekçi sınır varyantı üretilir:
  gevşek: adayın eyer grubunun (prob>0,15 bileşeni) prob>0,30 kısmı
  sıkı  : adayın prob>0,55 kısmı
Varyantın hedefi y, tüm anotatörlere karşı gerçek IoU'dan yeniden hesaplanır (scorer ile aynı tanım:
anotatör ortalaması IoU·[IoU>0,5]). Neredeyse aynı (IoU>0,95) veya 100 px'den küçük varyantlar atılır.
Varyantlar yalnız eğitimde kullanılır; değerlendirme doğal adaylarda kalır, aday havuzu değişmez.
"""
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pycocotools.mask as mu

from cnn_selector import crop_one, norm_img
from data import TRAIN_IMG, load_records, poly_to_rle
from metric import mask_to_rle
from tree_dp import y_of

REC = load_records()


def _image(args):
    stem, prob_path, fold, rows = args
    prob = np.load(prob_path)
    img = norm_img(TRAIN_IMG / f"{stem}.jpeg")
    group = cv2.connectedComponents((prob > 0.15 * 255).astype(np.uint8), connectivity=8)[1]
    gts = [[poly_to_rle(a["seg"]) for a in an] for an in REC[stem].values()]
    masks, crops = [], []
    for rle in rows:
        m = mu.decode({"size": [2048, 2048], "counts": rle.encode()}).astype(bool)
        g = np.isin(group, np.unique(group[m]))
        for v in ((prob > 0.30 * 255) & g, m & (prob > 0.55 * 255)):
            inter, union = (v & m).sum(), (v | m).sum()
            if v.sum() < 100 or inter / union > 0.95:
                continue
            masks.append(mask_to_rle(v))
            crops.append(crop_one(img, prob, v))
    y = y_of(masks, gts) if masks else np.zeros(0)
    return stem, fold, crops, y


def build_variants(df, oof_dir, folds, cache):
    jobs = [(s, oof_dir / f"fold{folds[s]}" / "val_probs" / f"{s}.npy", int(folds[s]), list(g.rle))
            for s, g in df.groupby("stem")]
    X, stems, fs, ys = [], [], [], []
    with ProcessPoolExecutor(8) as ex:
        for stem, fold, crops, y in ex.map(_image, jobs, chunksize=4):
            X += crops
            ys += list(y)
            stems += [stem] * len(crops)
            fs += [fold] * len(crops)
    np.savez(cache, X=np.stack(X), stem=np.array(stems), fold=np.array(fs), y=np.array(ys, np.float32))
    print(f"{len(X)} varyant, ortalama y={np.mean(ys):.3f}", flush=True)
