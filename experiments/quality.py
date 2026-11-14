"""Görüntü kalite ölçüleri: disk merkez kayması, keskinlik, bulut/gradyan.

Etiketsiz GONG karelerini train dağılımının dışında kalanlardan ayıklamak için.
Örnek: PYTHONPATH=src python experiments/quality.py  -> outputs/quality_train.csv, outputs/quality_gong.csv
"""
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from data import EXT_IMG, OUT, TRAIN_IMG


def measure(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)[::2, ::2].astype(np.float32)
    disk = im > 0.6 * np.median(im[im > 30])
    disk = cv2.erode(disk.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    ys, xs = np.nonzero(disk)
    off = float(np.hypot(ys.mean() - 512, xs.mean() - 512)) * 2
    r = float(np.sqrt(disk.sum() / np.pi)) * 2
    # disk içinde merkezde (limb'den uzak) keskinlik ve geniş ölçekli düzgünsüzlük
    inner = np.hypot(*np.mgrid[:1024, :1024] - np.array([ys.mean(), xs.mean()])[:, None, None]) < 0.8 * r / 2
    lap = cv2.Laplacian(im, cv2.CV_32F)[inner].var()
    blur = cv2.GaussianBlur(im, (0, 0), 40)[inner]
    grad = float(blur.std() / blur.mean())
    return dict(stem=path.stem, offset=off, radius=r, sharp=float(lap), nonuni=grad)


def main():
    for name, d in [("train", TRAIN_IMG), ("gong", EXT_IMG)]:
        with ProcessPoolExecutor(16) as ex:
            df = pd.DataFrame(ex.map(measure, sorted(d.glob("*.jpeg")), chunksize=8))
        df.to_csv(OUT / f"quality_{name}.csv", index=False)
        print(name, len(df))
        print(df.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(3).to_string())


if __name__ == "__main__":
    main()
