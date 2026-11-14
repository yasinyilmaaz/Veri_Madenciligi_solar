"""Eğitim görüntü artırmaları (train.py'den ayrıldı).

Albumentations 2.x kendi üreteçlerini kullanır; global random/numpy/torch tohumları onu etkilemez
Bu yüzden Compose'a açık tohum verilir ve her DataLoader
worker'ı kendi tohumuyla yeniden tohumlanır: aynı --seed aynı artırma dizisini üretir.
"""
import albumentations as A
import cv2
import numpy as np
import torch


def cloud_field(img, strength=0.15):
    """Bulut/şeffaflık değişimini taklit eden düşük frekanslı çarpımsal alan."""
    h, w = img.shape
    small = np.random.randn(4, 4).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    field = 1 + strength * field / (np.abs(field).max() + 1e-6)
    return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)


def make_aug(kind):
    """Artırma hattı. Tohum torch'tan alınır: ana süreçte --seed ile, worker'larda seed_worker ile."""
    base = [A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90()]
    if kind == "base":
        return A.Compose(base + [
            A.Affine(scale=(0.85, 1.15), rotate=(-30, 30), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ], seed=int(torch.initial_seed() % 2**32))
    if kind == "rot360":
        # yalnızca geometrik: disk her açıda fiziksel olarak geçerli; fotometrik kısım base ile aynı
        return A.Compose(base + [
            A.Affine(scale=(0.85, 1.15), rotate=(-180, 180), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ], seed=int(torch.initial_seed() % 2**32))
    if kind == "strong":
        # disk her açıda döndürülebilir; görüş (seeing), JPEG ve istasyon kontrast farkları
        return A.Compose(base + [
            A.Affine(scale=(0.85, 1.15), rotate=(-180, 180), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.25, p=0.6),
            A.RandomGamma(gamma_limit=(70, 130), p=0.4),
            # MotionBlur varsayılanı (allow_shifted, tek yönlü çekirdek) görüntüyü maskeye göre 3 px'e kadar kaydırıyor;
            # merkezli simetrik çekirdekte kütle merkezi kayması 0
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7)),
                     A.MotionBlur(blur_limit=(3, 7), allow_shifted=False, direction_range=(0, 0))], p=0.3),
            A.GaussNoise(std_range=(0.01, 0.04), p=0.3),
            A.ImageCompression(quality_range=(50, 95), p=0.3),
        ], seed=int(torch.initial_seed() % 2**32))
    raise ValueError(kind)


def seed_worker(worker_id):
    """DataLoader worker_init_fn: worker'ın torch tohumundan (temel tohum + worker no) artırmayı tohumla."""
    info = torch.utils.data.get_worker_info()
    info.dataset.aug.set_random_seed(info.seed % 2**32)
