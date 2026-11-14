"""Bölütleme kayıpları (train.py ve refiner.py tarafından kullanılır)."""
import torch


def dice_loss(logits, y, valid=None):
    """valid: 0/1 maske (yok sayılan pseudo bölgeler dışarıda kalır)."""
    p = torch.sigmoid(logits)
    if valid is not None:
        p, y = p * valid, y * valid
    inter = (p * y).sum((2, 3))
    return 1 - ((2 * inter + 1) / (p.sum((2, 3)) + y.sum((2, 3)) + 1)).mean()


def dml_loss(logits, y, valid=None):
    """Dice Semimetric Loss (Wang ve ark. 2023, arXiv:2303.16296): ikili hedefte soft Dice ile aynı,
    yumuşak hedefte minimumu p = y'de (soft Dice hedefi ikiliye iter). Deney 32."""
    p = torch.sigmoid(logits)
    if valid is not None:
        p, y = p * valid, y * valid
    s = p.sum((2, 3)) + y.sum((2, 3))
    return ((p - y).abs().sum((2, 3)) / (s + 1)).mean()


def tversky_loss(logits, y, valid=None, alpha=0.3, beta=0.7):
    """Tversky kaybı (Salehi ve ark. 2017): beta > alpha kaçırılan (FN) pikselleri yanlış alarmdan (FP)
    ağır cezalandırır; alpha = beta = 0,5 soft Dice ile aynı. Yumuşak hedefle çalışır. Deney 41."""
    p = torch.sigmoid(logits)
    if valid is not None:
        p, y = p * valid, y * valid
    tp = (p * y).sum((2, 3))
    fp, fn = (p * (1 - y)).sum((2, 3)), ((1 - p) * y).sum((2, 3))
    # yumuşatma 0,5: alpha = beta = 0,5'te dice_loss ile birebir aynı ((2TP+1)/(P+Y+1))
    return 1 - ((tp + 0.5) / (tp + alpha * fp + beta * fn + 0.5)).mean()


def skeleton_recall_loss(logits, skel):
    """Skeleton Recall Loss (Kirchhoff ve ark. 2024, arXiv:2404.03010): yalnız hedefin (kalınlaştırılmış)
    iskeletinde yumuşak recall. Arka plana ve filament kenarlarına baskı uygulamaz. Deney 43."""
    p = torch.sigmoid(logits)
    s = skel.sum((2, 3))
    rec = ((p * skel).sum((2, 3)) + 1e-6) / (s + 1e-6)
    has = (s > 0).float()
    return ((1 - rec) * has).sum() / has.sum().clamp(min=1)


def skeleton_target(mask):
    """Deney 43: ikili maskenin 3x3 kalınlaştırılmış iskeleti (CPU, veri yükleyicide)."""
    import cv2
    import numpy as np
    from skimage.morphology import skeletonize
    return cv2.dilate(skeletonize(mask).astype(np.uint8), np.ones((3, 3), np.uint8))[None].astype(np.float32)


def sdf_target(mask, trunc=8):
    """Deney 50: çoğunluk maskesinin işaretli sınır uzaklığı (dışarıda +, içeride −), ±trunc px'e kırpılıp
    [-1, 1]'e ölçeklenir (CPU, veri yükleyicide). Yardımcı kanal yalnız eğitimde kullanılır."""
    import cv2
    import numpy as np
    m = mask.astype(np.uint8)
    d_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    d_in = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return (np.clip(d_out - d_in, -trunc, trunc) / trunc)[None].astype(np.float32)
