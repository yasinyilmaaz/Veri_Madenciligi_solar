"""Olasılık haritası -> birbiriyle çakışmayan filament instance'ları."""
import cv2
import numpy as np


def prob_to_instances(prob, thr=0.5, min_area=200, join=0, thr_high=None, link_thr=None):
    """Eşikle, yakın parçaları birleştir, küçükleri at.

    thr_high verilirse hysteresis: thr ile bulunan bileşenlerden yalnızca
    içinde thr_high'ı geçen piksel olanlar tutulur (soluk uçlar kalır,
    zayıf gürültü elenir).
    join > 0 ise ikili maske join piksel genişletilip bağlı bileşenler
    bulunur; orijinal pikseller bu bileşenlere atanır. Böylece aynı
    filamentin kopuk parçaları tek instance olur ama maske büyümez.
    link_thr verilirse (eyer bağlantısı): parçalar prob > link_thr bileşeni
    aynıysa birleştirilir; mesafe yerine aradaki olasılık sırtı belirleyicidir.
    """
    binary = (prob > thr).astype(np.uint8)
    if link_thr is not None:
        _, lab = cv2.connectedComponents((prob > link_thr).astype(np.uint8), connectivity=8)
        lab = lab * binary
    elif join > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * join + 1, 2 * join + 1))
        _, lab = cv2.connectedComponents(cv2.dilate(binary, k), connectivity=8)
        lab = lab * binary
    else:
        _, lab = cv2.connectedComponents(binary, connectivity=8)
    areas = np.bincount(lab.ravel())
    keep = areas >= min_area
    if thr_high is not None:
        strong = np.zeros(len(areas), bool)
        strong[np.unique(lab[prob > thr_high])] = True
        keep &= strong
    keep[0] = False
    remap = np.zeros(len(areas), np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    return remap[lab]
