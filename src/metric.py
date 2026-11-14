"""Panoptic Quality (yarışmadaki gibi) ve submission yardımcıları.

PQ = sum(IoU, TP) / (TP + 0.5 FP + 0.5 FN); TP: IoU > 0.5.
TP/FP/FN tüm anotatör-görüntüleri üzerinden toplanır (global PQ).
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils

SIZE = 2048


def mask_to_rle(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def _enc(counts):
    return [{"size": [SIZE, SIZE], "counts": c.encode("ascii")} for c in counts]


def _group(df):
    g = defaultdict(list)
    for fid, c in zip(df["filament_id"], df["segmentation_rle"]):
        g[str(fid).rsplit("_", 1)[0]].append(str(c))
    return g


def compute_pq(gt_df, pred_df):
    gt = _group(gt_df)
    pred = _group(pred_df)
    s_iou, tp, fp, fn = 0.0, 0, 0, 0
    for ai, g in gt.items():
        p = pred.get(ai.split("-", 1)[1], [])
        if not p:
            fn += len(g)
            continue
        iou = np.asarray(mask_utils.iou(_enc(g), _enc(p), [0] * len(p))).reshape(len(g), len(p))
        hit = iou > 0.5
        s_iou += iou[hit].sum()
        tp += int(hit.sum())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = s_iou / denom if denom else 0.0
    sq = s_iou / tp if tp else 0.0
    rq = tp / denom if denom else 0.0
    return dict(pq=pq, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn)


def rubric_stats(gt_df, pred_df, return_arrays=False):
    """Resmi rubriğin PQ dışı ölçütleri (organizatörün self-evaluation notebook'u ile aynı tanımlar).

    - IoU / Dice dağılımı: anotatör-görüntü başına, örtüşen (IoU > 0) tüm GT-tahmin çiftleri
    - bire-çok: GT başına IoU > 0 ile örtüşen tahmin sayısı (>= 2 ise parçalanma)
    - çoka-bir: tahmin başına IoU > 0 ile örtüşen GT sayısı (>= 2 ise birleşme)
    return_arrays=True: özetlerin yanında dağılım dizileri de döner (rapor histogramları için).
    """
    gt = _group(gt_df)
    pred = _group(pred_df)
    ious, gt_deg, pred_deg = [], [], []
    for ai, g in gt.items():
        p = pred.get(ai.split("-", 1)[1], [])
        if not p:
            gt_deg += [0] * len(g)
            continue
        iou = np.asarray(mask_utils.iou(_enc(g), _enc(p), [0] * len(p))).reshape(len(g), len(p))
        ious += iou[iou > 0].tolist()
        hit = iou > 0
        gt_deg += hit.sum(1).tolist()
        pred_deg += hit.sum(0).tolist()
    ious, gt_deg, pred_deg = np.array(ious), np.array(gt_deg), np.array(pred_deg)
    dice = 2 * ious / (1 + ious)
    if return_arrays:
        return dict(iou=ious, dice=dice, gt_deg=gt_deg, pred_deg=pred_deg)
    return dict(
        iou_mean=ious.mean(), iou_median=np.median(ious), dice_mean=dice.mean(), dice_median=np.median(dice),
        gt_miss=(gt_deg == 0).mean(), gt_split=(gt_deg >= 2).mean(),
        pred_miss=(pred_deg == 0).mean(), pred_merge=(pred_deg >= 2).mean(),
    )


def instances_to_rows(stem, labels):
    """Etiket haritasından (0=arka plan) submission satırları."""
    rows = []
    for n, k in enumerate(np.unique(labels[labels > 0]), 1):
        rows.append((f"{stem}_{n}", mask_to_rle(labels == k)))
    return rows


def check_submission(df):
    """Kaggle'ın reddedeceği hataları yakala: boyut, tekil id, çakışma."""
    assert list(df.columns) == ["filament_id", "segmentation_rle"]
    assert df["filament_id"].is_unique, "filament_id tekil değil"
    for stem, counts in _group(df).items():
        acc = np.zeros((SIZE, SIZE), np.uint8)
        for c in counts:
            m = mask_utils.decode(_enc([c])[0])
            assert m.shape == (SIZE, SIZE)
            acc += m
        assert acc.max() <= 1, f"{stem}: maskeler çakışıyor"
    return True
