"""Test seti tahmini -> submission.csv (fold modellerinin ortalaması + flip TTA).

Örnek: PYTHONPATH=src python experiments/predict.py --runs outputs/unet_r34 --thr 0.5 --min_area 200 --join 0
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch

from data import OUT, TEST_IMG, read_image
from metric import check_submission, instances_to_rows
from post import prob_to_instances
from refiner import load_refiners, refine_instances
from scorer import CAND, PQModel, select_instances
from train import predict


def load_fold_model(fd):
    a = json.loads((fd / "args.json").read_text())
    m = getattr(smp, a["arch"])(a["encoder"], encoder_weights=None, in_channels=1, classes=a.get("n_out", 1))
    m.load_state_dict(torch.load(fd / "best.pt", map_location="cpu"))
    return m.cuda().eval()


def load_models(run_dirs, grouped=False):
    """run_dirs: 'outputs/run' veya 'outputs/run:0,1' (yalnızca belirtilen fold'lar; eğitimi süren
    fold'ların ara best.pt dosyalarını almamak için). grouped=True: run başına model listeleri."""
    groups = []
    for spec in run_dirs:
        rd, _, fl = spec.partition(":")
        wanted = {int(x) for x in fl.split(",")} if fl else None
        ms = []
        for fd in sorted(Path(rd).glob("fold*")):
            if (fd / "best.pt").exists() and (wanted is None or int(fd.name[4:]) in wanted):
                ms.append(load_fold_model(fd))
                print("model:", fd)
        groups.append(ms)
    return groups if grouped else [m for g in groups for m in g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min_area", type=int, default=200)
    ap.add_argument("--join", type=int, default=0)
    ap.add_argument("--thr_high", type=float, default=None)
    ap.add_argument("--tta", type=int, default=2, help="0 yok, 1 flip(4), 2 D4(8)")
    ap.add_argument("--scorer", default=None, help="scorer.py'nin kaydettiği LightGBM modeli")
    ap.add_argument("--cut", type=float, default=0.20)
    ap.add_argument("--scorer_pq", default=None, help="p.txt,q.txt: engel (p·q) seçici (deney 04)")
    ap.add_argument("--link_thr", type=float, default=None, help="eyer bağlantısı (deney 16); seçici aynı ayarla eğitilmiş olmalı")
    ap.add_argument("--refiner", default=None, help="refiner run klasörü (deney 02); seçilen adayları düzeltir")
    ap.add_argument("--ref_thr", type=float, default=0.5)
    ap.add_argument("--ref_blend", type=float, default=0.5)
    ap.add_argument("--per_run_mean", type=int, default=1, help="1: run (mimari) başına ortalama, sonra eşit ağırlık")
    ap.add_argument("--save_probs", default=None, help="test olasılık haritalarını bu klasöre kaydet (uint8 npy)")
    ap.add_argument("--load_probs", default=None, help="kaydedilmiş olasılık haritalarını kullan (modelleri çalıştırma)")
    ap.add_argument("--out", default=str(OUT / "submission.csv"))
    args = ap.parse_args()
    if args.link_thr is not None:
        CAND.update(join=0, link_thr=args.link_thr)

    groups = [] if args.load_probs else load_models(args.runs, grouped=True)
    if args.per_run_mean:
        print("mimari başına ortalama:", [len(g) for g in groups])
    if args.save_probs:
        Path(args.save_probs).mkdir(parents=True, exist_ok=True)
    scorer = lgb.Booster(model_file=args.scorer) if args.scorer else None
    if args.scorer_pq:
        scorer = PQModel(*args.scorer_pq.split(","))
    refiners = list(load_refiners(args.refiner).values()) if args.refiner else None
    if refiners:
        print("refiner modeli:", len(refiners))
    rows = []
    for p in sorted(TEST_IMG.glob("*.jpeg")):
        img = read_image(p)
        if args.load_probs:
            prob = np.load(Path(args.load_probs) / f"{p.stem}.npy").astype(np.float32) / 255
        else:
            if args.per_run_mean:  # önce her run'ın fold'ları, sonra run'lar eşit ağırlıkla
                prob = sum(sum(predict(m, img, tta=args.tta) for m in g) / len(g) for g in groups) / len(groups)
            else:
                ms = [m for g in groups for m in g]
                prob = sum(predict(m, img, tta=args.tta) for m in ms) / len(ms)
        if args.save_probs:
            np.save(Path(args.save_probs) / f"{p.stem}.npy", (prob * 255).astype(np.uint8))
        if scorer is not None and refiners:
            lab, scores = select_instances(prob, img, p.stem, scorer, args.cut, return_scores=True)
            lab = refine_instances(img, prob, lab, scores, refiners, cut=-1.0, thr=args.ref_thr,
                                   blend=args.ref_blend)
        elif scorer is not None:
            lab = select_instances(prob, img, p.stem, scorer, args.cut)
        else:
            lab = prob_to_instances(prob, args.thr, args.min_area, args.join, args.thr_high)
        rows += instances_to_rows(p.stem, lab)
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    check_submission(df)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} filament, {df.filament_id.str.rsplit('_', n=1).str[0].nunique()} görüntü -> {args.out}")


if __name__ == "__main__":
    main()
