"""Son sistemin son-işlem varyantları: OOF tahmini + test gönderimi (03.11, LB yoklamaları).

Seçici skoru = 0,5·LightGBM + 0,5·CNN (son sistem). Varyantlar seçilen adaylara uygulanır:
  cutXX   : kesme değişir (0,25 yerine)
  fill300 : maskelerdeki delikler doldurulur, 300 px altı seçilenler atılır
  mergeK  : aralarında K px'ten az boşluk olan seçilmiş parçalar tek örnek olur
  thrXX / linkXX / area200: aday üretimi değişir (CAND_VARIANTS; yalnız test)
  ensXXX  : olasılık haritası başka run birleşiminden (ENS_VARIANTS; yalnız test)
  mixMMcCC: seçici karışımı MM/100·LightGBM + (1−MM/100)·CNN, kesme CC/100 (yalnız test)
OOF: outputs/oof/ens1536_tta2 aday tablosu (rle) + iç içe skorlar. Test: outputs/test_probs/ens1536_tta2
önbelleği (uint8, üye başına aşağı kesme; paket çıktısından ≤1/255 farklı), aday skorları weights/ens1536 ile.

Örnek: PYTHONPATH=src python experiments/lb_variants.py --variants cut15 cut35 fill300 merge6
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils
from scipy import ndimage as ndi

from artifacts import load_candidates_scored, mean_cnn, merge_exact
from data import OUT, gt_frame, load_records
from metric import check_submission, compute_pq, instances_to_rows

MIX, CUT = 0.5, 0.25
TEST_IMG = Path("MAGFiLO_1.0_Kaggle_2026/test/test_images")


def apply(variant, masks):
    """masks: seçilmiş adayların ikili maskeleri -> etiket haritası (örtüşmesiz)."""
    lab = np.zeros((2048, 2048), np.int32)
    if variant.startswith("fill"):
        masks = [ndi.binary_fill_holes(m) for m in masks]
        masks = [m for m in masks if m.sum() >= int(variant[4:])]
    for k, m in enumerate(masks, 1):
        lab[m & (lab == 0)] = k
    if variant.startswith("merge") and lab.max() > 1:
        r = int(variant[5:])
        near, _ = ndi.label(cv2.dilate((lab > 0).astype(np.uint8), np.ones((r, r), np.uint8)))
        # örnek ile yakınlık bileşeni çizgesinde bağlı olanlar birleşir (çok parçalı örnekler bölünmez)
        parent = list(range(lab.max() + 1))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        first = {}
        for k, c in np.unique(np.stack([lab[lab > 0], near[lab > 0]]), axis=1).T:
            if c in first:
                parent[find(k)] = find(first[c])
            else:
                first[c] = k
        lab = np.array([find(k) for k in range(len(parent))], np.int32)[lab]
    return lab


def cut_of(variant):
    if variant.startswith("mix"):  # mixMMcCC: LightGBM ağırlığı MM/100, kesme CC/100
        return int(variant[6:8]) / 100
    return int(variant[3:]) / 100 if variant.startswith("cut") else CUT


def mix_of(variant):
    return int(variant[3:5]) / 100 if variant.startswith("mix") else MIX


def oof_image(args):
    stem, rles, scores, variants = args
    masks = [mask_utils.decode({"size": [2048, 2048], "counts": r.encode("ascii")}).astype(bool) for r in rles]
    out = {}
    for v in variants:
        keep = [m for m, s in zip(masks, scores) if s > cut_of(v)]
        out[v] = instances_to_rows(stem, apply(v, keep))
    return out


def oof(variants):
    d = OUT / "oof" / "ens1536_tta2"
    df = merge_exact(load_candidates_scored(d), mean_cnn(d), ["stem", "inst"], "aday", "cnn")
    df["sel"] = MIX * df.score + (1 - MIX) * df.cnn
    records = load_records()
    gt = gt_frame(records, sorted(records))
    jobs = [(s, g.rle.tolist(), g.sel.tolist(), ["base"] + variants) for s, g in df.groupby("stem")]
    rows = {v: [] for v in ["base"] + variants}
    with ProcessPoolExecutor(16) as ex:
        for r in ex.map(oof_image, jobs, chunksize=8):
            for v in r:
                rows[v] += r[v]
    for v, rr in rows.items():
        pred = pd.DataFrame(rr, columns=["filament_id", "segmentation_rle"])
        res = compute_pq(gt, pred)
        print(f"OOF {v:8s} PQ {res['pq']:.4f} SQ {res['sq']:.3f} RQ {res['rq']:.3f} n {len(pred)}", flush=True)


def test(variants):
    import torch
    from infer import load_bundle, score_candidates
    cfg, _, scorer, cnns = load_bundle("weights/ens1536")
    rows = {v: [] for v in variants}
    for p in sorted(TEST_IMG.glob("*.jpeg")):
        prob = np.load(OUT / "test_probs" / "ens1536_tta2" / f"{p.stem}.npy").astype(np.float32) / 255
        from data import read_image
        with torch.no_grad():
            lab, X = score_candidates(prob, read_image(p), p, scorer, cnns, cfg)
        for v in variants:
            sc = mix_of(v) * X.lgb.to_numpy() + (1 - mix_of(v)) * X.cnn.to_numpy() if len(X) else []
            keep = X.inst.to_numpy()[sc > cut_of(v)] if len(X) else []
            rows[v] += instances_to_rows(p.stem, apply(v, [lab == k for k in keep]))
    for v, rr in rows.items():
        df = pd.DataFrame(rr, columns=["filament_id", "segmentation_rle"])
        check_submission(df)
        df.to_csv(OUT / f"sub_var_{v}.csv", index=False)
        print(f"test {v}: {len(df)} örnek -> outputs/sub_var_{v}.csv", flush=True)


CAND_VARIANTS = {"thr35": dict(thr=0.35), "thr45": dict(thr=0.45), "link10": dict(link_thr=0.10),
                 "link25": dict(link_thr=0.25), "area200": dict(min_area=200)}


def test_cand(variants):
    """Aday üretimi değişen varyantlar (seçici aynı, kesme 0,25; yalnız test, LB yoklaması)."""
    import torch
    from data import read_image
    from infer import load_bundle, score_candidates
    from scorer import CAND
    cfg, _, scorer, cnns = load_bundle("weights/ens1536")
    base = dict(CAND)
    paths = sorted(TEST_IMG.glob("*.jpeg"))
    probs = {p.stem: np.load(OUT / "test_probs" / "ens1536_tta2" / f"{p.stem}.npy").astype(np.float32) / 255 for p in paths}
    for v in variants:
        CAND.clear(), CAND.update(base, **CAND_VARIANTS[v])
        rows = []
        for p in paths:
            with torch.no_grad():
                lab, X = score_candidates(probs[p.stem], read_image(p), p, scorer, cnns, cfg)
            keep = X.inst.to_numpy()[X.score.to_numpy() > CUT] if len(X) else []
            rows += instances_to_rows(p.stem, apply(v, [lab == k for k in keep]))
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        check_submission(df)
        df.to_csv(OUT / f"sub_var_{v}.csv", index=False)
        print(f"test {v} {dict(CAND)}: {len(df)} örnek -> outputs/sub_var_{v}.csv", flush=True)
    CAND.clear(), CAND.update(base)


ENS_VARIANTS = {"ensS": ["crop1536_effv2s"], "ensM": ["effv2m_c1536b"],
                "ensSMssl": ["crop1536_effv2s", "effv2m_c1536b", "ssl1536_effv2s"],
                "ensSMstrong": ["crop1536_effv2s", "effv2m_c1536b", "strong1536_effv2s"],
                "ensSMsslstrong": ["crop1536_effv2s", "effv2m_c1536b", "ssl1536_effv2s", "strong1536_effv2s"]}


def test_ens(variants):
    """Olasılık haritası farklı run birleşiminden (run'lar eşit ağırlık); aday, seçici ve kesme aynı (yalnız test)."""
    import torch
    from data import read_image
    from infer import load_bundle, score_candidates
    cfg, _, scorer, cnns = load_bundle("weights/ens1536")
    paths = sorted(TEST_IMG.glob("*.jpeg"))
    for v in variants:
        runs = ENS_VARIANTS[v]
        rows = []
        for p in paths:
            prob = sum(np.load(OUT / "test_probs" / f"{r}_tta2" / f"{p.stem}.npy").astype(np.float32) / 255 for r in runs) / len(runs)
            prob = np.round(prob * 255).astype(np.float32) / 255
            with torch.no_grad():
                lab, X = score_candidates(prob, read_image(p), p, scorer, cnns, cfg)
            keep = X.inst.to_numpy()[X.score.to_numpy() > CUT] if len(X) else []
            rows += instances_to_rows(p.stem, apply(v, [lab == k for k in keep]))
        df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        check_submission(df)
        df.to_csv(OUT / f"sub_var_{v}.csv", index=False)
        print(f"test {v} {runs}: {len(df)} örnek -> outputs/sub_var_{v}.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--skip_test", action="store_true")
    a = ap.parse_args()
    oof(a.variants)
    if not a.skip_test:
        test(a.variants)
