"""Etiket denetimi (confident learning tarzı): anotatör başına şüpheli bölgeler.

OOF tahminleri (görüntüyü görmemiş modeller) + aday seçici skorları kullanılır.
  - Kaçırılmış filament: seçici skoru > S_OMIT olan aday, bu anotatörün hiçbir çizimine dokunmuyor
  - Sahte etiket: anotatörün çizdiği filament içinde OOF ortalama olasılığı < P_SPUR
Kalibrasyon (çok anotatörlü görüntüler): şüphenin başka bir anotatörce doğrulanma oranı.
Çıktı: outputs/label_audit/{stem}.pkl -> {anotatör_öneki: (kaçırma_rle, sahte_rle)}, özet tablo.

Örnek: PYTHONPATH=src python experiments/label_audit.py --s_omit 0.5 --p_spur 0.1
"""
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mu

from data import OUT, load_records, poly_to_mask
from post import prob_to_instances
from scorer import CAND

OOF = OUT / "oof" / "ens3_mean_tta2"
DST = OUT / "label_audit"
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


def enc(m):
    return mu.encode(np.asfortranarray(m.astype(np.uint8)))


def audit_image(args):
    stem, rec, scores, s_omit, p_spur = args
    prob = np.load(next(OOF.glob(f"fold*/val_probs/{stem}.npy"))).astype(np.float32) / 255
    lab = prob_to_instances(prob, **CAND)
    ann = {}
    for ai, anns in rec.items():
        polys = [poly_to_mask(a["seg"]) for a in anns]
        ann[ai.split("-")[0]] = (polys, np.any(polys, 0) if polys else np.zeros((2048, 2048), bool))
    strong = [k for k, sc in scores.items() if sc > s_omit]
    out, rows = {}, []
    for a, (polys, m_a) in ann.items():
        others = [m for b, (_, m) in ann.items() if b != a]
        omit = np.zeros((2048, 2048), bool)
        for k in strong:
            inst = lab == k
            if not (inst & m_a).any():
                omit |= cv2.dilate(inst.astype(np.uint8), K5) > 0
                rows.append(dict(stem=stem, annot=a, kind="kaçırma", score=scores[k], n_annot=len(ann),
                                 confirmed=any((inst & m).any() for m in others) if others else None))
        spur = np.zeros((2048, 2048), bool)
        for pm in polys:
            mp = float(prob[pm].mean()) if pm.any() else 1.0
            if mp < p_spur:
                spur |= pm
                rows.append(dict(stem=stem, annot=a, kind="sahte", score=mp, n_annot=len(ann),
                                 confirmed=(not any((pm & m).any() for m in others)) if others else None))
        out[a] = (enc(omit), enc(spur))
    with open(DST / f"{stem}.pkl", "wb") as f:
        pickle.dump(out, f)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s_omit", type=float, default=0.5)
    ap.add_argument("--p_spur", type=float, default=0.1)
    args = ap.parse_args()
    DST.mkdir(parents=True, exist_ok=True)
    records = load_records()
    cand = pd.read_csv(OOF / "candidates_scored.csv")
    sc = {s: dict(zip(g.inst, g.score)) for s, g in cand.groupby("stem")}
    jobs = [(s, records[s], sc.get(s, {}), args.s_omit, args.p_spur) for s in sorted(records)]
    with ProcessPoolExecutor(16) as ex:
        rows = [r for img in ex.map(audit_image, jobs, chunksize=4) for r in img]
    df = pd.DataFrame(rows)
    df.to_csv(DST / "suspects.csv", index=False)
    n_ann_fil = sum(len(a) for rec in records.values() for a in rec.values())
    print(f"toplam anotatör-filament: {n_ann_fil}")
    for kind, g in df.groupby("kind"):
        multi = g[g.n_annot > 1]
        print(f"\n== {kind}: {len(g)} şüphe ({g.stem.nunique()} görüntü)")
        print(f"   çok anotatörlü görüntülerde başka anotatörce doğrulanma oranı: {multi.confirmed.mean():.1%} (n={len(multi)})")
        if kind == "kaçırma":
            for t in [0.4, 0.5, 0.6, 0.7]:
                mm = multi[multi.score > t]
                print(f"   skor>{t}: n={len(mm)}, doğrulanma {mm.confirmed.mean():.1%}")
        else:
            for t in [0.05, 0.1, 0.2]:
                mm = multi[multi.score < t]
                print(f"   ort. olasılık<{t}: n={len(mm)}, başkası çizmemiş {mm.confirmed.mean():.1%}")


if __name__ == "__main__":
    main()
