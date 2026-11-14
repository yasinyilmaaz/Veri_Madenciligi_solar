"""Hata atlası: modelin desteği düşük ama görüntüde kontrastı
belirgin GT kayıtlarını nihai sistemle (ens1536 + 0,5·LightGBM + 0,5·CNN > 0,25) yan yana gösterir.

Seçim: outputs/diagnostics/photometry_20261031 CSV'sinde kind == 'sinyal_yok' ve kontrast, eşleşen
grubun medyanından (0,12143) büyük ya da eşit (97 kayıt, 58 görüntü; tek S modelinin f0+1 OOF'una göre).
Her hedef için alan, kontrast ve diskteki konumu en yakın, havuzda eşleşen bir kontrol kaydı eklenir.
Paneller: ham JPEG | anotatör katmanları (her anotatör ayrı renk, hedef kalın) | ens1536 OOF olasılığı |
adaylar (gri: seçilmeyen, yeşil: teslim edilen). Yalnız keşif amaçlı; eğitim veya eşik seçimi için
kullanılmaz (yöntem seçiminde kullanılırsa bu örnekler bağımsız doğrulama sayılmaz).
Örnek: PYTHONPATH=src python experiments/error_atlas.py
"""
import hashlib
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pycocotools.mask as mu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from artifacts import load_candidates_scored, mean_cnn, merge_exact  # noqa: E402
from data import TRAIN_IMG, get_folds, load_records, poly_to_rle, read_image  # noqa: E402

DIAG = Path("outputs/diagnostics/photometry_20261031/filament_photometry.csv")
OOF = Path("outputs/oof/ens1536_tta2")
OUT = Path("outputs/diagnostics/error_atlas")
MATCHED_MEDIAN = 0.1214285705612245
COLORS = ["#e6194b", "#4363d8", "#f58231", "#911eb4", "#46f0f0"]
ROWS_PER_PAGE = 6


def gt_records(rec):
    """Görüntünün GT kayıtları, fotometri CSV'sindeki k sırasıyla (anotatör sırası, sonra kişi-içi sıra)."""
    out = []
    for ai, anns in rec.items():
        for n, a in enumerate(anns, 1):
            r = poly_to_rle(a["seg"])
            out.append(dict(ai=ai, n=n, rle=r, rle_sha1=hashlib.sha1(r["counts"]).hexdigest()[:12]))
    return out


def selection():
    """Teslim seçimi: ens1536 adayları, skor = 0,5·LightGBM + 0,5·(3 tohum CNN ortalaması)."""
    df = merge_exact(load_candidates_scored(OOF), mean_cnn(OOF), ["stem", "inst"], "aday", "CNN skoru")
    df["final"] = 0.5 * df.score + 0.5 * df.cnn
    df["sel"] = df.final > 0.25
    return df


def best_iou(g, rles):
    if not rles:
        return 0.0, -1
    iou = np.asarray(mu.iou([g], rles, [0] * len(rles))).ravel()
    return float(iou.max()), int(iou.argmax())


def pick_controls(diag, targets):
    """Her hedefe alan (log), kontrast ve disk merkezine uzaklıkça en yakın, farklı görüntüden eşleşen kayıt."""
    pool = diag[diag.kind == "eşleşen"].copy()
    feats = lambda d: np.stack([np.log(d.area), d.contrast, np.hypot(d.cx - 1024, d.cy - 1024) / 1024], 1)
    P, T = feats(pool), feats(targets)
    mu_, sd = P.mean(0), P.std(0)
    P, T = (P - mu_) / sd, (T - mu_) / sd
    ctrl = []
    for i, t in enumerate(T):
        dist = ((P - t) ** 2).sum(1)
        dist[pool.stem.to_numpy() == targets.stem.iloc[i]] = np.inf
        ctrl.append(pool.index[int(dist.argmin())])
    return pool.loc[ctrl].reset_index(drop=True)


def describe(row, rec, cands, folds):
    gts = gt_records(rec[row.stem])
    g = gts[int(row.k)]
    c = cands[cands.stem == row.stem]
    rles = [{"size": [2048, 2048], "counts": r.encode()} for r in c.rle]
    pool_iou, pi = best_iou(g["rle"], rles)
    sel = c[c.sel]
    sel_iou, _ = best_iou(g["rle"], [{"size": [2048, 2048], "counts": r.encode()} for r in sel.rle])
    status = ("teslimde_eşleşti" if sel_iou > 0.5 else "havuzda_seçilmedi" if pool_iou > 0.5 else "havuzda_yok")
    return dict(stem=row.stem, k=int(row.k), annotator_image_id=g["ai"], kisi_ici_sira=g["n"], rle_sha1=g["rle_sha1"],
                fold=int(folds[row.stem]), n_annot=len(rec[row.stem]), grup=row.grup, area=int(row.area),
                contrast=round(float(row.contrast), 4), havuz_iou=round(pool_iou, 3), teslim_iou=round(sel_iou, 3),
                havuz_skor=round(float(c.final.iloc[pi]), 3) if pi >= 0 else np.nan, durum=status)


def draw_row(axes, info, rec, cands, title):
    s = info["stem"]
    gts = gt_records(rec[s])
    target = gts[info["k"]]
    m = mu.decode(target["rle"]).astype(bool)
    ys, xs = np.nonzero(m)
    half = max(160, int(max(np.ptp(ys), np.ptp(xs)) * 0.75))
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    y0, y1 = max(0, cy - half), min(2048, cy + half)
    x0, x1 = max(0, cx - half), min(2048, cx + half)
    win = np.s_[y0:y1, x0:x1]
    img = read_image(TRAIN_IMG / f"{s}.jpeg")[win]
    prob = np.load(OOF / f"fold{info['fold']}" / "val_probs" / f"{s}.npy")[win]
    for ax in axes:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]), ax.set_yticks([])
    axes[0].set_title(title, fontsize=7, loc="left")
    annots = list(rec[s])
    for g in gts:
        gm = mu.decode(g["rle"])[win]
        col = COLORS[annots.index(g["ai"]) % len(COLORS)]
        axes[1].contour(gm, levels=[0.5], colors=[col], linewidths=2.2 if g is target else 0.7)
    axes[1].set_title(" ".join(f"{a.split('-')[0]}" for a in annots), fontsize=6)
    axes[2].imshow(prob, cmap="magma", vmin=0, vmax=255, alpha=0.75)
    axes[2].contour(m[win], levels=[0.5], colors=["cyan"], linewidths=0.8)
    c = cands[cands.stem == s]
    for r, sel in zip(c.rle, c.sel):
        cm = mu.decode({"size": [2048, 2048], "counts": r.encode()})[win]
        if cm.any():
            axes[3].contour(cm, levels=[0.5], colors=["#3cb44b" if sel else "#aaaaaa"], linewidths=1.0)
    axes[3].contour(m[win], levels=[0.5], colors=["cyan"], linewidths=0.8, linestyles="dashed")
    axes[3].set_title(f"havuz IoU {info['havuz_iou']:.2f} · teslim IoU {info['teslim_iou']:.2f}", fontsize=6)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec, folds = load_records(), get_folds().set_index("stem")["fold"]
    diag = pd.read_csv(DIAG)
    targets = diag[(diag.kind == "sinyal_yok") & (diag.contrast >= MATCHED_MEDIAN)].reset_index(drop=True)
    controls = pick_controls(diag, targets)
    cands = selection()
    tinfo = [describe(r, rec, cands, folds) for r in targets.itertuples()]
    cinfo = [describe(r, rec, cands, folds) for r in controls.itertuples()]
    tab = pd.DataFrame(tinfo)
    tab = tab.join(pd.DataFrame(cinfo).add_prefix("kontrol_")[["kontrol_stem", "kontrol_k", "kontrol_annotator_image_id",
                                                               "kontrol_area", "kontrol_contrast", "kontrol_durum"]])
    tab.to_csv(OUT / "atlas.csv", index=False)
    print(f"{len(tab)} hedef kayıt, {tab.stem.nunique()} görüntü")
    print("nihai sistemde hedefler:", tab.durum.value_counts().to_dict())
    print("hedefler × uzlaşı grubu:\n", pd.crosstab(tab.grup, tab.durum).to_string())
    print("kontroller:", tab.kontrol_durum.value_counts().to_dict())
    order = tab.sort_values(["durum", "stem", "k"]).index
    for p in range(0, len(order), ROWS_PER_PAGE):
        idx = order[p:p + ROWS_PER_PAGE]
        fig, axes = plt.subplots(2 * len(idx), 4, figsize=(11, 5.6 * len(idx)))
        for j, i in enumerate(idx):
            t, c = tinfo[i], cinfo[i]
            draw_row(axes[2 * j], t, rec, cands, f"HEDEF {t['stem']} k{t['k']} {t['annotator_image_id']} · {t['grup']} · "
                     f"alan {t['area']} · C {t['contrast']:.3f} · {t['durum']}")
            draw_row(axes[2 * j + 1], c, rec, cands, f"kontrol {c['stem']} k{c['k']} · alan {c['area']} · C {c['contrast']:.3f}")
        fig.tight_layout()
        fig.savefig(OUT / f"atlas_{p // ROWS_PER_PAGE + 1:02d}.png", dpi=90)
        plt.close(fig)
    print("sayfalar:", OUT)


if __name__ == "__main__":
    main()
