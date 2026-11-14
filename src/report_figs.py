"""Rapor figürleri (İngilizce): teslim sisteminin rubrik dağılımları (5-fold OOF, 707 görüntü).

Girdi: outputs/rubric_final_dist.npz (src/rubric_eval.py üretir). Çıktı: docs/figures/fig_rubric_dist.{pdf,png}.
Paneller: (a) örtüşen GT–tahmin çiftlerinin IoU dağılımı, PQ eşleşme eşiği 0,5 işaretli; (b) Dice dağılımı;
(c) GT başına örtüşen tahmin sayısı (0 = kaçan, ≥2 = parçalanma); (d) tahmin başına örtüşen GT sayısı
(0 = karşılıksız, ≥2 = birleştirme).
Ayrıca docs/figures/fig_pipeline.{pdf,png} (hat şeması) ve docs/figures/fig_qualitative.{pdf,png}: görüntü başına PQ dağılımının 90./50./10. yüzdeliğinden birer tek-anotatörlü
OOF görüntüsü (ham | GT | teslim tahmini). Test görüntüsü kullanılmaz.
Örnek: python src/report_figs.py
"""
from pathlib import Path

import matplotlib
import cv2
import numpy as np
import pycocotools.mask as mu

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SERIES = "#2a78d6"  # tek seri: referans paletin 1. kategorik rengi
GT_COLOR = "#eb6834"  # referans paletin 2. kategorik rengi
# 90./50./10. yüzdelik görüntü PQ'su (outputs/per_image_pq_final.csv; ≥6 GT, tek anotatör)
EXAMPLES = [("20120803164614Mh", "90th percentile"), ("20131116185334Ch", "median"), ("20130921060834Lh", "10th percentile")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OUT = Path("docs/figures")


def style(ax, title):
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def hist(ax, x, title, xlabel, mark=None):
    ax.hist(x, bins=np.linspace(0, 1, 26), color=SERIES, edgecolor="white", linewidth=1.0)
    if mark is not None:
        ax.axvline(mark, color=INK, linewidth=1.0, linestyle="--")
        ax.text(mark - 0.02, ax.get_ylim()[1] * 0.92, f"PQ match\nthreshold {mark}", fontsize=7, color=INK, ha="right", va="top")
    style(ax, title)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel("GT–prediction pairs", fontsize=8, color=MUTED)


def degree(ax, deg, title, labels):
    k = np.clip(deg, 0, 3)
    share = np.array([(k == i).mean() for i in range(4)])
    bars = ax.bar(range(4), share, color=SERIES, width=0.7, edgecolor="white", linewidth=1.0)
    for b, v in zip(bars, share):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{100 * v:.1f}%", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(4), labels, fontsize=7)
    ax.set_ylim(0, 1)
    style(ax, title)
    ax.set_ylabel("share", fontsize=8, color=MUTED)


def contours(ax, rles, color, size=512):
    for r in rles:
        m = cv2.resize(mu.decode(r), (size, size), interpolation=cv2.INTER_NEAREST)
        ax.contour(m, levels=[0.5], colors=[color], linewidths=0.9)


def qualitative():
    from data import TRAIN_IMG, load_records, poly_to_rle, read_image
    from rubric_eval import FINAL, SELECTED, selected
    import pandas as pd

    rec, pred = load_records(), selected(*SELECTED[FINAL])
    pq = pd.read_csv("outputs/per_image_pq_final.csv").set_index("stem")["pq"]
    fig, ax = plt.subplots(len(EXAMPLES), 3, figsize=(7.2, 2.5 * len(EXAMPLES)))
    for i, (s, label) in enumerate(EXAMPLES):
        img = cv2.resize(read_image(TRAIN_IMG / f"{s}.jpeg"), (512, 512), interpolation=cv2.INTER_AREA)
        gts = [poly_to_rle(a["seg"]) for anns in rec[s].values() for a in anns]
        prs = [{"size": [2048, 2048], "counts": r.encode()} for r in pred[pred.filament_id.str.startswith(s + "_")].segmentation_rle]
        for a in ax[i]:
            a.imshow(img, cmap="gray", vmin=0, vmax=255)
            a.set_xticks([]), a.set_yticks([])
            for side in a.spines.values():
                side.set_visible(False)
        contours(ax[i, 1], gts, GT_COLOR)
        contours(ax[i, 2], prs, SERIES)
        ax[i, 0].set_title(f"{label}, image PQ {pq[s]:.2f}", fontsize=8, loc="left", color=INK)
        ax[i, 0].set_ylabel(s, fontsize=7, color=MUTED)
        ax[i, 1].set_title(f"annotation ({len(gts)} filaments)", fontsize=8, loc="left", color=INK)
        ax[i, 2].set_title(f"prediction ({len(prs)} instances)", fontsize=8, loc="left", color=INK)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_qualitative.{ext}", dpi=200)
    print("yazıldı:", OUT / "fig_qualitative.pdf")


STEPS = [
    ("Input", "2048×2048 GONG\nH-alpha JPEG"),
    ("Normalise", "on-disk mean / std"),
    ("U-Net × 10", "EffNetV2-S, -M\n× 5 folds, D4 TTA"),
    ("Probability", "mean over\n10 models"),
    ("Candidates", "p > 0.4, saddle\nlink p > 0.15,\narea ≥ 100 px"),
    ("Selector", "0.5·LightGBM\n+ 0.5·ResNet-18\n> 0.25"),
    ("Output", "disjoint\ninstance RLE"),
]


def pipeline():
    """Hat şeması: kutular ve oklar (İngilizce)."""
    fig, ax = plt.subplots(figsize=(7.2, 1.6))
    w, gap = 0.118, 0.025
    for i, (head, body) in enumerate(STEPS):
        x = i * (w + gap)
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, 0.12), w, 0.76, boxstyle="round,pad=0.004,rounding_size=0.015",
                                                       facecolor="#eef4fc" if 2 <= i <= 5 else "#f4f4f2",
                                                       edgecolor=SERIES if 2 <= i <= 5 else MUTED, linewidth=0.9))
        ax.text(x + w / 2, 0.76, head, ha="center", va="center", fontsize=7.5, color=INK, weight="bold")
        ax.text(x + w / 2, 0.42, body, ha="center", va="center", fontsize=6.3, color=INK, linespacing=1.25)
        if i < len(STEPS) - 1:
            ax.annotate("", xy=(x + w + gap - 0.002, 0.5), xytext=(x + w + 0.002, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, shrinkA=0, shrinkB=0))
    ax.set_xlim(-0.005, len(STEPS) * (w + gap) - gap + 0.005)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_pipeline.{ext}", dpi=200)
    plt.close(fig)
    print("yazıldı:", OUT / "fig_pipeline.pdf")


def main():
    d = np.load("outputs/rubric_final_dist.npz")
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2))
    hist(ax[0, 0], d["iou"], f"(a) IoU of overlapping pairs (n={len(d['iou'])}, mean {d['iou'].mean():.3f})", "IoU",
         mark=0.5)
    hist(ax[0, 1], d["dice"], f"(b) Dice of overlapping pairs (mean {d['dice'].mean():.3f})", "Dice")
    degree(ax[1, 0], d["gt_deg"], f"(c) Overlapping predictions per GT (n={len(d['gt_deg'])})",
           ["0\nmissed", "1", "2\nsplit", "≥3"])
    degree(ax[1, 1], d["pred_deg"], f"(d) Overlapping GTs per prediction (n={len(d['pred_deg'])})",
           ["0\nunmatched", "1", "2\nmerge", "≥3"])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_rubric_dist.{ext}", dpi=200)
    print("yazıldı:", OUT / "fig_rubric_dist.pdf")
    plt.close(fig)
    pipeline()
    qualitative()


if __name__ == "__main__":
    main()
