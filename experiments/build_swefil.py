"""SWEFil (Reche & Cid, CC BY 4.0) etiketlerini Kaggle formatındaki GONG karelerine taşır.

SWEFil görüntüleri farklı işlenmiş; aynı zaman damgalı GONG 'hag' karesi arşivden
indirilir, SWEFil görüntüsü disk çemberiyle (ölçek+kayma) bu kareye hizalanır ve poligonlar aynı
dönüşümle taşınır. Test günlerinin +-3 günü atılır; hizalaması zayıf kareler atılır.
Çıktı: external/swefil_gong/{img,mask}/, outputs/swefil_pairs.csv (img, mask, date)

Örnek: PYTHONPATH=src python experiments/build_swefil.py
"""
import json
import urllib.request
from datetime import date, timedelta

import cv2
import numpy as np
import pandas as pd

from data import OUT, ROOT, TEST_IMG, poly_to_mask
from download_gong import BASE, clean

SW = ROOT / "external" / "swefil"
DST = ROOT / "external" / "swefil_gong"
MIN_CC = 0.05  # hizalama doğrulama payı alt sınırı


def load_swefil():
    P = lambda v: json.loads(v) if isinstance(v, str) else v
    polys = {}
    for k in ["train", "test"]:
        d = json.load(open(SW / f"{k}.json"))
        ims = {str(i["id"]): i["file_name"] for i in d["images"]}
        for a in d["annotations"]:
            fn = ims[str(a["image_id"])]
            if "raw" in fn:
                continue
            stem = fn.split("/")[-1].split(".")[0]
            polys.setdefault(stem, [])
            if str(a["category_id"]) != "3":  # güneş lekeleri hariç; 3 filament sınıfı tek sınıf
                polys[stem].extend(P(a["segmentation"]))
    return polys


def fit_disk(img):
    """Güneş diskine daire oturt: Otsu ile disk maskesi, en büyük bileşenin konturuna en küçük kareler çemberi."""
    g = cv2.GaussianBlur(img, (0, 0), 3)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    A = np.c_[2 * c, np.ones(len(c))]
    sol = np.linalg.lstsq(A, (c ** 2).sum(1), rcond=None)[0]
    cx, cy = sol[0], sol[1]
    return cx, cy, np.sqrt(sol[2] + cx ** 2 + cy ** 2)


def bandpass(img):
    x = img.astype(np.float32)
    return cv2.GaussianBlur(x, (0, 0), 3) - cv2.GaussianBlur(x, (0, 0), 20)


def register(src, dst):
    """src -> dst benzerlik dönüşümü (disk çemberinden) ve doğrulama payı.

    Pay = hizalı görüntüde ince yapı (band-pass) korelasyonu - 40 px kaydırılmış haldeki korelasyon.
    Sakin (yapısız) günlerde pay ~0 olur; hizalama doğru olsa bile kanıtlanamaz, bu kareler atılır.
    """
    sx, sy, sr = fit_disk(src)
    dx, dy, dr = fit_disk(dst)
    s = dr / sr
    M = np.array([[s, 0, dx - s * sx], [0, s, dy - s * sy]])
    w, g = bandpass(cv2.warpAffine(src, M, (2048, 2048))), bandpass(dst)
    inner = np.zeros((2048, 2048), np.uint8)
    cv2.circle(inner, (int(dx), int(dy)), int(0.85 * dr), 1, -1)
    m = inner > 0
    c0 = np.corrcoef(w[m], g[m])[0, 1]
    cs = np.mean([np.corrcoef(np.roll(w, sh, ax)[m], g[m])[0, 1] for sh in (40, -40) for ax in (0, 1)])
    return M, float(c0 - cs)


def main():
    polys = load_swefil()
    bad = set()
    for p in TEST_IMG.glob("*.jpeg"):
        d = date(int(p.stem[:4]), int(p.stem[4:6]), int(p.stem[6:8]))
        bad |= {d + timedelta(k) for k in range(-3, 4)}
    (DST / "img").mkdir(parents=True, exist_ok=True)
    (DST / "mask").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], {"test_günü": 0, "indirilemedi": 0, "hizalama": 0}
    for stem in sorted(polys):
        d = date(int(stem[:4]), int(stem[4:6]), int(stem[6:8]))
        if d in bad:
            skipped["test_günü"] += 1
            continue
        out_img = DST / "img" / f"{stem}.jpeg"
        if not out_img.exists():
            try:
                buf = urllib.request.urlopen(f"{BASE}/{stem[:6]}/{stem[:8]}/{stem}.jpg", timeout=60).read()
                g = clean(buf)
            except Exception:
                g = None
            if g is None:
                skipped["indirilemedi"] += 1
                continue
            cv2.imwrite(str(out_img), g, [cv2.IMWRITE_JPEG_QUALITY, 95])
        g = cv2.imread(str(out_img), 0)
        sw = cv2.imread(str(SW / "images" / f"{stem}.jpg"), 0)
        try:
            M, cc = register(sw, g)
        except cv2.error:
            cc = 0
        if cc < MIN_CC:
            skipped["hizalama"] += 1
            continue
        mask = np.zeros((2048, 2048), bool)
        moved = []
        for poly in polys[stem]:
            pts = np.asarray(poly, np.float64).reshape(-1, 2)
            pts = pts @ M[:, :2].T + M[:, 2]
            moved.append(pts.ravel().tolist())
        if moved:
            mask = poly_to_mask(moved)
        cv2.imwrite(str(DST / "mask" / f"{stem}.png"), mask.astype(np.uint8) * 255)
        sx = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        rows.append(dict(img=str(out_img), mask=str(DST / "mask" / f"{stem}.png"), date=str(d), stem=stem,
                         n_fil=len(moved), align_margin=round(float(cc), 4), scale=round(sx, 4),
                         dx=round(float(M[0, 2]), 1), dy=round(float(M[1, 2]), 1)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "swefil_pairs.csv", index=False)
    print(f"{len(df)} kare kullanılabilir, {int(df.n_fil.sum())} filament | atlanan: {skipped}")
    print(df[["align_margin", "scale", "dx", "dy"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
