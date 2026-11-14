"""Deney 24: eğitim etiketlerini dakikalar içindeki komşu GONG karelerine taşır (forum 748045'te izinli).

Her eğitim görüntüsü için aynı istasyonun 3–10 dakika önceki ve sonraki birer karesi seçilir (farklı istasyon
kareleri seeing farkı yüzünden güvenle hizalanamadı). Test zaman damgalarının ±1 saati dışlanır.
Komşu kare disk çemberiyle hizalanır (build_swefil.register); doğrulama payı düşükse atılır.
Hedef, tüm anotatörlerin taşınmış maskelerinin ortalamasıdır (0–255 PNG).
Fold f listesi yalnızca kaynağı fold f dışında olan komşuları içerir (validasyon sızıntısı yok).
İndirme tek bağlantıyla ve 2 sn aralıkla yapılır (forum 749225: yoğun istekte 403).
Örnek: PYTHONPATH=src python experiments/label_transfer.py && train.py ... --ext_pairs outputs/label_transfer/pairs_fold{fold}.csv --pseudo_n 560
"""
import argparse
import re
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd

from build_swefil import register
from data import OUT, ROOT, TEST_IMG, TRAIN_IMG, get_folds, load_records, poly_to_mask, read_image
from download_gong import BASE, clean, fetch

DST = ROOT / "external" / "gong_transfer"
MIN_CC = 0.05


def ts(stem):
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S")


def pick_neighbors(stem, names, test_times, k=2):
    t0, st0 = ts(stem), stem[-2]
    cands = []
    for n in names:
        if n == stem:
            continue
        dt = abs((ts(n) - t0).total_seconds()) / 60
        if dt > 10 or any(abs((ts(n) - tt).total_seconds()) < 3600 for tt in test_times):
            continue
        cands.append((n, dt))
    # farklı istasyon kareleri güvenle hizalanamıyor (ince yapı korelasyonu ~0,01; seeing farkı),
    # bu yüzden yalnızca aynı istasyonun 3–10 dk önceki ve sonraki birer karesi alınır
    same = [c for c in cands if c[0][-2] == st0 and c[1] >= 3]
    before = sorted([c for c in same if ts(c[0]) < t0], key=lambda c: c[1])
    after = sorted([c for c in same if ts(c[0]) > t0], key=lambda c: c[1])
    return [c[0] for c in (before[:1] + after[:1])][:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    (DST / "img").mkdir(parents=True, exist_ok=True)
    (DST / "mask").mkdir(parents=True, exist_ok=True)
    out = OUT / "label_transfer"
    out.mkdir(parents=True, exist_ok=True)
    records = load_records()
    folds = get_folds().set_index("stem")["fold"]
    test_times = [ts(p.stem) for p in TEST_IMG.glob("*.jpeg")]
    stems = sorted(records)[: args.limit or None]
    listing, rows, stat = {}, [], {"komşu_yok": 0, "indirilemedi": 0, "hizalama": 0, "tamam": 0}
    for i, s in enumerate(stems):
        day = s[:8]
        if day not in listing:
            html = fetch(f"{BASE}/{day[:6]}/{day}/")
            time.sleep(args.delay)
            listing[day] = sorted(set(re.findall(r"(\d{14}[A-Z]h)\.jpg", html.decode("latin1")))) if html else []
        nbrs = pick_neighbors(s, listing[day], test_times)
        if not nbrs:
            stat["komşu_yok"] += 1
        src = None
        for n in nbrs:
            img_p, mask_p = DST / "img" / f"{n}.jpeg", DST / "mask" / f"{n}__{s}.png"
            if mask_p.exists():
                rows.append(dict(img=str(img_p), mask=str(mask_p), src=s, fold=int(folds[s]), nbr=n))
                stat["tamam"] += 1
                continue
            if not img_p.exists():
                buf = fetch(f"{BASE}/{day[:6]}/{day}/{n}.jpg")
                time.sleep(args.delay)
                g = clean(buf) if buf else None
                if g is None:
                    stat["indirilemedi"] += 1
                    continue
                cv2.imwrite(str(img_p), g, [cv2.IMWRITE_JPEG_QUALITY, 95])
            g = cv2.imread(str(img_p), 0)
            src = read_image(TRAIN_IMG / f"{s}.jpeg") if src is None else src
            try:
                M, cc = register(src, g)
            except (cv2.error, ValueError):
                cc = 0
            if cc < MIN_CC:
                stat["hizalama"] += 1
                continue
            acc = np.zeros((2048, 2048), np.float32)
            for anns in records[s].values():
                moved = [(np.asarray(a["seg"][0], np.float64).reshape(-1, 2) @ M[:, :2].T + M[:, 2]).ravel().tolist()
                         for a in anns]
                if moved:
                    acc += poly_to_mask(moved)
            acc /= len(records[s])
            cv2.imwrite(str(mask_p), np.round(acc * 255).astype(np.uint8))
            rows.append(dict(img=str(img_p), mask=str(mask_p), src=s, fold=int(folds[s]), nbr=n, cc=round(cc, 3)))
            stat["tamam"] += 1
        if i % 50 == 0:
            print(i, len(stems), stat, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "pairs_all.csv", index=False)
    for f in range(5):
        df[df.fold != f][["img", "mask"]].to_csv(out / f"pairs_fold{f}.csv", index=False)
    print("bitti", stat, "| istasyon farklı:", int((df.nbr.str[-2] != df.src.str[-2]).sum()), "/", len(df), flush=True)


if __name__ == "__main__":
    main()
