"""GONG H-alpha arşivinden etiketsiz görüntü indirme (pseudo-labeling için).

Test günlerinin +-3 günü ve train günleri hariç tutulur. Arşivdeki köşe
yazıları/logo, Kaggle görüntülerindeki gibi arka planla doldurulur.

Örnek: PYTHONPATH=src python experiments/download_gong.py --step 2 --per_day 2
"""
import argparse
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import cv2
import numpy as np

from data import ROOT, TEST_IMG, TRAIN_IMG

BASE = "https://gong2.nso.edu/HA/hag"
OUT_DIR = ROOT / "external" / "gong"
STATIONS = set("BCLMTU")
# arşiv görüntüsündeki yazı/logo kutuları (x, y, w, h), 2048x2048
BOXES = [(0, 0, 580, 170), (1580, 0, 468, 120), (0, 1945, 540, 103), (1870, 1870, 178, 178)]


def fetch(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def clean(buf):
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (2048, 2048):
        return None
    for x, y, w, h in BOXES:
        ring = np.concatenate([
            img[max(y - 10, 0):y, x:x + w].ravel(), img[y + h:y + h + 10, x:x + w].ravel(),
            img[y:y + h, max(x - 10, 0):x].ravel(), img[y:y + h, x + w:x + w + 10].ravel()])
        img[y:y + h, x:x + w] = int(np.median(ring)) if ring.size else 0
    return img


def excluded_days():
    bad = set()
    for p in TEST_IMG.glob("*.jpeg"):
        d = date(int(p.stem[:4]), int(p.stem[4:6]), int(p.stem[6:8]))
        bad |= {d + timedelta(k) for k in range(-3, 4)}
    for p in TRAIN_IMG.glob("*.jpeg"):
        bad.add(date(int(p.stem[:4]), int(p.stem[4:6]), int(p.stem[6:8])))
    return bad


def do_day(d, per_day, rng):
    ymd = d.strftime("%Y%m%d")
    html = fetch(f"{BASE}/{ymd[:6]}/{ymd}/")
    if html is None:
        return 0
    names = sorted(set(re.findall(r"(\d{14}([A-Z])h)\.jpg", html.decode("latin1"))))
    by_station = {}
    for stem, st in names:
        if st in STATIONS:
            by_station.setdefault(st, []).append(stem)
    got = 0
    for st in rng.sample(sorted(by_station), min(per_day, len(by_station))):
        stem = rng.choice(by_station[st])
        out = OUT_DIR / f"{stem}.jpeg"
        if out.exists():
            got += 1
            continue
        buf = fetch(f"{BASE}/{ymd[:6]}/{ymd}/{stem}.jpg")
        img = clean(buf) if buf else None
        if img is not None:
            cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            got += 1
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--per_day", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--out", default=None, help="çıktı klasörü (varsayılan external/gong)")
    args = ap.parse_args()
    global OUT_DIR
    if args.out:
        OUT_DIR = ROOT / args.out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bad = excluded_days()
    d0, d1 = date.fromisoformat(args.start), date.fromisoformat(args.end)
    days = [d0 + timedelta(k) for k in range(0, (d1 - d0).days + 1, args.step)]
    days = [d for d in days if d not in bad]
    print(f"{len(days)} gün", flush=True)
    rngs = [random.Random(d.toordinal()) for d in days]
    total = 0
    with ThreadPoolExecutor(args.threads) as ex:
        for i, n in enumerate(ex.map(lambda a: do_day(*a), [(d, args.per_day, r) for d, r in zip(days, rngs)])):
            total += n
            if i % 100 == 0:
                print(f"{i}/{len(days)} gün, {total} görüntü", flush=True)
    print("bitti:", total)


if __name__ == "__main__":
    main()
