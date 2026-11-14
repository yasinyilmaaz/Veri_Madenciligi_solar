"""Etiketli dış veri setlerini indirir (kurallara uygun, MAGFiLO DEĞİL).

- Zhu et al. 2019 BBSO H-alpha (2013): gün başına birkaç kare, test günleri +-2 hariç
- CHASE MHAS/AHAS (Hu et al. 2026, Zenodo 21582245, CC BY 4.0, 2024-2025)

Örnek: PYTHONPATH=src python experiments/download_external.py --zhu_per_day 3 --chase
"""
import argparse
import random
import re
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from data import ROOT, TEST_IMG

ZHU = "https://sun10.bao.ac.cn/hsos_data/download/filaments-unet-dataset"
CHASE = "https://zenodo.org/api/records/21582245/files/data.zip/content"
EXT = ROOT / "external"


def fetch(url, out):
    if out.exists():
        return True
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(out)
        return True
    except Exception as e:
        print("hata", url, e)
        return False


def test_days(margin):
    bad = set()
    for p in TEST_IMG.glob("*.jpeg"):
        d = date(int(p.stem[:4]), int(p.stem[4:6]), int(p.stem[6:8]))
        bad |= {d + timedelta(k) for k in range(-margin, margin + 1)}
    return bad


def download_zhu(per_day):
    html = urllib.request.urlopen(f"{ZHU}/img/", timeout=120).read().decode()
    names = re.findall(r"(bbso_halph_fi_(\d{8})_\d{6})_nimg\.jpg", html)
    bad = test_days(2)
    by_day = defaultdict(list)
    for stem, ymd in names:
        d = date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
        if d not in bad:
            by_day[ymd].append(stem)
    rng = random.Random(0)
    pick = [s for day in sorted(by_day) for s in rng.sample(sorted(by_day[day]), min(per_day, len(by_day[day])))]
    print(f"Zhu: {len(names)} kare, {len(by_day)} gün (test +-2 hariç), {len(pick)} seçildi", flush=True)
    (EXT / "zhu_bbso" / "img").mkdir(parents=True, exist_ok=True)
    (EXT / "zhu_bbso" / "mask").mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in pick:
        jobs.append((f"{ZHU}/img/{s}_nimg.jpg", EXT / "zhu_bbso" / "img" / f"{s}.jpg"))
        jobs.append((f"{ZHU}/mask/{s}_ntimg.jpg", EXT / "zhu_bbso" / "mask" / f"{s}.jpg"))
    with ThreadPoolExecutor(4) as ex:
        ok = sum(ex.map(lambda j: fetch(*j), jobs))
    print(f"Zhu: {ok}/{len(jobs)} dosya", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zhu_per_day", type=int, default=3)
    ap.add_argument("--chase", action="store_true")
    args = ap.parse_args()
    if args.zhu_per_day:
        download_zhu(args.zhu_per_day)
    if args.chase:
        (EXT / "chase").mkdir(parents=True, exist_ok=True)
        print("CHASE:", fetch(CHASE, EXT / "chase" / "data.zip"), flush=True)


if __name__ == "__main__":
    main()
