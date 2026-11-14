"""Veri okuma: COCO anotasyonları, maske çizimi, fold'lar."""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_IMG = DATA / "train" / "train_images"
TEST_IMG = DATA / "test" / "test_images"
ANN_FILE = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
EXT_IMG = ROOT / "external" / "gong"
OUT = ROOT / "outputs"
SIZE = 2048


def load_records():
    """Her dosya için {stem: {annotator_image_id: [polygon, ...]}} döndürür.

    image_id formatı '<annotator>-<stem>'; aynı dosya birden fazla anotatör
    tarafından etiketlenmiş olabilir ve metrik her birini ayrı skorlar.
    """
    d = json.loads(ANN_FILE.read_text())
    stem_of = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    recs = defaultdict(lambda: defaultdict(list))
    for im in d["images"]:
        recs[stem_of[im["id"]]][im["id"]]  # anotasyonsuz anotatör-görüntüleri de tut
    for a in d["annotations"]:
        recs[stem_of[a["image_id"]]][a["image_id"]].append(
            {"seg": a["segmentation"], "spine": a.get("spine"), "cat": a["category_id"]}
        )
    return {k: dict(v) for k, v in recs.items()}


def poly_to_rle(seg):
    rles = mask_utils.frPyObjects(seg, SIZE, SIZE)
    return mask_utils.merge(rles)


def poly_to_mask(seg):
    return mask_utils.decode(poly_to_rle(seg)).astype(bool)


def gt_frame(records, stems):
    """Metrik için ground truth tablosu: filament_id='<annotator-image>_<n>'."""
    rows = []
    for s in stems:
        for ai, anns in records[s].items():
            for n, a in enumerate(anns, 1):
                counts = poly_to_rle(a["seg"])["counts"].decode("ascii")
                rows.append((f"{ai}_{n}", counts))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def read_image(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def make_folds(records, n_folds=5, seed=42):
    """Dosya bazında grupla, istasyon harfine göre katmanla."""
    from sklearn.model_selection import StratifiedKFold

    stems = sorted(records)
    station = [s[-2] for s in stems]
    df = pd.DataFrame({"stem": stems, "station": station, "n_annot": [len(records[s]) for s in stems]})
    df["fold"] = -1
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    strat = df["station"] + df["n_annot"].astype(str)
    for f, (_, va) in enumerate(skf.split(df, strat)):
        df.loc[va, "fold"] = f
    return df


def make_team_folds(records):
    """Deney 45: ekip dışı doğrulama. Fold = anotatör ekibi (kimliğin ilk 2 hanesi, 01–06 → 0–5).
    Çok anotatörlü görüntülerin hepsi tek ekipten olduğu için her görüntü tam bir ekibe düşer."""
    rows = []
    for s in sorted(records):
        teams = {ai.split("-")[0][:2] for ai in records[s]}
        assert len(teams) == 1, (s, teams)
        rows.append((s, s[-2], len(records[s]), int(teams.pop()) - 1))
    return pd.DataFrame(rows, columns=["stem", "station", "n_annot", "fold"])


def get_folds():
    """FOLD_SCHEME=team ortam değişkeniyle ekip fold'ları (deney 45); varsayılan görüntü fold'ları."""
    import os
    team = os.environ.get("FOLD_SCHEME") == "team"
    path = OUT / ("folds_team.csv" if team else "folds.csv")
    if not path.exists():
        OUT.mkdir(exist_ok=True)
        (make_team_folds if team else make_folds)(load_records()).to_csv(path, index=False)
    return pd.read_csv(path)
