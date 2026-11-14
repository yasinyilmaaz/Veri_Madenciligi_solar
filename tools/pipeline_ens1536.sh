#!/bin/bash
# Deney 17: 1536 ikili ensemble (EffNetV2-S + EffNetV2-M) 5 fold OOF -> eyer bağlantılı seçici -> test gönderimi.
# OOF işleri bitene kadar bekler (GPU bellek çakışmasını önlemek için test çıkarımı sonra).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
while pgrep -f "^.venv/bin/python src/oof.py --runs outputs/(crop1536_effv2s|effv2m_c1536b)" >/dev/null || \
      [ "$(ls outputs/oof/effv2m_c1536b_tta2/fold4/val_probs 2>/dev/null | wc -l)" -lt 140 ]; do sleep 30; done
.venv/bin/python - <<'PY'
import numpy as np
from pathlib import Path
runs = ["crop1536_effv2s", "effv2m_c1536b"]
for f in range(5):
    out = Path(f"outputs/oof/ens1536_tta2/fold{f}/val_probs"); out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(f"outputs/oof/{runs[0]}_tta2/fold{f}/val_probs").glob("*.npy")):
        m = sum(np.load(f"outputs/oof/{r}_tta2/fold{f}/val_probs/{p.name}").astype(np.float32) for r in runs) / len(runs)
        np.save(out / p.name, np.round(m).astype(np.uint8))
print("ens1536 OOF hazır")
PY
nice .venv/bin/python src/scorer.py outputs/oof/ens1536_tta2 --link_thr 0.15 > outputs/scorer_ens1536.log 2>&1
CUT=$(grep "en iyi kesme" outputs/scorer_ens1536.log | awk '{print $NF}')
.venv/bin/python experiments/predict.py --runs outputs/crop1536_effv2s outputs/effv2m_c1536b --save_probs outputs/test_probs/ens1536_tta2 \
    --scorer outputs/oof/ens1536_tta2/scorer.txt --cut "$CUT" --link_thr 0.15 --out outputs/sub_ens1536_sad15.csv > outputs/test_ens1536.log 2>&1
echo "bitti cut=$CUT"
