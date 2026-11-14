#!/bin/bash
# Genel ensemble hattı: eksik OOF'ları üret -> eşit ağırlıklı ensemble OOF -> eyer bağlantılı seçici
# -> test tahmini -> gönderim kuyruğu. Kuyruk (queue_daemon) içinde tek iş (@1) olarak çalıştırılmalı.
# Kullanım: tools/pipeline_ens.sh <ens_adı> <run1> <run2> ...
# Örnek (deney 21): tools/pipeline_ens.sh ens3scale crop1536_effv2s effv2m_c1536b crop2048b_effv2s
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
NAME=$1; shift
RUNS=("$@")
for r in "${RUNS[@]}"; do
  miss=()
  for f in 0 1 2 3 4; do
    [ "$(ls outputs/oof/${r}_tta2/fold$f/val_probs 2>/dev/null | wc -l)" -ge 140 ] || miss+=($f)
  done
  [ ${#miss[@]} -eq 0 ] || .venv/bin/python src/oof.py --runs outputs/$r --tta 2 --folds "${miss[@]}" --out outputs/oof/${r}_tta2
done
.venv/bin/python - "$NAME" "${RUNS[@]}" <<'PY'
import sys
from pathlib import Path
import numpy as np
name, runs = sys.argv[1], sys.argv[2:]
for f in range(5):
    out = Path(f"outputs/oof/{name}_tta2/fold{f}/val_probs"); out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(f"outputs/oof/{runs[0]}_tta2/fold{f}/val_probs").glob("*.npy")):
        m = sum(np.load(f"outputs/oof/{r}_tta2/fold{f}/val_probs/{p.name}").astype(np.float32) for r in runs) / len(runs)
        np.save(out / p.name, np.round(m).astype(np.uint8))
print(name, "OOF hazır")
PY
nice .venv/bin/python src/scorer.py outputs/oof/${NAME}_tta2 --link_thr 0.15 > outputs/scorer_${NAME}.log 2>&1
CUT=$(grep "en iyi kesme" outputs/scorer_${NAME}.log | awk '{print $NF}')
PQ=$(awk -F, -v c="$CUT" 'NR>1 && $1+0==c+0 {printf "%.4f", $2}' outputs/oof/${NAME}_tta2/scorer_cut.csv)
.venv/bin/python experiments/predict.py --runs $(printf 'outputs/%s ' "${RUNS[@]}") --save_probs outputs/test_probs/${NAME}_tta2 \
    --scorer outputs/oof/${NAME}_tta2/scorer.txt --cut "$CUT" --link_thr 0.15 --out outputs/sub_${NAME}_sad15.csv
echo "outputs/sub_${NAME}_sad15.csv|$PQ|$NAME (${RUNS[*]}) + eyer + seçici cut $CUT" >> outputs/submit_queue.txt
echo "bitti $NAME PQ=$PQ cut=$CUT"
