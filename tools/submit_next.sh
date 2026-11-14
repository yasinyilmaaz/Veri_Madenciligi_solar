#!/usr/bin/env bash
# Submission kuyruğundan sıradaki varyantı Kaggle'a gönderir ve skorunu günlüğe yazar.
# cron ile her saat başı çalışır: kuyrukta dosya ve günlük hak varsa gönderir, böylece hak boşa gitmez
# (Kaggle günlük 5 hak, UTC 00:00 = TR 03:00'te sıfırlanır).
#   kuyruk : outputs/submit_queue.txt  (her satır: csv_yolu|OOF|açıklama)
#   gönderilen: outputs/submitted.txt   günlük: docs/experiments/submissions.md
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
COMP=filament-segmentation-2026
QUEUE=outputs/submit_queue.txt
DONE=outputs/submitted.txt
LOG=docs/experiments/submissions.md
exec 9>outputs/.submit.lock
flock -n 9 || exit 0
touch "$DONE"
stamp() { date "+%Y-%m-%d %H:%M"; }

remaining=$(kaggle competitions submission-limits $COMP 2>/dev/null | grep -i "remaining today" | grep -oE "[0-9]+")
if [ -z "${remaining:-}" ] || [ "$remaining" -le 0 ]; then
  echo "$(stamp) hak yok veya okunamadı (${remaining:-?})" >> outputs/submit_cron.log; exit 0
fi
line=""
while IFS= read -r l; do
  [ -z "$l" ] && continue
  case "$l" in \#*) continue ;; esac
  # aynı csv daha önce gönderildiyse (açıklama değişmiş olsa bile) atla
  cut -d'|' -f1 "$DONE" | grep -qxF "${l%%|*}" || { line="$l"; break; }
done < "$QUEUE"
if [ -z "$line" ]; then echo "$(stamp) kuyruk boş" >> outputs/submit_cron.log; exit 0; fi

IFS='|' read -r csv oof msg <<< "$line"
if [ ! -f "$csv" ]; then echo "$(stamp) dosya yok: $csv" >> outputs/submit_cron.log; echo "$line" >> "$DONE"; exit 0; fi
# format kontrolü (çakışma, boyut, tekil id); başarısızsa gönderme
if ! .venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
import pandas as pd
from metric import check_submission
check_submission(pd.read_csv('$csv'))" 2>>outputs/submit_cron.log; then
  echo "$(stamp) format hatası, atlandı: $csv" >> outputs/submit_cron.log; echo "$line" >> "$DONE"; exit 0
fi
out=$(kaggle competitions submit $COMP -f "$csv" -m "$msg" 2>&1)
echo "$line" >> "$DONE"
echo "$(stamp) gönderildi: $csv | $(echo "$out" | grep -iE 'success|error' | head -1)" >> outputs/submit_cron.log
score="?"
fname=$(basename "$csv")
for i in $(seq 1 30); do
  row=$(kaggle competitions submissions $COMP 2>/dev/null | grep -F "$fname" | head -1)
  if echo "$row" | grep -q "COMPLETE"; then score=$(echo "$row" | awk '{print $NF}'); break; fi
  if echo "$row" | grep -qi "ERROR"; then score="HATA"; break; fi
  sleep 30
done
[ -f "$LOG" ] || printf '# Submission günlüğü\n\nKararlar OOF ile verilir; LB yalnızca kayıt ve tutarlılık kontrolü içindir.\n\n| Tarih | Dosya | Açıklama | OOF | Public LB |\n|---|---|---|---|---|\n' > "$LOG"
echo "| $(stamp) | $fname | $msg | $oof | $score |" >> "$LOG"
echo "$(stamp) skor: $fname -> $score" >> outputs/submit_cron.log
