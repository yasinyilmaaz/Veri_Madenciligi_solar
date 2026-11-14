#!/usr/bin/env bash
# Kalıcı deney kuyruğu: outputs/backlog.txt dosyasını izler, GPU boşaldıkça sıradaki işi başlatır.
# Dosya bitince kapanmaz; yeni satır eklenmesini bekler (GPU boş kalmasın diye).
#   satır biçimi : <isim> <komut...>   (komut "python ..." ile başlıyorsa venv python'u kullanılır)
#   eşzamanlılık : outputs/maxjobs dosyasındaki sayı (yoksa 2); her iş satırı başında "@1" -> tek başına
#   başlatılanlar: outputs/backlog_started.txt   günlük: outputs/backlog_daemon.log
cd "$(dirname "$0")/.."
BACKLOG=outputs/backlog.txt
STARTED=outputs/backlog_started.txt
touch "$BACKLOG" "$STARTED"
gpu_jobs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l; }
while true; do
  next=""
  while IFS= read -r l; do
    [ -z "$l" ] && continue
    case "$l" in \#*) continue ;; esac
    grep -qxF "$l" "$STARTED" || { next="$l"; break; }
  done < "$BACKLOG"
  if [ -z "$next" ]; then sleep 60; continue; fi
  max=$(cat outputs/maxjobs 2>/dev/null || echo 2)
  solo=0
  case "$next" in @1*) solo=1 ;; esac
  # tek başına işler GPU tamamen boşken, diğerleri max sınırına kadar
  if [ "$solo" = 1 ] && [ "$(gpu_jobs)" -gt 0 ]; then sleep 60; continue; fi
  if [ "$(gpu_jobs)" -ge "$max" ]; then sleep 60; continue; fi
  # çalışan bir tek-başına iş varsa yenisini başlatma
  if [ -f outputs/.solo_running ] && kill -0 "$(cat outputs/.solo_running)" 2>/dev/null; then sleep 60; continue; fi
  echo "$next" >> "$STARTED"
  body=${next#@1 }
  name=${body%% *}
  cmd=${body#* }
  cmd=${cmd/#python /.venv\/bin\/python }
  echo "$(date '+%m-%d %H:%M') başlıyor: $name :: $cmd" >> outputs/backlog_daemon.log
  bash -c "$cmd" > "outputs/$name.log" 2>&1 &
  pid=$!
  [ "$solo" = 1 ] && echo "$pid" > outputs/.solo_running
  sleep 120  # model GPU'ya yüklensin, sayaç doğru görsün
done
