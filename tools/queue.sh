#!/usr/bin/env bash
# Deney kuyruğu: aynı anda en fazla MAXJOBS eğitim çalıştırır.
# Kullanım: bash tools/queue.sh jobs.txt   (her satır: <isim> <train.py argümanları>)
cd "$(dirname "$0")/.."
MAXJOBS=${MAXJOBS:-2}
while read -r name args; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  # DataLoader worker'ları GPU kullanmadığı için yalnızca GPU süreçleri sayılır
  while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -ge "$MAXJOBS" ]; do sleep 60; done
  echo "$(date +%H:%M) başlıyor: $name $args"
  .venv/bin/python src/train.py $args 2>&1 | grep --line-buffered -v -i warn > "outputs/$name.log" &
  sleep 90
done < "$1"
wait
echo "kuyruk bitti"
