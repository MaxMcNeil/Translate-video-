#!/usr/bin/env bash
# ffmpeg + polices (dont l'arabe), avec réessais. Sans effet si déjà installés.
set -u
if command -v ffmpeg >/dev/null && fc-list | grep -q "Noto Sans Arabic"; then
  ffmpeg -version | head -1; exit 0
fi
for i in 1 2 3 4; do
  sudo apt-get update -q && sudo apt-get install -y -q --no-install-recommends ffmpeg fonts-noto-core && break
  echo "apt : essai $i raté, nouvel essai"; sleep $((15 * i))
  [ "$i" = 4 ] && { echo "::error::Installation de ffmpeg impossible."; exit 1; }
done
ffmpeg -version | head -1
fc-list | grep -q "Noto Sans Arabic" || echo "::warning::Police arabe absente"
df -h / | tail -1
