#!/usr/bin/env bash
# Usage : download_asset.sh <id> <taille attendue> <nom> → in/<nom> (taille vérifiée, 4 essais)
set -uo pipefail
ID="$1"; SIZE="$2"; NAME="$3"
mkdir -p in
F="in/$NAME"
for i in 1 2 3 4; do
  if gh api -H "Accept: application/octet-stream" "repos/$GITHUB_REPOSITORY/releases/assets/$ID" > "$F" \
     && [ "$(stat -c %s "$F")" = "$SIZE" ]; then
    echo "Vidéo : $NAME ($(du -h "$F" | cut -f1))"
    exit 0
  fi
  echo "Téléchargement : essai $i raté ou incomplet, nouvel essai"; sleep $((10 * i))
done
echo "::error::Impossible de télécharger « $NAME » complètement depuis la release (a-t-elle été supprimée ou remplacée pendant le run ?)."
exit 1
