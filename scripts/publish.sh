#!/usr/bin/env bash
# Usage : publish.sh <tag> <dossier> — envoie vidéos, .srt et transcription dans la release (4 essais),
# puis supprime les anciens fichiers de CETTE vidéo devenus obsolètes.
set -uo pipefail
TAG="$1"; DIR="$2"
shopt -s nullglob
FILES=("$DIR"/*.mp4 "$DIR"/*.srt "$DIR"/*.txt)
[ ${#FILES[@]} -gt 0 ] || { echo "::error::Aucun fichier à publier."; exit 1; }
for i in 1 2 3 4; do
  if gh release upload "$TAG" "${FILES[@]}" -R "$GITHUB_REPOSITORY" --clobber; then
    printf 'Publié : %s\n' "${FILES[@]}"
    T=$(ls "$DIR"/*__transcription_*.txt 2>/dev/null | head -1 || true)
    if [ -n "$T" ]; then
      STEM=$(basename "$T"); STEM="${STEM%%__transcription_*}"
      NEW=$(printf '%s\n' "${FILES[@]}" | xargs -n1 basename)
      gh api "repos/$GITHUB_REPOSITORY/releases?per_page=100" \
        | jq -r --arg t "$TAG" --arg p "${STEM}__" '.[] | select(.tag_name==$t) | .assets[] | select(.name | startswith($p)) | "\(.id) \(.name)"' \
        | while read -r id name; do
            if ! grep -qxF "$name" <<< "$NEW"; then
              gh api -X DELETE "repos/$GITHUB_REPOSITORY/releases/assets/$id" && echo "Ancien fichier supprimé : $name" || true
            fi
          done
    fi
    exit 0
  fi
  echo "Envoi : essai $i raté, nouvel essai"; sleep $((15 * i))
done
echo "::error::Envoi dans la release impossible. Le résultat est dans les artefacts de ce run (7 jours)."
exit 1
