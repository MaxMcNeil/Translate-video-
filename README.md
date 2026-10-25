# Translate video

Traduit une vidéo (arabe, italien, allemand, chinois, japonais… 40 langues) en **français, italien, arabe ou anglais**, entièrement sur GitHub Actions, gratuit, sans service externe payant.

## Utilisation
1. **Releases → Draft a new release** : crée un tag (ex. `v1`), dépose ta vidéo (mp4, mov, mkv, webm), publie.
2. **Actions → Traduire une vidéo → Run workflow** : choisis la langue (`fr`, `it`, `ar`, `en` ou plusieurs : `fr,en`).
3. 15–30 min plus tard, la vidéo traduite `…__FR.mp4` et ses sous-titres `.srt` apparaissent dans la même release.

## Ce que fait le workflow
- **Transcription** : faster-whisper (local, horodatage au mot près).
- **Traduction** : NLLB-200 de Meta (local, hors ligne).
- **Voix off** : Piper (local). Chaque phrase démarre quand la personne commence à parler et son débit est ajusté pour finir avec elle.
- **Voix originale** à 5 % (réglable).
- **Sous-titres** dans une bande noire ajoutée sous l'image → ils ne cachent jamais les visages.

Lancement manuel uniquement, aucune exécution automatique.
