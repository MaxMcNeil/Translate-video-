# Translate video

Traduit une vidéo (arabe, italien, allemand, chinois, japonais… 40 langues) en **français, italien, arabe ou anglais**, entièrement sur GitHub Actions, gratuit, sans service externe payant.

## Utilisation
1. **Releases → Draft a new release** : crée un tag (ex. `v1`), dépose ta vidéo (mp4, mov, mkv, webm), publie.
2. **Actions → Traduire une vidéo → Run workflow** : choisis la langue de sortie (`fr`, `it`, `ar`, `en` ou plusieurs : `fr,en`). La langue parlée dans la vidéo est détectée automatiquement.
3. 15–30 min plus tard, la vidéo traduite `…__FR.mp4` et ses sous-titres `.srt` apparaissent dans la même release.

## Les champs du formulaire
| Champ | Ce qu'il faut mettre |
|---|---|
| 0) Nombre de vidéos | Combien de vidéos de la release traduire en même temps : `1`, `3`… ou `toutes`. Chaque vidéo tourne sur sa propre machine, en parallèle ; si une échoue, les autres continuent |
| 1) Langue de SORTIE | La langue dans laquelle tu veux la vidéo : `fr`, `it`, `ar`, `en` (ou `fr,en`) |
| 2) Voix off | homme / femme |
| 3) Sous-titres | **sous l'image** : bande noire ajoutée en bas, l'image n'est jamais recouverte · **sur l'image** : texte posé en bas de la vidéo, sans bande · **aucun** : pas de sous-titres incrustés (le fichier .srt est quand même fourni) |
| 4) Tag | Vide = dernière release |
| 5) Nom de la vidéo | Laisser **vide**. Sert seulement à traduire UNE vidéo précise (le champ 0 est alors ignoré) |
| 6) Langue PARLÉE | Laisser `auto` (détection automatique). À forcer (`ar`, `de`, `zh`…) seulement si la détection se trompe |
| 7) Volume d'origine | 0.05 = à peine audible |
| 8) Qualité | `medium` conseillé |

## Ce que fait le workflow
- **Transcription** : faster-whisper (local, horodatage au mot près).
- **Traduction** : NLLB-200 de Meta (local, hors ligne).
- **Voix off** : Piper (local). Chaque phrase démarre quand la personne commence à parler et son débit est ajusté pour finir avec elle.
- **Voix originale** à 5 % (réglable).
- **Sous-titres** dans une bande noire ajoutée sous l'image → ils ne cachent jamais les visages.

Lancement manuel uniquement, aucune exécution automatique.

## Fiabilité et suivi
- **Champs vérifiés avant de démarrer** : une erreur de saisie s'affiche en clair en 10 secondes, sans lancer de traduction.
- **Une machine par vidéo** : si une vidéo échoue, les autres continuent.
- **Réessais automatiques** sur chaque téléchargement, installation et envoi. Les modèles sont mis en cache.
- **Une phrase dont la voix échoue est sautée** (son sous-titre reste) au lieu de faire planter la vidéo.
- **Contrôle final** : la durée et le son de la vidéo produite sont vérifiés avant publication.
- **Traçable** : chaque job affiche un tableau (langue détectée, phrases, accélérations, retards, durée de chaque étape, avertissements). Un **Bilan** récapitule toutes les vidéos à la fin du run. Le rapport détaillé `rapport_….json` et une copie de la vidéo sont dans les artefacts du run pendant 7 jours.
- **Refus clairs** : vidéo sans son, muette, corrompue, sans parole ou de plus de 15 min → message d'erreur qui dit quoi faire.
