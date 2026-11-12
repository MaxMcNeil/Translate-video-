#!/usr/bin/env python3
"""Rassemble les plans des vidéos (un par vidéo) et fabrique la liste des tâches parallèles.

Vidéo courte → 1 tâche (traduite et publiée directement).
Vidéo longue → 1 tâche par partie, puis 1 recollage.
Écrit dans GITHUB_OUTPUT : taches (JSON), longues (JSON).
"""
import json
import os
import sys
from pathlib import Path

MAX_TASKS = 250   # limite GitHub : 256 jobs par matrice


def main():
    plans_dir, user_src, user_voice = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    expected = json.loads(sys.argv[4])          # vidéos sélectionnées par « lister »
    plans = {}
    for f in sorted(plans_dir.rglob("*.json")):
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
            plans[str(p["id"])] = p
        except Exception as e:
            print(f"::warning::Plan illisible {f.name} : {e}")
    tasks, longues, md = [], [], ["## Plan de traitement", "", "| Vidéo | Durée | Traitement |", "|---|---|---|"]
    missing = []
    for v in expected:
        p = plans.get(str(v["id"]))
        if p is None:
            missing.append(v["name"])
            md.append(f"| {v['name']} | ? | ❌ préparation en échec (voir le job « Préparer ») |")
            continue
        base = {"id": v["id"], "name": v["name"], "size": v["size"]}
        dur = f"{p.get('duree', 0) / 60:.0f} min"
        if p.get("mode") == "direct":
            tasks.append({**base, "k": 1, "n": 1, "start": "", "end": "", "total": "",
                          "src": user_src, "voix": user_voice})
            md.append(f"| {v['name']} | {dur} | d'un seul tenant |")
        else:
            parts = p["parties"]
            src = p.get("src") or user_src
            voix = p.get("voix") or user_voice
            if voix == "auto" and user_voice in ("homme", "femme"):
                voix = user_voice
            for x in parts:
                tasks.append({**base, "k": x["k"], "n": len(parts), "start": f"{x['start']:.3f}",
                              "end": f"{x['end']:.3f}", "total": f"{p['duree']:.3f}", "src": src, "voix": voix})
            longues.append(base)
            md.append(f"| {v['name']} | {dur} | découpée en **{len(parts)} parties** en parallèle "
                      f"(langue {src}, voix {voix}), puis recollée |")
    if len(tasks) > MAX_TASKS:
        print(f"::error::{len(tasks)} tâches : trop pour un seul lancement (max {MAX_TASKS}). "
              "Mets moins de vidéos dans le champ 0.")
        sys.exit(1)
    if not tasks:
        print("::error::Aucune vidéo n'a pu être préparée (voir les jobs « Préparer »).")
        sys.exit(1)
    for m in missing:
        print(f"::error::Préparation en échec pour « {m} » : cette vidéo ne sera pas traduite.")
    out = os.environ.get("GITHUB_OUTPUT")
    payload = f"taches={json.dumps(tasks, ensure_ascii=False)}\nlongues={json.dumps(longues, ensure_ascii=False)}\n"
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(payload)
    else:
        print(payload)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write("\n".join(md) + f"\n\n**{len(tasks)} tâche(s) en parallèle.**\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
