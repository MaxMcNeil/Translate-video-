#!/usr/bin/env python3
"""Recolle les parties d'une vidéo longue traduites en parallèle.

Entrée : le plan (plan.json) et un dossier contenant part-1/, part-2/, … (sorties de translate_video.py
en mode partie). Sortie : pour chaque langue, une vidéo, un .srt et une transcription uniques, avec des
horaires recalés sur la vidéo d'origine. Tout est vérifié (parties manquantes, durées, formats) avant
d'écrire quoi que ce soit : en cas de problème, rien n'est publié et le message dit quelle partie pose
problème.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


class Fatal(Exception):
    pass


def log(msg):
    print(msg, flush=True)


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise Fatal(f"{Path(str(cmd[0])).name} en échec : {r.stderr.strip()[-500:]}")
    return r.stdout


def probe(path):
    info = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path]))
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    return {"w": int(v["width"]), "h": int(v["height"]), "dur": float(info["format"]["duration"]),
            "audio": any(s["codec_type"] == "audio" for s in info["streams"]),
            "vcodec": v.get("codec_name"), "fps": v.get("r_frame_rate"), "pix": v.get("pix_fmt")}


TS = re.compile(r"(\d+):(\d\d):(\d\d),(\d{3})")


def srt_t(t):
    ms = int(round(max(t, 0) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def parse_srt(text):
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        idx = 1 if "-->" in lines[1] else 0
        m = TS.findall(lines[idx])
        if len(m) != 2:
            continue
        s, e = [int(h) * 3600 + int(mi) * 60 + int(se) + int(ms) / 1000 for h, mi, se, ms in m]
        cues.append((s, e, "\n".join(lines[idx + 1:])))
    return cues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--parts", required=True, help="dossier contenant part-1/, part-2/, …")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    try:
        plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
        parts = plan["parties"]
        total = float(plan["duree"])
        root = Path(a.parts)
        log(f"Vidéo : {plan['video']} — {len(parts)} parties — {total / 60:.1f} min")

        # 1) toutes les parties sont-elles là et réussies ?
        problems, reports = [], []
        for p in parts:
            d = root / f"part-{p['k']}"
            reps = list(d.glob("rapport_*.json")) if d.is_dir() else []
            if not reps:
                problems.append(f"partie {p['k']} ({p['start'] / 60:.1f}→{p['end'] / 60:.1f} min) : absente "
                                "(son job a échoué — voir le job « Traduire … partie " f"{p['k']}/{len(parts)} »)")
                continue
            r = json.loads(reps[0].read_text(encoding="utf-8"))
            reports.append(r)
            if r.get("statut") != "OK":
                problems.append(f"partie {p['k']} : échec — {r.get('erreur', '?')}")
        if problems:
            raise Fatal("vidéo NON recollée, aucune partie publiée :\n- " + "\n- ".join(problems))

        first = root / "part-1"
        vids = sorted(first.glob("*__??.mp4"))
        if not vids:
            raise Fatal("aucune vidéo trouvée dans la partie 1.")
        langs = [re.search(r"__(\w\w)\.mp4$", v.name).group(1) for v in vids]
        stem = vids[0].name[: -len(f"__{langs[0]}.mp4")]
        log(f"Langues : {', '.join(langs)} — nom : {stem}")

        for lang in langs:
            files = []
            ref = None
            for p in parts:
                f = root / f"part-{p['k']}" / f"{stem}__{lang}.mp4"
                if not f.exists():
                    raise Fatal(f"partie {p['k']} : fichier {f.name} manquant.")
                g = probe(f)
                expect = p["end"] - p["start"]
                if abs(g["dur"] - expect) > 1.0:
                    raise Fatal(f"partie {p['k']} : durée {g['dur']:.1f} s au lieu de {expect:.1f} s.")
                key = (g["w"], g["h"], g["vcodec"], g["fps"], g["pix"])
                if ref is None:
                    ref = key
                elif key != ref:
                    raise Fatal(f"partie {p['k']} : format {key} différent de la partie 1 {ref}.")
                files.append(f.resolve())

            # 2) vidéo : image copiée telle quelle (aucune perte), son ré-encodé d'un bloc (aucun clic)
            lst = out / f"concat_{lang}.txt"
            lst.write_text("".join(f"file '{str(f).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
                                   for f in files), encoding="utf-8")
            final = out / f"{stem}__{lang}.mp4"
            run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                 "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                 "-movflags", "+faststart", final])
            lst.unlink()
            g = probe(final)
            if abs(g["dur"] - total) > 2.0 or not g["audio"] or (g["w"], g["h"]) != ref[:2]:
                raise Fatal(f"vidéo recollée incohérente ({g['dur']:.1f} s au lieu de {total:.1f} s, "
                            f"{g['w']}x{g['h']}).")
            size = final.stat().st_size / 1e6
            if size > 2000:
                raise Fatal(f"vidéo finale de {size:.0f} Mo : dépasse 2 Go, publication impossible.")

            # 3) sous-titres : horaires recalés sur la vidéo d'origine
            cues = []
            for p in parts:
                f = root / f"part-{p['k']}" / f"{stem}__{lang}.srt"
                if f.exists():
                    for s, e, t in parse_srt(f.read_text(encoding="utf-8")):
                        cues.append((s + p["start"], min(e + p["start"], p["end"]), t))
            (out / f"{stem}__{lang}.srt").write_text(
                "".join(f"{i}\n{srt_t(s)} --> {srt_t(e)}\n{t}\n\n" for i, (s, e, t) in enumerate(cues, 1)),
                encoding="utf-8")
            summary.append(f"| {lang} | {final.name} | {g['dur'] / 60:.1f} min | {size:.0f} Mo | {len(cues)} sous-titres |")
            log(f"OK → {final.name} ({size:.0f} Mo, {g['dur']:.1f} s, {len(cues)} sous-titres)")

        # 4) transcription unique
        src = plan.get("src") or "auto"
        lines = [f"# Vidéo découpée en {len(parts)} parties traduites en parallèle puis recollées",
                 f"# Langue commune : {src} — proposées : {plan.get('langues_candidates')}",
                 f"# Voix commune : {plan.get('voix')} — {plan.get('voix_detail')}"]
        tsrc = src
        for p in parts:
            for f in sorted((root / f"part-{p['k']}").glob("*__transcription_*.txt")):
                tsrc = re.search(r"__transcription_(\w+)\.txt$", f.name).group(1)
                for line in f.read_text(encoding="utf-8").splitlines():
                    m = re.match(r"^\[(\d+(?:\.\d+)?)\]\s?(.*)$", line)
                    if m:
                        lines.append(f"[{float(m.group(1)) + p['start']:.1f}] {m.group(2)}")
        (out / f"{stem}__transcription_{tsrc}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        # 5) rapport global
        warns = [f"partie {r.get('partie')} : {w}" for r in reports for w in r.get("avertissements", [])]
        rapport = {"video": plan["video"], "statut": "OK", "mode": "découpée", "duree_s": total,
                   "plan": plan, "parties": reports, "avertissements": warns}
        (out / f"rapport_{stem}.json").write_text(json.dumps(rapport, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
        md = [f"### ✅ {plan['video']} — recollée ({len(parts)} parties)", "",
              "| Langue | Fichier | Durée | Taille | Sous-titres |", "|---|---|---|---|---|", *summary]
        if warns:
            md += ["", "**Avertissements des parties :**"] + [f"- {w}" for w in warns]
        status = 0
    except Fatal as e:
        md = [f"### ❌ {Path(a.plan).name} — recollage impossible", "", str(e)]
        print(f"::error::{str(e).splitlines()[0]}", flush=True)
        for line in str(e).splitlines()[1:]:
            print(f"::error::{line.lstrip('- ')}", flush=True)
        status = 2
    except Exception as e:
        md = [f"### ❌ recollage : erreur imprévue", "", f"{type(e).__name__}: {e}"]
        print(f"::error::Recollage : {type(e).__name__}: {e}", flush=True)
        status = 1
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n\n")
    sys.exit(status)


if __name__ == "__main__":
    main()
