#!/usr/bin/env python3
"""Translate video — tout en local sur le runner GitHub, gratuit.

Chaîne : faster-whisper (transcription + horodatage des mots)
       → NLLB-200 (traduction hors ligne)
       → Piper (voix off locale, débit ajusté phrase par phrase)
       → ffmpeg (voix originale très basse + sous-titres dans une bande sous l'image).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
PIPER_BIN = MODELS / "piper" / "piper" / "piper"
VOICES_DIR = MODELS / "voices"
VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

# Voix Piper par langue de sortie (homme / femme)
VOICES = {
    "fr": {"homme": "fr/fr_FR/tom/medium/fr_FR-tom-medium",
           "femme": "fr/fr_FR/siwis/medium/fr_FR-siwis-medium"},
    "it": {"homme": "it/it_IT/riccardo/x_low/it_IT-riccardo-x_low",
           "femme": "it/it_IT/paola/medium/it_IT-paola-medium"},
    "en": {"homme": "en/en_US/ryan/medium/en_US-ryan-medium",
           "femme": "en/en_US/lessac/medium/en_US-lessac-medium"},
    "ar": {"homme": "ar/ar_JO/kareem/medium/ar_JO-kareem-medium",
           "femme": "ar/ar_JO/kareem/medium/ar_JO-kareem-medium"},
}

# Codes Whisper → codes NLLB (langues sources acceptées)
NLLB = {
    "fr": "fra_Latn", "it": "ita_Latn", "en": "eng_Latn", "ar": "arb_Arab",
    "de": "deu_Latn", "es": "spa_Latn", "pt": "por_Latn", "nl": "nld_Latn",
    "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang", "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl", "pl": "pol_Latn", "tr": "tur_Latn", "fa": "pes_Arab",
    "he": "heb_Hebr", "hi": "hin_Deva", "ur": "urd_Arab", "ro": "ron_Latn",
    "el": "ell_Grek", "sv": "swe_Latn", "da": "dan_Latn", "no": "nob_Latn",
    "fi": "fin_Latn", "cs": "ces_Latn", "hu": "hun_Latn", "bg": "bul_Cyrl",
    "sr": "srp_Cyrl", "hr": "hrv_Latn", "id": "ind_Latn", "ms": "zsm_Latn",
    "vi": "vie_Latn", "th": "tha_Thai", "bn": "ben_Beng", "ta": "tam_Taml",
    "sw": "swh_Latn", "am": "amh_Ethi", "ca": "cat_Latn",
}

SENT_END = re.compile(r"[.!?。！？؟…]['\"»”）)]*$")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise RuntimeError(f"Échec : {' '.join(map(str, cmd[:6]))} …")
    return r.stdout


def fetch(url, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            return
        except Exception as e:  # réseau capricieux → on réessaie
            log(f"Téléchargement raté ({e}), nouvel essai…")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Impossible de télécharger {url}")


# ---------------------------------------------------------------- vidéo
def probe(path):
    info = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                           "-of", "json", path]))
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise SystemExit("::error::Le fichier ne contient pas de piste vidéo.")
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int(float(v.get("tags", {}).get("rotate", 0)))
    if abs(rot) % 180 == 90:
        w, h = h, w
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    dur = float(info["format"].get("duration") or v.get("duration") or 0)
    return {"w": w - w % 2, "h": h - h % 2, "dur": dur, "audio": has_audio}


# ---------------------------------------------------------- transcription
def transcribe(video, work, model_size, src):
    from faster_whisper import WhisperModel
    wav = work / "audio16.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", wav])
    log(f"Transcription (whisper {model_size})…")
    model = WhisperModel(model_size, device="cpu", compute_type="int8",
                         cpu_threads=os.cpu_count() or 2)
    segs, info = model.transcribe(
        str(wav), language=None if src == "auto" else src, vad_filter=True,
        word_timestamps=True, beam_size=5, condition_on_previous_text=False)
    words = []
    for s in segs:
        for w in (s.words or []):
            if w.word.strip():
                words.append({"start": float(w.start), "end": float(w.end), "word": w.word})
    lang = info.language if src == "auto" else src
    log(f"Langue source : {lang} (probabilité {info.language_probability:.2f}), {len(words)} mots")
    return words, lang


def resegment(words, max_dur=9.0, gap=0.6):
    """Découpe en phrases courtes (fin de phrase, pause ou durée max) → meilleure synchro."""
    groups, cur = [], []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > gap or w["end"] - cur[0]["start"] > max_dur):
            groups.append(cur)
            cur = []
        cur.append(w)
        if SENT_END.search(w["word"].strip()) and cur[-1]["end"] - cur[0]["start"] > 1.0:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        text = re.sub(r"\s+", " ", "".join(x["word"] for x in g)).strip()
        if text:
            out.append({"start": g[0]["start"], "end": g[-1]["end"], "text": text})
    return out


# -------------------------------------------------------------- traduction
_MT = {}


def translate(texts, src, tgt):
    if src == tgt:
        return list(texts)
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch.set_num_threads(os.cpu_count() or 2)
    if "model" not in _MT:
        log("Chargement du modèle de traduction NLLB…")
        _MT["tok"] = AutoTokenizer.from_pretrained(NLLB_MODEL)
        _MT["model"] = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).eval()
    tok, model = _MT["tok"], _MT["model"]
    tok.src_lang = NLLB[src]
    bos = tok.convert_tokens_to_ids(NLLB[tgt])
    out = []
    for i in range(0, len(texts), 8):
        batch = texts[i:i + 8]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.inference_mode():
            gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=4,
                                 max_new_tokens=min(256, enc["input_ids"].shape[1] * 3 + 16),
                                 no_repeat_ngram_size=4)
        out += [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
    return out


# ------------------------------------------------------------------- voix
def voice_model(lang, gender):
    rel = VOICES[lang][gender]
    onnx = VOICES_DIR / (Path(rel).name + ".onnx")
    fetch(VOICE_URL + rel + ".onnx", onnx)
    fetch(VOICE_URL + rel + ".onnx.json", Path(str(onnx) + ".json"))
    return onnx


def piper(text, model, out: Path, length_scale=1.0):
    env = dict(os.environ, LD_LIBRARY_PATH=str(PIPER_BIN.parent))
    cmd = [str(PIPER_BIN), "--model", str(model), "--output_file", str(out),
           "--length_scale", f"{length_scale:.3f}", "--sentence_silence", "0.05"]
    for _ in range(3):
        r = subprocess.run(cmd, input=text.encode(), capture_output=True, env=env)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 2000:
            return
    raise RuntimeError(f"Piper a échoué : {r.stderr.decode()[-500:]}")


def load_48k(path, tempo=1.0):
    """Lit un wav, applique un éventuel accélérateur (atempo) et passe en 48 kHz mono."""
    out = path.with_name(path.stem + "_48k.wav")
    af = []
    t = tempo
    while t > 2.0:
        af.append("atempo=2.0")
        t /= 2.0
    if abs(t - 1.0) > 0.005:
        af.append(f"atempo={t:.4f}")
    af.append(f"aresample={SR}")
    run(["ffmpeg", "-y", "-v", "error", "-i", path, "-af", ",".join(af), "-ac", "1", out])
    a, _ = sf.read(out, dtype="float32")
    return trim_silence(a)


def trim_silence(a, thr_db=-42):
    if a.size == 0:
        return a
    thr = 10 ** (thr_db / 20) * max(np.abs(a).max(), 1e-6)
    idx = np.where(np.abs(a) > thr)[0]
    if idx.size == 0:
        return a
    s = max(idx[0] - int(0.01 * SR), 0)
    e = min(idx[-1] + int(0.04 * SR), a.size)
    return a[s:e]


def build_dub(segs, texts, model, video_dur, work):
    """Place chaque phrase traduite au moment où la personne parle, débit ajusté."""
    track = np.zeros(int((video_dur + 1) * SR), dtype=np.float32)
    timing = []
    cursor = 0.0
    for i, (seg, text) in enumerate(zip(segs, texts)):
        if not text:
            timing.append(None)
            continue
        pos = max(seg["start"], cursor)
        nxt = segs[i + 1]["start"] if i + 1 < len(segs) else video_dur
        avail = max(nxt - pos - 0.06, 0.4)
        target = max(min(seg["end"], nxt) - pos, 0.4)
        raw = work / f"tts_{i:04d}.wav"
        piper(text, model, raw, 1.0)
        d0 = sf.info(raw).duration
        # 1) ajustement naturel du débit par Piper (durée des lèvres = cible)
        ls = min(max(target / max(d0, 0.05), 0.74), 1.12)
        if abs(ls - 1.0) > 0.03:
            piper(text, model, raw, ls)
        a = load_48k(raw)
        # 2) si ça déborde encore sur la phrase suivante, léger accéléré (max ×1.3)
        d1 = a.size / SR
        if d1 > avail:
            a = load_48k(raw, min(d1 / avail, 1.3))
        start = int(pos * SR)
        end = min(start + a.size, track.size)
        if end > start:
            track[start:end] += a[:end - start]
        cursor = pos + a.size / SR + 0.05
        timing.append((pos, pos + a.size / SR))
    track = track[:int(video_dur * SR) + 1]
    peak = float(np.abs(track).max() or 1.0)
    track *= 0.89 / peak
    out = work / "dub.wav"
    sf.write(out, track, SR)
    return out, timing


# ------------------------------------------------------------- sous-titres
def ass_time(t):
    cs = int(round(max(t, 0) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def srt_time(t):
    ms = int(round(max(t, 0) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def chunk_text(text, max_chars):
    words, chunks, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def build_cues(texts, timing, line_chars):
    cues = []
    for text, tm in zip(texts, timing):
        if not text or tm is None:
            continue
        s, e = tm
        e = max(e, s + 0.8)
        parts = chunk_text(text, line_chars * 2)
        total = sum(len(p) for p in parts) or 1
        t = s
        for p in parts:
            d = (e - s) * len(p) / total
            cues.append((t, t + d, p))
            t += d
    return cues


def write_subs(cues, lang, geo, mode, work, stem):
    W, H = geo["w"], geo["h"]
    if mode == "bande":
        fs = max(int(min(W, H) * 0.052), 18)
        band = int(fs * 3.4)
        band += band % 2
        total_h = H + band
    else:
        fs = max(int(min(W, H) * 0.045), 16)
        band, total_h = 0, H
    font = "Noto Sans Arabic" if lang == "ar" else "Noto Sans"
    border = 1 if mode == "bande" else 3
    hdr = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {total_h}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{fs},&H00FFFFFF,&H000000FF,&H00000000,&H90000000,0,0,0,0,"
        f"100,100,0,0,{border},{max(fs // 18, 2)},0,2,{int(W * 0.04)},{int(W * 0.04)},"
        f"{int(H * 0.03)},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    pos_tag = f"{{\\an5\\pos({W // 2},{H + band // 2})}}" if mode == "bande" else ""
    for s, e, txt in cues:
        safe = txt.replace("\\", "/").replace("{", "(").replace("}", ")")
        lines.append(f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Default,,0,0,0,,{pos_tag}{safe}")
    ass = work / f"subs_{lang}.ass"
    ass.write_text(hdr + "\n".join(lines) + "\n", encoding="utf-8")
    srt = work / f"{stem}__{lang.upper()}.srt"
    srt.write_text("".join(f"{i}\n{srt_time(s)} --> {srt_time(e)}\n{t}\n\n"
                           for i, (s, e, t) in enumerate(cues, 1)), encoding="utf-8")
    return ass, srt, band


def line_chars_for(geo, mode):
    W, H = geo["w"], geo["h"]
    fs = max(int(min(W, H) * (0.052 if mode == "bande" else 0.045)), 16)
    return max(int(W * 0.92 / (fs * 0.55)), 14)


# ------------------------------------------------------------------ rendu
def render(video, dub, ass, band, geo, orig_vol, mode, out):
    W, H = geo["w"], geo["h"]
    a_dub = "[1:a]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000"
    if geo["audio"]:
        audio = (f"[0:a]aresample=48000,volume={orig_vol}[o];{a_dub}[d];"
                 "[o][d]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                 "alimiter=limit=0.95[a]")
    else:
        audio = f"{a_dub},apad[a]"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", Path(video).resolve(), "-i", Path(dub).resolve()]
    if mode == "non":
        cmd += ["-filter_complex", audio, "-map", "0:v:0", "-map", "[a]", "-c:v", "copy"]
    else:
        vf = f"[0:v]scale={W}:{H}"
        if band:
            vf += f",pad={W}:{H + band}:0:0:black"
        vf += f",subtitles={ass.name}[v]"
        cmd += ["-filter_complex", f"{vf};{audio}", "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-t", f"{geo['dur']:.3f}",
            "-movflags", "+faststart", out.resolve()]
    run(cmd, cwd=ass.parent)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--langs", default="fr")
    ap.add_argument("--src", default="auto")
    ap.add_argument("--voice", default="homme", choices=["homme", "femme"])
    ap.add_argument("--subs", default="bande", choices=["bande", "bas", "non"])
    ap.add_argument("--orig-vol", type=float, default=0.05)
    ap.add_argument("--whisper", default="medium")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()

    langs = [l.strip().lower() for l in a.langs.split(",") if l.strip()]
    bad = [l for l in langs if l not in VOICES]
    if not langs or bad:
        raise SystemExit(f"::error::Langue(s) de sortie non gérée(s) : {bad}. Choix : {list(VOICES)}")
    if a.src != "auto" and a.src not in NLLB:
        raise SystemExit(f"::error::Langue source inconnue : {a.src}")
    if not PIPER_BIN.exists():
        raise SystemExit("::error::Piper absent (étape d'installation ratée).")

    video = Path(a.input).resolve()
    outdir = Path(a.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    work = Path("work").resolve()
    work.mkdir(exist_ok=True)
    stem = re.sub(r"[^\w\-]+", "_", video.stem)[:80]

    geo = probe(video)
    log(f"Vidéo {geo['w']}x{geo['h']}, {geo['dur']:.1f} s, audio={geo['audio']}")
    if not geo["audio"]:
        raise SystemExit("::error::La vidéo n'a pas de son : rien à traduire.")
    if geo["dur"] > 420:
        log("Attention : vidéo de plus de 7 min, le traitement sera long.")

    words, src = transcribe(video, work, a.whisper, a.src)
    if src not in NLLB:
        raise SystemExit(f"::error::Langue détectée « {src} » non gérée par la traduction.")
    segs = resegment(words)
    if not segs:
        raise SystemExit("::error::Aucune parole détectée dans la vidéo.")
    (outdir / f"{stem}__transcription_{src}.txt").write_text(
        "\n".join(f"[{s['start']:.1f}] {s['text']}" for s in segs), encoding="utf-8")

    for lang in langs:
        log(f"=== Sortie {lang.upper()} ===")
        texts = translate([s["text"] for s in segs], src, lang)
        model = voice_model(lang, a.voice)
        log("Synthèse de la voix off…")
        dub, timing = build_dub(segs, texts, model, geo["dur"], work)
        cues = build_cues(texts, timing, line_chars_for(geo, a.subs))
        ass, srt, band = write_subs(cues, lang, geo, a.subs, work, stem)
        srt.replace(outdir / srt.name)
        out = outdir / f"{stem}__{lang.upper()}.mp4"
        log("Montage final…")
        render(video, dub, ass, band, geo, a.orig_vol, a.subs, out)
        log(f"OK → {out.name} ({out.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
