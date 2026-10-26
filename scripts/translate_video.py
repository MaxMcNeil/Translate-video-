#!/usr/bin/env python3
"""Translate video — tout en local sur le runner GitHub, gratuit.

Étapes : 1 analyse → 2 transcription (faster-whisper) → 3 traduction (NLLB-200)
         → 4 voix off (Piper, débit ajusté phrase par phrase) → 5 sous-titres → 6 montage (ffmpeg).

Chaque étape est tracée dans le log et dans un rapport (rapport_<vidéo>.json + résumé du run).
Une phrase dont la voix échoue est sautée (sous-titre conservé) au lieu de faire planter la vidéo.
"""
import argparse
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import unicodedata
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
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}

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

# Codes Whisper → codes NLLB (toutes les langues Whisper que NLLB sait traduire)
NLLB = {
    "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "as": "asm_Beng", "az": "azj_Latn",
    "ba": "bak_Cyrl", "be": "bel_Cyrl", "bg": "bul_Cyrl", "bn": "ben_Beng", "bo": "bod_Tibt",
    "bs": "bos_Latn", "ca": "cat_Latn", "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn",
    "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn", "et": "est_Latn",
    "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fo": "fao_Latn", "fr": "fra_Latn",
    "gl": "glg_Latn", "gu": "guj_Gujr", "ha": "hau_Latn", "he": "heb_Hebr", "hi": "hin_Deva",
    "hr": "hrv_Latn", "ht": "hat_Latn", "hu": "hun_Latn", "hy": "hye_Armn", "id": "ind_Latn",
    "is": "isl_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "jw": "jav_Latn", "ka": "kat_Geor",
    "kk": "kaz_Cyrl", "km": "khm_Khmr", "kn": "kan_Knda", "ko": "kor_Hang", "lb": "ltz_Latn",
    "ln": "lin_Latn", "lo": "lao_Laoo", "lt": "lit_Latn", "lv": "lvs_Latn", "mg": "plt_Latn",
    "mi": "mri_Latn", "mk": "mkd_Cyrl", "ml": "mal_Mlym", "mn": "khk_Cyrl", "mr": "mar_Deva",
    "ms": "zsm_Latn", "mt": "mlt_Latn", "my": "mya_Mymr", "ne": "npi_Deva", "nl": "nld_Latn",
    "nn": "nno_Latn", "no": "nob_Latn", "oc": "oci_Latn", "pa": "pan_Guru", "pl": "pol_Latn",
    "ps": "pbt_Arab", "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "sd": "snd_Arab",
    "si": "sin_Sinh", "sk": "slk_Latn", "sl": "slv_Latn", "sn": "sna_Latn", "so": "som_Latn",
    "sq": "als_Latn", "sr": "srp_Cyrl", "su": "sun_Latn", "sv": "swe_Latn", "sw": "swh_Latn",
    "ta": "tam_Taml", "te": "tel_Telu", "tg": "tgk_Cyrl", "th": "tha_Thai", "tk": "tuk_Latn",
    "tl": "tgl_Latn", "tr": "tur_Latn", "tt": "tat_Cyrl", "uk": "ukr_Cyrl", "ur": "urd_Arab",
    "uz": "uzn_Latn", "vi": "vie_Latn", "yi": "ydd_Hebr", "yo": "yor_Latn", "zh": "zho_Hans",
    "yue": "yue_Hant",
}

# Langues probables en entrée : une langue rare n'est retenue que si Whisper en est vraiment sûr
COMMON_LANGS = {"en", "fr", "it", "ar", "de", "es", "pt", "zh", "ja", "ko", "ru", "tr", "nl", "pl",
                "uk", "fa", "he", "hi", "ur", "ro", "el", "sv", "id", "vi", "th", "ms", "bn"}

SENT_END = re.compile(r"[.!?。！？؟…]['\"»”）)]*$")
SOFT_END = re.compile(r"[,;:，、؛،]['\"»”）)]*$")
# « U.S. », « Mr. », « Dr. », « etc. », chiffres « 3. » … : un point qui ne termine pas une phrase
ABBREV = re.compile(r"^(?:[A-Za-zÀ-ÿ]\.){1,4}$|^(?:mr|mrs|ms|dr|st|vs|etc|no|jr|sr|prof|gen|sen|rep|gov|inc|ltd|co|corp|e\.g|i\.e)\.$|^\d+\.$", re.I)

REPORT = {"etapes": [], "avertissements": []}
_STAGE = {"nom": "démarrage", "t0": time.time()}


# ------------------------------------------------------------ traçabilité
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def warn(msg):
    print(f"::warning::{msg}", flush=True)
    REPORT["avertissements"].append(msg)


def stage(nom):
    """Ferme l'étape en cours (durée) et ouvre la suivante — visible et repliable dans le log."""
    if _STAGE["nom"] != "démarrage":
        print("::endgroup::", flush=True)
        REPORT["etapes"].append({"etape": _STAGE["nom"],
                                 "secondes": round(time.time() - _STAGE["t0"], 1)})
    _STAGE.update(nom=nom, t0=time.time())
    if nom:
        print(f"::group::{nom}", flush=True)
        log(f"▶ {nom}")


def fail(msg):
    raise SystemExit(f"::error::[{_STAGE['nom']}] {msg}")


def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-3000:])
        raise RuntimeError(f"commande en échec ({Path(str(cmd[0])).name}) : {r.stderr.strip()[-400:]}")
    return r.stdout


def retry(what, fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise RuntimeError(f"{what} : échec après {tries} essais ({e})")
            log(f"{what} : essai {i + 1} raté ({e}), nouvel essai dans {15 * (i + 1)} s")
            time.sleep(15 * (i + 1))


def fetch(url, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return

    def _dl():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        if tmp.stat().st_size == 0:
            raise RuntimeError("fichier vide")
        tmp.rename(dest)
    retry(f"téléchargement {dest.name}", _dl, tries=6)


def hf_load(what, fn):
    """Modèle Hugging Face : d'abord depuis le cache (sans réseau), sinon téléchargement avec essais."""
    try:
        return fn(True)
    except Exception:
        log(f"{what} absent du cache → téléchargement")
        return retry(f"chargement {what}", lambda: fn(False))


# ---------------------------------------------------------------- vidéo
def probe(path):
    try:
        info = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                               "-of", "json", path]))
    except Exception as e:
        fail(f"fichier illisible ou corrompu ({e})")
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"
              and s.get("disposition", {}).get("attached_pic", 0) == 0), None)
    if v is None:
        fail("le fichier ne contient pas d'image vidéo.")
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    if w < 16 or h < 16:
        fail(f"dimensions vidéo invalides ({w}x{h}).")
    sar = v.get("sample_aspect_ratio", "1:1")
    try:
        n, d = (int(x) for x in sar.split(":"))
        if n > 0 and d > 0 and n != d:
            w = int(round(w * n / d))
    except Exception:
        pass
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int(float(v.get("tags", {}).get("rotate", 0) or 0))
    if abs(rot) % 180 == 90:
        w, h = h, w
    audio = [s for s in info["streams"] if s.get("codec_type") == "audio"]
    dur = 0.0
    for cand in (info.get("format", {}).get("duration"), v.get("duration"),
                 audio[0].get("duration") if audio else None):
        try:
            if cand and float(cand) > 0:
                dur = float(cand)
                break
        except ValueError:
            pass
    if dur <= 0:  # durée absente des métadonnées → on la mesure en décodant
        out = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
                             capture_output=True, text=True).stderr
        t = re.findall(r"time=(\d+):(\d+):([\d.]+)", out)
        if t:
            hh, mm, ss = t[-1]
            dur = int(hh) * 3600 + int(mm) * 60 + float(ss)
    if dur <= 0.5:
        fail("durée de la vidéo introuvable ou trop courte.")
    return {"w": max(w - w % 2, 16), "h": max(h - h % 2, 16), "dur": dur, "audio": bool(audio),
            "vcodec": v.get("codec_name"), "fps": v.get("avg_frame_rate")}


# ---------------------------------------------------------- transcription
def transcribe(video, work, model_size, src):
    from faster_whisper import WhisperModel
    wav = work / "audio16.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-map", "0:a:0", "-vn", "-ac", "1",
         "-ar", "16000", wav])
    a, _ = sf.read(wav, dtype="float32")
    if a.size == 0 or float(np.abs(a).max()) < 1e-4:
        fail("aucune voix audible dans la piste son : rien à traduire.")
    model = hf_load(f"Whisper {model_size}", lambda local: WhisperModel(
        model_size, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 2,
        local_files_only=local))

    def one_pass(lang):
        segs, info = model.transcribe(
            str(wav), language=lang, vad_filter=True, word_timestamps=True, beam_size=5,
            condition_on_previous_text=False)
        words, dropped, lps = [], 0, []
        for s in segs:
            # Hallucinations typiques de Whisper (silence, musique, boucles « mot mot mot… »)
            if (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0) or s.compression_ratio > 2.4:
                dropped += 1
                continue
            lps.append(s.avg_logprob)
            for w in (s.words or []):
                if w.word.strip() and w.end > w.start:
                    words.append({"start": float(w.start), "end": float(w.end), "word": w.word})
        toks = [w["word"].strip().lower() for w in words]
        rep = sum(1 for i in range(1, len(toks)) if toks[i] == toks[i - 1]) / max(len(toks), 1)
        score = (float(np.mean(lps)) if lps else -9.0) - 2.0 * rep - 0.05 * dropped
        return {"lang": lang or info.language, "info": info, "words": words, "dropped": dropped,
                "rep": rep, "score": score}

    if src != "auto":
        best = one_pass(src)
        prob, cands = 1.0, [(src, 1.0)]
    else:
        # 1) Whisper propose ses langues ; on corrige les confusions classiques
        #    (accent britannique → gallois, etc.) en favorisant les langues courantes.
        first = one_pass(None)
        probs = dict(first["info"].all_language_probs or [(first["lang"], 1.0)])
        cands = sorted(probs.items(), key=lambda kv: kv[1] * (1.0 if kv[0] in COMMON_LANGS else 0.15),
                       reverse=True)[:3]
        log("Langues candidates : " + ", ".join(f"{l} {p:.0%}" for l, p in cands))
        best = first if cands[0][0] == first["lang"] else one_pass(cands[0][0])
        # 2) transcription douteuse (boucles, faible confiance) → on essaie la candidate suivante
        if (best["rep"] > 0.12 or best["score"] < -1.0) and len(cands) > 1:
            alt_lang = next(l for l, _ in cands if l != best["lang"])
            alt = one_pass(alt_lang)
            log(f"Transcription douteuse en « {best['lang']} » (score {best['score']:.2f}) → essai en "
                f"« {alt_lang} » (score {alt['score']:.2f})")
            if alt["score"] > best["score"]:
                best = alt
        prob = probs.get(best["lang"], 0.0)
        if best["lang"] != first["lang"]:
            warn(f"Langue corrigée : Whisper proposait « {first['lang']} », retenu « {best['lang']} ».")
    words, lang, dropped = best["words"], best["lang"], best["dropped"]
    REPORT.update(langue_source=lang, langue_source_proba=round(prob, 2), langue_source_mode=src,
                  langues_candidates=[[l, round(p, 2)] for l, p in cands],
                  mots=len(words), segments_ecartes=dropped, score_transcription=round(best["score"], 2))
    log(f"Langue source : {lang} (certitude {prob:.0%}{', forcée' if src != 'auto' else ''}), "
        f"{len(words)} mots, {dropped} passage(s) douteux écarté(s), score {best['score']:.2f}")
    if src == "auto" and prob < 0.5:
        warn(f"Détection de langue peu sûre ({lang}, {prob:.0%}). Si le résultat est faux, "
             "relance en remplissant le champ « Langue PARLÉE ».")
    if best["score"] < -1.2:
        warn("Transcription de qualité faible (son difficile ?) : vérifie le fichier transcription.")
    del model
    gc.collect()
    return words, lang


def resegment(words, soft=7.0, hard=12.0, gap=0.7):
    """Phrases courtes pour la synchro, mais sans couper au milieu d'une idée quand c'est évitable."""
    groups, cur = [], []

    def close():
        nonlocal cur
        if cur:
            groups.append(cur)
        cur = []

    for i, w in enumerate(words):
        if cur:
            prev = cur[-1]["word"].strip()
            ponct = (SENT_END.search(prev) or SOFT_END.search(prev)) and not ABBREV.match(prev)
            pause = w["start"] - cur[-1]["end"]
            # une pause coupe la phrase seulement après une ponctuation, ou si elle est longue
            # (évite « all U.S. | military bases »)
            if (pause > gap and ponct) or pause > 1.6 or w["end"] - cur[0]["start"] > hard:
                close()
        cur.append(w)
        tok = w["word"].strip()
        dur = cur[-1]["end"] - cur[0]["start"]
        nxt_gap = (words[i + 1]["start"] - w["end"]) if i + 1 < len(words) else 9
        if SENT_END.search(tok) and not ABBREV.match(tok) and dur > 1.0:
            close()
        elif dur > soft and (SOFT_END.search(tok) or nxt_gap > 0.2):
            close()
    close()
    out = []
    for g in groups:
        text = re.sub(r"\s+", " ", "".join(x["word"] for x in g)).strip()
        text = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", text, flags=re.I)  # « mot mot mot » → « mot »
        if text and re.search(r"\w", text):
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
        _MT["tok"] = hf_load("tokenizer NLLB", lambda local: AutoTokenizer.from_pretrained(
            NLLB_MODEL, local_files_only=local))
        _MT["model"] = hf_load("modèle NLLB", lambda local: AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_MODEL, local_files_only=local)).eval()
    tok, model = _MT["tok"], _MT["model"]
    tok.src_lang = NLLB[src]
    bos = tok.convert_tokens_to_ids(NLLB[tgt])
    out = []
    for i in range(0, len(texts), 8):
        batch = texts[i:i + 8]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=400)
        with torch.inference_mode():
            gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=4,
                                 max_new_tokens=min(400, int(enc["input_ids"].shape[1] * 2.5) + 16),
                                 no_repeat_ngram_size=4, repetition_penalty=1.1)
        for src_t, t in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
            t = t.strip()
            # garde-fou : traduction vide ou délirante (beaucoup trop longue) → phrase sautée
            if not t or len(t) > 4 * len(src_t) + 40:
                t = ""
            out.append(t)
        log(f"  traduit {min(i + 8, len(texts))}/{len(texts)}")
    bad = sum(1 for t in out if not t)
    if bad:
        warn(f"{bad} phrase(s) n'ont pas pu être traduites (sautées).")
    return out


# ------------------------------------------------------------------- voix
def voice_model(lang, gender):
    rel = VOICES[lang][gender]
    onnx = VOICES_DIR / (Path(rel).name + ".onnx")
    fetch(VOICE_URL + rel + ".onnx", onnx)
    fetch(VOICE_URL + rel + ".onnx.json", Path(str(onnx) + ".json"))
    return onnx


def clean_for_tts(text):
    """Retire émojis, symboles et caractères de contrôle qui font échouer la synthèse."""
    t = "".join(c for c in unicodedata.normalize("NFC", text)
                if unicodedata.category(c)[0] in "LNPZM" or c in " '-")
    return re.sub(r"\s+", " ", t).strip()


def piper(text, model, out: Path, length_scale=1.0):
    env = dict(os.environ, LD_LIBRARY_PATH=str(PIPER_BIN.parent))
    cmd = [str(PIPER_BIN), "--model", str(model), "--output_file", str(out),
           "--length_scale", f"{length_scale:.3f}", "--sentence_silence", "0.05"]
    err = ""
    for _ in range(2):
        out.unlink(missing_ok=True)
        try:
            r = subprocess.run(cmd, input=text.encode(), capture_output=True, env=env, timeout=120)
            err = r.stderr.decode(errors="ignore")[-300:]
            if r.returncode == 0 and out.exists() and out.stat().st_size > 2000:
                return
        except subprocess.TimeoutExpired:
            err = "délai dépassé"
    raise RuntimeError(f"Piper : {err}")


def load_48k(path, tempo=1.0):
    out = path.with_name(path.stem + "_48k.wav")
    af, t = [], tempo
    while t > 2.0:
        af.append("atempo=2.0")
        t /= 2.0
    if abs(t - 1.0) > 0.005:
        af.append(f"atempo={t:.4f}")
    af.append(f"aresample={SR}")
    run(["ffmpeg", "-y", "-v", "error", "-i", path, "-af", ",".join(af), "-ac", "1", out])
    a, _ = sf.read(out, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return trim_silence(a)


def trim_silence(a, thr_db=-42):
    if a.size == 0:
        return a
    thr = 10 ** (thr_db / 20) * max(float(np.abs(a).max()), 1e-6)
    idx = np.where(np.abs(a) > thr)[0]
    if idx.size == 0:
        return a[:0]
    return a[max(idx[0] - int(0.01 * SR), 0):min(idx[-1] + int(0.04 * SR), a.size)]


def build_dub(segs, texts, model, video_dur, work):
    """Chaque phrase démarre quand la personne parle ; le débit s'adapte pour finir avec elle.
    Si la voix prend du retard, l'accélération autorisée augmente pour rattraper."""
    track = np.zeros(int((video_dur + 2) * SR), dtype=np.float32)
    timing, cursor = [], 0.0
    st = {"phrases": 0, "voix_ratees": 0, "accelerees": 0, "retard_max_s": 0.0,
          "coupee_fin_s": 0.0, "ecart_moyen_s": 0.0}
    ecarts = []
    for i, (seg, text) in enumerate(zip(segs, texts)):
        say = clean_for_tts(text) if text else ""
        if not re.search(r"\w", say):
            timing.append(None)
            continue
        pos = max(seg["start"], cursor)
        behind = pos - seg["start"]
        nxt = segs[i + 1]["start"] if i + 1 < len(segs) else video_dur
        nxt = min(nxt, video_dur)
        avail = max(nxt - pos - 0.06, 0.35)
        target = max(min(seg["end"], nxt) - pos, 0.35)
        raw = work / f"tts_{i:04d}.wav"
        try:
            piper(say, model, raw, 1.0)
            d0 = sf.info(raw).duration
            ls = min(max(target / max(d0, 0.05), 0.74), 1.12)
            if abs(ls - 1.0) > 0.03:
                piper(say, model, raw, ls)
            a = load_48k(raw)
            if a.size == 0:
                raise RuntimeError("voix vide")
            d1 = a.size / SR
            max_tempo = 1.3 if behind < 0.8 else 1.5   # rattrapage si retard
            if pos + d1 > video_dur:                      # fin de vidéo : on tient dedans
                max_tempo = 1.6
            if d1 > avail:
                tempo = min(d1 / avail, max_tempo)
                a = load_48k(raw, tempo)
                st["accelerees"] += 1
        except Exception as e:
            st["voix_ratees"] += 1
            warn(f"Voix off ratée pour la phrase {i + 1} ({seg['start']:.1f} s) — sous-titre gardé. ({e})")
            timing.append((seg["start"], max(seg["end"], seg["start"] + 1.0)))
            continue
        start = int(pos * SR)
        end = min(start + a.size, int(video_dur * SR))
        if end <= start:
            timing.append(None)
            continue
        if end - start < a.size:
            st["coupee_fin_s"] += round((a.size - (end - start)) / SR, 2)
            fade = min(int(0.15 * SR), end - start)
            a = a.copy()
            a[end - start - fade:end - start] *= np.linspace(1, 0, fade, dtype=np.float32)
        track[start:end] += a[:end - start]
        st["phrases"] += 1
        st["retard_max_s"] = round(max(st["retard_max_s"], behind), 2)
        ecarts.append(behind)
        cursor = pos + a.size / SR + 0.05
        timing.append((pos, min(pos + a.size / SR, video_dur)))
        raw.unlink(missing_ok=True)
        raw.with_name(raw.stem + "_48k.wav").unlink(missing_ok=True)
    if ecarts:
        st["ecart_moyen_s"] = round(float(np.mean(ecarts)), 2)
    if st["phrases"] == 0:
        fail("aucune phrase n'a pu être doublée (voix off entièrement en échec).")
    if st["voix_ratees"] > max(3, 0.2 * len(segs)):
        warn(f"Beaucoup de voix ratées ({st['voix_ratees']}/{len(segs)}).")
    if st["coupee_fin_s"] > 0.5:
        warn(f"La fin de la voix off dépassait la vidéo : {st['coupee_fin_s']} s coupées en fondu.")
    track = track[:int(video_dur * SR) + 1]
    peak = float(np.abs(track).max() or 1.0)
    track *= 0.89 / peak
    out = work / "dub.wav"
    sf.write(out, track, SR)
    log(f"Voix off : {st['phrases']} phrases, {st['accelerees']} accélérées, "
        f"{st['voix_ratees']} ratées, retard moyen {st['ecart_moyen_s']} s / max {st['retard_max_s']} s")
    return out, timing, st


# ------------------------------------------------- séparation voix / musique
def separate(video, work):
    """Isole la voix (Demucs htdemucs) : la musique de fond n'est JAMAIS gardée.
    Retourne (voix.wav, infos). En cas d'échec : (None, infos) → voix d'origine coupée."""
    full = work / "orig44.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-map", "0:a:0", "-vn", "-ac", "2",
         "-ar", "44100", full])
    sep = work / "sep"
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
           "-o", str(sep), "--filename", "{stem}.{ext}", str(full)]
    err = ""
    for attempt in range(2):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=2700)
            err = (r.stderr or "")[-600:]
            voc, acc = sep / "htdemucs" / "vocals.wav", sep / "htdemucs" / "no_vocals.wav"
            if r.returncode == 0 and voc.exists() and voc.stat().st_size > 10000:
                v, _ = sf.read(voc, dtype="float32")
                m, _ = sf.read(acc, dtype="float32") if acc.exists() else (np.zeros(1), 0)
                rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
                rv, rm = rms(v), rms(m)
                ratio_db = 20 * math.log10(max(rm, 1e-9) / max(rv, 1e-9))
                music = rm > 0.01 and ratio_db > -20
                info = {"separation": "OK", "musique_detectee": bool(music),
                        "niveau_musique_vs_voix_db": round(ratio_db, 1)}
                log(f"Séparation OK — musique de fond {'DÉTECTÉE et supprimée' if music else 'absente ou négligeable'} "
                    f"({ratio_db:+.0f} dB par rapport à la voix)")
                return voc, info
        except subprocess.TimeoutExpired:
            err = "délai dépassé (45 min)"
        log(f"Séparation : essai {attempt + 1} raté ({err.strip()[-300:]})")
    warn("Séparation voix/musique impossible : par sécurité, la voix d'origine est COUPÉE "
         "(aucune musique ne peut passer). La traduction continue normalement.")
    return None, {"separation": "ECHEC → voix d'origine coupée", "erreur_separation": err.strip()[-300:]}


def build_orig_voice(vocals, segs, video_dur, work):
    """Voix d'origine isolée, filtrée, et gardée UNIQUEMENT pendant que la personne parle.
    Intro / outro / passages musicaux = silence total."""
    tmp = work / "voc48.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", vocals, "-af",
         "highpass=f=90,lowpass=f=7500,aresample=48000", "-ac", "1", tmp])
    v, _ = sf.read(tmp, dtype="float32")
    n = int(video_dur * SR) + 1
    v = np.pad(v, (0, max(0, n - v.size)))[:n]
    mask = np.zeros(n, dtype=np.float32)
    for s in segs:
        a, b = int(max(s["start"] - 0.15, 0) * SR), int(min(s["end"] + 0.25, video_dur) * SR)
        mask[a:b] = 1.0
    k = int(0.06 * SR)  # fondus de 60 ms : pas de clic
    if k > 1:
        mask = np.convolve(mask, np.ones(k, dtype=np.float32) / k, mode="same")
    out = work / "orig_voice.wav"
    sf.write(out, (v * mask).astype(np.float32), SR)
    return out


# ------------------------------------------------------------- sous-titres
def ass_time(t):
    cs = int(round(max(t, 0) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def srt_time(t):
    ms = int(round(max(t, 0) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def chunk_text(text, max_chars):
    """Découpe équilibrée : pas de petit bout orphelin (« possibilité » tout seul)."""
    words = text.split()
    if not words:
        return []
    n = max(1, math.ceil(len(text) / max_chars))
    target = len(text) / n
    chunks, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max(target, 1) * 1.15 and len(chunks) < n - 1:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    chunks.append(cur)
    return chunks


def word_weights(words):
    """Durée relative de chaque mot à l'oral : longueur + petite pause après la ponctuation."""
    out = []
    for w in words:
        k = max(len(re.sub(r"\W", "", w)), 1) + 1.5
        if SENT_END.search(w):
            k += 4
        elif SOFT_END.search(w):
            k += 2.5
        out.append(k)
    return out


def build_cues(texts, timing, line_chars):
    """Chaque sous-titre (2 lignes max) + l'horaire de chaque mot pour le karaoké."""
    cues = []
    for text, tm in zip(texts, timing):
        if not text or tm is None:
            continue
        s, e = tm
        e = max(e, s + 0.8)
        parts = chunk_text(text.replace("\n", " "), line_chars * 2)
        total = sum(len(p) for p in parts) or 1
        t = s
        for p in parts:
            d = (e - s) * len(p) / total
            ws = p.split()
            wt = word_weights(ws)
            tw, acc, words = sum(wt), t, []
            for w, k in zip(ws, wt):
                dw = d * k / tw
                words.append((acc, acc + dw, w))
                acc += dw
            cues.append({"start": t, "end": t + d, "text": p, "words": words})
            t += d
    for k in range(len(cues) - 1):  # jamais deux sous-titres superposés
        if cues[k]["end"] > cues[k + 1]["start"]:
            cues[k]["end"] = max(cues[k + 1]["start"] - 0.01, cues[k]["start"] + 0.2)
    return cues


def layout(geo, mode):
    """Bandeau en HAUT, dans le cadre : la sortie garde exactement le format de l'entrée.
    La vidéo est légèrement réduite sous le bandeau (rien n'est caché), côtés en flou artistique."""
    W, H = geo["w"], geo["h"]
    if mode == "non":
        return {"band": 0, "fs": 0, "stripe": 0}
    portrait = H > W * 1.1
    fs = int(W * 0.058) if portrait else int(H * 0.058)
    fs = max(fs, 20)
    band = int(fs * 3.35)
    band += band % 2
    stripe = max(int(fs * 0.12), 3)
    return {"band": band, "fs": fs, "stripe": stripe}


def line_chars_for(geo, lay):
    if not lay["fs"]:
        return 40
    return max(int(geo["w"] * 0.90 / (lay["fs"] * 0.60)), 12)


# couleurs ASS : &HBBGGRR
C_WAIT, C_NOW, C_DONE = "&HB4B4B4&", "&H3DC8FF&", "&HFFFFFF&"   # gris / or / blanc
C_GLOW = "&H0060B0&"                                               # halo orangé du mot en cours


def ass_escape(t):
    return t.replace("\\", "/").replace("{", "(").replace("}", ")")


def write_subs(cues, lang, geo, lay, work, stem):
    W, H = geo["w"], geo["h"]
    fs, band = lay["fs"] or 20, lay["band"]
    font = "Noto Sans Arabic" if lang == "ar" else "Noto Sans"
    hdr = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: K,{font},{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,"
        f"100,100,{max(fs // 40, 0)},0,1,{max(fs / 14, 1.5):.1f},{max(fs / 22, 1):.1f},5,"
        f"{int(W * 0.05)},{int(W * 0.05)},0,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    cy = (band - lay["stripe"]) // 2
    lines = []
    for c in cues:
        s0 = c["start"]
        parts = []
        for ws, we, w in c["words"]:
            t1, t2 = int((ws - s0) * 1000), int((we - s0) * 1000)
            parts.append(
                f"{{\\1c{C_WAIT}\\3c&H000000&"
                f"\\t({t1},{t1 + 70},\\1c{C_NOW}\\3c{C_GLOW})"
                f"\\t({t2},{t2 + 140},\\1c{C_DONE}\\3c&H000000&)}}{ass_escape(w)}")
        body = " ".join(parts)
        lines.append(f"Dialogue: 0,{ass_time(s0)},{ass_time(c['end'])},K,,0,0,0,,"
                     f"{{\\an5\\pos({W // 2},{cy})\\fad(120,90)\\blur0.6}}{body}")
    ass = work / f"subs_{lang}.ass"
    ass.write_text(hdr + "\n".join(lines) + "\n", encoding="utf-8")
    srt = work / f"{stem}__{lang.upper()}.srt"
    srt.write_text("".join(f"{i}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n{c['text']}\n\n"
                           for i, c in enumerate(cues, 1)), encoding="utf-8")
    return ass, srt


def check_fonts(lang):
    want = "Noto Sans Arabic" if lang == "ar" else "Noto Sans"
    try:
        got = run(["fc-match", f"{want}:bold", "family"]).strip()
    except Exception:
        got = "?"
    if want.lower() not in got.lower():
        warn(f"Police « {want} » absente (remplacée par « {got} ») : rendu des sous-titres dégradé.")


# ------------------------------------------------------------------ rendu
def video_filter(geo, lay, ass_name):
    W, H = geo["w"], geo["h"]
    band = lay["band"]
    if not band:
        return f"[0:v:0]scale={W}:{H}:flags=lanczos,setsar=1,format=yuv420p[v]"
    vh = H - band
    st = lay["stripe"]
    third = W // 3
    return (
        f"[0:v:0]setsar=1,split=2[a][b];"
        # fond : la vidéo elle-même agrandie, floutée et assombrie (remplit les côtés)
        f"[a]scale={W}:{vh}:force_original_aspect_ratio=increase:flags=bilinear,crop={W}:{vh},"
        f"boxblur=luma_radius=24:luma_power=2,eq=brightness=-0.18:saturation=0.8[bg];"
        # premier plan : la vidéo entière, réduite pour tenir sous le bandeau (rien n'est coupé)
        f"[b]scale={W}:{vh}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos[fg];"
        f"[bg]setsar=1[bg1];[fg]setsar=1[fg1];[bg1][fg1]overlay=(W-w)/2:(H-h)/2[vid];"
        f"[vid]pad={W}:{H}:0:{band}:color=0x0A0A0E,"
        # filet tricolore sous le bandeau
        f"drawbox=x=0:y={band - st}:w={third}:h={st}:color=0x1F4FB8:t=fill,"
        f"drawbox=x={third}:y={band - st}:w={W - 2 * third}:h={st}:color=0xF2F2F2:t=fill,"
        f"drawbox=x={W - third}:y={band - st}:w={third}:h={st}:color=0xD7263D:t=fill,"
        f"subtitles={ass_name},setsar=1,format=yuv420p[v]"
    )


def render(video, dub, orig_voice, ass, geo, lay, orig_vol, out):
    inputs = ["-i", Path(video).resolve(), "-i", Path(dub).resolve()]
    a = "[1:a]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[d]"
    if orig_voice is not None and orig_vol > 0:
        inputs += ["-i", Path(orig_voice).resolve()]
        a += (f";[2:a]aresample=48000,volume={orig_vol:.3f}[o];"
              "[d][o]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
              "alimiter=limit=0.95,apad[a]")
    else:
        a += ";[d]apad[a]"
    # Le son d'origine de la vidéo (piste 0:a) n'est JAMAIS utilisé : aucune musique possible.
    cmd = ["ffmpeg", "-y", "-v", "error", *inputs,
           "-filter_complex", f"{video_filter(geo, lay, ass.name)};{a}",
           "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "192k", "-t", f"{geo['dur']:.3f}",
           "-movflags", "+faststart", Path(out).resolve()]
    run(cmd, cwd=ass.parent)
    chk = probe(out)
    problems = []
    if abs(chk["dur"] - geo["dur"]) > 1.5:
        problems.append(f"durée {chk['dur']:.1f} s au lieu de {geo['dur']:.1f} s")
    if (chk["w"], chk["h"]) != (geo["w"], geo["h"]):
        problems.append(f"format {chk['w']}x{chk['h']} au lieu de {geo['w']}x{geo['h']}")
    if not chk["audio"]:
        problems.append("pas de son")
    else:
        vd = subprocess.run(["ffmpeg", "-v", "info", "-i", str(out), "-map", "0:a:0",
                             "-af", "volumedetect", "-f", "null", "-"],
                            capture_output=True, text=True).stderr
        mv = re.search(r"max_volume:\s*(-?[\d.]+)", vd)
        if mv and float(mv.group(1)) < -45:
            problems.append("son quasi muet")
    if problems:
        fail("vidéo finale incohérente : " + ", ".join(problems) + ".")


# ----------------------------------------------------------------- rapport
def write_report(outdir, stem):
    path = outdir / f"rapport_{stem}.json"
    path.write_text(json.dumps(REPORT, ensure_ascii=False, indent=2), encoding="utf-8")
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summ:
        return
    r = REPORT
    L = [f"### {'✅' if r.get('statut') == 'OK' else '❌'} {r.get('video', '?')}", ""]
    if r.get("erreur"):
        L += [f"**Erreur** (étape « {r.get('etape_en_echec')} ») : {r['erreur']}", ""]
    L += ["| Élément | Valeur |", "|---|---|",
          f"| Durée vidéo | {r.get('duree_s', '?')} s ({r.get('dimensions', '?')}) |",
          f"| Langue parlée | {r.get('langue_source', '?')} (certitude {r.get('langue_source_proba', '?')}) |",
          f"| Musique de fond | {'détectée → supprimée' if r.get('musique_detectee') else 'absente'} "
          f"(séparation : {r.get('separation', '?')}) |",
          f"| Mots transcrits | {r.get('mots', '?')} — phrases : {r.get('phrases', '?')} |"]
    for lang, s in r.get("sorties", {}).items():
        L.append(f"| Sortie {lang.upper()} | {s.get('fichier')} — {s.get('phrases')} phrases doublées, "
                 f"{s.get('accelerees')} accélérées, {s.get('voix_ratees')} ratées, "
                 f"retard moyen {s.get('ecart_moyen_s')} s |")
    L.append("| Durées des étapes | " + ", ".join(f"{e['etape']} {e['secondes']} s"
                                                 for e in r.get("etapes", [])) + " |")
    if r.get("avertissements"):
        L += ["", "**Avertissements :**"] + [f"- {w}" for w in r["avertissements"]]
    with open(summ, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n\n")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--langs", default="fr")
    ap.add_argument("--src", default="auto")
    ap.add_argument("--voice", default="homme", choices=["homme", "femme"])
    ap.add_argument("--subs", default="bandeau", choices=["bandeau", "bande", "bas", "non"])
    ap.add_argument("--orig-vol", type=float, default=0.05)
    ap.add_argument("--whisper", default="medium")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()

    video = Path(a.input).resolve()
    outdir = Path(a.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w\-]+", "_", video.stem, flags=re.ASCII).strip("_")[:80] or "video"
    if video.suffix.lower() != ".mp4":  # a.mp4 et a.mov ne s'écrasent pas
        stem += "_" + video.suffix.lower().lstrip(".")
    REPORT.update(video=video.name, sorties={}, parametres=vars(a))
    mode = "non" if a.subs == "non" else "bandeau"

    try:
        stage("1. Vérifications et analyse de la vidéo")
        langs = list(dict.fromkeys(l.strip().lower() for l in re.split(r"[,; ]+", a.langs) if l.strip()))
        bad = [l for l in langs if l not in VOICES]
        if not langs or bad:
            fail(f"langue(s) de sortie non gérée(s) : {bad or 'aucune'}. Choix possibles : fr, it, ar, en.")
        src = a.src.strip().lower() or "auto"
        if src != "auto" and src not in NLLB:
            fail(f"langue parlée « {src} » inconnue. Mets « auto » ou un code comme ar, it, de, zh, ja.")
        if not 0 <= a.orig_vol <= 1:
            fail("le volume d'origine doit être entre 0 et 1 (ex. 0.05).")
        if not PIPER_BIN.exists():
            fail("moteur de voix Piper absent (étape d'installation ratée).")
        if not video.exists() or video.stat().st_size == 0:
            fail("vidéo absente ou vide.")
        geo = probe(video)
        REPORT.update(duree_s=round(geo["dur"], 1), dimensions=f"{geo['w']}x{geo['h']}")
        log(f"Vidéo {geo['w']}x{geo['h']}, {geo['dur']:.1f} s, codec {geo['vcodec']}, son={geo['audio']}")
        if not geo["audio"]:
            fail("la vidéo n'a pas de son : rien à traduire.")
        if geo["dur"] > 900:
            fail(f"vidéo de {geo['dur'] / 60:.0f} min : limite fixée à 15 min pour rester fiable.")
        if geo["dur"] > 420:
            warn(f"Vidéo de {geo['dur'] / 60:.1f} min (plus de 7 min) : traitement plus long.")
        work = Path("work").resolve() / stem
        work.mkdir(parents=True, exist_ok=True)

        stage("2. Séparation voix / musique (Demucs)")
        vocals, sep_info = separate(video, work)
        REPORT.update(sep_info)

        stage("3. Transcription (Whisper)")
        words, src = transcribe(vocals or video, work, a.whisper, src)
        if src not in NLLB:
            fail(f"langue détectée « {src} » non traduisible. Force la langue dans le champ « Langue PARLÉE ».")
        segs = resegment(words)
        REPORT["phrases"] = len(segs)
        if not segs:
            fail("aucune parole détectée dans la vidéo (musique seule ou silence ?).")
        (outdir / f"{stem}__transcription_{src}.txt").write_text(
            "\n".join(f"[{s['start']:.1f}] {s['text']}" for s in segs), encoding="utf-8")
        log(f"{len(segs)} phrases")
        orig_voice = build_orig_voice(vocals, segs, geo["dur"], work) if vocals and a.orig_vol > 0 else None
        lay = layout(geo, mode)

        for lang in langs:
            stage(f"4. Traduction {src} → {lang}")
            texts = [clean_for_tts(t) for t in translate([s["text"] for s in segs], src, lang)]
            if not any(texts):
                fail("la traduction n'a rien produit.")
            stage(f"5. Voix off {lang.upper()} ({a.voice})")
            model = voice_model(lang, a.voice)
            if lang == "ar" and a.voice == "femme":
                warn("Pas de voix féminine arabe disponible : voix masculine utilisée.")
            dub, timing, st = build_dub(segs, texts, model, geo["dur"], work)
            stage(f"6. Sous-titres karaoké {lang.upper()} ({mode})")
            check_fonts(lang)
            cues = build_cues(texts, timing, line_chars_for(geo, lay))
            ass, srt = write_subs(cues, lang, geo, lay, work, stem)
            srt.replace(outdir / srt.name)
            stage(f"7. Montage final {lang.upper()}")
            out = outdir / f"{stem}__{lang.upper()}.mp4"
            render(video, dub, orig_voice, ass, geo, lay, a.orig_vol, out)
            size = out.stat().st_size / 1e6
            if size > 1900:
                warn(f"Fichier de {size:.0f} Mo : proche de la limite de 2 Go des releases.")
            st["fichier"] = out.name
            REPORT["sorties"][lang] = st
            log(f"OK → {out.name} ({size:.1f} Mo)")
        stage("")
        REPORT["statut"] = "OK"
    except SystemExit as e:
        REPORT.update(statut="ECHEC", etape_en_echec=_STAGE["nom"],
                      erreur=re.sub(r"^::error::\[[^\]]*\]\s*", "", str(e.code)))
        stage("")
        write_report(outdir, stem)
        raise
    except Exception as e:
        traceback.print_exc()
        REPORT.update(statut="ECHEC", etape_en_echec=_STAGE["nom"], erreur=f"{type(e).__name__}: {e}")
        stage("")
        write_report(outdir, stem)
        print(f"::error::[{REPORT['etape_en_echec']}] {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
    write_report(outdir, stem)


if __name__ == "__main__":
    main()
