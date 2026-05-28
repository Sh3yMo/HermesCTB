"""RC8: align pipeline-authored lyrics to generated audio for exact section
timestamps (fixes segment cuts landing mid-vocal + enables chorus frame reuse).

Path B (gate-proven): Demucs vocal stem -> WhisperX medium ASR + align ->
fuzzy-match ASR words to the [Section] lyric blocks -> per-section [start,end].

FAIL-SOFT: every public entry returns None on ANY problem (missing deps,
demucs/whisperx error, no vocals, bad fixture). Callers MUST fall back to the
existing proportional segmentation when None is returned. This module must
never raise into the pipeline.
"""
from __future__ import annotations

# Fix 15 windows bootstrap: register FFmpeg shared-DLL dir before demucs
# subprocess invocation (demucs depends on torchaudio→torchcodec→avcodec).
import _ffmpeg_init  # noqa: F401

import difflib
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional


def _strip_parens(s: str) -> str:
    return re.sub(r"\(.*?\)", "", s)


def parse_sections(lyrics_path: str) -> List[Dict[str, Any]]:
    """[{'label','lyrics','is_vocal'}] in order; instrumental = no lyric lines."""
    if not lyrics_path or not os.path.exists(lyrics_path):
        return []
    txt = open(lyrics_path, encoding="utf-8").read()
    parts = re.split(r"\[([^\]]+)\]", txt)
    out: List[Dict[str, Any]] = []
    i = 1
    while i < len(parts) - 1:
        label = parts[i].strip()
        body = parts[i + 1].strip()
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        out.append({"label": label, "lyrics": "\n".join(lines),
                    "is_vocal": bool(lines)})
        i += 2
    return out


def _known_tokens(sections: List[Dict[str, Any]]):
    """[(section_idx, token)] for vocal sections (lowercased word tokens)."""
    toks = []
    for si, s in enumerate(sections):
        if not s["is_vocal"]:
            continue
        for t in re.findall(r"[^\W\d_]+", _strip_parens(s["lyrics"]).lower()):
            toks.append((si, t))
    return toks


def _demucs_vocals(audio_path: str, work: str) -> Optional[str]:
    # Fix 18: use sys.executable so Windows (no `python3`) and venvs both work.
    import sys
    try:
        subprocess.run(
            [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
             "-o", work, audio_path],
            check=True, capture_output=True, timeout=900,
        )
        base = os.path.splitext(os.path.basename(audio_path))[0]
        voc = os.path.join(work, "htdemucs", base, "vocals.wav")
        return voc if os.path.exists(voc) else None
    except Exception as e:
        print(f"[lyric_align] demucs failed (non-fatal): {e!r}")
        return None


def _whisperx_words(audio_file: str, lang: str) -> Optional[List[Dict[str, Any]]]:
    try:
        import whisperx  # heavy optional dep
    except Exception as e:
        print(f"[lyric_align] whisperx import failed (non-fatal): {e!r}")
        return None
    try:
        audio = whisperx.load_audio(audio_file)
        model = whisperx.load_model("medium", "cpu", compute_type="int8",
                                    language=lang)
        asr = model.transcribe(audio, batch_size=16)
        model_a, meta = whisperx.load_align_model(language_code=lang,
                                                  device="cpu")
        aligned = whisperx.align(asr["segments"], model_a, meta, audio, "cpu",
                                 return_char_alignments=False)
        words = [{"w": w["word"], "s": w.get("start"), "e": w.get("end")}
                 for seg in aligned["segments"] for w in seg.get("words", [])
                 if w.get("start") is not None]
        return words or None
    except Exception as e:
        print(f"[lyric_align] whisperx align failed (non-fatal): {e!r}")
        return None


def _section_times_from_words(words, sections):
    """Fuzzy-align ASR word seq to known lyric tokens; per-vocal-section
    [start,end]. Returns {section_idx: (start, end)} for vocal sections only."""
    ktoks = _known_tokens(sections)
    if not ktoks or not words:
        return {}
    K = [t for _, t in ktoks]
    A = [re.sub(r"[^\w]", "", w["w"].lower()) for w in words]
    sm = difflib.SequenceMatcher(None, K, A, autojunk=False)
    times: Dict[int, list] = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op not in ("equal", "replace"):
            continue
        for k in range(i1, i2):
            j = j1 + (k - i1)
            if j >= j2 or j >= len(words):
                continue
            s, e = words[j]["s"], words[j]["e"]
            if s is None:
                continue
            si = ktoks[k][0]
            d = times.setdefault(si, [s, e])
            d[0] = min(d[0], s)
            d[1] = max(d[1], e)
    return {si: (round(v[0], 2), round(v[1], 2)) for si, v in times.items()}


def _proportional_fallback(
    sections: List[Dict[str, Any]], audio_dur: float
) -> Optional[List[Dict[str, Any]]]:
    """Distribute audio_dur across sections without word alignment.

    Vocal sections weighted by lyric length; instrumental sections get a small
    fixed share. CRITICAL: preserves the gender-suffix labels from
    parse_sections so portrait routing still works. Boundaries are coarse —
    inaSpeech VAD (Fix 15c/16) refines them downstream.
    """
    if not sections or audio_dur <= 0:
        return None
    vocal_lens = [len(s["lyrics"]) for s in sections if s["is_vocal"] and s["lyrics"]]
    avg_vocal = (sum(vocal_lens) / len(vocal_lens)) if vocal_lens else 40.0
    weights: List[float] = []
    for s in sections:
        if s["is_vocal"]:
            weights.append(float(max(1, len(s["lyrics"]))))
        else:
            weights.append(0.3 * avg_vocal)  # short instrumental beat
    total_w = sum(weights) or 1.0
    out: List[Dict[str, Any]] = []
    cursor = 0.0
    for s, w in zip(sections, weights):
        dur = audio_dur * (w / total_w)
        d = dict(s)
        d["start"] = round(cursor, 2)
        d["end"] = round(min(cursor + dur, audio_dur), 2)
        cursor = d["end"]
        out.append(d)
    if out:
        out[-1]["end"] = round(audio_dur, 2)  # absorb rounding drift
    return out


def align_sections(
    audio_path: str,
    lyrics_path: str,
    audio_dur: float,
    lang: str = "en",
) -> Optional[List[Dict[str, Any]]]:
    """Returns ordered sections with real [start,end] covering [0, audio_dur],
    or None (caller falls back). Vocal sections get aligned times; instrumental
    sections fill the gaps; first starts at 0, last ends at audio_dur.
    """
    try:
        sections = parse_sections(lyrics_path)
        if not sections or audio_dur <= 0:
            return None
        if not any(s["is_vocal"] for s in sections):
            return None

        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="ctb_align_") as work:
            voc = _demucs_vocals(audio_path, work)
            target = voc or audio_path  # fall back to mix if demucs unavailable
            words = _whisperx_words(target, lang)
        if not words:
            # Fix 21: no word alignment (whisperx missing / failed). Don't lose
            # gender routing — emit proportional sections that KEEP the
            # gender-suffix labels. Downstream inaSpeech VAD (Fix 15c/16)
            # refines the coarse boundaries.
            print("[lyric_align] no whisperx words → proportional gender fallback")
            return _proportional_fallback(sections, audio_dur)

        vtimes = _section_times_from_words(words, sections)
        if not vtimes:
            print("[lyric_align] no word-to-section matches → proportional gender fallback")
            return _proportional_fallback(sections, audio_dur)

        # Order vocal anchors; sanity: monotonic-ish, within track.
        n = len(sections)
        out: List[Dict[str, Any]] = [dict(s) for s in sections]
        # Seed vocal sections with aligned times.
        for si, (s, e) in vtimes.items():
            s = max(0.0, min(s, audio_dur))
            e = max(s + 0.1, min(e, audio_dur))
            out[si]["start"] = s
            out[si]["end"] = e

        # Fill sections without a time (instrumental, or unmatched vocal) by
        # interpolating between known neighbors; clamp to [0, audio_dur].
        known = sorted(i for i in range(n) if "start" in out[i])
        if not known:
            return None
        # leading gap -> first known section's start
        first_k = known[0]
        for i in range(0, first_k):
            out[i]["start"] = 0.0 if i == 0 else out[i - 1]["end"]
            out[i]["end"] = out[first_k]["start"]
        # trailing gap -> audio end
        last_k = known[-1]
        for i in range(last_k + 1, n):
            prev_end = out[i - 1]["end"]
            out[i]["start"] = prev_end
            out[i]["end"] = audio_dur
        # interior gaps between consecutive known anchors
        for a, b in zip(known, known[1:]):
            if b == a + 1:
                # adjacent: snap boundary to midpoint of the silence between
                mid = (out[a]["end"] + out[b]["start"]) / 2.0
                out[a]["end"] = mid
                out[b]["start"] = mid
            else:
                gap_s = out[a]["end"]
                gap_e = out[b]["start"]
                span = max(0.0, gap_e - gap_s)
                k = b - a - 1
                step = span / (k + 1) if k else 0.0
                for idx in range(1, k + 1):
                    seg = a + idx
                    out[seg]["start"] = round(gap_s + step * (idx - 1), 2)
                    out[seg]["end"] = round(gap_s + step * idx, 2)
        # global clamp + ensure monotonic non-negative durations
        out[0]["start"] = 0.0
        out[-1]["end"] = round(audio_dur, 2)
        for i in range(1, n):
            if out[i]["start"] < out[i - 1]["end"]:
                out[i]["start"] = out[i - 1]["end"]
            if out[i]["end"] < out[i]["start"] + 0.1:
                out[i]["end"] = round(out[i]["start"] + 0.1, 2)

        # repeated-section (chorus) groups by identical normalized lyrics
        groups: Dict[str, List[int]] = {}
        for i, s in enumerate(out):
            if s["is_vocal"]:
                key = re.sub(r"\s+", " ", s["lyrics"].strip().lower())
                groups.setdefault(key, []).append(i)
        for s in out:
            s["reuse_of"] = None
        for idxs in groups.values():
            if len(idxs) > 1:
                anchor = idxs[0]
                for j in idxs[1:]:
                    out[j]["reuse_of"] = anchor

        print(f"[lyric_align] aligned {len(vtimes)}/{n} vocal sections "
              f"in {round(time.time() - t0, 1)}s "
              f"(reuse groups: {sum(1 for g in groups.values() if len(g) > 1)})")
        for s in out:
            s["start"] = round(float(s["start"]), 2)
            s["end"] = round(float(s["end"]), 2)
        return out
    except Exception as e:
        print(f"[lyric_align] align_sections failed (non-fatal): {e!r}")
        return None
