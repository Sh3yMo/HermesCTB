"""Prompt-hygiene helpers for MSR/PromptRelay constructions.

Pure functions, no I/O, easy to unit-test. Used by `music_video_pipeline`,
`api`, and `msr_refs` to keep beats, globals, portraits, and wardrobe
contracts clean and LTX-2.3-compatible.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable

DANGLING_CONNECTORS = frozenset(
    {
        "as",
        "while",
        "and",
        "but",
        "or",
        "because",
        "though",
        "when",
        "if",
        "until",
        "since",
        "with",
        "of",
        "to",
        "for",
        "by",
        "from",
        "into",
        "onto",
        "than",
    }
)

INSTRUCTIONAL_PATTERNS = (
    re.compile(r"\buse the\b", re.IGNORECASE),
    re.compile(r"\bshould\b", re.IGNORECASE),
    re.compile(r"\bmust\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\bdo not\b", re.IGNORECASE),
    re.compile(r"\bas the real location\b", re.IGNORECASE),
    re.compile(r"\bnot pasted over\b", re.IGNORECASE),
    re.compile(r"\bphysically present in\b", re.IGNORECASE),
    re.compile(r"\bkeep that\b", re.IGNORECASE),
    re.compile(r"\bclothing lock\b", re.IGNORECASE),
)

FRAMING_TOKENS = (
    "standing",
    "head to toe",
    "full body",
    "facing the camera",
    "front-facing",
    "head-to-toe",
    "full-body",
)

NEUTRAL_BG_HEX = "#808080"
NEUTRAL_BG_RGB = (128, 128, 128)


def _normalize_for_similarity(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def beats_similarity_ratio(a: str, b: str) -> float:
    """Token-level similarity ratio between two beat strings (0.0–1.0)."""
    if not a or not b:
        return 0.0
    na = _normalize_for_similarity(a)
    nb = _normalize_for_similarity(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _ends_with_dangling_connector(sentence: str) -> bool:
    stripped = sentence.rstrip(" .,;:")
    if not stripped:
        return True
    last_word = stripped.split()[-1].lower().strip(",.;:!?")
    return last_word in DANGLING_CONNECTORS


def clean_beat_text(
    text: str | None,
    *,
    min_length: int = 4,
    add_trailing_period: bool = False,
) -> str | None:
    """Return cleaned beat text or `None` if nothing meaningful remains.

    - Drops trailing dangling connectors (e.g. "she trembles as" -> "she trembles").
    - Drops empty sentences.
    - Collapses whitespace.
    - Returns `None` if result is empty or shorter than `min_length`.
    - If `add_trailing_period` is True, ensures the result ends with `.` (off
      by default to keep existing callers' expectations).
    """
    if not text:
        return None
    sentences = _split_sentences(text)
    cleaned: list[str] = []
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        while s and _ends_with_dangling_connector(s):
            tokens = s.rstrip(" .,;:").split()
            if len(tokens) <= 1:
                s = ""
                break
            s = " ".join(tokens[:-1]).rstrip(" .,;:")
        if s:
            cleaned.append(s)
    out = " ".join(cleaned).strip()
    out = re.sub(r"\s+", " ", out).strip(" ,;:")
    if not out:
        return None
    if len(out) < min_length:
        return None
    if add_trailing_period and not re.search(r"[.!?]$", out):
        out = out + "."
    return out


def is_instructional_language(text: str | None) -> bool:
    """True if `text` contains LLM-instruction-style tokens."""
    if not text:
        return False
    for pat in INSTRUCTIONAL_PATTERNS:
        if pat.search(text):
            return True
    return False


def _count_framing_token(text: str, token: str) -> int:
    pattern = re.compile(r"\b" + re.escape(token) + r"\b", re.IGNORECASE)
    return len(pattern.findall(text))


def assert_no_duplicate_framing_phrases(prompt: str) -> None:
    """Raise `ValueError` if any framing token appears more than once."""
    if not prompt:
        return
    duplicates: list[str] = []
    for token in FRAMING_TOKENS:
        if _count_framing_token(prompt, token) > 1:
            duplicates.append(token)
    if duplicates:
        raise ValueError(
            f"portrait prompt contains duplicate framing phrases: {duplicates!r}"
        )


def has_duplicate_framing_phrases(prompt: str) -> bool:
    try:
        assert_no_duplicate_framing_phrases(prompt)
    except ValueError:
        return True
    return False


def collapse_duplicate_beats(
    beats: Iterable[str],
    *,
    threshold: float = 0.85,
) -> list[str]:
    """Drop beats that are near-duplicates of an earlier beat."""
    kept: list[str] = []
    for b in beats:
        if not b:
            continue
        if any(beats_similarity_ratio(b, prev) >= threshold for prev in kept):
            continue
        kept.append(b)
    return kept


def normalize_ltx_text(text: str | None) -> str:
    """Deterministic, content-preserving LTX wording cleanup (R2-4 safety net).

    LTX-2.3 prompts must be single flowing natural-language phrasing — no
    colons, no line breaks, no label syntax. This applies ONLY safe transforms
    (it never drops words/sentences, so it cannot change meaning):
    - newlines/tabs collapsed to spaces
    - colons turned into commas ("X: Y" -> "X, Y")
    - repeated spaces collapsed, dangling/duplicated punctuation tidied

    Used as the always-on net under the LLM polish pass: even if the LLM call
    fails, beats/globals reach PromptRelaySmartEncode colon-free and single-line.
    """
    if not text:
        return ""
    # Drop the scene-lock meta phrase — LTX rules forbid meta-instructions ("must"),
    # and the scene description that follows is kept.
    out = re.sub(r"\bScene location must be exactly\b[,:]?\s*", "", text, flags=re.IGNORECASE)
    out = re.sub(r"[\r\n\t]+", " ", out)
    out = out.replace(":", ", ")
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;!?])", r"\1", out)
    out = re.sub(r",\s*([,.;])", r"\1", out)
    out = re.sub(r"(,\s*){2,}", ", ", out)
    out = re.sub(r"\.\s*\.+", ".", out)
    return out.strip(" ,;")


# LTX video-grading / scene-lock artifacts that must never leak into a Flux2 still
# prompt. The music-video segment plan writes these into frame_variant_prompt
# (LTX-flavored), but Flux2 wants clean prose with no video-grade/colon language.
_FLUX2_STRIP_PATTERNS = (
    r"Render this video frame with this overall visual medium and color grade\.?",
    r"The overall video color grade and cinematography use palette tones[^.]*\.?",
    r"Do not recolor[^.]*\.?",
    r"\bNo movement\b\.?",
    r"Scene location must be exactly[,:]?",
)


def normalize_flux2_text(text: str | None) -> str:
    """Deterministic, content-preserving cleanup for Flux2 still prompts.

    Flux2 wants natural-language prose without colons and without LTX video-grading
    language. Strips the known LTX / scene-lock artifacts that leak in from the
    music-video segment plan, converts style-enum underscores to spaces
    (chiaroscuro_caravaggio -> chiaroscuro caravaggio), then applies the shared
    single-line/colon cleanup. Always-on fallback under the LLM Flux2-format pass.
    """
    if not text:
        return ""
    out = text
    for pat in _FLUX2_STRIP_PATTERNS:
        out = re.sub(pat, " ", out, flags=re.IGNORECASE)
    out = re.sub(r"(?<=[A-Za-z])_(?=[A-Za-z])", " ", out)
    return normalize_ltx_text(out)


def wardrobe_contract_to_compact_string(contract: dict) -> str:
    """Flatten a structured wardrobe dict into a single deterministic string.

    Expected shape:
        {
          "top":      {"garment": "...", "color": "...", "fit": "..."},
          "bottom":   {"garment": "...", "color": "..."},
          "footwear": {"garment": "...", "color": "..."},
          "accessories": ["..."],
        }
    Returns a single comma-separated phrase with no ambiguous "or"
    disjunctions. Missing fields are silently skipped; the caller is
    responsible for validating completeness.
    """
    if not isinstance(contract, dict):
        return ""
    parts: list[str] = []

    def _piece(slot: dict | None, *, include_fit: bool = False) -> str:
        if not isinstance(slot, dict):
            return ""
        color = (slot.get("color") or "").strip()
        garment = (slot.get("garment") or "").strip()
        fit = (slot.get("fit") or "").strip() if include_fit else ""
        chunks = [c for c in (fit, color, garment) if c]
        return " ".join(chunks)

    top = _piece(contract.get("top"), include_fit=True)
    if top:
        parts.append(top)
    bot = _piece(contract.get("bottom"))
    if bot:
        parts.append(bot)
    foot = _piece(contract.get("footwear"))
    if foot:
        parts.append(foot)
    accessories = contract.get("accessories") or []
    if isinstance(accessories, list):
        for a in accessories:
            if isinstance(a, str) and a.strip():
                parts.append(a.strip())
    out = ", ".join(parts)
    if " or " in out.lower():
        out = re.sub(r"\s+or\s+", ", ", out, flags=re.IGNORECASE)
    return out


def validate_wardrobe_contract(contract: dict) -> tuple[bool, list[str]]:
    """Return (ok, errors) — requires at least named top.color and bottom.color."""
    errors: list[str] = []
    if not isinstance(contract, dict):
        return False, ["contract is not a dict"]
    top = contract.get("top") or {}
    bot = contract.get("bottom") or {}
    if not isinstance(top, dict) or not (top.get("color") or "").strip():
        errors.append("top.color missing")
    if not isinstance(bot, dict) or not (bot.get("color") or "").strip():
        errors.append("bottom.color missing")
    if not isinstance(top, dict) or not (top.get("garment") or "").strip():
        errors.append("top.garment missing")
    if not isinstance(bot, dict) or not (bot.get("garment") or "").strip():
        errors.append("bottom.garment missing")
    return (len(errors) == 0), errors
