"""Fix 33 — verify strip_lyrics_from_image_prompt removes ALL lyric leaks
from frame_variant_prompt, including the "The text:" / "The lyrics:" /
substring patterns the LLM uses to bypass the quote ban.

Pure / network-free.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_video_pipeline import (  # noqa: E402
    strip_lyrics_from_image_prompt,
    _SEG_DIRECTOR_RULES,
)


# ---------------------------------------------------------------------------
# Existing Fix 26 cleanup paths must keep working.
# ---------------------------------------------------------------------------

def test_strip_removes_double_quoted_lyric_run():
    out = strip_lyrics_from_image_prompt(
        'Singer in cafe, she sings "c\'est le temps des nostalgies" softly.'
    )
    assert "nostalgies" not in out
    assert "Singer in cafe" in out


def test_strip_removes_sings_lead_in_with_unquoted_tail():
    out = strip_lyrics_from_image_prompt(
        "Medium shot. He sings: c'est le temps des nostalgies. Background blurred."
    )
    assert "nostalgies" not in out
    assert "Medium shot" in out
    assert "Background blurred" in out


# ---------------------------------------------------------------------------
# Fix 33 — the new introducer patterns must be stripped.
# ---------------------------------------------------------------------------

def test_strip_removes_the_text_introducer():
    """Regression from c72fa617: frame_variant_prompt leaked lyrics via
    'The text: <line>' construct, T2I rendered it as on-screen caption."""
    out = strip_lyrics_from_image_prompt(
        "Close-up of woman in knit sweater. The text: c'est le temps des nostalgies. Warm cafe lighting."
    )
    assert "nostalgies" not in out
    assert "Close-up" in out
    assert "Warm cafe lighting" in out


def test_strip_removes_the_lyrics_introducer():
    out = strip_lyrics_from_image_prompt(
        "Wide shot of cafe at night. The lyrics: ou mon ame s'est endormie. Rain on the window."
    )
    assert "endormie" not in out
    assert "Wide shot" in out
    assert "Rain on the window" in out


def test_strip_removes_the_line_introducer():
    out = strip_lyrics_from_image_prompt(
        "Singer at the bar. The line: feel the magic in the air. Blue lighting."
    )
    assert "feel the magic" not in out.lower()
    assert "Singer at the bar" in out


def test_strip_removes_the_words_introducer():
    out = strip_lyrics_from_image_prompt(
        "Profile shot. The words: dancing in the moonlight. Subtle camera shake."
    )
    assert "dancing in the moonlight" not in out.lower()
    assert "Profile shot" in out


def test_strip_removes_duplicated_introducer_pattern():
    """The actual c72fa617 prompt contained 'The text:: ... The text: ...'."""
    bad = (
        "Medium close-up of singer. The text:: The camera performs a subtle "
        "dolly-in. The text: c'est le temps des nostalgies, où mon âme s'est "
        "endormie. The scene ends with a clean hard cut."
    )
    out = strip_lyrics_from_image_prompt(bad)
    assert "nostalgies" not in out
    assert "endormie" not in out
    assert "Medium close-up of singer" in out


# ---------------------------------------------------------------------------
# Fix 33 — lyrics-substring removal (defense in depth).
# ---------------------------------------------------------------------------

def test_strip_removes_full_lyrics_substring_when_passed():
    fvp = (
        "Close-up of woman in cafe, soft cream knit sweater, hair down "
        "naturally, looking out the window. c'est le temps des nostalgies, "
        "ou mon ame s'est endormie. Warm cafe lighting subtle."
    )
    out = strip_lyrics_from_image_prompt(
        fvp,
        lyrics="c'est le temps des nostalgies, ou mon ame s'est endormie.",
    )
    assert "nostalgies" not in out
    assert "endormie" not in out
    assert "Close-up of woman in cafe" in out
    assert "Warm cafe lighting" in out


def test_strip_removes_substring_case_insensitive():
    out = strip_lyrics_from_image_prompt(
        "She gazes. C'EST LE TEMPS DES NOSTALGIES. Distant rain.",
        lyrics="c'est le temps des nostalgies",
    )
    assert "nostalgies" not in out.lower()
    assert "She gazes" in out
    assert "Distant rain" in out


def test_strip_no_lyrics_arg_falls_back_to_pattern_strip():
    """Calling without lyrics= still strips via patterns (back-compat)."""
    out = strip_lyrics_from_image_prompt(
        'Singer in rain, she sings "feel the rhythm" warmly.'
    )
    assert "feel the rhythm" not in out.lower()


def test_strip_ignores_short_lyric_lines():
    """Very short tokens (<12 chars) are not substring-stripped to avoid
    nuking common English words that happen to appear in the prompt."""
    out = strip_lyrics_from_image_prompt(
        "Close-up of singer in the rain.",
        lyrics="oh",  # 2 chars — must NOT touch "Close-up", etc.
    )
    assert "Close-up of singer in the rain" in out


def test_strip_handles_empty_prompt():
    assert strip_lyrics_from_image_prompt("") == ""
    assert strip_lyrics_from_image_prompt("", lyrics="anything") == ""


def test_strip_handles_no_match_clean():
    """A clean prompt with no lyric leaks should remain intact."""
    clean = "Close-up of singer in soft cream knit sweater, warm cafe lighting"
    out = strip_lyrics_from_image_prompt(clean, lyrics="something unrelated longer than twelve")
    assert out == clean


# ---------------------------------------------------------------------------
# System prompt — Fix 33 introducer ban present.
# ---------------------------------------------------------------------------

def test_seg_director_rules_explicitly_bans_text_introducers():
    text = _SEG_DIRECTOR_RULES.lower()
    assert "fix 33" in text
    for needle in ("the text:", "the lyrics:", "the line:", "the words:"):
        assert needle in text, f"missing ban for introducer {needle!r}"
