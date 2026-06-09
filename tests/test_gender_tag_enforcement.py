"""Stage L6: gender modifier on every vocal section."""

from __future__ import annotations

import pytest

from audio_enhancer import AudioEnhancer, AudioSettings


def _enforce(settings: AudioSettings) -> None:
    enh = AudioEnhancer.__new__(AudioEnhancer)
    AudioEnhancer._enforce_gender_tags(enh, settings)


def _make(voice: str, structure: list[str], lyrics_text: str = "lyrics here") -> AudioSettings:
    s = AudioSettings()
    s.voice = voice
    s.structure = list(structure)
    s.lyrics = {t: lyrics_text for t in structure}
    return s


# ── voice="male": every vocal section forced to include - male ────


def test_male_voice_injects_male_into_style_only_vocal_tags():
    s = _make("male", [
        "Intro", "Verse 1 - raspy", "Pre-Chorus - male",
        "Chorus - anthemic", "Verse 2 - raspy", "Bridge - male",
        "Final Chorus - anthemic", "Outro",
    ])
    _enforce(s)
    labels = " | ".join(s.structure)
    assert "Verse 1 - male - raspy" in labels
    assert "Pre-Chorus - male" in labels
    assert "Chorus - male - anthemic" in labels
    assert "Verse 2 - male - raspy" in labels
    assert "Bridge - male" in labels
    assert "Final Chorus - male - anthemic" in labels
    # Intro/Outro untouched
    assert "Intro" in s.structure
    assert "Outro" in s.structure
    # lyrics dict re-keyed
    assert all(k in s.lyrics for k in s.structure)


def test_male_voice_leaves_already_tagged_unchanged():
    s = _make("male", ["Intro", "Verse 1 - male", "Chorus - male - powerful", "Outro"])
    _enforce(s)
    assert s.structure == ["Intro", "Verse 1 - male", "Chorus - male - powerful", "Outro"]


# ── voice="female": symmetric ────────────────────────────────────


def test_female_voice_injects_female():
    s = _make("female", ["Intro", "Verse 1 - breathy", "Chorus - powerful", "Outro"])
    _enforce(s)
    assert "Verse 1 - female - breathy" in s.structure
    assert "Chorus - female - powerful" in s.structure


# ── voice="any": majority gender wins, default male fallback ─────


def test_any_voice_uses_majority_gender_when_some_tags_exist():
    s = _make("any", [
        "Intro", "Verse 1 - female - breathy", "Pre-Chorus - anthemic",
        "Chorus - female", "Verse 2 - raspy", "Outro",
    ])
    _enforce(s)
    # 2 female already → inject female
    assert "Pre-Chorus - female - anthemic" in s.structure
    assert "Verse 2 - female - raspy" in s.structure


def test_any_voice_falls_back_to_male_when_no_gender_anywhere():
    s = _make("any", ["Intro", "Verse 1 - raspy", "Chorus - anthemic", "Outro"])
    _enforce(s)
    assert "Verse 1 - male - raspy" in s.structure
    assert "Chorus - male - anthemic" in s.structure


# ── Intro / Outro / instrumental blocks left alone ───────────────


def test_intro_outro_never_get_gender_even_with_lyrics():
    s = _make("male", ["Intro", "Verse 1", "Outro"], lyrics_text="some line")
    _enforce(s)
    assert s.structure[0] == "Intro"
    assert s.structure[-1] == "Outro"


def test_instrumental_blocks_not_touched():
    s = _make("male", [
        "Intro", "Verse 1 - raspy", "Guitar Solo",
        "Chorus - anthemic", "Piano Interlude", "Outro",
    ])
    _enforce(s)
    assert "Guitar Solo" in s.structure
    assert "Piano Interlude" in s.structure
    assert any(t.startswith("Verse 1 - male") for t in s.structure)
    assert any(t.startswith("Chorus - male") for t in s.structure)


# ── duet majority gets normalized ─────────────────────────────────


def test_any_voice_duet_majority_yields_duet_tag():
    s = _make("any", [
        "Intro", "Verse 1 - duet",
        "Pre-Chorus - whispered", "Chorus - duet",
        "Bridge - powerful", "Outro",
    ])
    _enforce(s)
    assert "Pre-Chorus - duet - whispered" in s.structure
    assert "Bridge - duet - powerful" in s.structure


# ── no lyrics → still no injection (vocal tag without lyrics is unusual) ─


def test_empty_lyrics_in_vocal_section_skipped():
    s = AudioSettings()
    s.voice = "male"
    s.structure = ["Intro", "Verse 1 - raspy", "Chorus - anthemic", "Outro"]
    s.lyrics = {"Intro": "", "Verse 1 - raspy": "", "Chorus - anthemic": "lyrics", "Outro": ""}
    _enforce(s)
    # Verse 1 had no lyrics text → leave alone (treated as instrumental verse)
    assert "Verse 1 - raspy" in s.structure
    # Chorus had lyrics → injected
    assert "Chorus - male - anthemic" in s.structure
