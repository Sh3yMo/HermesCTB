"""Stage L4: enforce ACE-Step section budget per duration band."""

from __future__ import annotations

import pytest

from audio_enhancer import (
    AudioEnhancer,
    AudioSettings,
    DURATION_SECTION_BUDGET,
    _section_budget_for,
)


def _enforce(settings: AudioSettings) -> None:
    """Helper: call the private enforcer without constructing the full enhancer."""
    # _enforce_section_budget is an instance method but uses only self.logger refs;
    # we mimic by constructing a bare instance through __new__ to skip __init__.
    enh = AudioEnhancer.__new__(AudioEnhancer)
    AudioEnhancer._enforce_section_budget(enh, settings)


# ── Budget lookup ──────────────────────────────────────────────────


def test_budget_for_60s():
    n, ab, asol, afc = _section_budget_for(60)
    assert n == 6
    assert not ab
    assert not asol
    assert not afc


def test_budget_for_30s():
    n, ab, asol, afc = _section_budget_for(30)
    assert n == 4
    assert not ab and not asol and not afc


def test_budget_for_150s():
    n, ab, asol, afc = _section_budget_for(150)
    assert n == 9
    assert ab
    assert not asol
    assert not afc


def test_budget_for_180s():
    n, ab, asol, afc = _section_budget_for(180)
    assert n == 10
    assert ab and asol and afc


def test_budget_for_240s():
    n, ab, asol, afc = _section_budget_for(240)
    assert n == 13


# ── Drop behavior at 60s ───────────────────────────────────────────


def test_60s_drops_bridge_solo_final_chorus():
    s = AudioSettings()
    s.duration = 60
    s.structure = [
        "Intro", "Verse 1 - raspy", "Pre-Chorus - male",
        "Chorus - anthemic", "Verse 2 - raspy", "Pre-Chorus - male",
        "Chorus - anthemic", "Bridge - male", "Guitar Solo",
        "Final Chorus - anthemic", "Outro",
    ]
    s.lyrics = {t: "some line" for t in s.structure}
    _enforce(s)
    assert len(s.structure) <= 6
    # Bridge / Solo / Final Chorus banned at 60s
    assert not any("bridge" in t.lower() for t in s.structure)
    assert not any("solo" in t.lower() for t in s.structure)
    assert not any("final chorus" in t.lower() for t in s.structure)
    assert s.structure[0] == "Intro"
    # lyrics dict drops keys for removed tags
    assert all(k in s.structure for k in s.lyrics.keys())


def test_60s_keeps_intro_verse_chorus_outro():
    s = AudioSettings()
    s.duration = 60
    s.structure = ["Intro", "Verse 1", "Chorus", "Outro"]
    s.lyrics = {t: "" for t in s.structure}
    _enforce(s)
    assert s.structure == ["Intro", "Verse 1", "Chorus", "Outro"]


# ── 150s: bridge allowed, solo+final still dropped ────────────────


def test_150s_allows_bridge_drops_solo_and_final():
    s = AudioSettings()
    s.duration = 150
    s.structure = [
        "Intro", "Verse 1", "Pre-Chorus", "Chorus",
        "Verse 2", "Pre-Chorus", "Chorus", "Bridge",
        "Guitar Solo", "Final Chorus", "Outro",
    ]
    s.lyrics = {t: "" for t in s.structure}
    _enforce(s)
    assert any("bridge" in t.lower() for t in s.structure)
    assert not any(t.lower() == "guitar solo" for t in s.structure)
    assert not any("final chorus" in t.lower() for t in s.structure)
    assert len(s.structure) <= 9


# ── 180s: everything allowed up to 10 ─────────────────────────────


def test_180s_allows_all_extras():
    s = AudioSettings()
    s.duration = 180
    s.structure = [
        "Intro", "Verse 1", "Pre-Chorus", "Chorus",
        "Verse 2", "Pre-Chorus", "Chorus", "Bridge",
        "Guitar Solo", "Final Chorus", "Outro",
    ]
    s.lyrics = {t: "" for t in s.structure}
    _enforce(s)
    assert any("bridge" in t.lower() for t in s.structure)
    assert any("solo" in t.lower() for t in s.structure)
    assert len(s.structure) <= 10


# ── Edge: empty structure ──────────────────────────────────────────


def test_empty_structure_no_crash():
    s = AudioSettings()
    s.duration = 60
    s.structure = []
    s.lyrics = {}
    _enforce(s)
    assert s.structure == []


def test_under_budget_unchanged():
    s = AudioSettings()
    s.duration = 60
    s.structure = ["Intro", "Verse 1", "Chorus"]
    s.lyrics = {t: "" for t in s.structure}
    _enforce(s)
    assert s.structure == ["Intro", "Verse 1", "Chorus"]


# ── Coverage table consistency ─────────────────────────────────────


def test_budget_table_monotonic():
    """Section count must not decrease as duration grows."""
    prev = 0
    for lo, hi, n, *_ in DURATION_SECTION_BUDGET:
        assert n >= prev
        prev = n
