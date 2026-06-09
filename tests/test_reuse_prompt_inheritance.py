"""Stage L7: reuse rows inherit the anchor's prompts (in-process unit test).

Smoke check on the inheritance pattern itself — the integration test for the
full plan_segments() flow already lives in test_mv_director_integration.py.
"""

from __future__ import annotations

import pytest


def _apply_inheritance(rows: list[dict], specs: list[dict]) -> list[dict]:
    """Local reimplementation of the inheritance block in plan_segments()
    so the rule is asserted independently of the surrounding async pipeline."""
    for i, r in enumerate(rows):
        anchor = r.get("reuse_of")
        if anchor is not None and 0 <= anchor < len(specs):
            for fld in ("video_prompt", "frame_variant_prompt", "video_prompt_relay"):
                if fld in specs[anchor]:
                    specs[i][fld] = specs[anchor][fld]
    return specs


def test_reuse_row_inherits_anchor_prompts():
    rows = [
        {"label": "Intro", "reuse_of": None},
        {"label": "Verse 1 - male", "reuse_of": None},
        {"label": "Chorus - male", "reuse_of": None},  # anchor
        {"label": "Verse 2 - male", "reuse_of": None},
        {"label": "Chorus - male", "reuse_of": 2},     # repeat of idx 2
    ]
    specs = [
        {"video_prompt": "intro_p", "frame_variant_prompt": "intro_fp"},
        {"video_prompt": "v1_p", "frame_variant_prompt": "v1_fp"},
        {"video_prompt": "chorus_anchor_p", "frame_variant_prompt": "chorus_anchor_fp",
         "video_prompt_relay": {"global": "g", "beats": ["b1"]}},
        {"video_prompt": "v2_p", "frame_variant_prompt": "v2_fp"},
        {"video_prompt": "chorus2_diverged_p", "frame_variant_prompt": "chorus2_diverged_fp"},
    ]
    out = _apply_inheritance(rows, specs)
    assert out[4]["video_prompt"] == "chorus_anchor_p"
    assert out[4]["frame_variant_prompt"] == "chorus_anchor_fp"
    assert out[4]["video_prompt_relay"] == {"global": "g", "beats": ["b1"]}
    # Anchor itself unchanged
    assert out[2]["video_prompt"] == "chorus_anchor_p"


def test_anchor_rows_untouched():
    rows = [
        {"label": "Intro", "reuse_of": None},
        {"label": "Chorus", "reuse_of": None},
    ]
    specs = [
        {"video_prompt": "a"}, {"video_prompt": "b"},
    ]
    out = _apply_inheritance(rows, specs)
    assert out[0]["video_prompt"] == "a"
    assert out[1]["video_prompt"] == "b"


def test_invalid_anchor_index_skipped():
    rows = [{"label": "Chorus", "reuse_of": 99}]
    specs = [{"video_prompt": "x"}]
    out = _apply_inheritance(rows, specs)
    assert out[0]["video_prompt"] == "x"


def test_missing_field_on_anchor_not_overwritten():
    rows = [
        {"label": "Anchor", "reuse_of": None},
        {"label": "Repeat", "reuse_of": 0},
    ]
    specs = [
        {"video_prompt": "anchor"},
        {"video_prompt": "repeat_orig", "frame_variant_prompt": "repeat_fp_orig"},
    ]
    out = _apply_inheritance(rows, specs)
    assert out[1]["video_prompt"] == "anchor"
    # frame_variant_prompt missing on anchor → repeat keeps its own
    assert out[1]["frame_variant_prompt"] == "repeat_fp_orig"
