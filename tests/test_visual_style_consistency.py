"""Stage P1 — producer visual medium applied consistently to portrait + MSR grid.

Root cause of the test-run style war: the producer profile's art medium (Cole
Bennett = cell-shaded cartoon) leaked only into the LLM segment/background
prompts while the portrait + reference grid stayed photoreal. These tests lock
in the two helpers that close the gap:

  - MVDirector.visual_style_descriptor(profile) -> render-ready medium string
  - msr_refs.build_view_prompts(style, appearance) -> views pinned to one look
"""

from __future__ import annotations

import json

from mv_director import DEFAULT_PROFILES_PATH, MVDirector
from msr_refs import MSR_VIEW_PROMPTS, build_view_prompts


def _profile_by_id(pid: str) -> dict:
    data = json.loads(DEFAULT_PROFILES_PATH.read_text(encoding="utf-8"))
    for p in data.get("producers", []):
        if p.get("id") == pid:
            return p
    raise AssertionError(f"profile {pid!r} not found")


# ── visual_style_descriptor ─────────────────────────────────────────


def test_cole_bennett_descriptor_is_cel_shaded() -> None:
    """The cartoon/cell-shaded ethos must yield a stylized (non-photoreal) medium."""
    d = MVDirector()
    desc = d.visual_style_descriptor(_profile_by_id("cole_bennett")).lower()
    assert "cel-shaded" in desc or "cartoon" in desc
    assert "photorealistic" not in desc
    # palette + grain tail still present
    assert "color palette" in desc


def test_non_stylized_profile_defaults_photoreal() -> None:
    d = MVDirector()
    fake = {
        "ethos_oneliner": "Gritty handheld realism, naturalistic available light.",
        "signature_shots": ["handheld_MS", "available_light_CU"],
        "color_palette": ["teal", "amber"],
        "film_grain": True,
    }
    desc = d.visual_style_descriptor(fake).lower()
    assert "photorealistic cinematic still" in desc
    assert "subtle film grain" in desc  # film_grain=True


def test_explicit_visual_medium_overrides_inference() -> None:
    d = MVDirector()
    fake = {
        "visual_medium": "claymation stop-motion",
        "ethos_oneliner": "cell-shaded cartoon",  # would otherwise infer cel-shaded
        "color_palette": [],
        "film_grain": False,
    }
    desc = d.visual_style_descriptor(fake)
    assert desc.startswith("claymation stop-motion")
    assert "cel-shaded" not in desc.lower()


# ── build_view_prompts ──────────────────────────────────────────────


def test_empty_args_match_static_prompts() -> None:
    """Back-compat: no style/appearance -> identical to the static constant."""
    assert build_view_prompts() == MSR_VIEW_PROMPTS


def test_injects_appearance_and_style_into_every_view() -> None:
    style = "bold cel-shaded 2D cartoon illustration style"
    appearance = "dark-skinned woman, long straight black hair, gold two-piece bikini, white heels"
    views = build_view_prompts(style_descriptor=style, appearance_desc=appearance)
    assert len(views) == len(MSR_VIEW_PROMPTS) == 4
    for v in views:
        assert "gold two-piece bikini" in v   # garment carried into every angle
        assert "cel-shaded" in v               # producer medium carried too
    # The 4 distinct view framings are preserved.
    assert "seen from behind" in views[0]
    assert "strict side profile" in views[1]
    assert "facing the camera" in views[2]
