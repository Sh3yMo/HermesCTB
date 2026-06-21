"""Iteration 2: model-aware batched prompt formatting (Flux2 stills + LTX video)."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

from mv_prompt_hygiene import normalize_flux2_text, normalize_ltx_text
import music_video_pipeline as mvp


def test_normalize_ltx_strips_scene_lock_meta():
    out = normalize_ltx_text(
        "Scene location must be exactly: a neon alley at night. A figure walks."
    )
    assert "must be exactly" not in out.lower()
    assert ":" not in out
    assert "neon alley" in out.lower()  # scene description kept


def test_normalize_flux2_strips_ltx_artifacts():
    raw = (
        "Scene location must be exactly: a neon alley. A lone figure stands. "
        "No movement Render this video frame with this overall visual medium and "
        "color grade. photorealistic still. The overall video color grade and "
        "cinematography use palette tones crimson red velvet, sepia bleach bypass. "
        "Lighting is chiaroscuro_caravaggio."
    )
    out = normalize_flux2_text(raw)
    low = out.lower()
    assert ":" not in out
    assert "render this video frame" not in low
    assert "color grade" not in low
    assert "no movement" not in low
    assert "scene location must be exactly" not in low
    assert "chiaroscuro caravaggio" in low  # underscore -> space
    assert "neon alley" in low  # content preserved


def _prompter():
    return mvp.MusicVideoPrompter({})


def test_polish_segment_prompts_failsoft_sets_baseline(monkeypatch):
    p = _prompter()

    async def _boom(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr(p, "_call_openrouter", _boom)
    segs = [
        SimpleNamespace(
            index=0,
            prompt="Scene location must be exactly: a rain-slick street.",
            frame_variant_prompt=(
                "A singer in a coat. Render this video frame with this overall "
                "visual medium and color grade. cinematic still."
            ),
            ltx_video_prompt="",
            flux2_frame_prompt="",
        ),
    ]
    asyncio.run(p.polish_segment_prompts(segs))
    s = segs[0]
    # deterministic baseline populated despite LLM failure
    assert s.ltx_video_prompt and ":" not in s.ltx_video_prompt
    assert s.flux2_frame_prompt
    assert "render this video frame" not in s.flux2_frame_prompt.lower()
    assert "rain-slick street" in s.ltx_video_prompt.lower()


def test_prompts_forced_to_english():
    # Bug B: wardrobe contract + batched formatter must force English output so
    # a German brief does not leak German garment names into the prompt.
    with open(os.path.join(os.path.dirname(__file__), "..", "music_video_pipeline.py"),
              encoding="utf-8") as f:
        src = f.read()
    wc = src[src.index("def generate_wardrobe_contract"):src.index("def generate_wardrobe_contract") + 2500]
    assert "respond in English" in wc.lower() or "in english" in wc.lower()
    ps = src[src.index("async def polish_segment_prompts"):]
    ps = ps[:ps.index("async def extract_scene_anchor")]
    assert "translate any non-english" in ps.lower()


def test_api_uses_model_formatted_prompts():
    with open(os.path.join(os.path.dirname(__file__), "..", "api.py"), encoding="utf-8") as f:
        src = f.read()
    assert "await MV_PROMPTER.polish_segment_prompts(segments)" in src
    assert "seg_prompt = seg.ltx_video_prompt or seg.prompt" in src
    assert "ff_still = seg.flux2_frame_prompt or (" in src
    assert "lf_still = seg.flux2_frame_prompt or (" in src
