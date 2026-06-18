"""MSR (Multiple Subject Reference) integration: workflow JSON wiring,
comfyui.py injection helpers, msr_refs slot allocation + sheet composer,
and the background/prop spec passthrough in the segment planner."""

from __future__ import annotations

import json
import os

import pytest

from comfyui import (
    MSR_TITLE_TAG,
    has_msr_nodes,
    inject_resolution,
    inject_input_image,
    inject_msr_images,
    msr_frame_count,
    summarize_resolution_state,
)
from msr_refs import (
    MSR_VIEW_PROMPTS,
    allocate_msr_subjects,
    build_msr_reference_block,
    compose_character_sheet,
    derive_background_prompt,
)
import music_video_pipeline as mvp

_WF_DIR = os.path.join(os.path.dirname(__file__), "..", "Workflows")
_MSR_WF = os.path.join(_WF_DIR, "LTX2.3 - IA2V-PromptRelay-MSR.json")
_STD_WF = os.path.join(_WF_DIR, "LTX2.3 - IA2V-PromptRelay.json")
_FLUX2_T2I_WF = os.path.join(_WF_DIR, "Flux2 Klein 9B - T2I.json")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── workflow JSON wiring ─────────────────────────────────────────


def test_msr_workflow_detected_and_standard_not():
    assert has_msr_nodes(_load(_MSR_WF))
    assert not has_msr_nodes(_load(_STD_WF))


def test_msr_workflow_wiring():
    wf = _load(_MSR_WF)
    # MSR LoRA sits between PowerLoraLoader (#211) and the relay encode (#1700)
    assert wf["2007"]["class_type"] == "LTXICLoRALoaderModelOnly"
    assert wf["2007"]["inputs"]["model"] == ["211", 0]
    assert wf["1700"]["inputs"]["model"] == ["2007", 0]
    # guide node feeds pass-1 guider conditioning + the AV concat latent
    assert wf["2008"]["class_type"] == "LTXAddVideoICLoRAGuide"
    assert wf["2008"]["inputs"]["image"] == ["2006", 0]
    assert wf["759:1052"]["inputs"]["positive"] == ["2008", 0]
    assert wf["759:1052"]["inputs"]["negative"] == ["2008", 1]
    # Sheet guide-planting (#2009, LTXVAddGuideMulti) seeds the base latent with the
    # reference sheet at frame 0 — without it the prompt dominates and identity drifts.
    assert wf["2009"]["class_type"] == "LTXVAddGuideMulti"
    assert wf["2009"]["inputs"]["num_guides.image_1"] == ["2001", 0]
    assert wf["2009"]["inputs"]["num_guides.image_2"] == ["2005", 0]
    assert wf["759:1055"]["inputs"]["video_latent"] == ["2009", 2]
    # crop guides strips the reference frames using the guide conditioning,
    # and the pass-2 upsampler consumes the CROPPED latent
    assert wf["759:1074"]["inputs"]["positive"] == ["2008", 0]
    assert wf["759:1070"]["inputs"]["samples"] == ["759:1074", 2]
    # LiconMSR resolution follows the generation resolution chain
    assert wf["2006"]["inputs"]["width"] == ["759:1080", 1]
    assert wf["2006"]["inputs"]["height"] == ["759:1081", 1]
    # all five reference LoadImage slots exist and are tagged for patching
    for nid in ("2001", "2002", "2003", "2004", "2005"):
        assert wf[nid]["class_type"] == "LoadImage"
        assert MSR_TITLE_TAG in wf[nid]["_meta"]["title"]


def test_standard_workflow_untouched_by_msr_nodes():
    std = _load(_STD_WF)
    # MSR nodes (2001-2008) and 10s-Likeness Guide (3002) must not exist
    # in the vanilla GitHub upstream IA2V-PromptRelay workflow. The 10s
    # variant lives in a separate file (LTX2.3 - IA2V-PromptRelay-10sNodes.json).
    for nid in ("2001", "2006", "2007", "2008", "3002"):
        assert nid not in std
    # Original wiring: positive conditioning chains directly through 759:1067
    # (no 10s-Guide hop).
    assert std["759:1052"]["inputs"]["positive"] == ["759:1067", 0]
    assert std["1700"]["inputs"]["model"] == ["211", 0]


def test_flux2_t2i_resolution_injection_preserves_resolution_chain():
    wf = _load(_FLUX2_T2I_WF)
    out = inject_resolution(wf, "2:3", megapixels=2.5)

    assert out["134"]["inputs"]["megapixel"] == "2.5"
    assert out["134"]["inputs"]["custom_ratio"] is True
    assert out["134"]["inputs"]["custom_aspect_ratio"] == "2:3"
    assert out["101:218"]["inputs"]["width"] == ["101:68", 0]
    assert out["101:218"]["inputs"]["height"] == ["101:69", 0]
    assert out["101:216"]["inputs"]["width"] == ["101:68", 0]
    assert out["101:216"]["inputs"]["height"] == ["101:69", 0]


def test_flux2_t2i_resolution_summary_warns_on_literal_latent_size():
    wf = _load(_FLUX2_T2I_WF)
    out = inject_resolution(wf, "16:9", megapixels=2.5)
    assert "WARNING" not in summarize_resolution_state(out)

    out["101:218"]["inputs"]["width"] = 1024
    out["101:218"]["inputs"]["height"] = 576
    assert "WARNING EmptyFlux2LatentImage" in summarize_resolution_state(out)


# ── msr_frame_count ──────────────────────────────────────────────


@pytest.mark.parametrize("images,expected", [
    (1, 17), (2, 17), (3, 25), (4, 33), (5, 41), (6, 41),
])
def test_msr_frame_count_mapping(images, expected):
    assert msr_frame_count(images) == expected


# ── inject_msr_images ────────────────────────────────────────────


def test_inject_msr_images_single_subject_prunes_unused_slots():
    wf = _load(_MSR_WF)
    wf = inject_msr_images(wf, ["sheet.png"], "bg.png")
    assert wf["2001"]["inputs"]["image"] == "sheet.png"
    assert wf["2005"]["inputs"]["image"] == "bg.png"
    # unused subject slots removed: LiconMSR keys AND orphaned LoadImages
    licon = wf["2006"]["inputs"]
    assert "1" in licon and "background" in licon
    for slot, nid in (("2", "2002"), ("3", "2003"), ("4", "2004")):
        assert slot not in licon
        assert nid not in wf
    # MSR-Guide deckt jetzt immer die volle Referenzvideo-Länge ab (41), damit die
    # Identität über den ganzen Clip hält statt nach den ersten Frames zu driften.
    assert licon["frame_count"] == 41


def test_inject_msr_images_four_subjects_keeps_all_slots():
    wf = _load(_MSR_WF)
    wf = inject_msr_images(wf, ["a.png", "b.png", "c.png", "d.png"], "bg.png")
    licon = wf["2006"]["inputs"]
    for slot in ("1", "2", "3", "4", "background"):
        assert slot in licon
    assert licon["frame_count"] == 41  # 5 images -> 41 frames
    assert wf["2004"]["inputs"]["image"] == "d.png"


def test_inject_msr_images_validates_subject_count():
    wf = _load(_MSR_WF)
    with pytest.raises(ValueError):
        inject_msr_images(wf, [], "bg.png")
    with pytest.raises(ValueError):
        inject_msr_images(wf, ["1", "2", "3", "4", "5"], "bg.png")


def test_inject_msr_images_noop_on_standard_workflow():
    std = _load(_STD_WF)
    before = json.dumps(std, sort_keys=True)
    out = inject_msr_images(std, ["a.png"], "bg.png")
    assert json.dumps(out, sort_keys=True) == before


def test_inject_input_image_skips_msr_slots():
    # Stage Q: the MSR workflow is FRAMELESS — there is no non-MSR "First Frame"
    # LoadImage, so inject_input_image is a no-op here and must never write the
    # file into an MSR reference slot.
    wf = _load(_MSR_WF)
    assert not any(
        n.get("class_type") == "LoadImage"
        and "[[P:MSR]]" not in n.get("_meta", {}).get("title", "")
        for n in wf.values()
    ), "frameless MSR workflow must have no start-frame LoadImage"
    wf = inject_input_image(wf, "first_frame.png")
    for nid in ("2001", "2002", "2003", "2004", "2005"):
        assert wf[nid]["inputs"]["image"] != "first_frame.png"


# ── character sheet composer ─────────────────────────────────────


def test_compose_character_sheet_grid(tmp_path):
    from PIL import Image
    views = []
    for i in range(4):
        p = tmp_path / f"v{i}.png"
        Image.new("RGB", (300, 500), (i * 40, 100, 150)).save(p)
        views.append(str(p))
    out = compose_character_sheet(views, str(tmp_path / "sheet.png"),
                                  cell_w=256, cell_h=384, target_aspect=None)
    img = Image.open(out)
    # Stage Q: seamless 2x2, no border, 2:3 portrait cells
    # (target_aspect=None keeps the legacy un-padded layout; Stage MSR-2026-06
    # adds 16:9 padding by default — covered by tests/test_mv_prompt_hygiene.py.)
    assert img.size == (2 * 256, 2 * 384)
    assert img.mode == "RGB"


def test_compose_character_sheet_requires_views(tmp_path):
    with pytest.raises(ValueError):
        compose_character_sheet([], str(tmp_path / "x.png"))


# ── slot allocation ──────────────────────────────────────────────


def _refs(**kwargs):
    return {k: v for k, v in kwargs.items()}


def test_allocate_solo_sheet_with_props():
    paths, descs = allocate_msr_subjects(
        "female",
        _refs(female=["sheet_f.png"], male=["sheet_m.png"]),
        {"female": "the female singer", "male": "the male singer"},
        prop_refs=[("prop.png", "a silver pendant")],
    )
    assert paths == ["sheet_f.png", "prop.png"]
    assert descs[0].startswith("the female singer")
    assert descs[1] == "a silver pendant"


def test_allocate_duet_mixed_uses_both_characters():
    paths, _ = allocate_msr_subjects(
        "duet",
        _refs(male=["sheet_m.png"], female=["sheet_f.png"]),
        {"male": "m", "female": "f"},
    )
    assert paths == ["sheet_m.png", "sheet_f.png"]


def test_allocate_duet_same_gender_uses_partner():
    paths, _ = allocate_msr_subjects(
        "duet",
        _refs(female=["sheet_f.png"], female2=["sheet_f2.png"]),
        {"female": "f", "female2": "f2"},
    )
    assert paths == ["sheet_f.png", "sheet_f2.png"]


def test_allocate_duet_views_overflow_reduces_to_front():
    paths, _ = allocate_msr_subjects(
        "duet",
        _refs(male=["m1", "m2", "m3", "m4"], female=["f1", "f2", "f3", "f4"]),
        {"male": "m", "female": "f"},
    )
    assert paths == ["m1", "f1"]


def test_allocate_story_without_explicit_role_uses_no_character_refs():
    paths, _ = allocate_msr_subjects(
        None, _refs(lead=["sheet.png"]), {"lead": "the lead performer"},
    )
    assert paths == []


def test_allocate_story_without_role_can_still_use_props():
    paths, descs = allocate_msr_subjects(
        None,
        _refs(lead=["sheet.png"]),
        {"lead": "the lead performer"},
        prop_refs=[("prop.png", "a wooden guitar")],
    )
    assert paths == ["prop.png"]
    assert descs == ["a wooden guitar"]


def test_allocate_caps_at_four_subjects():
    paths, descs = allocate_msr_subjects(
        "male",
        _refs(male=["v1", "v2", "v3", "v4"]),
        {"male": "m"},
        prop_refs=[("p1", "prop one"), ("p2", "prop two")],
    )
    assert len(paths) == 4
    assert paths == ["v1", "v2", "v3", "v4"]  # props dropped, no slot left


# ── reference block + background fallback ────────────────────────


def test_build_msr_reference_block():
    block = build_msr_reference_block(
        ["the female singer (character turnaround sheet: front, back, side views and face close-up)",
         "a silver pendant"],
        "neon-lit alley at night",
    )
    assert not block.startswith("References:")
    assert "Use the first subject reference as" in block
    assert "Use the second subject reference as" in block
    assert "Use the background reference as the real location" in block
    assert "[1]" not in block
    assert "Background reference:" not in block
    assert block.endswith("natural contact shadows and matching light.")
    assert build_msr_reference_block([], "x") == ""


def test_derive_background_prompt():
    # Stage O1: segment text must NEVER be copied into the background prompt
    # (a character description overrides the "no people" prefix and Flux
    # renders a person into the MSR background slot — job ffd3b9a6).
    seg = (
        "Close-up of a female singer with bold black tribal tattoos, "
        "wearing a neon-pink oversized hoodie, on a rooftop at night"
    )
    out = derive_background_prompt(seg)
    assert "no people" in out
    assert "rooftop" in out  # scene hint survives as a neutral phrase
    for leak in ("singer", "tattoo", "hoodie", "wearing", "female", "close-up"):
        assert leak not in out.lower()
    assert derive_background_prompt("")  # non-empty neutral fallback


def test_background_prompt_people_guard():
    # Stage O1 defense-in-depth: detect person/wardrobe vocabulary so the
    # render loop can swap in the neutral fallback before the T2I call.
    from msr_refs import background_prompt_mentions_people

    assert background_prompt_mentions_people(
        "a female singer wearing a pink hoodie in an alley"
    )
    assert background_prompt_mentions_people("portrait of a man at night")
    # negated exclusions must NOT trip the guard
    assert not background_prompt_mentions_people(
        "empty neon-lit alley at night, no people, no characters, no faces"
    )
    assert not background_prompt_mentions_people(
        "rain-slick rooftop overlooking the city skyline at dusk"
    )
    assert not background_prompt_mentions_people("")


def test_compose_character_sheet_four_views(tmp_path):
    # Stage Q: portrait + back + side + face-front = 4 cells -> seamless 2x2
    from PIL import Image

    paths = []
    for i in range(4):
        p = tmp_path / f"v{i}.png"
        Image.new("RGB", (200, 300), (i * 40, 10, 10)).save(p)
        paths.append(str(p))
    out = compose_character_sheet(paths, str(tmp_path / "sheet.png"),
                                  cell_w=128, cell_h=192, target_aspect=None)
    with Image.open(out) as sheet:
        assert sheet.size == (2 * 128, 2 * 192)


def test_msr_view_prompts_closed_mouth():
    # Stage M: every reference still must demand a closed mouth
    # Stage Q: 3 MCA views (back, side, face-front); portrait is the 4th cell
    assert len(MSR_VIEW_PROMPTS) == 3
    for p in MSR_VIEW_PROMPTS:
        assert "mouth closed" in p


# ── planner spec passthrough ─────────────────────────────────────


def test_parse_segment_plan_passes_background_and_prop():
    resp = json.dumps([{
        "label": "Chorus - female",
        "video_prompt": "vp",
        "frame_variant_prompt": "fvp",
        "lyrics": "la la",
        "background_prompt": "neon-lit alley at night",
        "prop_prompt": "a silver pendant on black velvet",
    }])
    specs = mvp.parse_segment_plan(resp)
    assert specs[0]["background_prompt"] == "neon-lit alley at night"
    assert specs[0]["prop_prompt"] == "a silver pendant on black velvet"
    # omitted fields default to empty strings
    specs2 = mvp.parse_segment_plan(json.dumps([{"video_prompt": "vp"}]))
    assert specs2[0]["background_prompt"] == ""
    assert specs2[0]["prop_prompt"] == ""


def test_segment_carries_msr_fields():
    seg = mvp.Segment(index=0, start_time=0.0, end_time=5.0,
                      background_prompt="bg", prop_prompt="prop")
    d = seg.to_dict()
    assert d["background_prompt"] == "bg"
    assert d["prop_prompt"] == "prop"
    # merged segments keep the anchor's background
    merged = mvp._build_merged_segment([seg, mvp.Segment(index=1, start_time=5.0, end_time=9.0)])
    assert merged.background_prompt == "bg"


def test_msr_assets_built_before_segment_frames():
    # Stage O2: the MSR reference stage (character sheet + backgrounds) must
    # run BEFORE the per-segment MCA frame stage — in job ffd3b9a6 segment
    # frames and even first renders existed before the sheet was composed.
    with open(os.path.join(os.path.dirname(__file__), "..", "api.py"),
              encoding="utf-8") as f:
        src = f.read()
    pos_msr = src.index("# 3b. MSR (Multiple Subject Reference)")
    pos_frames = src.index("# RC8 chorus reuse: only generate MCA frames")
    assert pos_msr < pos_frames, (
        "MSR reference assets (3b) must be built before per-segment "
        "MCA frames (3c)"
    )


def test_msr_solo_generative_path_uses_role_portrait_before_legacy_source():
    with open(os.path.join(os.path.dirname(__file__), "..", "api.py"),
              encoding="utf-8") as f:
        src = f.read()
    block = src[src.index("use_msr_solo_portrait ="):src.index("# 3b. MSR")]

    assert 'source_mode in ("auto", "describe")' in block
    assert "portraits[solo_role] = await _resolve_singer_portrait" in block
    assert block.index("_resolve_singer_portrait") < block.index("_resolve_source_image")


def test_aligned_system_prompt_mentions_msr_fields():
    import inspect
    src = inspect.getsource(mvp)
    # Stage O1: background_prompt is REQUIRED (missing field caused the
    # character-bleed fallback in job ffd3b9a6); prop_prompt stays optional.
    assert "REQUIRED FIELD background_prompt" in src
    assert "SCENE CONTRACT (graded)" in src
    assert "fixed-duration blocks" in src
    assert "OPTIONAL FIELD prop_prompt" in src
