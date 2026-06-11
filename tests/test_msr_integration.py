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
    inject_input_image,
    inject_msr_images,
    msr_frame_count,
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
    assert wf["759:1055"]["inputs"]["video_latent"] == ["2008", 2]
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
    for nid in ("2001", "2006", "2007", "2008"):
        assert nid not in std
    # baseline wiring still original
    assert std["759:1052"]["inputs"]["positive"] == ["759:1067", 0]
    assert std["1700"]["inputs"]["model"] == ["211", 0]


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
    # 1 subject + background = 2 images -> 17 frames
    assert licon["frame_count"] == 17


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
    wf = _load(_MSR_WF)
    wf = inject_input_image(wf, "first_frame.png")
    # the IA2V first-frame node gets the file, MSR slots stay untouched
    assert wf["149"]["inputs"]["image"] == "first_frame.png"
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
    out = compose_character_sheet(views, str(tmp_path / "sheet.png"), cell_size=256, border=8)
    img = Image.open(out)
    # 2x2 grid: 2*256 + 3*8 per axis
    assert img.size == (2 * 256 + 3 * 8, 2 * 256 + 3 * 8)
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


def test_allocate_story_falls_back_to_first_character():
    paths, _ = allocate_msr_subjects(
        None, _refs(lead=["sheet.png"]), {"lead": "the lead performer"},
    )
    assert paths == ["sheet.png"]


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
    assert block.startswith("References:")
    assert "[1] the female singer" in block
    assert "[2] a silver pendant." in block
    assert block.endswith("Background reference: neon-lit alley at night.")
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


def test_compose_character_sheet_five_views(tmp_path):
    # Stage O3: full-body front (portrait) + 4 MCA views = 5 cells -> 3x2 grid
    from PIL import Image

    paths = []
    for i in range(5):
        p = tmp_path / f"v{i}.png"
        Image.new("RGB", (200, 300), (i * 40, 10, 10)).save(p)
        paths.append(str(p))
    out = compose_character_sheet(paths, str(tmp_path / "sheet.png"),
                                  cell_size=128, border=8)
    with Image.open(out) as sheet:
        assert sheet.size == (3 * 128 + 4 * 8, 2 * 128 + 3 * 8)


def test_msr_view_prompts_closed_mouth():
    # Stage M: every reference still must demand a closed mouth
    assert len(MSR_VIEW_PROMPTS) == 4
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


def test_aligned_system_prompt_mentions_msr_fields():
    import inspect
    src = inspect.getsource(mvp)
    # Stage O1: background_prompt is REQUIRED (missing field caused the
    # character-bleed fallback in job ffd3b9a6); prop_prompt stays optional.
    assert "REQUIRED FIELD background_prompt" in src
    assert "OPTIONAL FIELD prop_prompt" in src
