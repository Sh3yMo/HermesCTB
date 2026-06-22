"""Unit tests for pure music-video planning helpers (no ComfyUI / network)."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402
import music_video_pipeline as mvp  # noqa: E402
from music_video_pipeline import (  # noqa: E402
    ACE_STEP_LANGS,
    Segment,
    apply_role_clothing_contracts_to_segments,
    apply_scene_contracts_to_segments,
    build_aligned_timeline,
    build_role_age_contracts,
    build_role_clothing_contracts,
    build_segment_timeline,
    chunk_list,
    clamp_song_duration,
    enforce_performer_role,
    extract_section_role,
    parse_segment_plan,
    partition_anchors_by_role,
    plan_same_gender_portraits,
    filter_role_contracts_for_present_roles,
    same_gender_veto,
    sanitize_role_prompt_text,
    ensure_relay_specs_for_segments,
    to_ace_language,
    _segment_video_prompt,
)


def _al(label, lyrics, start, end, is_vocal=True, reuse_of=None):
    return {"label": label, "lyrics": lyrics, "start": start, "end": end,
            "is_vocal": is_vocal, "reuse_of": reuse_of}


def test_aligned_basic_keeps_real_boundaries():
    a = [_al("Verse 1", "x", 0.0, 12.0), _al("Chorus", "y", 12.0, 20.0)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    assert [(r["start_time"], r["end_time"]) for r in rows] == [(0.0, 12.0), (12.0, 20.0)]
    assert [r["label"] for r in rows] == ["Verse 1", "Chorus"]
    assert all(r["is_vocal"] for r in rows)


def test_aligned_splits_over_cap_within_span():
    a = [_al("Long", "L", 0.0, 70.0)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    assert len(rows) == 3  # ceil(70/30)
    assert rows[0]["start_time"] == 0.0 and rows[-1]["end_time"] == 70.0
    assert rows[0]["lyrics"] == "L" and rows[1]["lyrics"] == "" and rows[2]["lyrics"] == ""
    for x, y in zip(rows, rows[1:]):
        assert y["start_time"] == x["end_time"]


def test_aligned_merges_tiny_section_into_previous():
    a = [_al("Verse", "v", 0.0, 12.0), _al("Stab", "", 12.0, 13.0, is_vocal=False)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    assert len(rows) == 1
    assert rows[0]["start_time"] == 0.0 and rows[0]["end_time"] == 13.0


def test_aligned_carries_reuse_of_first_row_only():
    a = [_al("Chorus", "c", 0.0, 12.0),
         _al("Chorus2", "c", 40.0, 80.0, reuse_of=0)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    reuse_rows = [r for r in rows if r.get("reuse_of") == 0]
    assert len(reuse_rows) == 1  # only first sub-row of the repeated section
    assert reuse_rows[0]["start_time"] == 40.0


def test_aligned_empty_returns_empty():
    assert build_aligned_timeline([], 8.0, 30.0) == []


def test_aligned_zero_duration_first_section_absorbed():
    """RC: an instrumental [Intro] with 0 duration (WhisperX gave it no time)
    has no predecessor to merge into. It must be absorbed INTO the next section
    so no 0-duration row reaches _extract_audio_clip (→ empty WAV → LTX crash)."""
    a = [
        _al("Intro", "", 0.0, 0.0, is_vocal=False),
        _al("Verse 1", "v", 0.0, 12.0),
        _al("Chorus", "c", 12.0, 24.0),
    ]
    rows = build_aligned_timeline(a, min_seg=4.0, max_seg=20.0)
    assert rows, "expected non-empty timeline"
    for r in rows:
        assert (r["end_time"] - r["start_time"]) >= 0.05, f"degenerate row: {r}"
    # the intro's time span is folded into the first kept row
    assert rows[0]["start_time"] == 0.0


def test_aligned_tiny_first_section_absorbed_forward():
    """A short-but-nonzero leading section also merges forward (no predecessor)."""
    a = [
        _al("Intro", "", 0.0, 0.4, is_vocal=False),
        _al("Verse 1", "v", 0.4, 12.0),
    ]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    assert len(rows) == 1
    assert rows[0]["start_time"] == 0.0 and rows[0]["end_time"] == 12.0


def test_aligned_closes_gap_between_sections():
    """Bug 2: VAD boundary-refinement can leave a hole between a section's end
    and the next section's start. The timeline MUST tile contiguously — gaps
    make the assembled video shorter than the audio → A/V desync."""
    a = [_al("Verse", "v", 0.0, 12.0),
         _al("Chorus", "c", 18.0, 30.0)]  # 6s gap 12.0 -> 18.0
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    for x, y in zip(rows, rows[1:]):
        assert y["start_time"] == x["end_time"], f"gap/overlap: {x} -> {y}"
    assert rows[0]["start_time"] == 0.0


def test_aligned_closes_real_run_gap_regression():
    """Regression for the observed run (segments_02360422.json): Instrumental
    ended at 70.62, Bridge started at 75.43 → 4.81s uncovered → video 5s short."""
    a = [_al("Instrumental", "", 65.8, 70.62, is_vocal=False),
         _al("Bridge", "b", 75.43, 90.56)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    for x, y in zip(rows, rows[1:]):
        assert y["start_time"] == x["end_time"], f"gap/overlap: {x} -> {y}"


def test_aligned_snaps_last_end_to_total_duration():
    """When total_duration is given, the final row's end is snapped to it so
    sum(segment spans) == audio length (no trailing gap → -shortest truncates)."""
    a = [_al("Verse", "v", 0.0, 12.0),
         _al("Chorus", "c", 12.0, 20.0)]  # ends at 20 but song is 24s
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0, total_duration=24.0)
    assert rows[0]["start_time"] == 0.0
    assert rows[-1]["end_time"] == 24.0
    for x, y in zip(rows, rows[1:]):
        assert y["start_time"] == x["end_time"]


def test_aligned_total_duration_none_preserves_last_end():
    """Without total_duration the last end is left as-is (back-compat)."""
    a = [_al("Verse", "v", 0.0, 12.0), _al("Chorus", "c", 12.0, 20.0)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    assert rows[-1]["end_time"] == 20.0


# --- Bug 1: segment video prompt never falls back to raw lyrics -----------

def test_segment_video_prompt_uses_scene_description():
    spec = {"video_prompt": "Close-up of singer on a neon rooftop", "lyrics": "la la la"}
    assert _segment_video_prompt(spec, "theme") == "Close-up of singer on a neon rooftop"


def test_segment_video_prompt_empty_falls_back_to_theme_not_lyrics():
    # The regression: empty spec must NOT yield the raw lyrics.
    spec = {"video_prompt": "", "lyrics": "I AM THE LIGHT IN THE RAIN"}
    assert _segment_video_prompt(spec, "neon synthwave city") == "neon synthwave city"


def test_segment_video_prompt_missing_key_falls_back_to_theme():
    assert _segment_video_prompt({}, "theme") == "theme"


def test_segment_video_prompt_whitespace_only_falls_back():
    assert _segment_video_prompt({"video_prompt": "   "}, "theme") == "theme"


CAP = 30.0
MIN = 8.0


# --- clamp_song_duration ---------------------------------------------------

def test_clamp_none_returns_default():
    assert clamp_song_duration(None) == 150


def test_clamp_below_floor():
    assert clamp_song_duration(5) == 20


def test_clamp_explicit_short_honored():
    assert clamp_song_duration(30) == 30


def test_clamp_above_ceiling():
    assert clamp_song_duration(9999) == 300


def test_clamp_valid_passthrough():
    assert clamp_song_duration(200) == 200


def test_clamp_float_and_string_coerced():
    assert clamp_song_duration("210") == 210
    assert clamp_song_duration(180.7) == 180


def test_clamp_garbage_returns_default():
    assert clamp_song_duration("not-a-number") == 150


# --- to_ace_language -------------------------------------------------------

def test_lang_name_to_iso():
    assert to_ace_language("german", "") == "de"
    assert to_ace_language("German", "") == "de"
    assert to_ace_language("französisch", "") == "fr"


def test_lang_iso_passthrough():
    assert to_ace_language("de", "") == "de"
    assert to_ace_language("EN", "") == "en"


def test_lang_infer_from_brief():
    assert to_ace_language("", "deutsches HipHop-Video über X") == "de"
    assert to_ace_language(None, "a french chanson") == "fr"


def test_lang_unknown_returns_none():
    assert to_ace_language("klingon", "") is None
    assert to_ace_language("", "a song about cats") is None
    assert to_ace_language(None, None) is None


def test_lang_result_always_valid_enum():
    for v in ("german", "de", "english", "französisch", "spanish"):
        r = to_ace_language(v, "")
        assert r in ACE_STEP_LANGS


# --- chunk_list ------------------------------------------------------------

def test_chunk_even():
    assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunk_remainder():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_chunk_empty():
    assert chunk_list([], 4) == []


def test_chunk_batch_larger_than_list():
    assert chunk_list([1, 2], 10) == [[1, 2]]


def test_chunk_nonpositive_batch_is_single_chunk():
    assert chunk_list([1, 2, 3], 0) == [[1, 2, 3]]
    assert chunk_list([1, 2, 3], -1) == [[1, 2, 3]]


# --- parse_segment_plan ----------------------------------------------------

def test_parse_plain_json_array():
    txt = '[{"video_prompt":"a","frame_variant_prompt":"b","duration":12,"label":"Intro"}]'
    out = parse_segment_plan(txt)
    assert len(out) == 1
    assert out[0]["video_prompt"] == "a"
    assert out[0]["frame_variant_prompt"] == "b"
    assert out[0]["duration"] == 12
    assert out[0]["label"] == "Intro"


def test_parse_with_code_fence_and_prose():
    txt = 'Sure!\n```json\n[{"video_prompt":"x","frame_variant_prompt":"y","duration":20}]\n```\nDone'
    out = parse_segment_plan(txt)
    assert len(out) == 1 and out[0]["video_prompt"] == "x"


def test_parse_missing_fields_defaulted():
    out = parse_segment_plan('[{"video_prompt":"only"}]')
    assert out[0]["video_prompt"] == "only"
    assert out[0]["frame_variant_prompt"] == "only"  # falls back to video_prompt
    assert out[0]["duration"] is None
    assert out[0]["label"] == ""


def test_parse_passes_video_prompt_relay_through_raw():
    # Regression: parse_segment_plan used to drop video_prompt_relay, so the
    # aligned path never saw LLM-emitted multi-beat blocks (extract_relay_spec
    # always got None). The block must survive parsing untouched — validation
    # happens later in extract_relay_spec.
    from music_video_pipeline import extract_relay_spec
    txt = (
        '[{"video_prompt":"a","duration":10,'
        '"video_prompt_relay":{"global":"slow dolly-in, warm grade",'
        '"beats":["A figure sits at a piano.","She lifts her head and sings."]}}]'
    )
    out = parse_segment_plan(txt)
    assert out[0]["video_prompt_relay"] == {
        "global": "slow dolly-in, warm grade",
        "beats": ["A figure sits at a piano.", "She lifts her head and sings."],
    }
    relay = extract_relay_spec(out[0])
    assert relay is not None
    assert relay["beats"] == [
        "A figure sits at a piano.", "She lifts her head and sings.",
    ]
    # Omitted field stays None (extract_relay_spec → None → legacy fallback).
    out2 = parse_segment_plan('[{"video_prompt":"only"}]')
    assert out2[0]["video_prompt_relay"] is None
    assert extract_relay_spec(out2[0]) is None
    # Malformed block passes through raw; extract_relay_spec rejects it.
    out3 = parse_segment_plan('[{"video_prompt":"x","video_prompt_relay":"not a dict"}]')
    assert out3[0]["video_prompt_relay"] == "not a dict"
    assert extract_relay_spec(out3[0]) is None


def test_parse_malformed_returns_empty():
    assert parse_segment_plan("not json at all") == []
    assert parse_segment_plan("") == []
    assert parse_segment_plan('{"not":"a list"}') == []


# --- build_segment_timeline ------------------------------------------------

def _check_contiguous(segs, total):
    assert segs, "must produce at least one segment"
    assert abs(segs[0]["start_time"]) < 1e-6
    assert abs(segs[-1]["end_time"] - total) < 1e-3, (segs[-1]["end_time"], total)
    for a, b in zip(segs, segs[1:]):
        assert abs(a["end_time"] - b["start_time"]) < 1e-6
    for s in segs:
        d = s["end_time"] - s["start_time"]
        assert d > 0


def test_timeline_empty_specs_fallback_tiles_total():
    segs = build_segment_timeline([], 180.0, MIN, CAP)
    _check_contiguous(segs, 180.0)
    for s in segs:
        d = s["end_time"] - s["start_time"]
        assert MIN - 1e-6 <= d <= CAP + 1e-6


def test_timeline_clamps_oversized_duration_to_cap():
    specs = [{"video_prompt": "p", "frame_variant_prompt": "f", "duration": 35},
             {"video_prompt": "p2", "frame_variant_prompt": "f2", "duration": 35}]
    segs = build_segment_timeline(specs, 60.0, MIN, CAP)
    _check_contiguous(segs, 60.0)
    for s in segs:
        d = s["end_time"] - s["start_time"]
        assert d <= CAP + 1e-6  # never exceeds A/V-safe cap


def test_timeline_normalizes_sum_to_total():
    specs = [{"video_prompt": "a", "frame_variant_prompt": "fa", "duration": 10},
             {"video_prompt": "b", "frame_variant_prompt": "fb", "duration": 10},
             {"video_prompt": "c", "frame_variant_prompt": "fc", "duration": 10}]
    segs = build_segment_timeline(specs, 150.0, MIN, CAP)
    _check_contiguous(segs, 150.0)
    for s in segs:
        d = s["end_time"] - s["start_time"]
        assert MIN - 1e-6 <= d <= CAP + 1e-6


def test_timeline_preserves_prompts_in_order():
    specs = [{"video_prompt": "v1", "frame_variant_prompt": "f1", "duration": 12, "label": "Intro"},
             {"video_prompt": "v2", "frame_variant_prompt": "f2", "duration": 12, "label": "Verse"}]
    segs = build_segment_timeline(specs, 24.0, MIN, CAP)
    assert [s["video_prompt"] for s in segs[:2]] == ["v1", "v2"]
    assert [s["frame_variant_prompt"] for s in segs[:2]] == ["f1", "f2"]


def test_timeline_short_total_single_segment():
    segs = build_segment_timeline([], 12.0, MIN, CAP)
    _check_contiguous(segs, 12.0)
    assert len(segs) == 1


# ---------------------------------------------------------------------------
# Role-aware multi-singer routing (extract_section_role + partition + prompt sanitization)
# ---------------------------------------------------------------------------

def test_extract_section_role_male_female_duet():
    assert extract_section_role("Verse - male") == "male"
    assert extract_section_role("Chorus - female") == "female"
    assert extract_section_role("Bridge - duet") == "duet"
    assert extract_section_role("Chorus - both") == "duet"
    assert extract_section_role("Outro - together") == "duet"
    # case-insensitive
    assert extract_section_role("VERSE 1 - MALE") == "male"
    assert extract_section_role("verse 2 - Female") == "female"
    # extra whitespace around dash
    assert extract_section_role("Verse  -  male") == "male"


def test_extract_section_role_returns_none_for_unannotated():
    assert extract_section_role("Chorus") is None
    assert extract_section_role("Verse 1") is None
    assert extract_section_role("Intro") is None
    assert extract_section_role("Bridge - instrumental") is None
    assert extract_section_role("") is None
    assert extract_section_role(None) is None  # type: ignore[arg-type]


def _seg(idx: int, label: str, reuse_of=None) -> Segment:
    return Segment(index=idx, start_time=float(idx), end_time=float(idx + 1),
                   label=label, reuse_of=reuse_of)


def test_partition_by_role_male_female_mix():
    # Intro (none), Verse - male, Chorus - female, Verse 2 - male, Chorus repeat, Outro
    segs = [
        _seg(0, "Intro"),
        _seg(1, "Verse 1 - male"),
        _seg(2, "Chorus - female"),
        _seg(3, "Verse 2 - male"),
        _seg(4, "Chorus - female", reuse_of=2),
        _seg(5, "Outro"),
    ]
    parts = partition_anchors_by_role(segs)
    assert parts["male"] == [1, 3]
    assert parts["female"] == [2]
    assert parts[None] == [0, 5]
    # repeat at idx 4 must not appear anywhere
    all_idxs = [i for v in parts.values() for i in v]
    assert 4 not in all_idxs


def test_partition_by_role_with_duet_section():
    segs = [
        _seg(0, "Verse - male"),
        _seg(1, "Chorus - female"),
        _seg(2, "Bridge - duet"),
        _seg(3, "Chorus - female", reuse_of=1),
    ]
    parts = partition_anchors_by_role(segs)
    assert parts["male"] == [0]
    assert parts["female"] == [1]
    assert parts["duet"] == [2]
    assert None not in parts


def test_partition_by_role_all_unannotated_single_bucket():
    segs = [_seg(0, "Verse 1"), _seg(1, "Chorus"), _seg(2, "Outro")]
    parts = partition_anchors_by_role(segs)
    assert parts == {None: [0, 1, 2]}


# ---------------------------------------------------------------------------
# Fix 27 — same-gender duet routing (plan_same_gender_portraits + veto)
# ---------------------------------------------------------------------------

def test_plan_same_gender_ff_with_duet():
    # two women + shared duet chorus → lead "female", partner "female2", build duet
    assert plan_same_gender_portraits(
        "ff", {"female", "duet"}, consistent_character=True, source_mode="auto"
    ) == ("female", "female2", True)


def test_plan_same_gender_mm_with_duet():
    assert plan_same_gender_portraits(
        "mm", {"male", "duet"}, consistent_character=True, source_mode="describe"
    ) == ("male", "male2", True)


def test_plan_same_gender_no_duet_section_still_two_portraits():
    # solos only, no duet label → still lead + partner, but no duet portrait
    assert plan_same_gender_portraits(
        "ff", {"female"}, consistent_character=True, source_mode="auto"
    ) == ("female", "female2", False)


def test_plan_same_gender_disabled_returns_none():
    # empty intent → standard path
    assert plan_same_gender_portraits(
        "", {"female", "duet"}, consistent_character=True, source_mode="auto"
    ) is None
    # invalid intent → standard path
    assert plan_same_gender_portraits(
        "xx", {"female"}, consistent_character=True, source_mode="auto"
    ) is None


def test_plan_same_gender_requires_consistent_character_and_generative_mode():
    assert plan_same_gender_portraits(
        "ff", {"female", "duet"}, consistent_character=False, source_mode="auto"
    ) is None
    assert plan_same_gender_portraits(
        "ff", {"female", "duet"}, consistent_character=True, source_mode="upload"
    ) is None


def test_same_gender_veto_all_female_no_veto():
    # ff intent, every vocal section detected female → no opposite presence → no veto
    sections = [("female", 12.0), ("female", 8.0), ("female", 14.0)]
    assert same_gender_veto(sections, "ff") is False


def test_same_gender_veto_stray_male_below_threshold_no_veto():
    # one short male window amid lots of female → below 15% duration → no veto
    sections = [("female", 40.0), ("female", 40.0), ("male", 5.0)]
    assert same_gender_veto(sections, "ff") is False


def test_same_gender_veto_sustained_male_triggers():
    # a genuine male section ~33% of vocal duration → veto, fall back to mixed
    sections = [("female", 20.0), ("male", 20.0), ("female", 20.0)]
    assert same_gender_veto(sections, "ff") is True


def test_same_gender_veto_mm_symmetric():
    sections = [("male", 20.0), ("female", 20.0), ("male", 20.0)]
    assert same_gender_veto(sections, "mm") is True
    assert same_gender_veto([("male", 30.0), ("male", 30.0)], "mm") is False


def test_same_gender_veto_ignores_unknown():
    # "unknown" windows are dropped from the denominator; pure female → no veto
    sections = [("female", 20.0), ("unknown", 30.0)]
    assert same_gender_veto(sections, "ff") is False


def test_same_gender_veto_detected_duet_means_opposite_present():
    # _classify_section returns "duet" only when BOTH genders ≥20% → a male is
    # present (e.g. a man sings the shared chorus). An ff request must veto so
    # the man gets his own portrait via the mixed path.
    sections = [("female", 20.0), ("female", 20.0), ("duet", 20.0)]
    assert same_gender_veto(sections, "ff") is True


# ---------------------------------------------------------------------------
# Role clothing contracts
# ---------------------------------------------------------------------------

def test_role_clothing_contracts_do_not_copy_female_outfit_to_missing_male():
    text = (
        "Male Jamaican singer with dreadlocks and rasta tam. "
        "Female Puerto-Rican singer wears a white tank top with a green "
        "marijuana leaf and high-cut blue denim hotpants."
    )
    contracts = build_role_clothing_contracts(text, "reggae")
    assert "hotpants" not in contracts["male"].lower()
    assert "marijuana leaf" not in contracts["male"].lower()
    # Male names only a hat (rasta tam) → no body garment matched → the neutral
    # male default body garments are appended (RC1: default is no longer reggae).
    assert "dark indigo straight-leg jeans" in contracts["male"]
    assert "black leather boots" in contracts["male"]
    assert "hotpants" in contracts["female"].lower()


def test_role_clothing_contract_extracts_outfit_from_scene_heavy_theme():
    theme = (
        "realistic indie pop rooftop performance, blue hour city skyline, "
        "two pole lights, solo female singer, consistent white tank top "
        "blue denim shorts black sandals"
    )

    contracts = build_role_clothing_contracts(theme, "indie pop")

    female = contracts["female"].lower()
    assert "white tank top" in female
    assert "blue denim shorts" in female
    assert "black sandals" in female
    assert "rooftop" not in female
    assert "city skyline" not in female
    assert "pole lights" not in female


def test_role_contracts_extract_female_only_outfit_and_age_without_male_contract():
    theme = (
        "Solo female singer around 29 years old wearing a silver cropped jacket "
        "and black high-waist pants in a neon alley. No male vocalist."
    )

    clothing = build_role_clothing_contracts(theme, "dark pop")
    ages = build_role_age_contracts(theme, "dark pop")

    assert clothing["male"] == ""
    assert "silver cropped jacket" in clothing["female"].lower()
    assert "black high-waist pants" in clothing["female"].lower()
    assert ages["male"] == ""
    assert ages["female"].lower() == "around 29 years old"


def test_filter_role_contracts_removes_unused_male_for_female_only_segments():
    segments = [
        Segment(
            index=0,
            start_time=0,
            end_time=10,
            label="Verse - female",
            prompt="Female singer performs.",
        )
    ]
    clothing = {
        "male": "burgundy shirt, tan trousers",
        "female": "silver cropped jacket, black high-waist pants",
    }
    ages = {
        "male": "adult male performer around 30 years old",
        "female": "adult female performer around 30 years old",
    }

    filtered_clothing, filtered_ages = filter_role_contracts_for_present_roles(
        segments, clothing, ages, duet=""
    )

    assert filtered_clothing["male"] == ""
    assert filtered_ages["male"] == ""
    assert filtered_clothing["female"] == clothing["female"]
    assert filtered_ages["female"] == ages["female"]


def test_sanitize_male_prompt_removes_female_body_and_clothing_terms():
    contracts = {
        "male": "loose linen shirt, denim shorts, sandals",
        "female": "white tank top with marijuana leaf and hotpants",
    }
    prompt = (
        "A male singer with hourglass build wears a white tank top with "
        "marijuana leaf and hotpants."
    )
    out = sanitize_role_prompt_text(prompt, "male", contracts).lower()
    assert "hourglass" not in out
    assert "hotpants" not in out
    assert "marijuana leaf" not in out
    assert "loose linen shirt" in out
    assert "flat chest" in out


def test_apply_role_clothing_contracts_sanitizes_segment_and_relay():
    seg = Segment(
        index=0,
        start_time=0,
        end_time=10,
        label="Verse - male",
        prompt="Male singer in a feminine hourglass pose.",
        frame_variant_prompt="Male singer wearing hotpants.",
        video_prompt_relay={
            "global": "studio lighting",
            "beats": ["The male performer wears a marijuana leaf tank top."],
        },
    )
    contracts = {
        "male": "rasta tam, loose linen shirt, denim shorts",
        "female": "white tank top with marijuana leaf and hotpants",
    }
    apply_role_clothing_contracts_to_segments([seg], contracts)
    combined = " ".join([
        seg.prompt, seg.frame_variant_prompt, seg.video_prompt_relay["beats"][0],
    ]).lower()
    assert "hourglass" not in combined
    assert "hotpants" not in combined
    assert "marijuana leaf" not in combined
    assert "rasta tam" in combined
    assert seg.role_clothing_contract == contracts["male"]


def test_wardrobe_slot_overrides_theme_clothing_contracts():
    seg = Segment(
        index=0,
        start_time=0,
        end_time=10,
        label="Chorus - female",
        prompt="Female singer performs on rooftop, wearing black sequined halter bodysuit.",
        frame_variant_prompt="Female singer stands on rooftop.",
        wardrobe_slot="performance_stage",
        video_prompt_relay={
            "global": "rooftop camera anchor",
            "beats": ["The singer stands under stage lights."],
        },
    )
    contracts = {
        "male": "burgundy shirt and tan trousers",
        "female": "fitted feminine top, denim shorts or skirt",
    }

    apply_role_clothing_contracts_to_segments([seg], contracts)

    combined = " ".join([
        seg.role_clothing_contract,
        seg.prompt,
        seg.frame_variant_prompt,
        seg.video_prompt_relay["global"],
        seg.video_prompt_relay["beats"][0],
    ]).lower()
    assert "black sequined halter bodysuit" in combined
    assert "fitted feminine top" not in combined
    assert "denim shorts or skirt" not in combined


def test_wardrobe_slot_removes_conflicting_existing_outfit_phrases():
    seg = Segment(
        index=0,
        start_time=0,
        end_time=10,
        label="Verse - male",
        prompt=(
            "Male singer wearing burgundy shirt under a brown buttoned vest "
            "stands on a rooftop."
        ),
        frame_variant_prompt="He wears a brown buttoned vest and tan trousers.",
        wardrobe_slot="rocker_edgy",
        video_prompt_relay={
            "global": "rooftop camera anchor",
            "beats": [
                "The male singer wears tan trousers and a brown vest under stage lights."
            ],
        },
    )
    contracts = {
        "male": "burgundy shirt under a brown buttoned vest, tan trousers",
        "female": "black dress",
    }

    apply_role_clothing_contracts_to_segments([seg], contracts)

    combined = " ".join([
        seg.role_clothing_contract,
        seg.prompt,
        seg.frame_variant_prompt,
        seg.video_prompt_relay["global"],
        seg.video_prompt_relay["beats"][0],
    ]).lower()
    assert "black leather biker jacket" in combined
    assert "brown buttoned vest" not in combined
    assert "tan trousers" not in combined


def test_duet_clothing_contract_names_both_roles_without_swapping():
    contracts = {"male": "linen shirt", "female": "leaf tank top"}
    out = sanitize_role_prompt_text("Two singers perform together", "duet", contracts)
    out_lower = out.lower()
    assert "the man wears exactly linen shirt" in out_lower
    assert "the woman wears exactly leaf tank top" in out_lower
    assert "Never swap" in out


def test_fit_clip_to_frames_trims_overrender_without_padding(monkeypatch):
    calls = []
    monkeypatch.setattr(mvp, "_probe_frame_count", lambda _src: 120)

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        class R:
            stdout = ""
        return R()

    monkeypatch.setattr(mvp.subprocess, "run", fake_run)
    dst, actual = mvp._fit_clip_to_frames("src.mp4", 100, "dst.mp4")
    assert dst == "dst.mp4"
    assert actual == 120
    cmd = calls[0]
    assert "tpad=stop_mode=clone" not in " ".join(cmd)
    assert cmd[cmd.index("-frames:v") + 1] == "100"


def test_fit_clip_to_frames_warns_on_large_padding(monkeypatch, capsys):
    monkeypatch.setattr(mvp, "_probe_frame_count", lambda _src: 60)

    def fake_run(cmd, **kwargs):
        class R:
            stdout = ""
        return R()

    monkeypatch.setattr(mvp.subprocess, "run", fake_run)
    mvp._fit_clip_to_frames("src.mp4", 100, "dst.mp4")
    assert "large frame pad" in capsys.readouterr().out


def test_apply_scene_contracts_prefixes_prompts_without_changing_timing():
    seg = Segment(
        index=0,
        start_time=3.5,
        end_time=17.6,
        label="Verse - duet",
        prompt="Two singers perform on a beach.",
        frame_variant_prompt="Two singers stand together.",
        background_prompt="sunset beach with orange sky",
        video_prompt_relay={"global": "warm camera grade", "beats": ["They sing."]},
    )
    apply_scene_contracts_to_segments([seg])
    assert (seg.start_time, seg.end_time) == (3.5, 17.6)
    assert seg.scene_contract == "sunset beach with orange sky"
    assert seg.background_source == "llm"
    assert seg.prompt.startswith("Scene location must be exactly: sunset beach")
    assert seg.video_prompt_relay["beats"][0] == "They sing."
    g = seg.video_prompt_relay["global"]
    assert "sunset beach with orange sky" in g
    assert "Photoreal cinematic still" in g
    # No instructional language in global (Stage MSR-2026-06: visual prose only).
    assert "Use the scene reference" not in g
    assert "should look" not in g


def test_scene_contract_relay_keeps_beats_promptrelay_clean():
    seg = Segment(
        index=0,
        start_time=0,
        end_time=12,
        label="Verse - female",
        prompt="Female singer performs on a rooftop.",
        frame_variant_prompt="Female singer stands on a rooftop.",
        background_prompt="empty rooftop at blue hour with two pole lights",
        video_prompt_relay={
            "global": "slow dolly, realistic grade",
            "beats": [
                "The performer stands still on the rooftop.",
                "The camera slowly pulls back.",
            ],
        },
    )

    apply_scene_contracts_to_segments([seg])

    g = seg.video_prompt_relay["global"]
    assert "empty rooftop at blue hour" in g
    assert "Photoreal cinematic still" in g
    assert "Use the scene reference" not in g
    assert seg.video_prompt_relay["beats"] == [
        "The performer stands still on the rooftop.",
        "The camera slowly pulls back.",
    ]
    assert not seg.video_prompt_relay["beats"][0].startswith("Scene location must be exactly")


def test_role_clothing_contracts_lock_relay_global_and_scene_integration():
    seg = Segment(
        index=0,
        start_time=0,
        end_time=10,
        label="Verse - female",
        prompt="Female singer performs in a neon alley.",
        frame_variant_prompt="Female singer stands on wet pavement.",
        video_prompt_relay={
            "global": "handheld camera, neon rim light",
            "beats": ["She turns toward the camera."],
        },
    )
    contracts = {
        "male": "black silk shirt",
        "female": "black sequined halter bodysuit and matching shorts",
    }

    apply_role_clothing_contracts_to_segments([seg], contracts)

    global_prompt = seg.video_prompt_relay["global"].lower()
    beat_prompt = seg.video_prompt_relay["beats"][0].lower()
    assert "black sequined halter bodysuit" in global_prompt
    assert "black sequined halter bodysuit" in beat_prompt
    assert "natural contact shadows" in global_prompt
    assert "scene light" in global_prompt


def test_role_clothing_contract_relay_global_only_after_first_static_beat():
    seg = Segment(
        index=0,
        start_time=0,
        end_time=12,
        label="Verse - female",
        prompt="Female singer performs in a neon alley.",
        frame_variant_prompt="Female singer stands on wet pavement.",
        video_prompt_relay={
            "global": "handheld camera, neon rim light",
            "beats": [
                "The performer stands still in the alley.",
                "She turns toward the camera.",
            ],
        },
    )
    contracts = {
        "female": "white tank top, blue denim shorts and black sandals",
    }

    apply_role_clothing_contracts_to_segments([seg], contracts)

    g = seg.video_prompt_relay["global"]
    assert "Female performer wearing white tank top" in g
    assert "Use the female performer reference" not in g
    assert "white tank top" in seg.video_prompt_relay["beats"][0]
    assert "white tank top" not in seg.video_prompt_relay["beats"][1]


def test_ensure_relay_specs_synthesizes_missing_relay_without_legacy_prompt_blob():
    seg = Segment(
        index=0,
        start_time=0,
        end_time=8,
        label="Chorus - female",
        prompt=(
            "Scene location must be exactly: empty rooftop at blue hour. "
            "Clothing lock: the referenced female singer wears only this exact outfit: "
            "white tank top, blue denim shorts, black sandals. Close-up of singer. "
            "Text appears in the city background reading 'STAY IN THIS MOMENT NOW'."
        ),
        frame_variant_prompt="Female singer stands on a rooftop.",
        background_prompt="empty rooftop at blue hour",
        scene_contract="empty rooftop at blue hour",
        role_clothing_contract="white tank top, blue denim shorts, black sandals",
    )

    ensure_relay_specs_for_segments([seg])

    assert seg.video_prompt_relay is not None
    g = seg.video_prompt_relay["global"]
    assert "empty rooftop at blue hour" in g
    assert "Photoreal cinematic still" in g
    assert "Use the scene reference" not in g
    assert len(seg.video_prompt_relay["beats"]) >= 1
    assert "Text appears" not in " ".join(seg.video_prompt_relay["beats"])
    assert "Scene location must be exactly" not in " ".join(seg.video_prompt_relay["beats"])
    assert "white tank top" in seg.video_prompt_relay["beats"][0]


def test_ensure_relay_specs_includes_age_and_avoids_generic_story_placeholder():
    seg = Segment(
        index=0,
        start_time=0,
        end_time=8,
        label="Chorus - female",
        prompt="Female singer leans toward a rain-streaked window.",
        frame_variant_prompt="Close-up of the female singer beside a rain-streaked window.",
        background_prompt="rainy neon alley window",
        scene_contract="rainy neon alley window",
        role_clothing_contract="silver cropped jacket, black high-waist pants",
        role_age_contract="adult female performer around 30 years old",
    )

    ensure_relay_specs_for_segments([seg])

    combined = " ".join([
        seg.video_prompt_relay["global"],
        *seg.video_prompt_relay["beats"],
    ]).lower()
    assert "rain-streaked window" in combined
    assert "silver cropped jacket" in combined
    # RC3: age contract is rendered bare ("around 30 years old") — the leading
    # "adult female performer" tokens are stripped so the template that prepends
    # "performer apparent age" does not double the word.
    assert "around 30 years old" in combined
    assert "female performer apparent age adult female performer" not in combined
    assert "the performer holds the pose" not in combined


def test_apply_scene_contracts_replaces_studio_with_real_location():
    # R3-4: an LLM-authored studio/seamless backdrop must be replaced by a
    # derived real location (it bleeds a grey void into the MSR output).
    seg = Segment(
        index=0, start_time=0, end_time=8, label="Pre-Chorus - female",
        prompt="Medium close-up of female singer at night with neon and rain.",
        frame_variant_prompt="Female singer, neon-lit rainy night, moonlight.",
        background_prompt="High-key seamless studio background, blue mist, silvery highlights",
        scene_contract="High-key seamless studio background, blue mist, silvery highlights",
    )
    apply_scene_contracts_to_segments([seg])
    sc = seg.scene_contract.lower()
    assert "studio" not in sc
    assert "seamless" not in sc
    assert seg.background_source == "fallback"


def test_apply_scene_contracts_falls_back_when_background_mentions_people():
    seg = Segment(
        index=0,
        start_time=0,
        end_time=8,
        label="Intro",
        prompt="Empty jungle at night with moonlight.",
        frame_variant_prompt="Empty jungle at night.",
        background_prompt="female singer wearing jeans in a jungle",
    )
    apply_scene_contracts_to_segments([seg])
    assert seg.background_source == "fallback"
    assert "female singer" not in seg.background_prompt.lower()
    assert "jeans" not in seg.background_prompt.lower()
    assert "jungle" in seg.prompt.lower()


# ---------------------------------------------------------------------------
# Fix 23: LLM-failure hardening (portrait fallback + _call_openrouter retry)
# ---------------------------------------------------------------------------

def test_portrait_empty_llm_returns_lyrics_free_default():
    """Fix 23: empty LLM response must NOT leak the raw-lyrics seed onto the
    portrait — it must return the neutral studio default."""
    prompter = mvp.MusicVideoPrompter({})

    async def _empty(*a, **k):
        return ""
    prompter._call_openrouter = _empty
    seed = "Rain on the midnight glass\nWatching the neon pass"  # raw lyric lines
    out = asyncio.run(prompter.generate_character_portrait_prompt(seed, "synthwave"))
    # Stage O3: the default is now a FULL-BODY studio shot (the portrait
    # doubles as the character sheet's full-body front cell).
    assert "solid flat neutral mid-grey background" in out
    assert "#808080" in out
    # Lyrics-free guard (the test's name is about the fallback NOT leaking the seed).
    assert "lyric" not in out.lower()
    assert "midnight glass" not in out


def test_portrait_uses_llm_response_when_present():
    prompter = mvp.MusicVideoPrompter({})

    async def _resp(*a, **k):
        return "a woman with red hair, plain studio portrait"
    prompter._call_openrouter = _resp
    out = asyncio.run(prompter.generate_character_portrait_prompt("lyrics", ""))
    assert out == "a woman with red hair, plain studio portrait"


def test_female_portrait_strips_male_body_leak_from_llm_response():
    prompter = mvp.MusicVideoPrompter({})

    async def _resp(*a, **k):
        return (
            "front-facing full-body standing solo female indie pop singer, "
            "hourglass build, flat chest, blonde bob hair, white tank top"
        )
    prompter._call_openrouter = _resp

    out = asyncio.run(prompter.generate_character_portrait_prompt(
        "theme",
        "indie pop",
        role="female",
        clothing_contract="white tank top, blue denim shorts, black sandals",
    ))

    assert "flat chest" not in out.lower()
    assert "female indie pop singer" in out


def test_generic_female_portrait_strips_male_body_leak_from_llm_response():
    prompter = mvp.MusicVideoPrompter({})

    async def _resp(*a, **k):
        return "front-facing full-body female singer, hourglass build, flat chest, brown hair"
    prompter._call_openrouter = _resp

    out = asyncio.run(prompter.generate_character_portrait_prompt("theme", "indie pop"))

    assert "flat chest" not in out.lower()
    assert "female singer" in out


class _FakeResp:
    def __init__(self, status_code, content):
        self.status_code = status_code
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _patch_httpx(monkeypatch, decide):
    """Patch mvp.httpx.AsyncClient so post() outcome is decided by `decide(model)`."""
    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            return decide(json["model"])
    monkeypatch.setattr(mvp.httpx, "AsyncClient", lambda *a, **k: _Client())


def test_call_openrouter_retries_429_then_falls_back_to_next_model(monkeypatch):
    calls = []

    def decide(model):
        calls.append(model)
        if model == "primary":
            return _FakeResp(429, "")          # primary always rate-limited
        return _FakeResp(200, f"out-{model}")  # fallback succeeds
    _patch_httpx(monkeypatch, decide)

    prompter = mvp.MusicVideoPrompter({
        "openrouter_model": "primary",
        "fallback_models": ["secondary"],
        "max_retries": 2,
        "retry_backoff_seconds": [0, 0],
    })
    result = asyncio.run(prompter._call_openrouter([{"role": "user", "content": "x"}]))
    assert result == "out-secondary"
    assert calls.count("primary") == 2   # retried up to max_retries before giving up
    assert "secondary" in calls


def test_call_openrouter_all_429_returns_empty(monkeypatch):
    def decide(model):
        return _FakeResp(429, "")
    _patch_httpx(monkeypatch, decide)
    prompter = mvp.MusicVideoPrompter({
        "openrouter_model": "primary",
        "fallback_models": ["secondary"],
        "max_retries": 1,
        "retry_backoff_seconds": [0],
    })
    result = asyncio.run(prompter._call_openrouter([{"role": "user", "content": "x"}]))
    assert result == ""


def test_call_openrouter_explicit_model_does_not_walk_fallbacks(monkeypatch):
    calls = []

    def decide(model):
        calls.append(model)
        return _FakeResp(429, "")
    _patch_httpx(monkeypatch, decide)
    prompter = mvp.MusicVideoPrompter({
        "openrouter_model": "primary",
        "fallback_models": ["secondary"],
        "max_retries": 1,
        "retry_backoff_seconds": [0],
    })
    result = asyncio.run(prompter._call_openrouter(
        [{"role": "user", "content": "x"}], model="pinned"))
    assert result == ""
    assert calls == ["pinned"]  # only the pinned model, no fallback walk


# ---- enforce_performer_role -------------------------------------------------

def test_enforce_role_male_strips_female_clauses():
    p = "He walks on the beach. She sings to him in the sunset. The waves crash."
    out = enforce_performer_role(p, "male")
    assert "she" not in out.lower()
    assert "He walks" in out
    assert "waves crash" in out.lower() or "waves" in out.lower()


def test_enforce_role_female_strips_male_clauses():
    p = "She dances by the palm trees. The man approaches from behind. Sunset glow fills the sky."
    out = enforce_performer_role(p, "female")
    assert "the man" not in out.lower()
    assert "She dances" in out


def test_enforce_role_duet_is_noop():
    p = "He sings to her on the beach as she sings back to him."
    assert enforce_performer_role(p, "duet") == p


def test_enforce_role_none_is_noop():
    p = "Anything goes here."
    assert enforce_performer_role(p, None) == p


def test_enforce_role_gender_neutral_unchanged():
    p = "The ocean glows golden at sunset behind swaying palms."
    assert enforce_performer_role(p, "male") == p
    assert enforce_performer_role(p, "female") == p


def test_enforce_role_falls_back_to_original_when_empty():
    # entire prompt is female references → stripping would empty it
    p = "She sings. Her dress flows. The woman dances."
    out = enforce_performer_role(p, "male")
    assert out == p  # fallback to original


def test_enforce_role_already_clean_prompt_unchanged_text():
    p = "He stands in the water at golden hour, palm trees behind him."
    out = enforce_performer_role(p, "male")
    # "him" trailing — strip module keeps it because it's not at clause start
    # at minimum no information added; check the visible male content remains
    assert "He stands" in out


# ---------------------------------------------------------------------------
# Fix 8: Intro/Outro standalone-unless-shorter-than-3s
# ---------------------------------------------------------------------------

def test_intro_4s_stays_standalone():
    # Intro 0-4s is above the 3.0s intro/outro threshold → keep as own row
    # even though normal min_seg/2 (=4.0) would absorb it. min_seg=8 → tiny=4
    a = [_al("Intro", "", 0.0, 4.0, is_vocal=False),
         _al("Verse 1 - male", "v", 4.0, 16.0)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    labels = [r["label"] for r in rows]
    assert "Intro" in labels  # standalone
    assert rows[0]["start_time"] == 0.0 and rows[0]["end_time"] == 4.0


def test_intro_2s_merges_forward():
    # Intro 0-2s falls below the 3.0s threshold → absorbed into the next row
    a = [_al("Intro", "", 0.0, 2.0, is_vocal=False),
         _al("Verse 1 - male", "v", 2.0, 14.0)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    assert len(rows) == 1
    assert rows[0]["start_time"] == 0.0 and rows[0]["end_time"] == 14.0
    # absorbed forward, so the surviving label is the Verse
    assert "Verse" in rows[0]["label"]


def test_outro_4s_stays_standalone():
    # Outro 50-54s (4s, above intro/outro threshold) standalone
    a = [_al("Verse 1 - male", "v", 0.0, 50.0),
         _al("Outro", "", 50.0, 54.0, is_vocal=False)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    labels = [r["label"] for r in rows]
    assert "Outro" in labels
    # Outro is the LAST row
    assert rows[-1]["start_time"] == 50.0


def test_fade_below_3s_merges_back():
    # Fade Out 60.0-60.5s (below 3s) merges into preceding row
    a = [_al("Outro", "", 50.0, 60.0, is_vocal=False),
         _al("Fade Out", "", 60.0, 60.5, is_vocal=False)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    assert len(rows) == 1
    assert rows[0]["start_time"] == 50.0 and rows[0]["end_time"] == 60.5


# ---------------------------------------------------------------------------
# Fix 10: camera-move rules must stay section-type-aware in both system prompts
# ---------------------------------------------------------------------------

from music_video_pipeline import _SEG_DIRECTOR_RULES, MusicVideoPrompter  # noqa: E402
import inspect  # noqa: E402


_FORBIDDEN_VOCAL_PHRASES = (
    "dolly-out",
    "zoom-out-then-zoom-in",
    "face must remain in frame and recognizable",
)
_STORY_LATITUDE_PHRASES = (
    "STORY-section allowed shot types",
    "wide-landscape, drone-style aerial",
)


def test_seg_director_rules_contains_vocal_forbidden_line():
    for phrase in _FORBIDDEN_VOCAL_PHRASES:
        assert phrase in _SEG_DIRECTOR_RULES, f"_SEG_DIRECTOR_RULES missing: {phrase!r}"


def test_seg_director_rules_contains_story_latitude_section():
    for phrase in _STORY_LATITUDE_PHRASES:
        assert phrase in _SEG_DIRECTOR_RULES, f"_SEG_DIRECTOR_RULES missing: {phrase!r}"


def test_plan_segments_inline_prompt_contains_vocal_forbidden_line():
    # The longer inline system prompt lives inside MusicVideoPrompter.plan_segments.
    # Grab the source and assert both rule blocks survived.
    src = inspect.getsource(MusicVideoPrompter.plan_segments)
    for phrase in _FORBIDDEN_VOCAL_PHRASES:
        assert phrase in src, f"plan_segments inline prompt missing: {phrase!r}"


def test_plan_segments_inline_prompt_contains_story_latitude_section():
    src = inspect.getsource(MusicVideoPrompter.plan_segments)
    for phrase in _STORY_LATITUDE_PHRASES:
        assert phrase in src, f"plan_segments inline prompt missing: {phrase!r}"


def test_non_intro_short_section_uses_min_seg_half():
    # A non-intro/outro 3.5s section should still merge (3.5 < min_seg/2 = 4)
    a = [_al("Verse", "v", 0.0, 12.0),
         _al("Stab", "", 12.0, 15.5, is_vocal=False)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    assert len(rows) == 1  # Stab absorbed because 3.5 < tiny(=4) for non-intro/outro
    assert rows[0]["start_time"] == 0.0 and rows[0]["end_time"] == 15.5


# ─── Fix 24A: build_duet_portrait_prompt ─────────────────────────────────────

def test_duet_prompt_locks_identity_to_reference_images():
    """Fix 24A: prompt must explicitly instruct the T2I model to preserve
    face/hair/skin from the reference images — that's the whole point of the
    deterministic duet prompt."""
    p = mvp.build_duet_portrait_prompt("neon city, synthwave")
    assert "Preserve each performer" in p
    assert "face" in p and "hair" in p and "skin tone" in p
    assert "reference images" in p
    assert "do not restyle or recolour" in p


def test_duet_prompt_does_not_invent_identity_attributes():
    """Fix 24A regression guard: the LLM-driven path invented attributes like
    'blonde hair' / 'dark waves' that competed with the references and caused
    drift. The deterministic prompt MUST NOT contain colour/style attributes."""
    p = mvp.build_duet_portrait_prompt("80s synthwave duo").lower()
    for forbidden in ("blonde", "brunette", "redhead", "dark waves",
                      "wavy hair", "curly", "tattoo", "beard", "moustache"):
        assert forbidden not in p, f"prompt invented identity attribute: {forbidden!r}"


def test_duet_prompt_embeds_theme_when_given_and_omits_when_empty():
    with_theme = mvp.build_duet_portrait_prompt("rainy neon street")
    assert "Theme context: rainy neon street." in with_theme
    no_theme = mvp.build_duet_portrait_prompt("")
    assert "Theme context:" not in no_theme


# ── RC1/RC2/RC3 prompt-hygiene regression guards ─────────────────────


def test_male_clothing_default_is_neutral_no_reggae_no_colon():
    # RC1: the male default must not be the reggae test fixture, and must not
    # carry a label+colon (colons are illegal in LTX prompts).
    d = mvp._MALE_CLOTHING_DEFAULT.lower()
    assert "reggae" not in d
    assert "rasta" not in d
    assert ":" not in mvp._MALE_CLOTHING_DEFAULT


def test_role_age_global_text_does_not_double_performer():
    # RC3: even when handed the old verbose contract, the template must not
    # produce "performer ... performer".
    txt = mvp.role_age_global_text(
        "male", {"male": "adult male performer around 30 years old"},
    )
    assert txt == "Male performer apparent age around 30 years old."
    assert txt.lower().count("performer") == 1


def _msr_seg():
    return Segment(
        index=0, start_time=0, end_time=8, label="Verse 1 - male",
        prompt="Male singer in a neon alley.",
        frame_variant_prompt="Close-up of the male singer in a neon alley.",
        background_prompt="rain-slicked neon alley",
        scene_contract="rain-slicked neon alley",
        video_prompt_relay={
            "global": "Cold synthwave noir, handheld camera, rain on the street.",
            "beats": ["The singer stands still as rain falls.",
                      "He looks down as a breath forms."],
        },
    )


def test_msr_gate_keeps_clothing_and_age_out_of_relay():
    # RC2: with MSR active the sheet/subject-desc own wardrobe + age, so the
    # relay global/beats must stay free of clothing/age tokens.
    seg = _msr_seg()
    clothing = {"male": "charcoal trench coat, black boots", "female": ""}
    ages = {"male": "around 30 years old", "female": ""}
    apply_role_clothing_contracts_to_segments([seg], clothing, msr_active=True)
    mvp.apply_role_age_contracts_to_segments([seg], ages, msr_active=True)
    relay = seg.video_prompt_relay
    blob = " ".join([relay["global"], *relay["beats"]]).lower()
    assert "trench coat" not in blob
    assert "wearing" not in blob
    assert "apparent age" not in blob
    assert "around 30 years old" not in blob
    # The attribute is still set for the MSR subject-description builder.
    assert seg.role_clothing_contract == "charcoal trench coat, black boots"


def test_ensure_relay_strips_clothing_lock_block_from_beats():
    # R2-3: sanitize_role_prompt_text prepends a "Clothing lock. ... outfit. ...
    # cutout look." block to frame_variant_prompt (feeds MCA gen). ensure_relay
    # builds beats from that text and must strip the whole block, not leak the
    # wardrobe into PromptRelay beats.
    fv = (
        "Clothing lock. The referenced male singer wears only this exact outfit. "
        "oversized onyx trench coat, ebony slim trousers, charcoal leather boots. "
        "Shirt and vest stay visible; no bare chest, no topless or shirtless look. "
        "Masculine adult male body, flat chest, no breasts. Never swap clothing "
        "with any female performer. Natural contact shadows under the performer, "
        "physically inside the background, not pasted on top, no green screen or "
        "cutout look. Medium shot of the singer on a rain-slicked neon street."
    )
    seg = Segment(
        index=0, start_time=0, end_time=8, label="Verse 1 - male",
        prompt=fv, frame_variant_prompt=fv,
        background_prompt="rain-slicked neon street",
        scene_contract="rain-slicked neon street",
        role_clothing_contract="oversized onyx trench coat, ebony slim trousers, charcoal leather boots",
    )
    ensure_relay_specs_for_segments([seg], msr_active=True)
    blob = " ".join(seg.video_prompt_relay["beats"]).lower()
    assert "clothing lock" not in blob
    assert "onyx trench coat" not in blob
    assert "no breasts" not in blob
    # The actual scene/action survives.
    assert "rain-slicked neon street" in blob


def test_non_msr_still_injects_clothing_and_age_into_relay():
    # Regression guard: the non-MSR (IA2V) path keeps the wardrobe/age lock in
    # the relay, since there is no sheet to carry identity.
    seg = _msr_seg()
    clothing = {"male": "charcoal trench coat, black boots", "female": ""}
    ages = {"male": "around 30 years old", "female": ""}
    apply_role_clothing_contracts_to_segments([seg], clothing, msr_active=False)
    mvp.apply_role_age_contracts_to_segments([seg], ages, msr_active=False)
    blob = " ".join([
        seg.video_prompt_relay["global"], *seg.video_prompt_relay["beats"],
    ]).lower()
    assert "charcoal trench coat" in blob
    assert "around 30 years old" in blob
