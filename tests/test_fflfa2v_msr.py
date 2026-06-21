"""FFLFA2V-MSR: the standalone LTXMSRICLoRAFLF workflow wiring + the FF/LF
injection helpers and api.py hook that drive it. Verifies the new MSR path
without disturbing the legacy LiconMSR path."""

from __future__ import annotations

import json
import os

from comfyui import (
    FF_TITLE_TAG,
    LF_TITLE_TAG,
    has_msr_nodes,
    inject_first_frame,
    inject_last_frame,
    inject_flf_clip_length,
    inject_msr_flf_resolution,
    inject_msr_images,
)

_WF_DIR = os.path.join(os.path.dirname(__file__), "..", "Workflows")
_FLF_WF = os.path.join(_WF_DIR, "LTX2.3 - FFLFA2V-MSR.json")
_STD_WF = os.path.join(_WF_DIR, "LTX2.3 - IA2V-PromptRelay.json")
_FLF_NODE = "LTXMSRICLoRAFLF_Experimental"


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _flf_id(wf):
    return next(nid for nid, n in wf.items() if n.get("class_type") == _FLF_NODE)


# ── detection ────────────────────────────────────────────────────


def test_flf_workflow_detected_and_standard_not():
    assert has_msr_nodes(_load(_FLF_WF))
    assert not has_msr_nodes(_load(_STD_WF))


# ── node presence + settings (from the repo FLF settings PNG) ─────


def test_flf_node_settings():
    wf = _load(_FLF_WF)
    inp = wf[_flf_id(wf)]["inputs"]
    assert inp["enable_msr_latent_injection"] is False
    assert inp["enable_msr_attention"] is True
    assert inp["lock_first_frame"] is True
    assert inp["lock_last_frame"] is False
    assert inp["msr_frame_count"] == "41"
    # static default (16:9 @ longer edge 1024); pipeline overrides per-segment from FF aspect
    assert inp["msr_width"] == 1024 and inp["msr_height"] == 576
    # first_frame_strength lowered 1.0 -> 0.85 (A/B: frame-0 colour flash 27.3 -> 3.7)
    assert inp["first_frame_strength"] == 0.85


def test_inject_first_frame_strength():
    from comfyui import inject_msr_flf_first_frame_strength
    wf = _load(_FLF_WF)
    inject_msr_flf_first_frame_strength(wf, 0.6)
    assert wf[_flf_id(wf)]["inputs"]["first_frame_strength"] == 0.6


# ── splice wiring ────────────────────────────────────────────────


def test_flf_outputs_feed_guider_crop_and_concat():
    wf = _load(_FLF_WF)
    mid = _flf_id(wf)
    assert wf["759:1052"]["inputs"]["positive"] == [mid, 0]
    assert wf["759:1052"]["inputs"]["negative"] == [mid, 1]
    assert wf["759:1074"]["inputs"]["positive"] == [mid, 0]
    assert wf["759:1074"]["inputs"]["negative"] == [mid, 1]
    assert wf["759:1055"]["inputs"]["video_latent"] == [mid, 2]


def test_flf_iclora_on_model_path():
    wf = _load(_FLF_WF)
    lid = next(nid for nid, n in wf.items()
               if n.get("class_type") == "LTXICLoRALoaderModelOnly")
    assert wf[lid]["inputs"]["model"] == ["211", 0]
    assert wf[lid]["inputs"]["lora_name"] == "LTX-2.3-Licon-MSR-V1.safetensors"
    assert wf["700"]["inputs"]["model"] == [lid, 0]


def test_flf_node_fed_by_clean_conditioning_not_old_guide_chain():
    # M must NOT read the stock 759:1067 (downstream of the old FF/LF
    # AddGuideMulti) or it would double-plant keyframes.
    wf = _load(_FLF_WF)
    mid = _flf_id(wf)
    cid = wf[mid]["inputs"]["positive"][0]
    assert cid != "759:1067"
    assert wf[cid]["class_type"] == "LTXVConditioning"
    assert wf[cid]["inputs"]["positive"] == ["121", 0]
    assert wf[cid]["inputs"]["negative"] == ["593", 0]


def test_flf_no_dangling_refs():
    wf = _load(_FLF_WF)
    ids = set(wf)
    for nid, n in wf.items():
        for v in n.get("inputs", {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0][:1].isdigit():
                assert v[0] in ids, f"dangling ref {nid} -> {v[0]}"


# ── injection helpers ────────────────────────────────────────────


def test_inject_first_and_last_frame():
    wf = _load(_FLF_WF)
    inject_first_frame(wf, "ff.png")
    inject_last_frame(wf, "lf.png")
    ff = next(n for n in wf.values()
              if n.get("class_type") == "LoadImage" and FF_TITLE_TAG in n.get("_meta", {}).get("title", ""))
    lf = next(n for n in wf.values()
              if n.get("class_type") == "LoadImage" and LF_TITLE_TAG in n.get("_meta", {}).get("title", ""))
    assert ff["inputs"]["image"] == "ff.png"
    assert lf["inputs"]["image"] == "lf.png"


def test_inject_msr_images_on_flf_node():
    wf = _load(_FLF_WF)
    mid = _flf_id(wf)
    inject_msr_images(wf, ["s1.png", "s2.png"], "bg.png")
    m = wf[mid]
    # used subject slots patched on their LoadImage nodes
    assert wf[m["inputs"]["msr_image_1"][0]]["inputs"]["image"] == "s1.png"
    assert wf[m["inputs"]["msr_image_2"][0]]["inputs"]["image"] == "s2.png"
    assert wf[m["inputs"]["msr_background"][0]]["inputs"]["image"] == "bg.png"
    # unused subject slots dropped from node inputs AND workflow (no orphan LoadImage)
    assert "msr_image_3" not in m["inputs"]
    assert "msr_image_4" not in m["inputs"]
    # FLF node bakes frame_count as a widget — must stay untouched (not the LiconMSR path)
    assert m["inputs"]["msr_frame_count"] == "41"


# ── api.py hook ──────────────────────────────────────────────────


def test_api_flf_hook_present():
    with open(os.path.join(os.path.dirname(__file__), "..", "api.py"), encoding="utf-8") as f:
        src = f.read()
    assert 'MSR_FLF_WORKFLOWS = {"LTX2.3 - FFLFA2V-MSR"}' in src
    assert "is_flf = video_workflow_id in MSR_FLF_WORKFLOWS and use_msr" in src
    assert "_run_flux2_last_frame" in src
    assert "inject_first_frame" in src and "inject_last_frame" in src
    assert "_init_run_dir()" in src


# ── per-run numbered output folder ───────────────────────────────


def test_api_gdrive_hook_present():
    with open(os.path.join(os.path.dirname(__file__), "..", "api.py"), encoding="utf-8") as f:
        src = f.read()
    assert 'GDRIVE_REMOTE = "gdrive:HermesCTB/MusicVideos"' in src
    assert "async def _upload_final_to_gdrive" in src
    assert "await _upload_final_to_gdrive(final_path)" in src
    assert "MSR_FLF_LONGER_EDGE = 1024" in src
    # upload must run before the job is marked done
    assert src.index("await _upload_final_to_gdrive(final_path)") < src.index("_job_done(jid, final_path, lyrics_path)")


# ── MSR resolution auto-match ─────────────────────────────────────


def test_inject_msr_flf_resolution():
    wf = _load(_FLF_WF)
    inject_msr_flf_resolution(wf, 768, 1024)
    inp = wf[_flf_id(wf)]["inputs"]
    assert inp["msr_width"] == 768 and inp["msr_height"] == 1024


def test_inject_flf_clip_length():
    wf = _load(_FLF_WF)
    inject_flf_clip_length(wf, 12.0)
    # "Clip Length (in seconds)" mxSlider -> 12; "Cut OFF End Frames" -> 0
    slider = next(n for n in wf.values()
                  if n.get("class_type") == "mxSlider"
                  and "clip length" in n.get("_meta", {}).get("title", "").lower())
    assert slider["inputs"]["Xi"] == 12 and slider["inputs"]["Xf"] == 12
    cut = next(n for n in wf.values()
               if n.get("class_type") == "JWInteger"
               and "cut off end" in n.get("_meta", {}).get("title", "").lower())
    assert cut["inputs"]["value"] == 0
    # one start frame trimmed to drop the locked-FF colour-flash frame
    cut_start = next(n for n in wf.values()
                     if n.get("class_type") == "JWInteger"
                     and "cut off start" in n.get("_meta", {}).get("title", "").lower())
    assert cut_start["inputs"]["value"] == 1
    # rounds fractional seconds, clamps >= 1
    inject_flf_clip_length(wf, 0.2)
    assert slider["inputs"]["Xi"] == 1


def test_api_flf_clip_length_wired():
    with open(os.path.join(os.path.dirname(__file__), "..", "api.py"), encoding="utf-8") as f:
        src = f.read()
    assert "inject_flf_clip_length" in src
    assert "wf = inject_flf_clip_length(wf, _flf_secs)" in src


def test_api_flf_first_frame_composite():
    # The locked first frame must be the singer-in-scene composite, not the raw
    # portrait. The is_flf branch rebuilds it when frame_by_seg is not a composite.
    with open(os.path.join(os.path.dirname(__file__), "..", "api.py"), encoding="utf-8") as f:
        src = f.read()
    assert 'already_composite = bool(frame) and os.path.basename(frame).startswith("seg_")' in src
    assert "role_portrait = portraits.get(msr_role) or frame or source" in src
    assert "composite = await _run_flux2_segment_frame(" in src


def test_msr_flf_dims_from_frame(tmp_path):
    import api
    from PIL import Image
    land = tmp_path / "land.png"
    port = tmp_path / "port.png"
    Image.new("RGB", (1600, 900)).save(land)
    Image.new("RGB", (900, 1600)).save(port)
    assert api._msr_flf_dims(str(land)) == (1024, 576)
    assert api._msr_flf_dims(str(port)) == (576, 1024)
    # both edges are /32 multiples
    for w, h in (api._msr_flf_dims(str(land)), api._msr_flf_dims(str(port))):
        assert w % 32 == 0 and h % 32 == 0


def test_run_dir_numbering(tmp_path, monkeypatch):
    import api
    from datetime import datetime
    monkeypatch.chdir(tmp_path)
    a = api._init_run_dir()
    assert api._outputs_dir() == a
    b = api._init_run_dir()
    c = api._init_run_dir()
    assert [os.path.basename(x) for x in (a, b, c)] == ["01", "02", "03"]
    day = os.path.join("outputs", datetime.now().strftime("%Y-%m-%d"))
    assert sorted(os.listdir(day)) == ["01", "02", "03"]
    # after the last init, the run-scoped dir is the current one
    assert api._outputs_dir() == c
