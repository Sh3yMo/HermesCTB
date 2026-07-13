"""Build `Workflows/LTX2.3 - FFLFA2V-MSR.json` from the stock FFLFA2V workflow.

Clones `LTX2.3 - FFLFA2V.json` (API format) and splices in the standalone
PsypmP `LTXMSRICLoRAFLF_Experimental` node, which combines MSR (cross-attention),
First Frame and Last Frame conditioning + latent planting into one node. This
REPLACES the workflow's LTXVAddGuideMulti FF/LF guide chain.

Topology (mirrors the repo v3 example, adapted to our 2-stage audio workflow):

    759:1067 LTXVConditioning ─┬─> M.positive
                               └─> Z (ConditioningZeroOut) ─> M.negative
    174 VAELoader(video) ─> M.vae
    759:1060 EmptyLTXVLatentVideo ─> M.latent
    149 (FF) ─> M.first_frame ; 786 (LF) ─> M.last_frame
    2110..2113 (MSR subjects) ─> M.msr_image_1..4 ; 2114 (bg) ─> M.msr_background

    M.positive/negative ─> 759:1052 CFGGuider (base) + 759:1074 LTXVCropGuides
    M.latent            ─> 759:1055 LTXVConcatAVLatent.video_latent

    211 Power Lora Loader ─> L (LTXICLoRALoaderModelOnly, Licon-MSR) ─> 700 ChunkFeedForward

Widget settings copied from the repo's "experimental node - settings for FLF.png":
MSR via cross-attention only (no latent injection), lock first frame, free last frame.

Run: py scripts/build_fflfa2v_msr_wf.py
"""
from __future__ import annotations

import copy
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "Workflows", "LTX2.3 - FFLFA2V.json")
DST = os.path.join(ROOT, "Workflows", "LTX2.3 - FFLFA2V-MSR.json")
DST_RELAY = os.path.join(ROOT, "Workflows", "LTX2.3 - FFLFA2V-MSR-PromptRelay.json")

# New node ids (verified free in the source workflow).
M = "2100"   # LTXMSRICLoRAFLF_Experimental
Z = "2101"   # ConditioningZeroOut (negative)
L = "2102"   # LTXICLoRALoaderModelOnly (IC-LoRA on model path)
C2 = "2103"  # fresh LTXVConditioning feeding M (clean text + frame_rate)
RELAY = "2105"  # PromptRelaySmartEncode (model-path, relay variant only)
SUBJ = ["2110", "2111", "2112", "2113"]
BG = "2114"
CLIP_LOADER = "146"  # DualCLIPLoader (clip source for the relay node)

# Existing node ids we wire to / rewire (verified present in FFLFA2V).
# NOTE: the stock 759:1067 LTXVConditioning is downstream of the old FF/LF
# LTXVAddGuideMulti chain, so its conditioning already carries planted keyframes.
# Feeding M from it would double-plant FF/LF. We instead build a fresh
# LTXVConditioning (C2) from the raw text encoders + the same frame_rate source.
COND = "759:1067"          # reference only: source of the frame_rate input
TEXT_POS = "121"           # CLIPTextEncode (positive)
TEXT_NEG = "593"           # CLIPTextEncode (negative)
VAE_VIDEO = "174"
EMPTY_LATENT = "759:1060"
FF_LOAD = "149"
LF_LOAD = "786"
GUIDER_BASE = "759:1052"   # CFGGuider (base)
CROP_GUIDES = "759:1074"   # LTXVCropGuides
CONCAT_AV = "759:1055"     # LTXVConcatAVLatent (base)
POWER_LORA = "211"
CHUNK_FF = "700"           # LTXVChunkFeedForward.model <- 211 (rewired to L)

PLACEHOLDER_IMG = "pasted/image (14).png"
LICON_MSR_LORA = "LTX-2.3-Licon-MSR-V1.safetensors"

# First-frame latent-injection strength. At 1.0 the external Flux2 first frame is
# hard-pinned at frame 0; its grade differs from the LTX decode and shows as a
# 1-frame colour flash. Lower values let the model harmonise frame 0's colour.
# Confirmed via A/B test (frame-0 YAVG delta).
MSR_FF_STRENGTH = 0.85

# MSR cross-attention strength. At 1.0 the (photoreal) reference sheet bleeds into
# mid-clip frames — the subject snaps to the standing reference pose with a rendered
# "plastic" look. Identity is independently anchored by the FF + LF attention
# (both 1.0), so we can lower MSR attention to suppress the bleed without losing
# the person. Keep msr_frame_count at 41 for identity robustness.
MSR_ATTENTION_STRENGTH = 0.6


def _loadimage(title: str) -> dict:
    return {
        "inputs": {"image": PLACEHOLDER_IMG, "upload": "image"},
        "class_type": "LoadImage",
        "_meta": {"title": title},
    }


def build() -> dict:
    with open(SRC, encoding="utf-8") as f:
        wf = json.load(f)

    for nid in (M, Z, L, C2, BG, *SUBJ):
        assert nid not in wf, f"id collision: {nid} already in source workflow"
    for nid in (COND, VAE_VIDEO, EMPTY_LATENT, FF_LOAD, LF_LOAD,
                GUIDER_BASE, CROP_GUIDES, CONCAT_AV, POWER_LORA, CHUNK_FF):
        assert nid in wf, f"expected node missing from source: {nid}"

    # IC-LoRA on the model path: 211 -> L -> 700
    wf[L] = {
        "inputs": {
            "model": [POWER_LORA, 0],
            "lora_name": LICON_MSR_LORA,
            "strength_model": 1,
        },
        "class_type": "LTXICLoRALoaderModelOnly",
        "_meta": {"title": "MSR IC-LoRA Loader [[P:MSR-FLF]]"},
    }
    wf[CHUNK_FF]["inputs"]["model"] = [L, 0]

    # Fresh LTXVConditioning from raw text — clean of the old FF/LF guide planting,
    # but reusing the stock frame_rate source so temporal positioning is preserved.
    wf[C2] = {
        "inputs": {
            "positive": [TEXT_POS, 0],
            "negative": [TEXT_NEG, 0],
            "frame_rate": wf[COND]["inputs"]["frame_rate"],
        },
        "class_type": "LTXVConditioning",
        "_meta": {"title": "MSR FLF Conditioning [[P:MSR-FLF]]"},
    }

    # Zeroed negative conditioning (keeps frame_rate metadata from C2).
    wf[Z] = {
        "inputs": {"conditioning": [C2, 0]},
        "class_type": "ConditioningZeroOut",
        "_meta": {"title": "MSR FLF Negative (zeroed) [[P:MSR-FLF]]"},
    }

    # MSR reference + background LoadImage nodes (injected by inject_msr_images).
    for i, nid in enumerate(SUBJ, start=1):
        wf[nid] = _loadimage(f"MSR Subject {i} [[P:MSR]]")
    wf[BG] = _loadimage("MSR Background [[P:MSR]]")

    # The combined MSR + FF/LF node. Widget values per the repo FLF settings PNG.
    wf[M] = {
        "inputs": {
            "positive": [C2, 0],
            "negative": [Z, 0],
            "vae": [VAE_VIDEO, 0],
            "latent": [EMPTY_LATENT, 0],
            "enable_msr_latent_injection": False,
            "enable_msr_attention": True,
            # Static default only — the pipeline overrides msr_width/height per
            # segment to match the first frame's aspect (api._msr_flf_dims).
            "msr_width": 1024,
            "msr_height": 576,
            "msr_frame_count": "41",
            "msr_image_1": [SUBJ[0], 0],
            "msr_strength_1": 1.0,
            "msr_image_2": [SUBJ[1], 0],
            "msr_strength_2": 1.0,
            "msr_image_3": [SUBJ[2], 0],
            "msr_strength_3": 1.0,
            "msr_image_4": [SUBJ[3], 0],
            "msr_strength_4": 1.0,
            "msr_background": [BG, 0],
            "msr_background_strength": 1.0,
            "msr_attention_strength": MSR_ATTENTION_STRENGTH,
            "first_frame": [FF_LOAD, 0],
            "lock_first_frame": True,
            "lock_first_frame_len": 1,
            "first_frame_strength": MSR_FF_STRENGTH,
            "first_frame_attention_strength": 1.0,
            "last_frame": [LF_LOAD, 0],
            "lock_last_frame": False,
            "last_frame_strength": 1.0,
            "last_frame_attention_strength": 1.0,
            "latent_downscale_factor": 1.0,
            "crop": "center",
            "use_tiled_encode": False,
            "tile_size": 256,
            "tile_overlap": 64,
        },
        "class_type": "LTXMSRICLoRAFLF_Experimental",
        "_meta": {"title": "MSR FF/LF [[P:MSR-FLF]]"},
    }

    # Tag the FF/LF LoadImage nodes so inject_first_frame / inject_last_frame find them.
    wf[FF_LOAD]["_meta"]["title"] = "First Frame [[P:FF]]"
    wf[LF_LOAD]["_meta"]["title"] = "Last Frame [[P:LF]]"

    # Splice M's outputs into the guider, crop-guides and AV-concat latent.
    wf[GUIDER_BASE]["inputs"]["positive"] = [M, 0]
    wf[GUIDER_BASE]["inputs"]["negative"] = [M, 1]
    wf[CROP_GUIDES]["inputs"]["positive"] = [M, 0]
    wf[CROP_GUIDES]["inputs"]["negative"] = [M, 1]
    wf[CONCAT_AV]["inputs"]["video_latent"] = [M, 2]

    return wf


def build_relay(base: dict) -> dict:
    """Variant of the FFLFA2V-MSR workflow with PromptRelay added (MSR + relay).

    Inserts a PromptRelaySmartEncode node on the MODEL path between the IC-LoRA
    loader (L) and LTXVChunkFeedForward — same wiring as the stock
    LTX2.3 - IA2V-PromptRelay-MSR.json. The relay node patches the model so the
    per-beat prompt timeline switches over the clip; the conditioning path
    (C2 -> M -> guiders) is unchanged, with the global/beat text injected by the
    pipeline's relay path (has_relay_smart_node -> build_smart_prompt). Requires
    the ComfyUI-PromptRelay (kijai) custom node — already used by the IA2V relay
    workflows."""
    wf = copy.deepcopy(base)
    assert RELAY not in wf, f"id collision: {RELAY}"
    assert CLIP_LOADER in wf, f"clip loader {CLIP_LOADER} missing"
    wf[RELAY] = {
        "inputs": {
            "global_prompt": "",
            "smart_prompt": "",
            "normalize_by_tokens": False,
            "epsilon": 0.001,
            "model": [L, 0],
            "clip": [CLIP_LOADER, 0],
            "latent": [EMPTY_LATENT, 0],
        },
        "class_type": "PromptRelaySmartEncode",
        "_meta": {"title": "Prompt Relay Encode (Smart) positive"},
    }
    wf[CHUNK_FF]["inputs"]["model"] = [RELAY, 0]
    return wf


def main() -> None:
    wf = build()
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
    print(f"wrote {DST} ({len(wf)} nodes)")
    rwf = build_relay(wf)
    with open(DST_RELAY, "w", encoding="utf-8") as f:
        json.dump(rwf, f, indent=2, ensure_ascii=False)
    print(f"wrote {DST_RELAY} ({len(rwf)} nodes)")


if __name__ == "__main__":
    main()
