"""Stage P (2D) — Flux2 Klein M-I Edit GGUF loader swap.

The fp8mixed Flux2-Klein build drifts the subject's ethnicity on edit; the Q8_0
GGUF holds identity. _flux2_use_gguf() rewires the UNETLoader (node 61) to the
GGUF loader at queue-time, leaving the static workflow JSON pristine. These
tests verify the rewire targets the right node, keeps the CFGGuider edge valid,
and is idempotent.
"""

from __future__ import annotations

import copy
import json
import os

import api
from api import _FLUX2_MIEDIT, _flux2_use_gguf, _strip_flux2_miedit_unused


def _load_wf() -> dict:
    path = os.path.join(
        api.CONFIG.get("workflows_dir", "./Workflows"), _FLUX2_MIEDIT["workflow"]
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_gguf_swap_rewires_unet_loader() -> None:
    wf = _flux2_use_gguf(_strip_flux2_miedit_unused(_load_wf()))
    nid = _FLUX2_MIEDIT["unet_loader"]
    assert wf[nid]["class_type"] == "UnetLoaderGGUF"
    assert wf[nid]["inputs"] == {"unet_name": _FLUX2_MIEDIT["gguf_unet"]}


def test_gguf_swap_keeps_cfgguider_edge() -> None:
    """CFGGuider must still read its model from the (now-GGUF) loader node."""
    wf = _flux2_use_gguf(_strip_flux2_miedit_unused(_load_wf()))
    # node 60 = CFGGuider in pipeline 2; model input points at the loader node.
    assert wf["60"]["inputs"]["model"] == [_FLUX2_MIEDIT["unet_loader"], 0]


def test_gguf_swap_idempotent() -> None:
    once = _flux2_use_gguf(_strip_flux2_miedit_unused(_load_wf()))
    twice = _flux2_use_gguf(copy.deepcopy(once))
    nid = _FLUX2_MIEDIT["unet_loader"]
    assert twice[nid]["class_type"] == "UnetLoaderGGUF"
    assert twice[nid]["inputs"] == {"unet_name": _FLUX2_MIEDIT["gguf_unet"]}


def test_primary_ref_node_drives_dimensions() -> None:
    """Stage P invariant: load_img_a (background slot) must be the node whose
    size feeds GetImageSize -> the latent, so the background sets output dims."""
    wf = _strip_flux2_miedit_unused(_load_wf())
    a = _FLUX2_MIEDIT["load_img_a"]
    # GetImageSize 22 reads a scaled copy of node a; trace 22 -> 35 -> a.
    assert wf["22"]["class_type"] == "GetImageSize"
    scaled = wf["22"]["inputs"]["image"][0]          # node 35 (ImageScaleToTotalPixels)
    assert wf[scaled]["inputs"]["image"][0] == a      # node 35 reads node 33
