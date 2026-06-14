"""Regenerate a CLEAN MSR character sheet from the gold master portrait.

The verify-test sheet (`sheet_verify_2x2.png`) was broken: it stitched raw
F2K9B-MCA *batch* variants (`mca_frames/frame_fb9e05d4_*`) via a naive
`glob[:4]`, which gave an inconsistent grid — bikini recoloured (front/back
red+silver, side gold) and TWO side views instead of side + face close-up.

This script mirrors the PRODUCTION sheet build (api.py:1633-1664) standalone:
the gold portrait IS the full-body front view; MCA renders only the 3 missing
views (back / side / face-front), pinned to a concrete GOLD appearance so it
neither recolours the garment nor invents per-angle details. The four cells
(front-gold + back + side + face) compose into one seamless 2x2 sheet.

ComfyUI must be running (start via supervisor /restart on :8787).

Run:  py -3 scripts/build_msr_sheet.py
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import api  # noqa: E402  (provides _run_mca_variants, MSR_PORTRAIT_ASPECT)
from msr_refs import build_view_prompts, compose_character_sheet  # noqa: E402

# Gold master portrait (full-body front, gold bikini) — the original color
# before the MCA batch drifted it.
PORTRAIT = ROOT / "outputs" / "2026-06-13" / "ComfyUI_temp_tiepv_00001_.png"
OUT_SHEET = ROOT / "outputs" / "2026-06-13" / "msr_refs" / "sheet_gold_v2.png"

# Concrete appearance pinned into every MCA view so back/side/face stay the
# SAME person in the SAME gold look (no recolor, no phantom details).
APPEARANCE = (
    "athletic woman, tan skin, long dark brown wavy hair, "
    "metallic GOLD bikini (gold triangle top and gold bottoms), "
    "black high-heel shoes, plain neutral grey studio background"
)


async def main() -> int:
    if not PORTRAIT.exists():
        print(f"[sheet] gold portrait missing: {PORTRAIT}")
        return 1

    view_prompts = build_view_prompts(style_descriptor="", appearance_desc=APPEARANCE)
    print(f"[sheet] generating {len(view_prompts)} MCA views (back/side/face) from gold portrait ...")
    views = await api._run_mca_variants(
        str(PORTRAIT), view_prompts, aspect_ratio=api.MSR_PORTRAIT_ASPECT,
    )
    print(f"[sheet] MCA views: {[Path(v).name for v in views]}")
    if len(views) < 3:
        print(f"[sheet] WARNING only {len(views)} views returned (expected 3)")

    sheet = compose_character_sheet([str(PORTRAIT)] + views, str(OUT_SHEET))
    from PIL import Image
    with Image.open(sheet) as im:
        print(f"[sheet] DONE -> {sheet}  size={im.size} (expect 1024x1536, all-gold, front/back/side/face)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
