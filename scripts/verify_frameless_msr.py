"""End-to-end verification of the frameless MSR path (Stage Q).

Drives ONE segment through the pipeline's real per-segment render helpers on the
frameless MSR workflow, using the existing 2026-06-13 assets:

  * composes a FRESH 2x2 seamless character sheet (Block 4) from the run's MCA
    view frames and feeds it as the MSR subject,
  * injects prompt (+MSR reference block) via PromptRelay, MSR images, and the
    segment-1 audio slice,
  * does NOT inject a start frame (frameless — Blocks 2+3),
  * queues on ComfyUI, waits, downloads the clip.

Run:  py -3 scripts/verify_frameless_msr.py
"""
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comfyui import (  # noqa: E402
    load_workflow, inject_prompt, inject_input_audio, inject_msr_images,
    inject_segment_duration, upload_file_to_comfy, queue_prompt_async,
    wait_for_completion_async, download_output_to_local, has_msr_nodes,
    has_relay_smart_node,
)
from msr_refs import build_msr_reference_block  # noqa: E402

WF = "LTX2.3 - IA2V-PromptRelay-MSR"
OUT = ROOT / "outputs" / "2026-06-13"
SEG_VIDEOS = OUT / "seg_videos"
BG = OUT / "ComfyUI_temp_rvehm_00001_.png"
SONG = OUT / "Nova Luxe - Chasing the Glow.mp4"
AUDIO_START, DURATION = 14.44, 9.86

PROMPT = (
    "Low-angle fisheye push-in shot of a female singer performing aggressively "
    "towards the camera on a sun-drenched Miami rooftop at golden hour."
)
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


async def main() -> int:
    SEG_VIDEOS.mkdir(parents=True, exist_ok=True)

    # Block 4: use the CLEAN gold sheet built by scripts/build_msr_sheet.py
    # (front-gold + back + side + face, color-consistent). The old path stitched
    # raw MCA batch variants via glob[:4] -> inconsistent grid (recolor + 2x side).
    sheet = str(OUT / "msr_refs" / "sheet_gold_v2.png")
    if not Path(sheet).exists():
        print(f"[verify] clean sheet missing: {sheet} — run scripts/build_msr_sheet.py first")
        return 1
    from PIL import Image
    with Image.open(sheet) as im:
        print(f"[verify] gold sheet: {sheet}  size={im.size}")

    # Audio slice for segment 1.
    audio = OUT / "verify_seg1_audio.wav"
    subprocess.run(
        [FFMPEG, "-y", "-ss", str(AUDIO_START), "-i", str(SONG), "-t", str(DURATION),
         "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(audio)],
        check=True, capture_output=True,
    )
    print(f"[verify] audio slice -> {audio.name}")

    wf = load_workflow(WF)
    print(f"[verify] workflow {WF!r}: nodes={len(wf)} "
          f"has_msr={has_msr_nodes(wf)} has_relay={has_relay_smart_node(wf)}")

    ref_block = build_msr_reference_block(
        ["the female lead singer in a metallic gold bikini (character turnaround "
         "sheet: full body front, back and side views, face close-up front)"],
        "rooftop terrace overlooking the Miami skyline at sunset, no people",
    )
    inject_prompt(wf, PROMPT + " " + ref_block)
    inject_segment_duration(wf, DURATION)

    sheet_name = await upload_file_to_comfy(Path(sheet).read_bytes(), Path(sheet).name, "image")
    bg_name = await upload_file_to_comfy(BG.read_bytes(), BG.name, "image")
    audio_name = await upload_file_to_comfy(audio.read_bytes(), audio.name, "audio")
    inject_msr_images(wf, [sheet_name], bg_name)
    inject_input_audio(wf, audio_name)
    # NOTE: deliberately NO inject_input_image — frameless.

    print("[verify] queuing (frameless: no start frame) ...")
    pid = await queue_prompt_async(wf)
    print(f"[verify] prompt_id={pid}, waiting ...")
    # Planting (LTXVAddGuideMulti encodet das 1024x1536-Sheet) verdoppelt ~ die
    # Renderzeit auf ~2400s — der alte 2400er-Timeout wurde knapp gerissen.
    info = await wait_for_completion_async(pid, timeout=3600)
    raw = await download_output_to_local(info, str(SEG_VIDEOS))
    dest = SEG_VIDEOS / "verify_frameless_msr.mp4"
    shutil.copyfile(raw, dest)
    print(f"[verify] DONE -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
