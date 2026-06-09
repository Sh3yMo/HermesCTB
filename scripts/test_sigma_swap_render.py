"""Single-segment render test for ManualSigmas → BasicScheduler swap.

Loads patched LTX2.3 IA2V-PromptRelay workflow, injects test image+audio+prompt,
queues to ComfyUI, waits for completion. Reports prompt_id + output path.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import comfyui
from config_loader import load_config

CFG = load_config()
comfyui.init_config(CFG)

WORKFLOW = "Workflows/LTX2.3 - IA2V-PromptRelay.json"
TEST_IMAGE = "hermes_sigma_test.png"
TEST_AUDIO = "hermes_sigma_test.mp3"

SMART_PROMPT = (
    "Male singer performing a guitar solo on an outdoor rooftop stage at dusk. "
    "He wears a shimmering gold metallic blazer, black silk shirt, black trousers. "
    "No sunglasses, full face clearly visible with sharp facial features and detailed eyes. "
    "Camera slowly pushes in from a full-body wide shot to a mid-shot framing his chest and face, "
    "smooth dolly forward, steady cinematic motion. He plays the electric guitar with intensity, "
    "subtle head movement, expressive face, eyes focused. Crisp skin texture, photoreal detail."
)
GLOBAL_PROMPT = (
    "Cinematic music video, golden hour to blue hour rooftop, "
    "high detail, sharp focus, photoreal, 24fps, professional film lighting."
)

DURATION_SECONDS = 5.0


async def main() -> int:
    wf_path = Path(WORKFLOW)
    if not wf_path.exists():
        print(f"[ERR] workflow not found: {wf_path}", file=sys.stderr)
        return 2
    wf = json.loads(wf_path.read_text(encoding="utf-8"))

    comfyui.inject_input_image(wf, TEST_IMAGE)
    comfyui.inject_input_audio(wf, TEST_AUDIO)

    if "196" in wf:
        wf["196"]["inputs"]["Xi"] = int(DURATION_SECONDS)
        wf["196"]["inputs"]["Xf"] = int(DURATION_SECONDS)

    if "1700" in wf:
        wf["1700"]["inputs"]["smart_prompt"] = SMART_PROMPT
        wf["1700"]["inputs"]["global_prompt"] = GLOBAL_PROMPT

    comfyui.randomize_seeds(wf)
    comfyui.preprocess_workflow(wf)

    print(f"[INFO] queueing patched workflow ({wf_path.name})")
    print(f"[INFO] image={TEST_IMAGE} audio={TEST_AUDIO} duration={DURATION_SECONDS}s")

    sigma_p1 = wf.get("759:1065", {})
    sigma_p2 = wf.get("759:1069", {})
    print(f"[VERIFY] Pass1 sigmas node: class={sigma_p1.get('class_type')} inputs={sigma_p1.get('inputs')}")
    print(f"[VERIFY] Pass2 sigmas node: class={sigma_p2.get('class_type')} inputs={sigma_p2.get('inputs')}")
    print(f"[VERIFY] 211_p1 LoRA chain: {wf.get('211_p1', {}).get('inputs', {}).get('lora_2')}")
    print(f"[VERIFY] Pass1 CFG: {wf.get('759:1052', {}).get('inputs', {}).get('cfg')}")

    alive = await comfyui.probe_comfyui_alive()
    if not alive:
        print("[ERR] ComfyUI not reachable", file=sys.stderr)
        return 3

    t0 = time.time()
    try:
        prompt_id, info = await comfyui.queue_and_wait_with_recovery(wf, job_timeout=1200)
    except Exception as exc:
        print(f"[ERR] queue/wait failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    elapsed = time.time() - t0

    print(f"[OK] prompt_id={prompt_id} elapsed={elapsed:.1f}s")
    print(f"[OK] output info: {json.dumps(info, default=str)[:500]}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
