"""Sequential 4-variant test for Option B (R3G4L) refinement.

Pass 1 stays: LTXVScheduler 14 steps, distill on.
Pass 2 varies across V1-V4.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import comfyui
from config_loader import load_config

comfyui.init_config(load_config())

WORKFLOW = Path("Workflows/LTX2.3 - IA2V-PromptRelay.json")

VARIANTS = [
    {"name": "V1_6steps_0.75_0.45",  "steps": 6, "rescale_start": 0.75, "denoise": 0.45},
    {"name": "V2_3steps_0.60_0.45",  "steps": 3, "rescale_start": 0.60, "denoise": 0.45},
    {"name": "V3_3steps_0.75_0.55",  "steps": 3, "rescale_start": 0.75, "denoise": 0.55},
    {"name": "V4_6steps_0.60_0.50",  "steps": 6, "rescale_start": 0.60, "denoise": 0.50},
]

SMART_PROMPT = (
    "Three female backup dancers performing on rooftop stage at dusk. "
    "They wear matching silver sequined outfits with shimmering sequins. "
    "Sharp facial features clearly visible, detailed eyes, expressive faces. "
    "Camera slowly pushes in from a full-body wide shot framing all three dancers, "
    "dollying forward to a tighter mid-shot focusing on their faces and upper bodies. "
    "Smooth cinematic camera motion. They execute synchronized sharp-edged power poses, "
    "energetic stomp choreography. Crisp skin texture, photoreal detail, professional studio lighting on faces."
)
GLOBAL_PROMPT = (
    "Cinematic music video, dusk rooftop stage, high detail, sharp focus, "
    "photoreal, 24fps, professional film lighting, vibrant color grading."
)


def patch_variant(wf: dict, variant: dict) -> None:
    wf["759:1069_base"]["inputs"]["steps"] = variant["steps"]
    wf["759:1069_base"]["inputs"]["denoise"] = variant["denoise"]
    wf["759:1069"]["inputs"]["start"] = variant["rescale_start"]


async def run_variant(variant: dict, results: list) -> None:
    wf = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    patch_variant(wf, variant)
    comfyui.inject_input_image(wf, "hermes_sigma_test.png")
    comfyui.inject_input_audio(wf, "hermes_sigma_test.mp3")
    wf["196"]["inputs"]["Xi"] = 6
    wf["196"]["inputs"]["Xf"] = 6
    wf["1700"]["inputs"]["smart_prompt"] = SMART_PROMPT
    wf["1700"]["inputs"]["global_prompt"] = GLOBAL_PROMPT
    comfyui.randomize_seeds(wf)
    comfyui.preprocess_workflow(wf)

    print(f"[{variant['name']}] queueing P2 steps={variant['steps']} rescale={variant['rescale_start']} denoise={variant['denoise']}")
    t0 = time.time()
    pid, info = await comfyui.queue_and_wait_with_recovery(wf, job_timeout=3600)
    elapsed = time.time() - t0

    # Resolve output filename
    outfile = None
    for v in (info or {}).values():
        if isinstance(v, dict) and "gifs" in v:
            for g in v["gifs"]:
                if g.get("filename", "").endswith(".mp4"):
                    outfile = g["filename"]
                    break

    entry = {"variant": variant["name"], "prompt_id": pid, "elapsed_s": round(elapsed, 1), "output": outfile}
    results.append(entry)
    print(f"[{variant['name']}] DONE prompt_id={pid} elapsed={elapsed:.1f}s output={outfile}")


async def main() -> int:
    results = []
    for variant in VARIANTS:
        try:
            await run_variant(variant, results)
        except Exception as exc:
            print(f"[{variant['name']}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            results.append({"variant": variant["name"], "error": str(exc)})

    print("\n=== SUMMARY ===")
    for r in results:
        print(json.dumps(r))

    Path("outputs/option_b_variants_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
