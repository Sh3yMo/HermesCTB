"""Manual MSR workflow smoke test against a running ComfyUI.

Builds dummy reference images (first frame, character sheet, background),
patches "LTX2.3 - IA2V-PromptRelay-MSR" via the SAME comfyui.py helpers the
pipeline uses, queues it, and polls history until completion. Exercises:
LiconMSR pseudo-video build, IC-LoRA guide on the GGUF model, audio encode,
two-pass sampling with the rewired CropGuides->Upsampler path.

Usage: python scripts/msr_queue_test.py [--seconds 4] [--timeout 1800]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import struct
import sys
import time
import wave

import requests
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfyui import (  # noqa: E402
    build_smart_prompt,
    inject_input_audio,
    inject_input_image,
    inject_msr_images,
    inject_prompt,
    inject_segment_duration,
    randomize_seeds,
    has_msr_nodes,
)

URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
WF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "Workflows", "LTX2.3 - IA2V-PromptRelay-MSR.json")


def _upload(name: str, blob: bytes, mime: str) -> str:
    r = requests.post(f"{URL}/upload/image",
                      files={"image": (name, blob, mime)},
                      data={"overwrite": "true"}, timeout=30)
    r.raise_for_status()
    return r.json()["name"]


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _dummy_sheet() -> Image.Image:
    """2x2 grid of labelled colour cells — stands in for a character sheet."""
    sheet = Image.new("RGB", (1040, 1040), "white")
    d = ImageDraw.Draw(sheet)
    colours = [(180, 60, 60), (60, 140, 60), (60, 60, 180), (160, 140, 40)]
    labels = ["front", "back", "side", "face"]
    for i, (c, lbl) in enumerate(zip(colours, labels)):
        x, y = 16 + (i % 2) * 512, 16 + (i // 2) * 512
        d.rectangle([x, y, x + 496, y + 496], fill=c)
        d.text((x + 20, y + 20), lbl, fill="white")
    return sheet


def _silent_wav(seconds: float, rate: int = 44100) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(seconds * rate)
        # quiet 220 Hz hum instead of pure silence so the audio VAE has signal
        frames = b"".join(
            struct.pack("<h", int(800 * math.sin(2 * math.pi * 220 * t / rate)))
            for t in range(n)
        )
        w.writeframes(frames)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    with open(WF_PATH, encoding="utf-8") as f:
        wf = json.load(f)
    assert has_msr_nodes(wf), "MSR workflow has no LiconMSR node?!"

    first = _upload("msr_test_first.png", _png(Image.new("RGB", (768, 432), (35, 50, 80))), "image/png")
    sheet = _upload("msr_test_sheet.png", _png(_dummy_sheet()), "image/png")
    bg = _upload("msr_test_bg.png", _png(Image.new("RGB", (768, 432), (20, 20, 28))), "image/png")
    audio = _upload("msr_test_audio.wav", _silent_wav(args.seconds), "audio/wav")
    print(f"uploaded: first={first} sheet={sheet} bg={bg} audio={audio}")

    wf = inject_prompt(
        wf,
        build_smart_prompt(["A figure stands still in a dim studio, soft haze."]),
        global_prompt=("static camera, moody studio lighting. References: "
                       "[1] the performer (character turnaround sheet). "
                       "Background reference: plain dark studio."),
    )
    wf = inject_input_image(wf, first)
    wf = inject_input_audio(wf, audio)
    wf = inject_msr_images(wf, [sheet], bg)
    wf = inject_segment_duration(wf, args.seconds)
    wf = randomize_seeds(wf)

    r = requests.post(f"{URL}/prompt", json={"prompt": wf}, timeout=60)
    body = r.json()
    if r.status_code != 200 or body.get("node_errors"):
        print(f"QUEUE FAILED (HTTP {r.status_code}):")
        print(json.dumps(body, indent=2)[:4000])
        return 1
    pid = body["prompt_id"]
    print(f"queued OK, prompt_id={pid}")

    t0 = time.time()
    while time.time() - t0 < args.timeout:
        time.sleep(10)
        h = requests.get(f"{URL}/history/{pid}", timeout=30).json()
        entry = h.get(pid)
        if not entry:
            continue
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            print("EXECUTION ERROR:")
            print(json.dumps(status, indent=2)[:6000])
            return 2
        if status.get("completed"):
            outs = {
                nid: [o.get("filename") for o in (node_out.get("images", []) + node_out.get("gifs", []))]
                for nid, node_out in entry.get("outputs", {}).items()
            }
            print(f"COMPLETED in {time.time() - t0:.0f}s, outputs: {json.dumps(outs)}")
            return 0
    print("TIMEOUT waiting for completion")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
