"""ComfyUI utility functions for HermesCTB — extracted from CTB/ctb.py."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

# ---------------------------------------------------------------------------
# Globals — overridden by init_config()
# ---------------------------------------------------------------------------
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
WORKFLOWS_DIR = "./Workflows"
COMFY_VIEW_TIMEOUT = 60
COMFY_VIEW_RETRIES = 3
COMFY_JOB_TIMEOUT = 2400
COMFY_HISTORY_RETRIES = 60
COMFY_HISTORY_RETRY_DELAY = 3


def init_config(config: dict) -> None:
    global COMFYUI_URL, WORKFLOWS_DIR, COMFY_VIEW_TIMEOUT, COMFY_VIEW_RETRIES
    global COMFY_JOB_TIMEOUT, COMFY_HISTORY_RETRIES, COMFY_HISTORY_RETRY_DELAY
    COMFYUI_URL = config.get("comfyui_url", COMFYUI_URL)
    WORKFLOWS_DIR = config.get("workflows_dir", WORKFLOWS_DIR)
    rv = config.get("comfy_recovery", {})
    COMFY_JOB_TIMEOUT = rv.get("job_timeout_seconds", COMFY_JOB_TIMEOUT)
    COMFY_HISTORY_RETRIES = rv.get("history_retry_attempts", COMFY_HISTORY_RETRIES)
    COMFY_HISTORY_RETRY_DELAY = rv.get("history_retry_delay_seconds", COMFY_HISTORY_RETRY_DELAY)
    COMFY_VIEW_TIMEOUT = config.get("comfy_view_timeout_seconds", COMFY_VIEW_TIMEOUT)
    COMFY_VIEW_RETRIES = config.get("comfy_view_retries", COMFY_VIEW_RETRIES)


# ---------------------------------------------------------------------------
# Workflow lookup
# ---------------------------------------------------------------------------

def get_workflow_path_by_name(name: str) -> Optional[str]:
    wf_dir = Path(WORKFLOWS_DIR)
    candidate = wf_dir / f"{name}.json"
    if candidate.exists():
        return str(candidate)
    # fuzzy: case-insensitive
    for p in wf_dir.glob("*.json"):
        if p.stem.lower() == name.lower():
            return str(p)
    return None


def load_workflow(name: str) -> dict:
    path = get_workflow_path_by_name(name)
    if path is None:
        raise ValueError(f"Workflow not found: {name!r}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Inject helpers (ported from CTB/ctb.py)
# ---------------------------------------------------------------------------

def inject_prompt(workflow: dict, prompt_text: str) -> dict:
    text_encode_nodes = []
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        if class_type == "CLIPTextEncode":
            if "inputs" in node and "text" in node["inputs"]:
                text_encode_nodes.append({
                    "id": node_id,
                    "title": node.get("_meta", {}).get("title", ""),
                    "node": node,
                    "input_field": "text",
                })
        elif "TextEncodeQwen" in class_type:
            if "inputs" in node and "prompt" in node["inputs"]:
                text_encode_nodes.append({
                    "id": node_id,
                    "title": node.get("_meta", {}).get("title", ""),
                    "node": node,
                    "input_field": "prompt",
                })
    if not text_encode_nodes:
        return workflow
    positive_nodes = [n for n in text_encode_nodes if "negative" not in n["title"].lower()] or [text_encode_nodes[0]]
    for n in positive_nodes:
        n["node"]["inputs"][n["input_field"]] = prompt_text
    return workflow


def inject_input_image(workflow: dict, image_filename: str) -> dict:
    nodes = []
    for node_id, node in workflow.items():
        if node.get("class_type") == "LoadImage" and "inputs" in node and "image" in node["inputs"]:
            nodes.append({"id": node_id, "title": node.get("_meta", {}).get("title", ""), "node": node})
    if not nodes:
        return workflow
    target = next((n for n in nodes if "input" in n["title"].lower()), nodes[0])
    target["node"]["inputs"]["image"] = image_filename
    return workflow


def inject_input_audio(workflow: dict, audio_filename: str) -> dict:
    for node_id, node in workflow.items():
        if node.get("class_type") == "LoadAudio":
            node["inputs"]["audio"] = audio_filename
            break
    return workflow


def inject_segment_duration(workflow: dict, duration_seconds: float, max_duration: float = 30.0) -> dict:
    clamped = min(duration_seconds, max_duration)
    LTX_FPS = 24  # LTX2.3 (4.2) renders at 24 fps (verified via ffprobe on real output)
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        if ct in ("ImpactInt", "PrimitiveNode"):
            title = node.get("_meta", {}).get("title", "").lower()
            if "duration" in title or "length" in title or "frames" in title:
                if "inputs" in node and "value" in node["inputs"]:
                    if "frames" in title:
                        node["inputs"]["value"] = max(1, int(clamped * 24))
                    else:
                        node["inputs"]["value"] = clamped
        elif ct == "EmptyLTXVLatentVideo":
            node.setdefault("inputs", {})
            node["inputs"]["length"] = max(1, int(clamped * LTX_FPS))
        elif ct == "LTXVEmptyLatentAudio":
            node.setdefault("inputs", {})
            node["inputs"]["frames_number"] = max(1, int(clamped * LTX_FPS))
    return workflow


_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1),
    "4:3": (4, 3), "3:4": (3, 4), "21:9": (21, 9),
    "2:3": (2, 3), "3:2": (3, 2), "9:7": (9, 7),
}

_RESOLUTION_LATENT_TYPES = (
    "EmptyFlux2LatentImage",
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyLTXVLatentVideo",
)


def inject_resolution(workflow: dict, aspect_ratio: str, megapixels: float = 0.59) -> dict:
    import math
    ratio = _ASPECT_RATIOS.get(aspect_ratio)
    if ratio is None:
        parts = aspect_ratio.split(":")
        if len(parts) == 2:
            try:
                ratio = (int(parts[0]), int(parts[1]))
            except ValueError:
                return workflow
        else:
            return workflow
    w_r, h_r = ratio
    total = int(megapixels * 1_000_000)
    height = int(math.sqrt(total * h_r / w_r) / 64) * 64
    width = int(math.sqrt(total * w_r / h_r) / 64) * 64
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") in _RESOLUTION_LATENT_TYPES:
            node.setdefault("inputs", {})
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height
    return workflow


def randomize_seeds(workflow: dict) -> dict:
    SEED_FIELDS = ("seed", "noise_seed")
    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        for field in SEED_FIELDS:
            if field not in inputs:
                continue
            val = inputs[field]
            if isinstance(val, (int, float)):
                inputs[field] = random.randint(0, 4294967295)
            elif isinstance(val, list) and len(val) == 2:
                src_id = str(val[0])
                if src_id in workflow:
                    src_inputs = workflow[src_id].get("inputs", {})
                    if "value" in src_inputs and isinstance(src_inputs["value"], (int, float)):
                        src_inputs["value"] = random.randint(0, 4294967295)
    return workflow


def preprocess_workflow(workflow: dict) -> dict:
    """Deepcopy only. Full preprocessing (ComfyMathExpression etc.) can be ported from CTB/ctb.py."""
    return copy.deepcopy(workflow)


def _normalize_image_ref(path: str) -> str:
    """Strip absolute/pasted path noise — return just the filename ComfyUI expects."""
    if path.startswith("pasted/"):
        return path
    return os.path.basename(path)


def inject_custom_loras(workflow: dict, custom_loras: list) -> dict:
    """Stub — inject LoRA overrides into workflow. Extend as needed."""
    return workflow


# ---------------------------------------------------------------------------
# Async ComfyUI API calls
# ---------------------------------------------------------------------------

async def upload_file_to_comfy(file_bytes: bytes, filename: str, file_type: str = "image") -> str:
    # ComfyUI exposes only /upload/image; it stores any uploaded file in the
    # input/ dir by filename (LoadAudio/LoadImage then read it by name).
    # There is no /upload/audio endpoint (POSTing to it returns HTTP 405).
    endpoint = "upload/image"
    content_type = "audio/wav" if file_type == "audio" else "image/png"
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{COMFYUI_URL}/{endpoint}",
            files={"image": (filename, file_bytes, content_type)},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("name", filename)


async def queue_prompt_async(workflow: dict) -> str:
    workflow = preprocess_workflow(workflow)
    workflow = randomize_seeds(workflow)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": str(uuid.uuid4())},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["prompt_id"]


async def wait_for_completion_async(prompt_id: str, timeout: Optional[int] = None) -> dict:
    max_wait = timeout or COMFY_JOB_TIMEOUT
    check_interval = 2
    import time
    start = time.time()
    async with httpx.AsyncClient() as client:
        while (time.time() - start) < max_wait:
            # Transient poll failures (ComfyUI GPU-saturated during a long
            # render → /queue slow/unreachable) must NOT kill the job. Skip
            # the tick; max_wait still bounds total wall time.
            try:
                resp = await client.get(f"{COMFYUI_URL}/queue", timeout=5)
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                elapsed = int(time.time() - start)
                print(f"Job {prompt_id}: queue poll transient error ({elapsed}s): {e!r} — retrying")
                await asyncio.sleep(check_interval)
                continue
            running = data.get("queue_running", [])
            pending = data.get("queue_pending", [])
            if not any(item[1] == prompt_id for item in running + pending):
                break
            elapsed = int(time.time() - start)
            print(f"Job {prompt_id}: waiting... ({elapsed}s)")
            await asyncio.sleep(check_interval)
        else:
            raise TimeoutError(f"Job {prompt_id} timed out after {max_wait}s")

        for attempt in range(COMFY_HISTORY_RETRIES):
            try:
                resp = await client.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
                resp.raise_for_status()
                history = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                print(f"Job {prompt_id}: history poll transient error (attempt {attempt}): {e!r} — retrying")
                await asyncio.sleep(COMFY_HISTORY_RETRY_DELAY)
                continue
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                if outputs:
                    return _extract_output(outputs)
            await asyncio.sleep(COMFY_HISTORY_RETRY_DELAY)

    raise Exception(f"Job {prompt_id}: no outputs after {COMFY_HISTORY_RETRIES} retries")


def _extract_output(outputs: dict) -> dict:
    for node_output in outputs.values():
        for key in ("audio", "gifs", "videos"):
            if node_output.get(key):
                return node_output[key][0]
        if node_output.get("filenames"):
            return {"filename": node_output["filenames"][0], "subfolder": "", "type": "output"}
    for node_output in outputs.values():
        images = node_output.get("images", [])
        saved = [i for i in images if i.get("type") != "temp"]
        if saved:
            return saved[0]
        if images:
            return images[0]
    raise Exception("No output found in ComfyUI job history")


async def get_file_bytes(filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    for attempt in range(max(1, COMFY_VIEW_RETRIES)):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{COMFYUI_URL}/view", params=params, timeout=COMFY_VIEW_TIMEOUT)
                resp.raise_for_status()
                return resp.content
        except Exception as e:
            if attempt < COMFY_VIEW_RETRIES - 1:
                await asyncio.sleep(1.5)
            else:
                raise Exception(f"Failed to fetch '{filename}': {e}") from e
    raise Exception("get_file_bytes: unreachable")


async def download_output_to_local(output_info: dict, output_dir: str) -> str:
    filename = output_info.get("filename", "output")
    subfolder = output_info.get("subfolder", "")
    folder_type = output_info.get("type", "output")
    data = await get_file_bytes(filename, subfolder, folder_type)
    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, filename)
    with open(dest, "wb") as f:
        f.write(data)
    return dest


# ---------------------------------------------------------------------------
# comfy_caller factory (for FilmPipeline)
# ---------------------------------------------------------------------------

def make_comfy_caller(output_dir: str) -> Callable:
    """Returns an async callable(workflow_name, params) -> dict for FilmPipeline."""

    async def comfy_caller(workflow_name: str, params: dict) -> dict:
        workflow = load_workflow(workflow_name)

        # Clear stale pasted image refs
        for node in workflow.values():
            if isinstance(node, dict) and node.get("class_type") == "LoadImage":
                node.get("inputs", {})["image"] = ""

        # Inject first/last/middle frame by title keyword
        FRAME_KEYS = {
            "first_frame": ["first frame", "first_frame", " ff"],
            "last_frame": ["last frame", "last_frame", " lf"],
            "middle_frame": ["middle frame", "middle_frame", " mf"],
        }
        if params:
            for node in workflow.values():
                if isinstance(node, dict) and node.get("class_type") == "LoadImage":
                    title_lower = node.get("_meta", {}).get("title", "").lower()
                    for param_key, keywords in FRAME_KEYS.items():
                        val = params.get(param_key)
                        if val and isinstance(val, str) and any(kw in title_lower for kw in keywords):
                            node["inputs"]["image"] = _normalize_image_ref(val)
                            break

        # Node-title overrides
        if params:
            for node in workflow.values():
                if isinstance(node, dict):
                    title = node.get("_meta", {}).get("title", "")
                    if title in params:
                        override = dict(params[title])
                        for key, val in override.items():
                            if key == "image" and isinstance(val, str) and val:
                                override[key] = _normalize_image_ref(val)
                        node.get("inputs", {}).update(override)

        # LoRAs
        loras = (params or {}).get("loras", [])
        if loras:
            workflow = inject_custom_loras(workflow, loras)

        # Prompt
        if params and params.get("prompt"):
            workflow = inject_prompt(workflow, params["prompt"])

        # Audio
        if params and params.get("audio_clip"):
            audio_path = params["audio_clip"]
            if os.path.exists(audio_path):
                with open(audio_path, "rb") as af:
                    audio_bytes = af.read()
                uploaded = await upload_file_to_comfy(audio_bytes, os.path.basename(audio_path), "audio")
                workflow = inject_input_audio(workflow, uploaded)

        # Duration
        if params and params.get("duration"):
            workflow = inject_segment_duration(workflow, float(params["duration"]))

        # Queue and wait
        prompt_id = await queue_prompt_async(workflow)
        output_info = await wait_for_completion_async(prompt_id)

        # Download primary output
        os.makedirs(output_dir, exist_ok=True)
        out_path = await download_output_to_local(output_info, output_dir)
        result = {"prompt_id": prompt_id, "output_path": out_path}

        # Collect all images for first/last frame detection
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
                history = resp.json().get(prompt_id, {})
            all_images = []
            for node_out in history.get("outputs", {}).values():
                for img in node_out.get("images", []):
                    img_path = await download_output_to_local(img, output_dir)
                    all_images.append(img_path)
            if len(all_images) >= 2:
                result["first_frame"] = all_images[0]
                result["last_frame"] = all_images[1]
            elif len(all_images) == 1:
                result["first_frame"] = all_images[0]
        except Exception:
            pass

        return result

    return comfy_caller
