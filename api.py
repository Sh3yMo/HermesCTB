"""HermesCTB API — exposes CTB generation capabilities as REST endpoints for Hermes."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from audio_enhancer import AudioEnhancer, AudioSettings
from config_loader import load_config
from comfyui import (
    download_output_to_local,
    get_file_bytes,
    inject_input_audio,
    inject_input_image,
    inject_prompt,
    init_config,
    make_comfy_caller,
    queue_prompt_async,
    upload_file_to_comfy,
    wait_for_completion_async,
    load_workflow,
    COMFYUI_URL,
)
from music_video_pipeline import (
    MVSession,
    MusicVideoPrompter,
    Segment,
    assemble_video,
    segment_audio,
)
from short_film_pipeline import FilmPipeline
from workflow_registry import load_registry

# ---------------------------------------------------------------------------
# Global state (set in lifespan)
# ---------------------------------------------------------------------------
CONFIG: dict = {}
AUDIO_ENHANCER: Optional[AudioEnhancer] = None
MV_PROMPTER: Optional[MusicVideoPrompter] = None
REGISTRY: dict = {}
JOBS: dict[str, dict] = {}  # job_id → {status, output_path, error}


def _outputs_dir() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    d = os.path.join("outputs", today)
    os.makedirs(d, exist_ok=True)
    return d


def _new_job(job_id: str | None = None) -> str:
    jid = job_id or str(uuid.uuid4())
    JOBS[jid] = {"status": "pending", "output_path": None, "error": None}
    return jid


def _job_running(jid: str) -> None:
    JOBS[jid]["status"] = "running"


def _job_done(jid: str, output_path: str) -> None:
    JOBS[jid]["status"] = "completed"
    JOBS[jid]["output_path"] = output_path


def _job_failed(jid: str, error: str) -> None:
    JOBS[jid]["status"] = "failed"
    JOBS[jid]["error"] = error


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global CONFIG, AUDIO_ENHANCER, MV_PROMPTER, REGISTRY
    CONFIG = load_config("config.json")
    # Docker env var override (host.docker.internal in bridge networks)
    comfyui_env = os.getenv("COMFYUI_URL")
    if comfyui_env:
        CONFIG["comfyui_url"] = comfyui_env
    init_config(CONFIG)
    AUDIO_ENHANCER = AudioEnhancer(CONFIG)
    MV_PROMPTER = MusicVideoPrompter(CONFIG)
    REGISTRY = load_registry()
    os.makedirs("outputs", exist_ok=True)
    print("HermesCTB API ready")
    yield


app = FastAPI(title="HermesCTB API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health & Workflow listing
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "comfyui_url": COMFYUI_URL}


@app.get("/workflows")
async def list_workflows():
    categories = REGISTRY.get("categories", {})
    result = {}
    for cat_id, cat in categories.items():
        result[cat_id] = {
            "name": cat.get("name", cat_id),
            "description": cat.get("description", ""),
            "workflows": cat.get("workflows", []),
            "needs_input_image": cat.get("needs_input_image", False),
            "needs_input_video": cat.get("needs_input_video", False),
            "needs_input_audio": cat.get("needs_input_audio", False),
        }
    return result


# ---------------------------------------------------------------------------
# Status & Output
# ---------------------------------------------------------------------------

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    # Check internal jobs (complex pipelines)
    if job_id in JOBS:
        return JOBS[job_id]
    # Fall back to ComfyUI history (direct workflow jobs)
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{COMFYUI_URL}/history/{job_id}", timeout=10)
            history = resp.json()
        if job_id in history:
            outputs = history[job_id].get("outputs", {})
            return {"status": "completed" if outputs else "running", "prompt_id": job_id}
        # Check queue
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{COMFYUI_URL}/queue", timeout=5)
            queue = resp.json()
        running = [item[1] for item in queue.get("queue_running", [])]
        pending = [item[1] for item in queue.get("queue_pending", [])]
        if job_id in running:
            return {"status": "running", "prompt_id": job_id}
        if job_id in pending:
            return {"status": "pending", "prompt_id": job_id}
        return {"status": "unknown", "prompt_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/output/{job_id}")
async def get_output(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=202, detail=f"Job status: {job['status']}")
    path = job["output_path"]
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Direct workflow dispatch helpers
# ---------------------------------------------------------------------------

async def _run_direct_workflow(
    workflow_name: str,
    prompt: str,
    input_image_bytes: Optional[bytes] = None,
    input_image_name: Optional[str] = None,
    input_audio_bytes: Optional[bytes] = None,
    input_audio_name: Optional[str] = None,
    input_video_bytes: Optional[bytes] = None,
    input_video_name: Optional[str] = None,
) -> dict:
    workflow = load_workflow(workflow_name)
    if prompt:
        workflow = inject_prompt(workflow, prompt)
    if input_image_bytes:
        uploaded = await upload_file_to_comfy(input_image_bytes, input_image_name or "input.png", "image")
        workflow = inject_input_image(workflow, uploaded)
    if input_audio_bytes:
        uploaded = await upload_file_to_comfy(input_audio_bytes, input_audio_name or "input.wav", "audio")
        workflow = inject_input_audio(workflow, uploaded)
    if input_video_bytes:
        uploaded = await upload_file_to_comfy(input_video_bytes, input_video_name or "input.mp4", "image")
        workflow = inject_input_image(workflow, uploaded)
    prompt_id = await queue_prompt_async(workflow)
    return {"job_id": prompt_id, "prompt_id": prompt_id}


# ---------------------------------------------------------------------------
# generate/image
# ---------------------------------------------------------------------------

@app.post("/generate/image")
async def generate_image(
    workflow_id: str = Form(...),
    prompt: str = Form(...),
    input_image: Optional[UploadFile] = File(None),
):
    img_bytes = await input_image.read() if input_image else None
    return await _run_direct_workflow(
        workflow_id, prompt,
        input_image_bytes=img_bytes,
        input_image_name=input_image.filename if input_image else None,
    )


# ---------------------------------------------------------------------------
# generate/video
# ---------------------------------------------------------------------------

@app.post("/generate/video")
async def generate_video(
    workflow_id: str = Form(...),
    prompt: str = Form(""),
    input_image: Optional[UploadFile] = File(None),
    input_video: Optional[UploadFile] = File(None),
    input_audio: Optional[UploadFile] = File(None),
):
    img_bytes = await input_image.read() if input_image else None
    vid_bytes = await input_video.read() if input_video else None
    aud_bytes = await input_audio.read() if input_audio else None
    return await _run_direct_workflow(
        workflow_id, prompt,
        input_image_bytes=img_bytes,
        input_image_name=input_image.filename if input_image else None,
        input_audio_bytes=aud_bytes,
        input_audio_name=input_audio.filename if input_audio else None,
        input_video_bytes=vid_bytes,
        input_video_name=input_video.filename if input_video else None,
    )


# ---------------------------------------------------------------------------
# generate/music  (ACEStep)
# ---------------------------------------------------------------------------

async def _run_music_generation(jid: str, settings_dict: dict, idea: str) -> None:
    _job_running(jid)
    try:
        settings = AudioSettings.from_dict(settings_dict)
        enriched = await AUDIO_ENHANCER.generate_song(settings, idea)
        workflow = load_workflow("ACE-Step 1.5")
        workflow = AUDIO_ENHANCER.inject_audio_settings(workflow, enriched)
        prompt_id = await queue_prompt_async(workflow)
        output_info = await wait_for_completion_async(prompt_id)
        out_path = await download_output_to_local(output_info, _outputs_dir())
        _job_done(jid, out_path)
    except Exception as e:
        _job_failed(jid, str(e))
        raise


@app.post("/generate/music")
async def generate_music(
    mode: str = Form("vocal"),
    caption: str = Form(...),
    idea: str = Form(""),
    bpm: Optional[int] = Form(None),
    key: Optional[str] = Form(None),
    duration: Optional[int] = Form(None),
    lyrics: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None),
):
    settings_dict: dict[str, Any] = {"caption": caption, "mode": mode}
    if bpm is not None:
        settings_dict["bpm"] = bpm
    if key:
        settings_dict["key"] = key
    if duration:
        settings_dict["duration"] = duration
    if lyrics:
        settings_dict["lyrics"] = lyrics
    if reference_audio:
        ref_bytes = await reference_audio.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(reference_audio.filename)[1])
        tmp.write(ref_bytes)
        tmp.flush()
        settings_dict["reference_audio_path"] = tmp.name

    jid = _new_job()
    asyncio.create_task(_run_music_generation(jid, settings_dict, idea or caption))
    return {"job_id": jid}


# ---------------------------------------------------------------------------
# generate/music-video
# ---------------------------------------------------------------------------

async def _run_music_video(
    jid: str,
    audio_path: str,
    video_workflow_id: str,
    theme: str,
    crossfade_duration: float,
    tmp_dir: str,
) -> None:
    _job_running(jid)
    try:
        seg_dir = os.path.join(tmp_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)
        segments = segment_audio(audio_path, seg_dir)
        prompts = await MV_PROMPTER.generate_segment_prompts(segments, theme)

        out_dir = _outputs_dir()
        for i, seg in enumerate(segments):
            prompt_text = prompts[i] if i < len(prompts) else theme
            seg_workflow = load_workflow(video_workflow_id)
            seg_workflow = inject_prompt(seg_workflow, prompt_text)
            if seg.audio_clip and os.path.exists(seg.audio_clip):
                with open(seg.audio_clip, "rb") as af:
                    aud_bytes = af.read()
                uploaded_audio = await upload_file_to_comfy(aud_bytes, os.path.basename(seg.audio_clip), "audio")
                seg_workflow = inject_input_audio(seg_workflow, uploaded_audio)
            prompt_id = await queue_prompt_async(seg_workflow)
            output_info = await wait_for_completion_async(prompt_id)
            seg_video_path = await download_output_to_local(output_info, os.path.join(out_dir, "seg_videos"))
            seg.video_clip = seg_video_path
            seg.status = "completed"

        session = MVSession(
            audio_path=audio_path,
            segments=segments,
            crossfade_duration=crossfade_duration,
        )
        final_path = os.path.join(out_dir, f"music_video_{jid[:8]}.mp4")
        assemble_video(session, final_path)
        _job_done(jid, final_path)
    except Exception as e:
        _job_failed(jid, str(e))
        raise


@app.post("/generate/music-video")
async def generate_music_video(
    audio: UploadFile = File(...),
    video_workflow_id: str = Form("LTX2.3 - IA2V"),
    theme: str = Form(...),
    crossfade_duration: float = Form(0.0),
):
    tmp_dir = tempfile.mkdtemp(prefix="ctb_mv_")
    audio_path = os.path.join(tmp_dir, audio.filename or "audio.wav")
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    jid = _new_job()
    asyncio.create_task(_run_music_video(jid, audio_path, video_workflow_id, theme, crossfade_duration, tmp_dir))
    return {"job_id": jid}


# ---------------------------------------------------------------------------
# generate/short-film
# ---------------------------------------------------------------------------

async def _run_short_film(
    jid: str,
    story: str,
    style: str,
    target_duration: float,
    director_mode: str,
    audio_path: Optional[str],
) -> None:
    _job_running(jid)
    try:
        out_dir = _outputs_dir()
        comfy_caller = make_comfy_caller(out_dir)
        pipeline = FilmPipeline(CONFIG, comfy_caller=comfy_caller)
        result_path = await pipeline.run(
            story=story,
            target_duration=target_duration,
            style=style,
            director_mode=director_mode,
            audio_path=audio_path,
        )
        _job_done(jid, result_path)
    except Exception as e:
        _job_failed(jid, str(e))
        raise


@app.post("/generate/short-film")
async def generate_short_film(
    premise: str = Form(...),
    style: str = Form("cinematic"),
    duration: float = Form(60.0),
    director_mode: str = Form("auto"),
    audio: Optional[UploadFile] = File(None),
):
    audio_path = None
    if audio:
        tmp_dir = tempfile.mkdtemp(prefix="ctb_sf_")
        audio_path = os.path.join(tmp_dir, audio.filename or "audio.wav")
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

    jid = _new_job()
    asyncio.create_task(_run_short_film(jid, premise, style, duration, director_mode, audio_path))
    return {"job_id": jid}


# ---------------------------------------------------------------------------
# enhance/prompt
# ---------------------------------------------------------------------------

@app.post("/enhance/prompt")
async def enhance_prompt(
    text: str = Form(...),
    workflow_id: Optional[str] = Form(None),
):
    from prompt_enhancer import PromptEnhancer
    enhancer = PromptEnhancer(CONFIG)
    enhanced = await enhancer.build_prompt(text, workflow_name=workflow_id)
    return {"original": text, "enhanced": enhanced}
