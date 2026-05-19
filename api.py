"""HermesCTB API — exposes CTB generation capabilities as REST endpoints for Hermes."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
import httpx
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
    inject_resolution,
    inject_segment_duration,
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
    clamp_song_duration,
    get_audio_duration,
    segment_audio,
    to_ace_language,
    _extract_audio_clip,
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
    JOBS[jid] = {"status": "pending", "output_path": None, "lyrics_path": None, "error": None, "prompt_id": None}
    return jid


def _job_running(jid: str) -> None:
    JOBS[jid]["status"] = "running"


def _job_done(jid: str, output_path: str, lyrics_path: str | None = None) -> None:
    JOBS[jid]["status"] = "completed"
    JOBS[jid]["output_path"] = output_path
    JOBS[jid]["lyrics_path"] = lyrics_path


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
    AUDIO_ENHANCER = AudioEnhancer(CONFIG["prompt_enhancer"])
    # openrouter_api_key lives under prompt_enhancer, NOT top-level — passing the
    # full CONFIG leaves api_key="" and silently kills every MV LLM call.
    MV_PROMPTER = MusicVideoPrompter(CONFIG["prompt_enhancer"])
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


@app.get("/presets")
async def list_presets():
    from prompt_enhancer import (
        DIRECTOR_PRESETS, FILMSTYLE_PRESETS, CAMERA_PRESETS,
        LIGHT_PRESETS, MOOD_PRESETS, MOTION_PRESETS, CINEMATIC_PATTERN_PRESETS,
    )
    return {
        "director":          {k: v["label"] for k, v in DIRECTOR_PRESETS.items()},
        "film_style":        {k: v["label"] for k, v in FILMSTYLE_PRESETS.items()},
        "camera":            list(CAMERA_PRESETS.keys()),
        "lighting":          list(LIGHT_PRESETS.keys()),
        "mood":              list(MOOD_PRESETS.keys()),
        "motion":            list(MOTION_PRESETS.keys()),
        "cinematic_pattern": list(CINEMATIC_PATTERN_PRESETS.keys()),
    }


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
            if outputs:
                from comfyui import download_output_to_local
                out_dir = _outputs_dir()
                local_paths = []
                for node_out in outputs.values():
                    for item in node_out.get("images", []) + node_out.get("videos", []):
                        try:
                            path = await download_output_to_local(item, out_dir)
                            local_paths.append(path)
                        except Exception:
                            pass
                return {
                    "status": "completed",
                    "prompt_id": job_id,
                    "output_path": local_paths[0] if local_paths else "",
                    "output_urls": local_paths,
                }
            return {"status": "running", "prompt_id": job_id}
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


@app.get("/outputs/{path:path}")
async def serve_output_file(path: str):
    full_path = os.path.join("outputs", path)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full_path)


# ---------------------------------------------------------------------------
# Direct workflow dispatch helpers
# ---------------------------------------------------------------------------

async def _run_direct_download_job(jid: str, prompt_id: str) -> None:
    _job_running(jid)
    try:
        output_info = await wait_for_completion_async(prompt_id)
        out_path = await download_output_to_local(output_info, _outputs_dir())
        _job_done(jid, out_path)
    except Exception as e:
        _job_failed(jid, str(e))
        raise


async def _run_direct_workflow(
    workflow_name: str,
    prompt: str,
    input_image_bytes: Optional[bytes] = None,
    input_image_name: Optional[str] = None,
    input_audio_bytes: Optional[bytes] = None,
    input_audio_name: Optional[str] = None,
    input_video_bytes: Optional[bytes] = None,
    input_video_name: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    duration: Optional[float] = None,
) -> dict:
    workflow = load_workflow(workflow_name)
    if prompt:
        workflow = inject_prompt(workflow, prompt)
    if aspect_ratio:
        workflow = inject_resolution(workflow, aspect_ratio)
    if duration is not None:
        workflow = inject_segment_duration(workflow, duration)
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
    jid = _new_job()
    JOBS[jid]["prompt_id"] = prompt_id
    asyncio.create_task(_run_direct_download_job(jid, prompt_id))
    return {"job_id": jid, "prompt_id": prompt_id}


# ---------------------------------------------------------------------------
# generate/image
# ---------------------------------------------------------------------------

@app.post("/generate/image")
async def generate_image(
    workflow_id: str = Form(...),
    prompt: str = Form(...),
    input_image: Optional[UploadFile] = File(None),
    aspect_ratio: Optional[str] = Form(None),
):
    img_bytes = await input_image.read() if input_image else None
    return await _run_direct_workflow(
        workflow_id, prompt,
        input_image_bytes=img_bytes,
        input_image_name=input_image.filename if input_image else None,
        aspect_ratio=aspect_ratio,
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
    aspect_ratio: Optional[str] = Form(None),
    duration: Optional[float] = Form(None),
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
        aspect_ratio=aspect_ratio,
        duration=duration,
    )


# ---------------------------------------------------------------------------
# generate/music  (ACEStep)
# ---------------------------------------------------------------------------

async def _generate_song(settings_dict: dict, idea: str) -> tuple[str, str | None]:
    """Generate a song via ACE-Step. Returns (audio_path, lyrics_path|None).

    Reusable by /generate/music and the music-video orchestration so lyrics are
    always pipeline-authored (never improvised by the caller).
    """
    settings = AudioSettings.from_dict(settings_dict)
    enriched = await AUDIO_ENHANCER.generate_song(settings, idea)
    workflow = load_workflow("ACE-Step 1.5")
    workflow = AUDIO_ENHANCER.inject_audio_settings(workflow, enriched)
    prompt_id = await queue_prompt_async(workflow)
    output_info = await wait_for_completion_async(prompt_id)
    out_path = await download_output_to_local(output_info, _outputs_dir())
    lyrics_path = enriched.export_lyrics(out_path)
    return out_path, lyrics_path


async def _run_music_generation(jid: str, settings_dict: dict, idea: str) -> None:
    _job_running(jid)
    try:
        out_path, lyrics_path = await _generate_song(settings_dict, idea)
        _job_done(jid, out_path, lyrics_path)
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
        base, _ = os.path.splitext(audio_path)
        sidecar = f"{base}_lyrics.txt"
        lyrics_path = sidecar if os.path.exists(sidecar) else None
        segments = segment_audio(audio_path, seg_dir, lyrics_path=lyrics_path)
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
    video_workflow_id: str = Form("LTX2.3 - IA2V"),  # non-4.2: 4.2 IA2V has no lip-sync (audio not driving video)
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
# create/music-video  (autonomous orchestration: song → source → MCA → video)
# ---------------------------------------------------------------------------

def _mca_cfg() -> dict:
    c = CONFIG.get("music_video", {}).get("mca", {})
    return {
        "workflow": c.get("workflow", "F2K9B MCA.json"),
        "input_node": str(c.get("input_node", "76")),
        "prompt_node": str(c.get("prompt_node", "101")),
        "output_node": str(c.get("output_node", "9")),
        "batch_size": int(c.get("mca_batch_size", 4)),
        "t2i_workflow": c.get("t2i_workflow", "Flux2 Klein 9B - T2I"),
    }


async def _generate_still(workflow_name: str, prompt: str, aspect_ratio: str) -> str:
    """Synchronously generate one still image; returns a local file path."""
    from comfyui import randomize_seeds
    wf = load_workflow(workflow_name)
    wf = inject_prompt(wf, prompt)
    if aspect_ratio:
        wf = inject_resolution(wf, aspect_ratio)
    wf = randomize_seeds(wf)
    prompt_id = await queue_prompt_async(wf)
    info = await wait_for_completion_async(prompt_id)
    return await download_output_to_local(info, _outputs_dir())


async def _resolve_source_image(
    mode: str,
    image_bytes: Optional[bytes],
    image_name: Optional[str],
    description: str,
    theme: str,
    lyrics_text: str,
    genre: str,
    aspect_ratio: str,
    consistent_character: bool,
    tmp_dir: str,
) -> Optional[str]:
    """Resolve the single source character image. Returns a local path or None.

    upload   -> caller-supplied image
    describe -> LLM turns description into a prompt -> generated still
    auto     -> derive from lyrics + theme -> prompt -> generated still
    none     -> no source image (segments render without a start frame)

    RC7a: when consistent_character (default), describe/auto generate a CLEAN
    front-facing singer PORTRAIT (identity anchor, face/mouth visible for
    lip-sync) instead of a cinematic scene. False -> legacy scene image
    (intentional varied/different characters per voice).
    """
    try:
        if mode == "upload" and image_bytes:
            p = os.path.join(tmp_dir, image_name or "source.png")
            with open(p, "wb") as f:
                f.write(image_bytes)
            return p
        if mode in ("describe", "auto"):
            if mode == "describe":
                seed = description or theme
            else:
                seed = (lyrics_text[:600] + "\n" + theme).strip() or theme
            if consistent_character:
                t2i = await MV_PROMPTER.generate_character_portrait_prompt(seed, genre)
            else:
                t2i = await MV_PROMPTER.generate_start_image_prompt(seed, genre)
            return await _generate_still(_mca_cfg()["t2i_workflow"], t2i, aspect_ratio)
    except Exception as e:
        print(f"[create-mv] source image resolution failed ({mode}): {e} — continuing without start frame")
    return None


async def _run_mca_variants(source_image_path: str, variant_prompts: list[str]) -> list[str]:
    """One source image -> N variant stills via F2K9B MCA, in VRAM-safe chunks.

    Returns local paths aligned to variant_prompts order (best-effort; may be
    shorter if a chunk yields fewer images).
    """
    from music_video_pipeline import chunk_list
    from comfyui import randomize_seeds
    cfg = _mca_cfg()
    wf_path = os.path.join(CONFIG.get("workflows_dir", "./Workflows"), cfg["workflow"])
    if not os.path.exists(wf_path):
        raise FileNotFoundError(f"MCA workflow not found: {wf_path}")

    src_bytes = open(source_image_path, "rb").read()
    out_dir = os.path.join(_outputs_dir(), "mca_frames")
    os.makedirs(out_dir, exist_ok=True)
    results: list[str] = []
    idx = 0

    for chunk in chunk_list(variant_prompts, cfg["batch_size"]):
        with open(wf_path, encoding="utf-8") as f:
            wf = json.load(f)
        async with httpx.AsyncClient(timeout=30) as client:
            up = await client.post(
                f"{COMFYUI_URL}/upload/image",
                files={"image": (f"mca_src_{uuid.uuid4().hex}.png", src_bytes, "image/png")},
                data={"overwrite": "true"},
            )
            up.raise_for_status()
            uploaded_name = up.json().get("name")
        wf[cfg["input_node"]]["inputs"]["image"] = uploaded_name
        wf[cfg["prompt_node"]]["inputs"]["prompts"] = "\n".join(p.strip() for p in chunk)
        wf = randomize_seeds(wf)

        async with httpx.AsyncClient(timeout=15) as client:
            pr = await client.post(
                f"{COMFYUI_URL}/prompt",
                json={"prompt": wf, "client_id": str(uuid.uuid4())},
            )
            pr.raise_for_status()
            prompt_id = pr.json()["prompt_id"]

        outputs = None
        for _ in range(120):
            await asyncio.sleep(3)
            async with httpx.AsyncClient(timeout=10) as client:
                hist = (await client.get(f"{COMFYUI_URL}/history/{prompt_id}")).json()
            if prompt_id in hist and hist[prompt_id].get("outputs"):
                outputs = hist[prompt_id]["outputs"]
                break
        if not outputs:
            raise RuntimeError(f"[MCA] timeout waiting for prompt {prompt_id}")

        images = outputs.get(cfg["output_node"], {}).get("images", [])
        for img in images:
            async with httpx.AsyncClient(timeout=30) as client:
                view = await client.get(
                    f"{COMFYUI_URL}/view",
                    params={
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    },
                )
                view.raise_for_status()
            dest = os.path.join(out_dir, f"frame_{idx:03d}.png")
            with open(dest, "wb") as fh:
                fh.write(view.content)
            results.append(dest)
            idx += 1

    return results


async def _run_create_music_video(
    jid: str,
    brief: str,
    song_path: Optional[str],
    theme: str,
    source_mode: str,
    source_image_bytes: Optional[bytes],
    source_image_name: Optional[str],
    source_description: str,
    duration: Optional[int],
    video_workflow_id: str,
    crossfade_duration: float,
    aspect_ratio: str,
    language: str,
    consistent_character: bool,
    tmp_dir: str,
) -> None:
    _job_running(jid)
    try:
        # 1. Song (pipeline-authored lyrics) unless caller supplied one.
        if song_path:
            audio_path = song_path
            base, _ = os.path.splitext(audio_path)
            sidecar = f"{base}_lyrics.txt"
            lyrics_path = sidecar if os.path.exists(sidecar) else None
        else:
            # RC2: AudioSettings defaults language to 'en' → English lyrics even
            # for "deutsches ..." briefs. The ACE-Step TextEncodeAceStepAudio1.5
            # node only accepts ISO codes (de/en/fr/...), NOT names — passing a
            # name 400s the graph. Resolve to a valid code or leave the default.
            lang = to_ace_language(language, brief)
            settings_dict = {
                "caption": brief,
                "mode": "vocal",
                "duration": clamp_song_duration(duration),
            }
            if lang:
                settings_dict["language"] = lang
            audio_path, lyrics_path = await _generate_song(settings_dict, brief)

        total_duration = get_audio_duration(audio_path)
        theme_eff = theme or brief
        lyrics_text = ""
        if lyrics_path and os.path.exists(lyrics_path):
            with open(lyrics_path, encoding="utf-8") as f:
                lyrics_text = f.read()

        # 2. Creative segment plan (dynamic lengths + per-segment frame variants).
        segments = await MV_PROMPTER.plan_segments(
            lyrics_text, theme_eff, total_duration, genre=brief
        )

        seg_dir = os.path.join(tmp_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)
        for seg in segments:
            clip = os.path.join(seg_dir, f"seg_{seg.index:03d}.wav")
            _extract_audio_clip(audio_path, seg.start_time, seg.end_time, clip)
            seg.audio_clip = clip

        # 3. Source image + per-segment MCA variant frames.
        source = await _resolve_source_image(
            source_mode, source_image_bytes, source_image_name,
            source_description, theme_eff, lyrics_text, brief,
            aspect_ratio, consistent_character, tmp_dir,
        )
        frames: list[str] = []
        if source:
            frames = await _run_mca_variants(
                source, [s.frame_variant_prompt or s.prompt for s in segments]
            )

        # 4. Per-segment IA2V render (fresh frame each segment, no chaining,
        #    no inject_resolution — IA2V resolution is driven by the input image).
        out_dir = _outputs_dir()
        for i, seg in enumerate(segments):
            wf = load_workflow(video_workflow_id)
            wf = inject_prompt(wf, seg.prompt)
            frame = frames[i] if i < len(frames) else (source or "")
            if frame and os.path.exists(frame):
                with open(frame, "rb") as fr:
                    up = await upload_file_to_comfy(fr.read(), os.path.basename(frame), "image")
                wf = inject_input_image(wf, up)
            if seg.audio_clip and os.path.exists(seg.audio_clip):
                with open(seg.audio_clip, "rb") as af:
                    ua = await upload_file_to_comfy(af.read(), os.path.basename(seg.audio_clip), "audio")
                wf = inject_input_audio(wf, ua)
            # Best-effort: ineffective on LTX2.3 (4.2) IA2V (latent length is a
            # wired compute chain that ignores a literal) — clips come out at the
            # workflow's fixed length there. Kept for workflows where it works.
            wf = inject_segment_duration(wf, seg.duration)
            prompt_id = await queue_prompt_async(wf)
            info = await wait_for_completion_async(prompt_id)
            seg.video_clip = await download_output_to_local(info, os.path.join(out_dir, "seg_videos"))
            seg.status = "completed"

        # RC3: per-segment audio chunks were planned to a length the (4.2)
        # workflow ignores, so splicing them against fixed-length video clips
        # produced an audible gap at every segment boundary. Drop the chunks and
        # mux the FULL continuous song over the assembled video instead — one
        # uninterrupted track, A/V coherent regardless of clip lengths.
        for seg in segments:
            seg.audio_clip = ""

        session = MVSession(
            audio_path=audio_path,
            start_image_path=source or "",
            segments=segments,
            crossfade_duration=crossfade_duration,
        )
        final_path = os.path.join(out_dir, f"music_video_{jid[:8]}.mp4")
        assemble_video(session, final_path)
        _job_done(jid, final_path, lyrics_path)
    except Exception as e:
        _job_failed(jid, str(e))
        raise


@app.post("/create/music-video")
async def create_music_video(
    brief: str = Form(...),
    song: Optional[UploadFile] = File(None),
    theme: str = Form(""),
    source_mode: str = Form("auto"),
    source_image: Optional[UploadFile] = File(None),
    source_description: str = Form(""),
    duration: Optional[int] = Form(None),
    video_workflow_id: str = Form("LTX2.3 - IA2V"),  # non-4.2: 4.2 IA2V has no lip-sync (audio not driving video)
    crossfade_duration: float = Form(0.0),
    aspect_ratio: str = Form("16:9"),
    language: str = Form(""),
    consistent_character: bool = Form(True),
):
    """Autonomous music-video creation. Lyrics are always pipeline-authored —
    callers pass a topic in `brief`, never finished lyrics."""
    tmp_dir = tempfile.mkdtemp(prefix="ctb_cmv_")
    song_path = None
    if song is not None:
        song_path = os.path.join(tmp_dir, song.filename or "song.wav")
        with open(song_path, "wb") as f:
            f.write(await song.read())
    src_bytes = await source_image.read() if source_image is not None else None
    src_name = source_image.filename if source_image is not None else None

    jid = _new_job()
    asyncio.create_task(_run_create_music_video(
        jid, brief, song_path, theme, source_mode, src_bytes, src_name,
        source_description, duration, video_workflow_id, crossfade_duration,
        aspect_ratio, language, consistent_character, tmp_dir,
    ))
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
async def enhance_prompt_endpoint(
    text: str = Form(...),
    workflow_id: Optional[str] = Form(None),
    selections: Optional[str] = Form(None),
):
    try:
        from prompt_enhancer import PromptEnhancer
        sel_dict: dict = json.loads(selections) if selections else {}
        enhancer = PromptEnhancer(CONFIG)
        result = enhancer.enhance_prompt(
            user_prompt=text,
            workflow_name=workflow_id or "",
            selections=sel_dict,
        )
        return {
            "original": text,
            "enhanced": result.final_prompt,
            "used_llm": result.used_llm,
            "status": result.status_message,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Prompt enhancer unavailable: {e}")
