"""HermesCTB API — exposes CTB generation capabilities as REST endpoints for Hermes."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
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
    build_smart_prompt,
    download_output_to_local,
    get_file_bytes,
    has_relay_smart_node,
    inject_input_audio,
    inject_input_image,
    inject_prompt,
    inject_resolution,
    inject_segment_duration,
    init_config,
    make_comfy_caller,
    probe_comfyui_alive,
    queue_prompt_async,
    queue_and_wait_with_recovery,
    free_comfy,
    upload_file_to_comfy,
    wait_for_completion_async,
    load_workflow,
    COMFYUI_URL,
)


async def _ensure_comfy_up_or_412() -> None:
    """Stage 8 pre-flight: fail fast when ComfyUI is unreachable.

    Long-running submit endpoints (/generate/video, /generate/music-video,
    /create/music-video) call this BEFORE accepting the job, so a down
    ComfyUI surfaces as an immediate 412 instead of a queued job that
    crashes at the first render step (and then triggers the Tier-2
    recovery cascade against a possibly-broken supervisor path).
    """
    if not await probe_comfyui_alive():
        raise HTTPException(
            status_code=412,
            detail=(
                f"ComfyUI not reachable at {COMFYUI_URL}. Start ComfyUI "
                "Desktop or check the host_supervisor before resubmitting."
            ),
        )
from music_video_pipeline import (
    MVSession,
    MusicVideoPrompter,
    Segment,
    assemble_video,
    build_duet_portrait_prompt,
    clamp_song_duration,
    enforce_performer_role,
    extract_section_role,
    get_audio_duration,
    nearest_annotated_role,
    partition_anchors_by_role,
    plan_same_gender_portraits,
    same_gender_veto,
    segment_audio,
    strip_lyrics_from_image_prompt,
    to_ace_language,
    _extract_audio_clip,
)


_nearest_annotated_role = nearest_annotated_role  # local alias for routing block
from lyric_align import align_sections
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


def _safe_filename(*parts: str, max_len: int = 80) -> str:
    """Build a filesystem-safe basename from one or more text parts.

    Joins non-empty parts with ' - ', strips characters that break Windows /
    POSIX paths, collapses whitespace, caps length. Returns '' when input
    yields no usable characters — callers must supply a fallback name.
    """
    import re as _re
    pieces = [p.strip() for p in parts if p and p.strip()]
    if not pieces:
        return ""
    joined = " - ".join(pieces)
    cleaned = _re.sub(r"[^\w\s\-'.]", "", joined, flags=_re.UNICODE)
    cleaned = _re.sub(r"\s+", " ", cleaned).strip(" .-")
    return cleaned[:max_len] if cleaned else ""


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
    # Bug #6 patch: adds absolute step-time ceiling to SlowdownMonitor so
    # initial CPU-fallback (constant slowdown from step 1) is detected. Lives
    # in a separate file because comfyui.py was OS-locked during the fix.
    import comfyui_patches
    comfyui_patches.apply_config(CONFIG)
    _audio_cfg = dict(CONFIG["prompt_enhancer"])
    _audio_cfg["ace_step_encoder"] = CONFIG.get("ace_step_encoder", {})
    AUDIO_ENHANCER = AudioEnhancer(_audio_cfg)
    # openrouter_api_key lives under prompt_enhancer, NOT top-level — passing the
    # full CONFIG leaves api_key="" and silently kills every MV LLM call.
    MV_PROMPTER = MusicVideoPrompter(CONFIG["prompt_enhancer"])
    REGISTRY = load_registry()
    os.makedirs("outputs", exist_ok=True)
    # Probe host_supervisor — if down, Tier-2 auto-recovery (full ComfyUI
    # restart) will silently fail on slowdown. Warn loud at startup so the
    # issue is caught before a 40-min render hangs.
    _sup_url = (
        CONFIG.get("comfy_recovery", {}).get("restart_service_url", "")
        .replace("/restart", "/health")
    )
    if _sup_url:
        try:
            async with httpx.AsyncClient(timeout=3) as _c:
                _r = await _c.get(_sup_url)
            if _r.status_code != 200:
                print(f"WARNING: host_supervisor health probe HTTP {_r.status_code} at {_sup_url}")
            else:
                print(f"host_supervisor OK at {_sup_url}")
        except Exception as _e:
            print(f"WARNING: host_supervisor unreachable at {_sup_url}: {_e!r} — Tier-2 restart will fail")
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
    await _ensure_comfy_up_or_412()
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
    await _ensure_comfy_up_or_412()
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

async def _generate_song(
    settings_dict: dict, idea: str,
) -> tuple[str, str | None, str, str]:
    """Generate a song via ACE-Step.

    Returns (audio_path, lyrics_path|None, artist, title). When the LLM emits
    an `artist`/`title` pair (and the resulting basename is filesystem-safe),
    the downloaded audio is renamed to `<artist> - <title>.mp3` so library /
    mixing workflows downstream get a meaningful filename instead of the
    generic ComfyUI output stem. The lyrics sidecar is renamed in lockstep.

    Reusable by /generate/music and the music-video orchestration so lyrics
    are always pipeline-authored (never improvised by the caller).
    """
    settings = AudioSettings.from_dict(settings_dict)
    enriched = await AUDIO_ENHANCER.generate_song(settings, idea)
    workflow = load_workflow("ACE-Step 1.5")
    workflow = AUDIO_ENHANCER.inject_audio_settings(workflow, enriched)
    # Bug #6: pre-job /free so ACE-Step gets a clean VRAM state. Without this,
    # fragmented VRAM from prior runs forces CPU-fallback (observed 132s/it
    # ETA 3h39m vs normal ~1-2s/it). Symmetrical to the IA2V-loop free_every.
    await free_comfy()
    prompt_id, output_info = await queue_and_wait_with_recovery(workflow)
    out_path = await download_output_to_local(output_info, _outputs_dir())
    lyrics_path = enriched.export_lyrics(out_path)

    artist = (enriched.artist or "").strip()
    title = (enriched.title or "").strip()
    basename = _safe_filename(artist, title)
    if basename:
        out_dir = os.path.dirname(out_path)
        ext = os.path.splitext(out_path)[1] or ".mp3"
        new_audio = os.path.join(out_dir, f"{basename}{ext}")
        if new_audio != out_path and not os.path.exists(new_audio):
            try:
                os.rename(out_path, new_audio)
                out_path = new_audio
                if lyrics_path and os.path.exists(lyrics_path):
                    new_lyrics = os.path.join(out_dir, f"{basename}_lyrics.txt")
                    if not os.path.exists(new_lyrics):
                        os.rename(lyrics_path, new_lyrics)
                        lyrics_path = new_lyrics
            except OSError as e:
                print(
                    f"[generate-song] rename to artist-title basename failed "
                    f"({e!r}); keeping ComfyUI default name {out_path}"
                )

    return out_path, lyrics_path, artist, title


async def _run_music_generation(jid: str, settings_dict: dict, idea: str) -> None:
    _job_running(jid)
    try:
        out_path, lyrics_path, _artist, _title = await _generate_song(
            settings_dict, idea,
        )
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
    finally:
        # Fix 25: clean per-job tmp_dir so /tmp/ctb_mv_* (host or container)
        # does not leak — accumulating leaks grow the WSL2 VHDX on I:.
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/generate/music-video")
async def generate_music_video(
    audio: UploadFile = File(...),
    video_workflow_id: str = Form("LTX2.3 - IA2V"),  # non-4.2: 4.2 IA2V has no lip-sync (audio not driving video)
    theme: str = Form(...),
    crossfade_duration: float = Form(0.0),
):
    await _ensure_comfy_up_or_412()
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
        "free_every": int(c.get("comfy_free_every", 3)),
    }


# RC9: deterministic lip-sync booster appended to VOCAL segment prompts (LLM
# may forget it; this guarantees it). Story/instrumental segments never get it.
LIPSYNC_BOOSTER = (
    "The lips are syncing naturally to the vocals. Every word is pronounced "
    "perfectly, facial expressions are lively, diction and lip sync are perfect."
)


async def _generate_still(workflow_name: str, prompt: str, aspect_ratio: str) -> str:
    """Synchronously generate one still image; returns a local file path."""
    from comfyui import randomize_seeds
    wf = load_workflow(workflow_name)
    wf = inject_prompt(wf, prompt)
    if aspect_ratio:
        wf = inject_resolution(wf, aspect_ratio)
    wf = randomize_seeds(wf)
    prompt_id, info = await queue_and_wait_with_recovery(wf)
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


_ROLE_SEED_PREFIX = {
    "male": (
        "Performer for sections sung by a MALE vocalist — a man, visually "
        "distinct from any female performer in this song; he leads only his "
        "own sections. Lyrics/theme context: "
    ),
    "female": (
        "Performer for sections sung by a FEMALE vocalist — a woman, visually "
        "distinct from any male performer in this song; she leads only her "
        "own sections. Lyrics/theme context: "
    ),
    # Fix 27: lead + partner portraits for an explicit same-gender duet. The
    # brief describes two same-gender singers; the "1" seed steers the LEAD to
    # the FIRST described person (the generic "female"/"male" prefix has no
    # "one of two" framing, so without this the lead and partner can collapse to
    # the same look). The "2" seed renders the SECOND, visually distinct, and is
    # used only in the shared duet frame.
    "female1": (
        "Performer for the FIRST of two FEMALE singers in this duet — a woman "
        "matching the first female described in the brief (her ethnicity, skin "
        "tone, hair and build); she sings all the solo sections and is one half "
        "of the shared duet. Lyrics/theme context: "
    ),
    "male1": (
        "Performer for the FIRST of two MALE singers in this duet — a man "
        "matching the first male described in the brief (his ethnicity, skin "
        "tone, hair and build); he sings all the solo sections and is one half "
        "of the shared duet. Lyrics/theme context: "
    ),
    "female2": (
        "Performer for the SECOND of two FEMALE singers in this duet — a woman "
        "visually DISTINCT from the first female singer (different ethnicity, "
        "skin tone, hair and build as described in the brief); she appears only "
        "in the shared duet. Lyrics/theme context: "
    ),
    "male2": (
        "Performer for the SECOND of two MALE singers in this duet — a man "
        "visually DISTINCT from the first male singer (different ethnicity, "
        "skin tone, hair and build as described in the brief); he appears only "
        "in the shared duet. Lyrics/theme context: "
    ),
}


async def _resolve_singer_portrait(
    role: str,
    theme: str,
    lyrics_text: str,
    genre: str,
    aspect_ratio: str,
) -> Optional[str]:
    """Generate a clean front-facing portrait for a specific performer role.

    role: "male" | "female" (mixed duet) | "female2" | "male2" (Fix 27 — the
    SECOND singer of an explicit same-gender duet). The role forces the LLM
    portrait seed with an identity-locking prefix so multi-singer songs produce
    distinct portraits instead of collapsing to whichever performer the LLM
    picks first.
    """
    prefix = _ROLE_SEED_PREFIX.get(role)
    if not prefix:
        raise ValueError(f"_resolve_singer_portrait: unsupported role {role!r}")
    try:
        seed_body = (lyrics_text[:600] + "\n" + theme).strip() or theme
        seed = prefix + seed_body
        t2i = await MV_PROMPTER.generate_character_portrait_prompt(seed, genre)
        return await _generate_still(_mca_cfg()["t2i_workflow"], t2i, aspect_ratio)
    except Exception as e:
        print(f"[create-mv] {role} portrait generation failed: {e}")
    return None


# Flux2 Klein M-I Edit workflow node IDs for the multi-reference chain.
# See Workflows/Flux2 Klein M-I Edit.json — pipeline 2 (SaveImage 80):
#   LoadImage 33 + 34 + 36 → VAEEncode → 3x ReferenceLatent → CFGGuider → SamplerCustomAdvanced 13
#   CLIPTextEncode at node 30 carries the positive prompt
# Pipeline 1 (SaveImage 19) and the SeedVR2 upscale branch are stripped at
# queue-time so a 2-input duet portrait runs cheaply.
_FLUX2_MIEDIT = {
    "workflow": "Flux2 Klein M-I Edit.json",
    "load_img_a": "33",
    "load_img_b": "34",
    "load_img_c": "36",
    "prompt_node": "30",
    "save_node": "80",
}


def _strip_flux2_miedit_unused(wf: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only nodes reachable from the M-I Edit SaveImage (node 80).

    Drops Pipeline 1 (SaveImage 19), the SeedVR2 upscaler branches (nodes 25
    + 70 + their dependencies), and any Image Comparer nodes. Avoids
    FileNotFoundError from stale default LoadImage names in unused branches.
    """
    keep: set[str] = set()
    stack = [_FLUX2_MIEDIT["save_node"]]
    while stack:
        nid = stack.pop()
        if nid in keep or nid not in wf:
            continue
        keep.add(nid)
        for v in (wf[nid].get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                stack.append(v[0])
    return {k: v for k, v in wf.items() if k in keep}


async def _resolve_duet_portrait(
    portrait_a: str,
    portrait_b: str,
    theme: str,
    genre: str,
    wardrobe_slot: str = "",
    duet_kind: str = "mixed",
) -> Optional[str]:
    """Generate a duet portrait depicting both performers together using the
    Flux2 Klein M-I Edit workflow. Takes any two already-rendered single-singer
    portraits as reference images so the faces stay consistent — works for
    male+female (mixed) and, since Fix 27, female+female / male+male.

    `portrait_a` is the lead reference (e.g. the male in a mixed duet, or the
    solo lead in a same-gender duet); `portrait_b` is the partner. The workflow
    needs 3 LoadImage inputs, so `portrait_a` is duplicated into slot 36
    (status-quo Fix 24B). This gives `portrait_a` a ~2:1 reference weight — an
    acceptable, documented lead bias; if a same-gender duet leans too hard
    toward the lead, alternating the duplicated ref is the mitigation.

    Returns a local PNG path or None on failure (caller falls back to None
    portrait for duet segments → MCA generates from segment prompt alone).
    """
    try:
        from comfyui import randomize_seeds
        wf_path = os.path.join(CONFIG.get("workflows_dir", "./Workflows"),
                               _FLUX2_MIEDIT["workflow"])
        if not os.path.exists(wf_path):
            raise FileNotFoundError(f"Flux2 M-I Edit workflow missing: {wf_path}")
        with open(wf_path, encoding="utf-8") as f:
            wf = json.load(f)
        wf = _strip_flux2_miedit_unused(wf)

        # Fix 24A: deterministic identity-neutral prompt — no LLM call. See
        # build_duet_portrait_prompt() docstring for the rationale.
        # Fix 30 (B1): pass the active wardrobe slot through so the duet
        # portrait anchors clothing to the same outfit as surrounding solo
        # segments. Empty string keeps the legacy generic-costume prompt.
        # Fix 32: duet_kind ('ff'/'mm'/'mixed') drives same-gender wording
        # so the T2I model does not invent an opposite-sex performer.
        prompt = build_duet_portrait_prompt(
            theme, wardrobe_slot=wardrobe_slot, duet_kind=duet_kind,
        )

        a_bytes = open(portrait_a, "rb").read()
        b_bytes = open(portrait_b, "rb").read()
        a_name = f"duet_a_{uuid.uuid4().hex}.png"
        b_name = f"duet_b_{uuid.uuid4().hex}.png"
        # Fix 24B (status quo): the Flux2 Klein M-I Edit workflow requires 3
        # LoadImage inputs feeding 3 sequential ReferenceLatent nodes. We
        # intentionally re-upload the lead (portrait_a) bytes as the 3rd
        # reference; the resulting lead-side weight bias is acceptable and
        # documented (see Fix 27 docstring).
        a_name_dup = f"duet_a2_{uuid.uuid4().hex}.png"
        async with httpx.AsyncClient(timeout=30) as client:
            for name, blob in (
                (a_name, a_bytes),
                (b_name, b_bytes),
                (a_name_dup, a_bytes),
            ):
                up = await client.post(
                    f"{COMFYUI_URL}/upload/image",
                    files={"image": (name, blob, "image/png")},
                    data={"overwrite": "true"},
                )
                up.raise_for_status()

        wf[_FLUX2_MIEDIT["load_img_a"]]["inputs"]["image"] = a_name
        wf[_FLUX2_MIEDIT["load_img_b"]]["inputs"]["image"] = b_name
        wf[_FLUX2_MIEDIT["load_img_c"]]["inputs"]["image"] = a_name_dup
        wf[_FLUX2_MIEDIT["prompt_node"]]["inputs"]["text"] = prompt
        wf = randomize_seeds(wf)

        prompt_id, _ = await queue_and_wait_with_recovery(wf)
        async with httpx.AsyncClient(timeout=10) as client:
            hist = (await client.get(f"{COMFYUI_URL}/history/{prompt_id}")).json()
        outputs = hist.get(prompt_id, {}).get("outputs", {})
        images = outputs.get(_FLUX2_MIEDIT["save_node"], {}).get("images", [])
        if not images:
            raise RuntimeError(f"[duet] history missing SaveImage {_FLUX2_MIEDIT['save_node']} output")

        img = images[0]
        out_dir = os.path.join(_outputs_dir(), "duet_portraits")
        os.makedirs(out_dir, exist_ok=True)
        local = os.path.join(out_dir, f"duet_{uuid.uuid4().hex}.png")
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(
                f"{COMFYUI_URL}/view",
                params={"filename": img["filename"], "type": img.get("type", "output"),
                        "subfolder": img.get("subfolder", "")},
            )
            r.raise_for_status()
            with open(local, "wb") as f:
                f.write(r.content)
        return local
    except Exception as e:
        print(f"[create-mv] duet portrait generation failed: {e}")
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
    # Unique per-call batch id so concurrent or sequential invocations don't
    # overwrite each other's frame_NNN.png files. Critical for multi-role
    # rendering where this helper is called once per role group.
    batch_id = uuid.uuid4().hex[:8]
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

        # Recovery-wrapped queue+wait. wait_for_completion_async's _extract_output
        # returns a single primary output; MCA needs ALL images from a specific
        # node, so we re-fetch full history after the wrapper confirms the job done.
        prompt_id, _ = await queue_and_wait_with_recovery(wf)
        async with httpx.AsyncClient(timeout=10) as client:
            hist = (await client.get(f"{COMFYUI_URL}/history/{prompt_id}")).json()
        outputs = hist.get(prompt_id, {}).get("outputs", {})
        if not outputs:
            raise RuntimeError(f"[MCA] history missing outputs for prompt {prompt_id}")

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
            dest = os.path.join(out_dir, f"frame_{batch_id}_{idx:03d}.png")
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
    duet: str = "",
    time_of_day_arc: str = "",
    wardrobe_arc: str = "",
    artist: str = "",
) -> None:
    _job_running(jid)
    # Threaded through the whole pipeline so the final output can be named
    # "<artist> - <title>.mp4" / ".mp3" instead of the generic job-id form.
    settings_artist = ""
    settings_title = ""
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
            if artist and artist.strip():
                settings_dict["artist"] = artist.strip()
            # Fix 27: steer the lyric author toward solo+duet structure for an
            # explicit same-gender request. Goes into the lyric-author `idea`
            # (NOT the ACE caption, which must not name voice gender).
            song_idea = brief
            if duet in ("ff", "mm"):
                g = "female" if duet == "ff" else "male"
                opp = "male" if duet == "ff" else "female"
                song_idea = (
                    f"{brief}\n\n[arrangement] Two {g} vocalists only: each solo "
                    f"section labelled [Verse - {g}] (one consistent lead voice), "
                    f"shared choruses labelled [Chorus - duet]; no {opp} voice."
                )
            audio_path, lyrics_path, settings_artist, settings_title = (
                await _generate_song(settings_dict, song_idea)
            )

        total_duration = get_audio_duration(audio_path)
        theme_eff = theme or brief
        lyrics_text = ""
        if lyrics_path and os.path.exists(lyrics_path):
            with open(lyrics_path, encoding="utf-8") as f:
                lyrics_text = f.read()

        # 2a. RC8: align lyrics to audio for REAL section timestamps (cuts on
        #     section boundaries, not mid-vocal) + chorus reuse. Fail-soft:
        #     None -> plan_segments uses legacy proportional segmentation.
        aligned = None
        if lyrics_path and os.path.exists(lyrics_path):
            lang_code = to_ace_language(language, brief) or "en"
            try:
                aligned = await asyncio.to_thread(
                    align_sections, audio_path, lyrics_path,
                    total_duration, lang_code,
                )
            except Exception as e:
                print(f"[create-mv] align_sections crashed (non-fatal): {e!r}")
                aligned = None
            print(f"[create-mv] lyric alignment: "
                  f"{'OK ' + str(len(aligned)) + ' sections' if aligned else 'unavailable → proportional fallback'}")

            # Fix 15: Audio-based gender detection overrides LLM lyrics-tag
            # gender when audio reality differs. Fix 15c: ALSO refines section
            # start/end boundaries using inaSpeech voice-activity transitions
            # (more frame-precise than whisperx word boundaries). Graceful
            # fallback on any failure (demucs/inaSpeechSegmenter unavailable
            # -> keep LLM tags and whisperx boundaries).
            if aligned:
                try:
                    import re as _re
                    import tempfile as _tempfile
                    from audio_gender_detect import (
                        _segment_audio,
                        _classify_section,
                        refine_section_boundaries,
                        split_sections_at_mid_swaps,
                    )
                    from lyric_align import _demucs_vocals
                    with _tempfile.TemporaryDirectory(prefix="gender_detect_") as _gd_work:
                        _vocals = await asyncio.to_thread(_demucs_vocals, audio_path, _gd_work)
                        if _vocals:
                            _segments = await asyncio.to_thread(_segment_audio, _vocals)
                            # Phase 1: refine section boundaries from VAD transitions
                            _refined = refine_section_boundaries(aligned, _segments)
                            for _i, _new in enumerate(_refined):
                                for _k in ("start", "end", "start_time", "end_time"):
                                    if _k in _new:
                                        aligned[_i][_k] = _new[_k]
                            # Fix 17: Split sections that contain a mid-section
                            # voice swap into two sub-sections at the swap point.
                            _pre_split_count = len(aligned)
                            _split = split_sections_at_mid_swaps(aligned, _segments)
                            mid_splits = len(_split) - _pre_split_count
                            if mid_splits > 0:
                                # Replace aligned IN PLACE so downstream code sees subsections
                                aligned[:] = _split
                            # Phase 2: classify gender per (refined+split) section
                            detected = {}
                            for _sec in aligned:
                                _label = _sec.get("label", "")
                                _s = float(_sec.get("start", _sec.get("start_time", 0.0)))
                                _e = float(_sec.get("end", _sec.get("end_time", 0.0)))
                                if _label and _e > _s:
                                    detected[_label] = _classify_section(_s, _e, _segments)
                            # Fix 27: same-gender request reconciliation. Audio
                            # cannot tell two women (or two men) apart, and a
                            # same-gender duet chorus reads as plain "female"/
                            # "male" to inaSpeech (the "duet" class needs BOTH
                            # genders ≥20%). So in same-gender mode the audio
                            # acts only as a VETO: if it finds sustained
                            # opposite-gender vocals the request is a mismatch →
                            # fall back to mixed routing (run the normal
                            # override). Otherwise keep the LLM labels verbatim
                            # so the "duet" choruses survive.
                            # Fix 36: detected is now Dict[label, (gender, confidence)].
                            same_gender = duet in ("ff", "mm")
                            if same_gender:
                                _vsecs = []
                                for _sec in aligned:
                                    _result = detected.get(_sec.get("label", ""))
                                    if not _result:
                                        continue
                                    _g2, _conf2 = _result
                                    _s2 = float(_sec.get("start", _sec.get("start_time", 0.0)))
                                    _e2 = float(_sec.get("end", _sec.get("end_time", 0.0)))
                                    _vsecs.append((_g2, max(0.0, _e2 - _s2)))
                                if same_gender_veto(_vsecs, duet):
                                    print(f"[create-mv] same-gender duet={duet!r} VETOED by audio "
                                          f"(sustained opposite-gender vocals) → mixed routing")
                                    duet = ""
                                    same_gender = False
                                else:
                                    print(f"[create-mv] same-gender duet={duet!r} confirmed by audio "
                                          f"→ keeping LLM labels (no gender override)")
                            corrections = 0
                            if not same_gender:
                                # Fix 36: asymmetric trust + confidence gate.
                                # LLM labels with an explicit "- male|female|
                                # duet" suffix carry lyric-aware semantic
                                # authority — they need a strong audio signal
                                # to be overridden. Generic LLM labels (no
                                # gender suffix) get the lower gate.
                                # Stage F (2026-06-07 evening): lowered
                                # _HIGH_CONF from 0.85 → 0.70 after job
                                # 63486a7a where a male-sung Verse 1 was
                                # mislabelled "- female" by the LLM and
                                # InaSpeechSegmenter's 0.78 confidence for
                                # the same section was insufficient to
                                # override. 0.70 matches the segmenter's
                                # `dominant_threshold` floor, so any audio
                                # classification the segmenter is willing
                                # to call "solo" can now override an LLM
                                # mistake.
                                _HIGH_CONF = 0.70
                                _LOW_CONF = 0.70
                                for _sec in aligned:
                                    _label = _sec.get("label", "")
                                    _result = detected.get(_label)
                                    if not _result:
                                        continue
                                    _g, _conf = _result
                                    if _g == "unknown":
                                        continue
                                    _has_explicit = bool(_re.search(
                                        r' - (male|female|duet)\b', _label
                                    ))
                                    _required = _HIGH_CONF if _has_explicit else _LOW_CONF
                                    if _conf < _required:
                                        print(
                                            f"[Fix 36] keeping LLM label "
                                            f"{_label!r} — audio says {_g!r} "
                                            f"with conf={_conf:.2f} < required "
                                            f"{_required:.2f}."
                                        )
                                        continue
                                    _new_label = _re.sub(
                                        r' - (male|female|duet)\b',
                                        f' - {_g}', _label, count=1,
                                    )
                                    if _new_label != _label:
                                        _sec["label"] = _new_label
                                        corrections += 1
                                        print(
                                            f"[Fix 36] overriding {_label!r} "
                                            f"→ {_new_label!r} "
                                            f"(audio conf={_conf:.2f})."
                                        )
                            # Fix 36: log detected as (gender, conf) tuples so
                            # the per-section confidence is visible.
                            _detected_compact = {
                                k: (v[0], round(v[1], 2)) for k, v in detected.items()
                            }
                            print(f"[create-mv] gender detection: {_detected_compact}; "
                                  f"gender corrections: {corrections}; "
                                  f"mid-section splits: {mid_splits}; "
                                  f"boundary refinements applied (voice_end tail +0.5s)")
                        else:
                            print("[create-mv] gender detection: demucs unavailable → skip")
                except Exception as _e:
                    print(f"[create-mv] gender detection failed (non-fatal): {_e!r}")

        # 2b. Creative segment plan (aligned timestamps if available).
        # Fix 29 + Fix 30: pass user-supplied time-of-day and wardrobe arc
        # overrides through. Empty string → plan_segments runs the LLM
        # mini-call (or genre default) for each.
        # Fix 32: pass duet_kind through ('ff'/'mm'/'mixed') so wardrobe
        # tags + system prompts use same-gender wording for ff/mm duets.
        segments = await MV_PROMPTER.plan_segments(
            lyrics_text, theme_eff, total_duration, genre=brief,
            aligned_sections=aligned,
            time_of_day_arc=(time_of_day_arc or None),
            wardrobe_arc=(wardrobe_arc or None),
            duet_kind=(duet if duet in ("ff", "mm") else "mixed"),
        )

        seg_dir = os.path.join(tmp_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)
        for seg in segments:
            clip = os.path.join(seg_dir, f"seg_{seg.index:03d}.wav")
            _extract_audio_clip(audio_path, seg.start_time, seg.end_time, clip)
            seg.audio_clip = clip

        # 3. Source image(s) + per-segment MCA variant frames.
        # Role hints come from LLM-authored section labels like
        # "[Verse - male]" / "[Chorus - female]" / "[Bridge - duet]". When the
        # song annotates ≥2 distinct roles AND consistent_character is on AND
        # source_mode is generative, render one portrait per role; for "duet",
        # combine the male+female portraits via the Flux2 M-I Edit workflow.
        role_groups = partition_anchors_by_role(segments)
        roles_present = {r for r in role_groups.keys() if r is not None}
        # Fix 32: also collect roles across ALL segments (including reuse_of
        # rows). partition_anchors_by_role only considers anchors, so when
        # the FIRST duet section happens to be a reuse of an earlier chorus
        # the "duet" role is dropped — and the ff/mm portrait path never
        # triggers. all_roles_present is used downstream for the
        # plan_same_gender_portraits decision; role_groups stays anchor-only
        # for per-anchor portrait routing.
        all_roles_present = {extract_section_role(s.label) for s in segments}
        all_roles_present.discard(None)

        # Fix 27: an explicit same-gender duet (ff/mm) takes a dedicated path —
        # ONE consistent lead portrait for all solo sections + a distinct
        # partner portrait that feeds ONLY the shared-duet frame. Section labels
        # stay "female"/"duet", so the per-segment routing below is unchanged.
        # Fix 32: feed all_roles_present (Anchors + reuse-rows) into the
        # ff/mm decision so a duet-section that first appears as a reuse
        # row still triggers the partner+duet portrait render.
        sg_plan = plan_same_gender_portraits(
            duet, all_roles_present, consistent_character, source_mode,
        )

        portraits: Dict[str, Optional[str]] = {}
        if sg_plan is not None:
            is_multi_role = True
            base, partner, make_duet = sg_plan
            # Lead portrait is stored under the plain section role ("female"/
            # "male") so per-segment routing finds it, but it's SEEDED with the
            # "1" prefix so the LLM steers it to the FIRST of the two described
            # singers (distinct from the partner).
            portraits[base] = await _resolve_singer_portrait(
                base + "1", theme_eff, lyrics_text, brief, aspect_ratio,
            )
            await free_comfy()
            # Partner = the SECOND same-gender singer; appears only in the duet
            # frame, never routed to a solo section. Skip it entirely when the
            # song has no duet chorus (nothing would use it).
            if make_duet:
                portraits[partner] = await _resolve_singer_portrait(
                    partner, theme_eff, lyrics_text, brief, aspect_ratio,
                )
                await free_comfy()
                if portraits.get(base) and portraits.get(partner):
                    # Fix 30 (B1): anchor the duet portrait to the wardrobe
                    # slot of the first duet segment so clothing matches the
                    # solo frames. Fallback: first segment's slot.
                    duet_slot = next(
                        (s.wardrobe_slot for s in segments
                         if extract_section_role(s.label) == "duet" and s.wardrobe_slot),
                        "",
                    ) or (segments[0].wardrobe_slot if segments else "")
                    portraits["duet"] = await _resolve_duet_portrait(
                        portraits[base], portraits[partner], theme_eff, brief,
                        wardrobe_slot=duet_slot,
                        duet_kind=duet,
                    )
                    await free_comfy()
            source: Optional[str] = (
                portraits.get(base)
                or next((v for v in portraits.values() if v), None)
            )
        else:
            # Fix 32: roles count must consider ALL segments (incl. reuse
            # rows) so a duet-section first-seen as reuse still flips the
            # multi-role branch and triggers the duet portrait render.
            is_multi_role = (
                consistent_character
                and len(all_roles_present) >= 2
                and source_mode in ("auto", "describe")
            )
            if is_multi_role:
                for role in ("male", "female"):
                    if role in roles_present:
                        portraits[role] = await _resolve_singer_portrait(
                            role, theme_eff, lyrics_text, brief, aspect_ratio,
                        )
                        await free_comfy()
                # Fix 32: see all_roles_present rationale above.
                if "duet" in all_roles_present:
                    # Stage G (2026-06-07): if duet sections exist but one
                    # solo role has no anchor segment (e.g. song with only
                    # female solos + mixed-gender duets), the solo loop
                    # above skipped generating that role's portrait, so
                    # _resolve_duet_portrait() below would fail the
                    # `male_p and female_p` guard and silently leave
                    # portraits["duet"] as None. Result on 63486a7a: duet
                    # segments fell back to the female portrait, and the
                    # video model invented a random male partner per shot
                    # instead of using a consistent duet composite. Force-
                    # generate the missing solo portrait(s) now so the
                    # composite always runs for mixed-gender duets.
                    for role in ("male", "female"):
                        if portraits.get(role) is None:
                            print(
                                f"[Stage G] no {role} solo anchor in song, "
                                f"force-generating {role} portrait for duet "
                                f"composite ({len(all_roles_present)} roles "
                                f"present, duet detected)"
                            )
                            portraits[role] = await _resolve_singer_portrait(
                                role, theme_eff, lyrics_text, brief, aspect_ratio,
                            )
                            await free_comfy()
                    male_p = portraits.get("male")
                    female_p = portraits.get("female")
                    if male_p and female_p:
                        # Fix 30 (B1): anchor duet portrait to first duet
                        # segment's wardrobe slot for clothing continuity.
                        duet_slot = next(
                            (s.wardrobe_slot for s in segments
                             if extract_section_role(s.label) == "duet" and s.wardrobe_slot),
                            "",
                        ) or (segments[0].wardrobe_slot if segments else "")
                        portraits["duet"] = await _resolve_duet_portrait(
                            male_p, female_p, theme_eff, brief,
                            wardrobe_slot=duet_slot,
                            duet_kind="mixed",
                        )
                        await free_comfy()
                source = (
                    portraits.get("male")
                    or portraits.get("female")
                    or next((v for v in portraits.values() if v), None)
                )
            else:
                # Legacy single-portrait path (upload/none source modes or only one role).
                source = await _resolve_source_image(
                    source_mode, source_image_bytes, source_image_name,
                    source_description, theme_eff, lyrics_text, brief,
                    aspect_ratio, consistent_character, tmp_dir,
                )

        # RC8 chorus reuse: only generate MCA frames for non-repeated segments.
        # Multi-role path: each anchor gets the portrait matching its role.
        # None-role anchors (Intro/Outro/Fade) are reassigned to their nearest
        # annotated neighbour's role so they share the same portrait instead
        # of triggering an extra MCA batch from the fallback `source`.
        frame_by_seg: dict[int, str] = {}
        if source:
            effective_role_groups: Dict[Optional[str], list[int]] = {
                r: list(idxs) for r, idxs in role_groups.items()
            }
            if is_multi_role and None in effective_role_groups:
                none_idxs = effective_role_groups.pop(None)
                # Stage C3 (2026-06-07): explicit-performer scan for story
                # segments. Before falling back to nearest-annotated-role
                # (which gave the Guitar Solo segment the Duet frame on
                # f3a5adf6, producing a 3-legged gold-blazer guitarist),
                # look at the segment's own prompt for an explicit
                # performer reference. A prompt that literally says
                # "male guitarist" must anchor to the male portrait, not
                # the neighbour's portrait.
                _male_re = re.compile(
                    r"\bmale (?:singer|guitarist|dancer|performer|vocalist|rapper)\b"
                    r"|\bhe wears\b|\bhis (?:guitar|microphone)\b",
                    re.IGNORECASE,
                )
                _female_re = re.compile(
                    r"\bfemale (?:singer|guitarist|dancer|performer|vocalist|rapper)\b"
                    r"|\bshe wears\b|\bher (?:guitar|microphone)\b",
                    re.IGNORECASE,
                )
                _duet_re = re.compile(
                    r"\b(?:male and female|female and male|duet|both sing(?:ers|ing)?)\b",
                    re.IGNORECASE,
                )

                def _prompt_performer_role(seg) -> Optional[str]:
                    text = " ".join(filter(None, (
                        seg.prompt or "",
                        seg.frame_variant_prompt or "",
                    )))
                    if not text.strip():
                        return None
                    # Order matters: duet keywords win over solo to avoid
                    # mis-classifying "male and female" as just "male".
                    if _duet_re.search(text):
                        return "duet"
                    has_m = bool(_male_re.search(text))
                    has_f = bool(_female_re.search(text))
                    if has_m and not has_f:
                        return "male"
                    if has_f and not has_m:
                        return "female"
                    if has_m and has_f:
                        return "duet"
                    return None

                for i in none_idxs:
                    target = _prompt_performer_role(segments[i])
                    if target is None or target not in effective_role_groups:
                        target = _nearest_annotated_role(segments, i)
                    if target is None or target not in effective_role_groups:
                        # No annotated neighbour at all (degenerate). Fall back
                        # to whichever portrait exists; insert into first bucket.
                        target = next(iter(effective_role_groups), None)
                        if target is None:
                            effective_role_groups[None] = [i]
                            continue
                    effective_role_groups.setdefault(target, []).append(i)
                # Keep each bucket's indices in original segment order so the
                # returned MCA frame list aligns with frame_by_seg lookups.
                for r in effective_role_groups:
                    effective_role_groups[r].sort()

            for role, idxs in effective_role_groups.items():
                if not idxs:
                    continue
                portrait = portraits.get(role) if role else None
                portrait = portrait or source
                # Fix 34: explicit fallback with sanitizer + log when fvp is
                # empty. The silent `fvp or prompt` chain used to feed the
                # full video_prompt (lyrics + camera moves + scene ends +
                # LIPSYNC_BOOSTER) to the T2I startframe model — which
                # rendered the lyrics literally as on-screen caption text.
                from music_video_pipeline import (
                    derive_still_prompt_from_video_prompt as _derive_still,
                )

                def _resolve_fvp_for_mca(seg_idx: int) -> str:
                    seg = segments[seg_idx]
                    if seg.frame_variant_prompt:
                        return strip_lyrics_from_image_prompt(
                            seg.frame_variant_prompt, lyrics=seg.lyrics,
                        )
                    print(
                        f"[Fix 34] frame_variant_prompt empty at MCA "
                        f"dispatch for segment {seg_idx} ({seg.label!r}); "
                        f"deriving sanitized fvp from video_prompt."
                    )
                    return _derive_still(seg.prompt, lyrics=seg.lyrics)

                prompts = [
                    enforce_performer_role(_resolve_fvp_for_mca(i), role)
                    for i in idxs
                ]
                if is_multi_role:
                    await free_comfy()  # RC10 VRAM hygiene between MCA passes
                frames = await _run_mca_variants(portrait, prompts)
                for k, i in enumerate(idxs):
                    frame_by_seg[i] = frames[k] if k < len(frames) else portrait
            for i, s in enumerate(segments):
                if s.reuse_of is not None:
                    frame_by_seg[i] = frame_by_seg.get(s.reuse_of, source)

        # Optional pause gate: lets the operator hold a job here (after T2I +
        # Flux2 + MCA, before the loud LTX video loop kicks off) by touching
        # the flag file. Job stays in "running" state but sits idle until the
        # flag disappears. Useful when wanting to time the noisy video phase.
        pause_flag = os.environ.get("MV_PAUSE_FLAG", "/tmp/mv_pause_before_ltx")
        if os.path.exists(pause_flag):
            print(f"[create-mv] pause flag {pause_flag} present — holding before LTX video loop")
            while os.path.exists(pause_flag):
                await asyncio.sleep(15)
            print("[create-mv] pause flag cleared — entering LTX video loop")

        # 4. Per-segment IA2V render (fresh frame each segment, no chaining,
        #    no inject_resolution — IA2V resolution is driven by the input image).
        out_dir = _outputs_dir()
        # Multi-role runs piled up T2I + Flux2 M-I Edit + ≥3 MCA passes before
        # LTX starts; without a defrag the first IA2V segment lands in LOWVRAM
        # mode (model partially offloaded to CPU → ~7 min/iter). One free here
        # tides the loop into the existing per-N free_every cadence.
        if is_multi_role:
            await free_comfy()
        free_every = _mca_cfg()["free_every"]
        for i, seg in enumerate(segments):
            # RC11: free VRAM BEFORE queuing this segment (not after the prior
            # one) so the next queue command lands on a fresh state. Boundary
            # is identical to the old RC10 placement; only the timing moved.
            if free_every > 0 and i > 0 and i % free_every == 0:
                await free_comfy()
            wf = load_workflow(video_workflow_id)
            # RC9: vocal segments (have lyrics) get the lip-sync booster so the
            # character actually sings; story/instrumental segments do not.
            seg_prompt = seg.prompt
            if (seg.lyrics or "").strip():
                seg_prompt = f"{seg.prompt} {LIPSYNC_BOOSTER}"
            # Stage 5: PromptRelay branch — used when the workflow contains a
            # PromptRelaySmartEncode node AND the LLM emitted a per-segment
            # multi-beat block. Falls back to the legacy single-prompt path
            # otherwise, so swapping the workflow file is the only switch
            # required to enable relay rendering per request.
            relay = seg.video_prompt_relay
            if relay and has_relay_smart_node(wf):
                beats = list(relay["beats"])
                if (seg.lyrics or "").strip() and beats:
                    # Append lipsync-booster to the LAST beat so it lands in
                    # the same latent region as the dialog beat per the
                    # recency rule encoded in the LLM system-prompt.
                    beats[-1] = f"{beats[-1]} {LIPSYNC_BOOSTER}"
                wf = inject_prompt(
                    wf,
                    build_smart_prompt(beats),
                    global_prompt=relay["global"],
                )
            else:
                wf = inject_prompt(wf, seg_prompt)
            frame = frame_by_seg.get(i, source or "")
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
            prompt_id, info = await queue_and_wait_with_recovery(wf)
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
            artist=settings_artist,
            title=settings_title,
        )
        mv_basename = _safe_filename(settings_artist, settings_title) \
            or f"music_video_{jid[:8]}"
        final_path = os.path.join(out_dir, f"{mv_basename}.mp4")
        assemble_video(session, final_path)

        # Fix 18 Component A: persist per-segment plan as JSON sidecar for
        # post-run analysis tools (analyze_mv_run.py).
        try:
            import re as _re_a
            def _portrait_role(label: str) -> str:
                m = _re_a.search(r' - (male|female|duet)\b', (label or "").lower())
                return m.group(1) if m else "story"
            seg_records = []
            for s in segments:
                lbl = getattr(s, "label", "") or ""
                seg_records.append({
                    "index": int(getattr(s, "index", 0)),
                    "start": float(getattr(s, "start_time", 0.0)),
                    "end": float(getattr(s, "end_time", 0.0)),
                    "section_label": lbl,
                    "is_vocal": bool(getattr(s, "lyrics", "")),
                    "portrait_role": _portrait_role(lbl),
                    "video_path": getattr(s, "video_clip", "") or "",
                    "prompt_snippet": (getattr(s, "prompt", "") or "")[:200],
                    "reuse_of": getattr(s, "reuse_of", None),
                    "status": getattr(s, "status", "completed"),
                })
            seg_json_path = os.path.join(out_dir, f"segments_{jid[:8]}.json")
            with open(seg_json_path, "w", encoding="utf-8") as _f:
                json.dump(seg_records, _f, ensure_ascii=False, indent=2)
            print(f"[create-mv] segments plan written: {seg_json_path} ({len(seg_records)} segments)")
        except Exception as _e:
            print(f"[create-mv] segments JSON write failed (non-fatal): {_e!r}")

        _job_done(jid, final_path, lyrics_path)
    except Exception as e:
        _job_failed(jid, str(e))
        raise
    finally:
        # Fix 25: clean per-job tmp_dir (ctb_cmv_*) to stop accumulating leaks.
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
    duet: str = Form(""),
    time_of_day_arc: str = Form(""),
    wardrobe_arc: str = Form(""),
    artist: str = Form(""),
):
    """Autonomous music-video creation. Lyrics are always pipeline-authored —
    callers pass a topic in `brief`, never finished lyrics.

    Multi-singer rendering is automatic: when ACE-Step-authored lyrics annotate
    section labels with role hints (e.g. "[Verse - male]", "[Chorus - female]",
    "[Bridge - duet]"), the pipeline generates one portrait per detected role
    and routes each segment to the matching portrait. Duet portraits use the
    Flux2 M-I Edit workflow with the two singer portraits as references.

    `duet` (Fix 27) is an explicit SAME-GENDER intent: "ff" = two female
    vocalists, "mm" = two male vocalists. When set (and the audio analysis does
    not contradict it with sustained opposite-gender vocals), solo sections are
    rendered with one consistent lead character and a SECOND distinct partner
    portrait is generated so the shared-duet frame depicts both people. Empty
    string keeps the standard male/female routing.
    """
    await _ensure_comfy_up_or_412()
    duet = duet if duet in ("ff", "mm") else ""
    # Fix 29 + Fix 30: validate arc form params against known arcs. Pass empty
    # string when caller's value is not recognized (pipeline will LLM-pick).
    try:
        from music_video_pipeline import (
            TIME_OF_DAY_ARCS as _TOD_ARCS,
            WARDROBE_ARCS as _WARDROBE_ARCS,
        )
        tod_arc = time_of_day_arc if time_of_day_arc in _TOD_ARCS else ""
        wardrobe_arc_eff = wardrobe_arc if wardrobe_arc in _WARDROBE_ARCS else ""
    except Exception:
        tod_arc = ""
        wardrobe_arc_eff = ""
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
        aspect_ratio, language, consistent_character, tmp_dir, duet, tod_arc,
        wardrobe_arc_eff, artist,
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
    tmp_dir: Optional[str] = None,
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
    finally:
        # Fix 25: clean per-job tmp_dir (ctb_sf_*) to stop accumulating leaks.
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/generate/short-film")
async def generate_short_film(
    premise: str = Form(...),
    style: str = Form("cinematic"),
    duration: float = Form(60.0),
    director_mode: str = Form("auto"),
    audio: Optional[UploadFile] = File(None),
):
    audio_path = None
    tmp_dir: Optional[str] = None
    if audio:
        tmp_dir = tempfile.mkdtemp(prefix="ctb_sf_")
        audio_path = os.path.join(tmp_dir, audio.filename or "audio.wav")
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

    jid = _new_job()
    asyncio.create_task(_run_short_film(jid, premise, style, duration, director_mode, audio_path, tmp_dir))
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
