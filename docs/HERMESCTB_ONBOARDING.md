# HermesCTB — Komplettes Onboarding (Codex Handoff)

Self-contained Onboarding für die **gesamte HermesCTB-Suite**. Annahme: kein
vorheriger Chat-Kontext. `AGENTS.md` ist die Kurz-Map; dieses Dokument ist die
ausführliche Variante. Aktiver Detail-Workstream: `docs/MSR_GRID_ONBOARDING.md`.

Projekt-Root: `I:\HermesCTB`. Shell: **PowerShell** (Windows). Python: `py -3`.
Tests: `py -3 -m pytest tests/ -q`.

---

## 1. Was ist das

Backend-Service für **vollautomatische Musikvideo-Generierung** (plus Short-Film-
und Einzel-Asset-Pfade). Pipeline: Song → Segmentierung → pro Segment Keyframes/
Prompts → ComfyUI-Render (LTX-Video 2.3) → Stitch zu einem Video mit Audio.
LLM-Planung (Director/Prompt-Enhancer) via OpenRouter; Bild/Video/Audio via lokale
**ComfyUI**-Workflows.

## 2. Laufzeit & Entry Points

| Dienst | URL | Start |
|--------|-----|-------|
| HermesCTB API (FastAPI) | `http://127.0.0.1:8766` | `docker compose up -d` (Service `hermes-ctb-api-1`) oder `py -3 api.py` |
| ComfyUI (headless) | `http://127.0.0.1:8188` | über Supervisor `/restart` |
| ComfyUI-Supervisor | `http://127.0.0.1:8787` | `host_supervisor/comfy_supervisor.py` |

ComfyUI startet **nicht** als Electron-App, sondern headless über den Supervisor
(`--base-directory I:\ComfyUI`). Neustart:
```powershell
py -3 -c "import urllib.request as u; print(u.urlopen(u.Request('http://127.0.0.1:8787/restart',data=b'{}',headers={'Content-Type':'application/json'},method='POST'),timeout=180).read())"
```
Timing-/VRAM-Log des Servers: **`I:\ComfyUI\user\comfyui.log`** (die Electron-Log
unter `%APPDATA%\ComfyUI\logs` ist stale).

## 3. API-Endpunkte (`api.py`)

| Route | Zweck |
|-------|-------|
| `GET /health`, `/presets`, `/workflows` | Status / Registry |
| `GET /status/{job_id}`, `/output/{job_id}`, `/outputs/{path}` | Job-Status / Ergebnis |
| `POST /generate/image` \| `/video` \| `/music` | Einzel-Assets (ein ComfyUI-Workflow) |
| `POST /generate/music-video` | MV aus vorgegebenen Inputs |
| `POST /create/music-video` | **Autonom**: Song → Source → MCA → Video (Haupteinstieg, `_run_create_music_video`) |
| `POST /generate/short-film` | Short-Film-Pfad |
| `POST /enhance/prompt` | Prompt-Enhancer standalone |

## 4. Module-Map

**Orchestrierung / Pipelines**
- `api.py` (2208) — FastAPI, Job-Handling, MSR-Asset-Orchestrierung, MCA-Aufrufe.
- `music_video_pipeline.py` (3473) — Kern: Segmentierung, Timeline, Wardrobe-/
  Tageszeit-Arcs, Duett-Logik, Relay-Specs, Stitch (`assemble_video`),
  `MusicVideoPrompter`, Datenklassen `Segment`/`MVSession`.
- `short_film_pipeline.py` (1127) — Director → Script → Prompt → Keyframe → Render → Stitch.
- `workflow_registry.py` (366) — Workflow-Katalog/Gruppen (`Workflows/registry`).

**Agents (LLM)**
- `director_agent.py` / `mv_director.py` — strukturierter Szenen-/Segment-Breakdown aus Story/Song.
- `prompt_enhancer.py` (2888) — Prompt-Veredelung, Genre-Validierung, OpenRouter-LLM-Calls.
- `keyframe_agent.py` — First/Last-Frame-Paare pro Segment via **Qwen IE Rapid** (+Multi-Angle/Next-Scene LoRAs).
- `frame_validator.py` — Vision-LLM-Validierung generierter Frames (Score, Retry).

**Audio**
- `audio_enhancer.py` (1356) — Musikgenerierung via **ACE-Step 1.5**.
- `audio_gender_detect.py` — Stimm-Geschlecht (steuert Performer-Rolle/Portraits).
- `lyric_align.py` / `voice_transcriber.py` — Lyrics-Timestamps, Segment-Cuts auf Vokal-Grenzen, Chorus-Reuse.

**ComfyUI-Anbindung**
- `comfyui.py` (1196) — Workflow laden/queuen, Injektion (Prompt, Audio, Startframe,
  **MSR** via `inject_msr_images`), Auto-Recovery + `SlowdownMonitor`.
- `comfyui_patches.py` — Workflow-Patches.
- `comfy_supervisor.py` / `host_supervisor/comfy_supervisor.py` — Headless-Launch + `/restart` (host-Copy ist die aktive).
- `msr_refs.py` — MSR-Referenz-Assets (Views, Character-Sheet, Reference-Block). Siehe MSR-Doc.

**Support**
- `config_loader.py` — lädt `config.json`. `_ffmpeg_init.py` — ffmpeg-Pfad.
- `llm_tester.py`, `analyze_mv_run.py` — Diagnose/Analyse.

## 5. MV-Flow (vereinfacht)

```
Song (gegeben/ACE-Step) → lyric_align (Timestamps)
  → segment_audio / build_segment_timeline (Segmente, Wardrobe-/TOD-Arcs, Rollen)
  → director/prompt_enhancer (pro Segment: Szene/Kamera/Prompt via OpenRouter)
  → Portrait(s) + (MSR) Character-Sheet / per-Segment Keyframes (Qwen MCA, Flux2)
  → ComfyUI Render pro Segment (LTX-2.3 Workflow; bei MSR frameless)
  → frame_validator (optional) → extract_last_frame (Kontinuität)
  → assemble_video (Crossfade + Audio-Mux) → Endvideo
```
Outputs landen unter `outputs/<YYYY-MM-DD>/` (`_outputs_dir()` in api.py).

## 6. Workflows (`Workflows/`)

~35 ComfyUI-JSONs. Familien: **LTX2.3 / LTX2.3 (4.2)** (T2V, I2V, IA2V, TA2V,
V2V, FFLF…, PromptRelay-Varianten, **-MSR**), **Flux2 Klein** (T2I/I2I/M-I Edit),
**F2K9B MCA** (Multi-Angle-Views), **Qwen IE Rapid** (Keyframes), **ACE-Step 1.5**
(Musik), **Music-Analyzer**, **Z-Image**. Auswahl/Gruppen über `workflow_registry.py`.
Aktiver MSR-Video-WF: `Workflows/LTX2.3 - IA2V-PromptRelay-MSR.json` (generiert).

## 7. Konfiguration (`config.json`)

Aus `config.example.json` ableiten (`config.json` enthält Secrets — nicht
committen; ein Hook schreibt `config.backup.json` mit geblankten Secrets). Keys:
- `comfyui_url`, `workflows_dir`, `comfy_view_timeout_seconds`, `comfy_view_retries`.
- `comfy_recovery` — `enabled`, `restart_service_url` (Supervisor), `max_restarts_per_job`,
  `job_timeout_seconds`, `slowdown_detection`.
- `prompt_enhancer` — `openrouter_enabled`, `openrouter_api_key`, `openrouter_model`,
  `fallback_models`, `audio_llm_model`, `genre_validation_model`.
- `music_video.mca` — F2K9B-MCA: `workflow`, `input_node`, `prompt_node`,
  `output_node`, `mca_batch_size`, `t2i_workflow`.

## 8. ComfyUI-Resilienz (`comfyui.py`)

Renders sind lang (LTX-2.3-22B, partielles VRAM-Offload). `queue_and_wait_with_recovery`:
- Tier 1 (Versuch 2): `/free` + Retry. Tier 2 (Versuch 3): voller ComfyUI-Restart via Supervisor.
- `SlowdownMonitor`: WS-Step-Timing; bricht bei Slowdown/WS-Stille ab (Frozen-Sampling-Schutz).
- **Gotcha**: kumulative VRAM-Degradation über viele Back-to-Back-Renders → vor
  Timing-Vergleichen ComfyUI neu starten.

## 9. MSR (aktiver Workstream)

Multiple Subject Reference: ein Character-Sheet (2×2-Turnaround) führt die
Video-Generierung, damit der Performer grid-treu bleibt. **Voller Stand + nächste
Schritte: `docs/MSR_GRID_ONBOARDING.md`.** Kurz:
- Frameless-MSR-WF wird via `scripts/build_frameless_msr_wf.py` aus TA2V + Delta
  generiert; alle Tweaks in `build()` post-delta (NICHT im JSON — wird überschrieben).
- Sheet-Planting (Node 2009) ankert Identität; Test: `scripts/build_msr_sheet.py`
  (sauberes Gold-Grid) → `scripts/verify_frameless_msr.py`.
- Referenz-Oracle: `Workflows/Preview/MSR LTX Sample WF distill-lora-API.json` (einstufig).
- Stand: Identität+Farbe+Schuhe matchen, Render ~12min. Offen: erfundene Accessoires,
  Top-Schnitt, Plastik-Look. Gerankte nächste Schritte in der MSR-Doc.

## 10. Tests & Dev-Workflow

- Tests: `py -3 -m pytest tests/ -q` (37 Dateien; viele "fixNN"-Regressionstests
  für Wardrobe-/Gender-/Lyrics-/Relay-/Recovery-Logik). Bei Änderungen grün halten.
- Naming der Test-Files spiegelt die Stage-/Fix-Historie der Pipeline.
- Vor Commits: relevante Tests laufen lassen. Commits direkt auf `master` (User-Vorgabe).

## 11. Gotchas (projektweit)

- **Workflows generiert, nicht handgepflegt** (MSR): Tweaks ins Build-Script.
- **ComfyUI neu starten** vor Timing-Vergleichen (VRAM-Frag).
- **Frame-Extraktion**: nie über Clip-EOF hinaus seeken (liefert Bogus-Frame).
- **Supervisor-Copy**: `host_supervisor/comfy_supervisor.py` editieren, nicht die Root-Copy.
- **Secrets**: `config.json` nicht committen; `config.example.json` pflegen.
- **PowerShell-Syntax** (kein `/dev/null`, `$env:VAR`); native exe stderr nicht via `2>&1` umleiten.
- **Codegraph** ist über `I:\HermesCTB` indiziert — für "wie hängt X zusammen" zuerst nutzen.

## 12. Wichtige Dateien (Schnellzugriff)

`api.py` · `music_video_pipeline.py` · `comfyui.py` · `prompt_enhancer.py` ·
`msr_refs.py` · `keyframe_agent.py` · `workflow_registry.py` ·
`host_supervisor/comfy_supervisor.py` · `config.example.json` · `tests/` ·
`Workflows/` · `docs/MSR_GRID_ONBOARDING.md`.
