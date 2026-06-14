# HermesCTB - Music-Video-Pipeline Orchestrator

## Was ist das

Docker-basierter Backend-Service für vollautomatische Musikvideo-Generierung. ComfyUI-Orchestrierung mit LTX-Video 2.3, MSR, IA2V und PromptRelay-Workflows.

## Architektur

- `api.py` - FastAPI, Port 8766, Haupteinstieg für Render-Jobs
- `comfy_supervisor.py` - ComfyUI-Supervisor, Port 8787, Restart-Bridge
- `host_supervisor/comfy_supervisor.py` - aktive Supervisor-Instanz, diese editieren, nicht Root-Copy
- `comfyui.py` - ComfyUI-Workflow-Injektion, Prompt, Audio, Startframe, MSR
- `music_video_pipeline.py` - Pipeline-Orchestrierung, Segmente, Startframes, Audio-Sync
- `msr_refs.py` - MSR-Referenzsheet-Generierung, 2x2 Portrait-Grid
- `Workflows/` - LTX-Workflow JSONs

## Laufzeit

- API: `http://127.0.0.1:8766`, Docker-Service `hermes-ctb-api-1`
- ComfyUI: `http://127.0.0.1:8188`
- ComfyUI-Supervisor: `http://127.0.0.1:8787`
- Start Docker: `docker compose up -d`
- Tests: `py -m pytest tests/ -q`

## Wichtige Konzepte

- Frameless MSR: MSR-Workflow nutzt Empty-Latent statt Startframe als Guide, verhindert Charakter-Swap und Referenz-Grid-Bleed im I2V-Modus.
- Supervisor-Restart: `host_supervisor/comfy_supervisor.py` startet ComfyUI headless direkt. `/restart` Endpoint auf Port 8787.
- Startframe-Gate: Startframe-Generierung deaktiviert, wenn MSR aktiv ist.
- MSR-Sheet: 2x2 Portrait-Layout, 4 Views, weiß-frei.

## Workflow-Dateien

- Aktiv: `Workflows/LTX2.3 - IA2V-PromptRelay-MSR.json`
- Backup: `Workflows/LTX2.3 - IA2V-PromptRelay-MSR.json.bak3`
- Frameless-Builder: `scripts/build_frameless_msr_wf.py`
- Codegraph-Projektpfad: `I:\HermesCTB`.

## Onboarding-Dokumente

- **Komplette Suite (zuerst lesen): `docs/HERMESCTB_ONBOARDING.md`** — Architektur,
  alle Module, MV-Flow, API, Workflows, Config, Resilienz, Tests, Gotchas.
- **Aktiver Workstream MSR Grid-Konsistenz: `docs/MSR_GRID_ONBOARDING.md`** — Deep-Dive
  + gerankte nächste Schritte zur grid-treuen Videogenerierung.

## Aktiver Workstream: MSR Grid-Konsistenz (Kurzfassung)

Onboarding + nächste Schritte zur Grid-treuen Videogenerierung (Identität,
Wardrobe, Farbe): **`docs/MSR_GRID_ONBOARDING.md`**.
Kurz: Frameless-MSR-WF wird via `build_frameless_msr_wf.py` aus TA2V + Delta
generiert (alle Tweaks in `build()` post-delta, NICHT im JSON); Sheet-Planting
(Node 2009) ankert die Identität; Test via `build_msr_sheet.py` (sauberes Gold-Grid)
→ `verify_frameless_msr.py`. Referenz-Oracle: `Workflows/Preview/MSR LTX Sample WF
distill-lora-API.json` (einstufig). Timing-Log: `I:\ComfyUI\user\comfyui.log`.
