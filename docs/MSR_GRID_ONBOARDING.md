# MSR Grid-Consistency — Onboarding & Next Steps (Codex handoff)

Self-contained handoff for continuing the **MSR (Multiple Subject Reference)
grid-consistency** workstream on HermesCTB with Codex. Assumes no prior chat
context. Read `AGENTS.md` first for the general project map.

Project root: `I:\HermesCTB`. Shell: PowerShell on Windows. Python: `py -3`.

---

## 1. Goal

A music-video segment is rendered by an LTX-2.3 ComfyUI workflow guided by an
**MSR character sheet** (a 2×2 turnaround grid of the performer). The rendered
person must match that grid — same face, hair, **wardrobe (incl. exact bikini
color/cut)**, body, shoes — and stay consistent across the whole clip.

"Grid consistency" = how faithfully the generated video reproduces the sheet.

## 2. Current state (what works)

Two commits on `master`:
- `948e27d` — identity fix (frameless MSR matched the grid for the first time).
- `890c067` — perf (base stage 8 steps) + clean test-grid regeneration.

After these, against a **clean gold grid**: identity (face/hair/skin), bikini
**color**, and shoes match; render ~**12 min** (was 40). Verified via
`scripts/verify_frameless_msr.py`.

### The three things that fixed identity (root causes, all vs the reference WF)
1. **`frame_count`** — `inject_msr_images` ([comfyui.py:239](../comfyui.py)) now
   pins LiconMSR `frame_count = _MSR_FRAME_COUNTS[-1]` (41), not the old
   "smallest valid" 17. Short reference video → identity only at clip start.
2. **reasoning_I2V LoRA** — `build_frameless_msr_wf.py` enables node `211`
   `lora_3` (`LTX2.3_reasoning_I2V_V3`) `on @0.5` (reference runs it; TA2V ships it off).
3. **Sheet guide-planting (decisive)** — added node `2009` `LTXVAddGuideMulti`
   that plants the sheet (+bg) at frame 0 into the **base latent**
   (`759:1055.video_latent <- 2009:2`). Without it only the weak IC-LoRA
   cross-attn guide remained → the prompt dominated → a *different* person.
   Mirrors reference node `85`.

## 3. Architecture (the MSR video workflow)

Active WF: `Workflows/LTX2.3 - IA2V-PromptRelay-MSR.json` (95 nodes, **frameless**:
no start frame; MSR guide + audio drive the clip).

It is **generated**, not hand-edited — `scripts/build_frameless_msr_wf.py`:
```
frameless-MSR = TA2V backbone  +  (PromptRelay + MSR) delta (from .bak3)
```
then applies post-delta tweaks in `build()`:
- (3) enable reasoning_I2V LoRA on `211`,
- (4) add planting node `2009` → `759:1055`,
- (5) base `LTXVScheduler` `759:1065` `steps → 8`.
⇒ **All scoped MSR tweaks live in `build()`, NOT in the JSON** — the JSON is
overwritten on every rebuild. Editing the JSON directly is lost on next build.

Key nodes:
- `2006` `LiconMSR` — turns the sheet (slot 1) + bg (slot 5) into a `frame_count`
  reference video. `inject_msr_images` patches the slots at runtime and drops unused ones.
- `2007` `LTXICLoRALoaderModelOnly` — MSR LoRA in the **model path**:
  `211` PowerLora (distill 0.6 + reasoning_I2V 0.5) → `2007` → `1700` → `700` → `759:1075` → both guiders.
- `2008` `LTXAddVideoICLoRAGuide` — IC-LoRA **cross-attn conditioning** (in-context reference frames).
- `2009` `LTXVAddGuideMulti` — **base-latent planting** of the sheet (identity anchor).
- Two-stage sampler: base `759:1054` (`LTXVScheduler 759:1065`, 8 steps) → upsample
  `759:1070` → refine `759:1066` (`BasicScheduler 759:1069_base`, linear_quadratic
  4 steps **denoise 0.45**, + `Sigmas Rescale 759:1069`). Decode `759:1071` VAEDecodeTiled.

**Reference oracle** (proven to match grids): `Workflows/Preview/MSR LTX Sample WF
distill-lora-API.json` — a clean **single-stage** 8-step euler MSR sampler
(node `9` IC-LoRA + node `85` AddGuideMulti + node `28` LiconMSR fc 41), **no audio**.
Use it to diff against ours whenever behavior diverges.

## 4. How to run the end-to-end test

ComfyUI is launched headless by the supervisor (NOT the Electron app):
```powershell
# 1. Start / restart ComfyUI (headless server on :8188 via supervisor on :8787)
#    POST /restart kills any running ComfyUI and relaunches; waits for online.
py -3 -c "import urllib.request as u; print(u.urlopen(u.Request('http://127.0.0.1:8787/restart',data=b'{}',headers={'Content-Type':'application/json'},method='POST'),timeout=180).read())"

# 2. Rebuild the workflow after any build_frameless_msr_wf.py change
py -3 scripts/build_frameless_msr_wf.py        # must print "validation OK"
py -3 -m pytest tests/test_msr_integration.py -q

# 3. (Re)build the CLEAN test grid from the gold master portrait
py -3 scripts/build_msr_sheet.py               # -> outputs/2026-06-13/msr_refs/sheet_gold_v2.png

# 4. Run one segment end-to-end (frameless, clean grid, audio)
py -3 scripts/verify_frameless_msr.py          # -> outputs/2026-06-13/seg_videos/verify_frameless_msr.mp4
```

Inspect result (extract frames WITHIN clip length — clip is ~8.4 s / 201 frames;
seeking past EOF returns a bogus/stale frame — a mistake made before):
```powershell
ffmpeg -y -ss 3.0 -i "outputs/2026-06-13/seg_videos/verify_frameless_msr.mp4" -frames:v 1 frame.png -loglevel error
```

**Per-node timing / progress** lives in the headless server log (the Electron
`%APPDATA%\ComfyUI\logs\comfyui.log` is STALE — wrong one):
`I:\ComfyUI\user\comfyui.log` — grep `got prompt`, `Prompt executed in`, `s/it`,
`loaded partially`/`offloaded` (VRAM).

## 5. Known remaining issues (grid drift)

With the clean gold grid, identity/color/shoes match, but these still drift:
- **Top cut** slightly off vs the sheet's triangle top.
- **Invented accessories** — bracelets + a necklace/choker appear (not in grid).
- **Body shape** slightly off.
- **"Plastic" look + somewhat unnatural body motion** (also present in the
  reference render → likely inherent to the LTX-2.3 distilled-LoRA / few-step / cfg=1 regime).

Hypothesis (unproven): the **two-stage refine** (`759:1069_base`, denoise 0.45)
regenerates fine detail post-upsample → invents accessories / restyles the top /
smooths skin. The single-stage reference doesn't have this stage.

## 6. Next planned steps (ranked, isolate ONE variable per ~12-min render)

1. **Negative guidance against accessories** (cheapest). Add explicit negatives
   ("no jewelry, no bracelets, no necklace, no choker") to the negative prompt /
   MSR reference block so the model stops inventing them. Entry points:
   `build_msr_reference_block` ([msr_refs.py:259](../msr_refs.py)) and the
   negative-conditioning path (`759:1067`/`759:1549`). Test on the verify render.
2. **Soften the refine** — `759:1069_base` (`BasicScheduler`) `denoise 0.45 → 0.2`
   in `build_frameless_msr_wf.py` `build()` (post-delta, like steps 3-5). Less
   regeneration → fewer invented details, less plastic. Re-render, compare.
3. **Inplace bypass** — set the 3× `LTXVImgToVideoInplace` (`759:1155/1163/1193`,
   strength 0.8) `bypass:true` and re-render to test their contribution to drift.
4. **Strategy B — single-stage rebase** (biggest, highest payoff). Re-base the
   frameless+audio MSR workflow on the **reference single-stage backbone**
   (`MSR LTX Sample WF distill-lora-API.json`) and graft audio + PromptRelay onto
   IT, instead of grafting MSR onto the TA2V two-stage backbone. The reference
   holds a 241-frame clip clean with one 8-step pass — this would address
   wardrobe drift, plastic look, AND perf together. Audio integration is the risk.
5. **Validate the production grid** — the bad test grid (recolor + 2× side view)
   came from the verify script's old `glob[:4]` stitching, now fixed. Confirm the
   PRODUCTION path (`api.py:1633-1664`: `build_view_prompts` appearance-pinning →
   `_run_mca_variants` → `compose_character_sheet`) yields an equally clean sheet
   for real songs (it should — it already pins appearance).

## 7. Gotchas

- **Rebuild after build-script edits** — the deployed JSON is regenerated; direct
  JSON edits are lost. Scoped MSR tweaks go in `build()` post-delta.
- **Cumulative VRAM degradation** — back-to-back long renders slow ComfyUI
  (a 64 s/it base stage degraded to 167 s/it across the day). **Restart ComfyUI**
  (supervisor `/restart`) before timing comparisons.
- **Planting cost** — the in-context reference frames make the base stage the
  dominant cost; keep base steps low (8) and watch sequence length if adding guides.
- **Frame extraction past clip EOF** returns a misleading frame — always seek `< duration`.
- **Two supervisor copies** — `comfy_supervisor.py` (root) vs
  `host_supervisor/comfy_supervisor.py`; AGENTS.md says edit the host copy. The
  running service on `:8787` answers `/restart` and `/status`.
- **verify timeout** is 3600 s (planting renders can approach it under VRAM pressure).

## 8. Key files

| Path | Role |
|------|------|
| `scripts/build_frameless_msr_wf.py` | Generates the active MSR WF (TA2V + delta + tweaks) |
| `scripts/build_msr_sheet.py` | Regenerates the clean gold test grid from the master portrait |
| `scripts/verify_frameless_msr.py` | One-segment end-to-end test render (frameless, clean grid, audio) |
| `comfyui.py` | `inject_msr_images`, `inject_prompt`, `inject_input_audio`, workflow load/queue |
| `msr_refs.py` | `build_view_prompts`, `compose_character_sheet`, `build_msr_reference_block`, `allocate_msr_subjects` |
| `api.py` | Production orchestration; MSR sheet build at `1633-1664`; `_run_mca_variants` at `1037` |
| `tests/test_msr_integration.py` | MSR wiring + injection tests (keep green) |
| `Workflows/LTX2.3 - IA2V-PromptRelay-MSR.json` | Active (generated) WF |
| `Workflows/Preview/MSR LTX Sample WF distill-lora-API.json` | Reference oracle (single-stage) |
| `I:\ComfyUI\user\comfyui.log` | Live headless server log (timing/VRAM) |
