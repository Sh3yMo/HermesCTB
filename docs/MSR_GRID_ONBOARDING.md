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

Hypothesis (partially tested): the **two-stage refine** (`759:1069_base`) regenerates
fine detail post-upsample → invents accessories / restyles the top.
**Tested: refine denoise 0.45 → 0.2** — top cut improved, BUT the **face got mushy**
(less refine = less sharpening), shoes still drifted, necklace still flickered.
Net **not a win** → reverted to 0.45. Takeaway: the two-stage refine is a sharpness
↔ invention tradeoff; you can't tune it to fix wardrobe without softening the face.
This is why **Strategy B (single-stage) is now the lead option** — it removes the
tradeoff entirely.

**cfg=1 caveat:** both CFGGuiders run `cfg=1` (distilled-LoRA requirement), so
**negative prompts are ignored** (classifier-free guidance off). Don't try to fix
invented accessories via a negative prompt — it has no effect. The only levers are
the reference image, the positive prompt, and the sampler/architecture.

## 6. Next planned steps (ranked, isolate ONE variable per render)

Already tested & rejected: **negative guidance** (ineffective — cfg=1) and
**refine denoise 0.2** (top better but mushy face — net not a win, reverted).
Those results point hard at the architecture, so:

1. **Strategy B — single-stage rebase (LEAD).** Re-base the frameless+audio MSR
   workflow on the **reference single-stage backbone**
   (`Workflows/Preview/MSR LTX Sample WF distill-lora-API.json`) and graft audio +
   PromptRelay onto IT, instead of grafting MSR onto the TA2V two-stage backbone.
   The reference holds a 241-frame clip clean with ONE 8-step pass → no refine
   sharpness/invention tradeoff, and roughly half the sampling work. Expected to
   fix wardrobe drift, mushy/plastic face, AND perf together. **Risk:** wiring the
   TA2V audio path (`LTXVConcatAVLatent`/`LTXVAudioVAEEncode`/mask) onto the
   single-stage graph. Approach: start from the reference JSON, add the audio
   encode + AV-concat into its single sampler's latent, add PromptRelay encode,
   keep LiconMSR + AddGuideMulti + IC-LoRA as-is. Validate with the same
   `verify_frameless_msr.py` + clean gold grid.
2. **Inplace bypass** (cheap, while still on two-stage) — set the 3×
   `LTXVImgToVideoInplace` (`759:1155/1163/1193`, strength 0.8) `bypass:true` and
   re-render to test their contribution to drift/shoes.
3. **PERF — VRAM/offload (systemic, biggest runtime win).** The GPU has only
   **12 GB dedicated VRAM**; LTX-2.3 **22B** (even Q4_K_M GGUF) doesn't fit →
   partial CPU offload (~5 GB offloaded) → base step time swings 32→206 s/it,
   render 12→40 min. NOT caused by the MSR changes. Levers: smaller UNET quant
   (Q3/Q2 GGUF in node `366`), `/free` after each render (a ~2 GB residual stays
   resident), reduce concurrent VRAM users, or `--reserve-vram` tuning. Strategy B
   also helps (one sampling stage instead of two).
4. **Validate the production grid** — the bad test grid (recolor + 2× side view)
   was the verify script's old `glob[:4]` stitching, now fixed. Confirm the
   PRODUCTION path (`api.py:1633-1664`: `build_view_prompts` appearance-pinning →
   `_run_mca_variants` → `compose_character_sheet`) yields an equally clean sheet
   for real songs (it should — it already pins appearance).

## 7. Gotchas

- **Rebuild after build-script edits** — the deployed JSON is regenerated; direct
  JSON edits are lost. Scoped MSR tweaks go in `build()` post-delta.
- **VRAM is the perf bottleneck** — 12 GB dedicated; the 22B model doesn't fit →
  partial offload → base step time is offload-bound and **highly variable**
  (32→206 s/it observed, same 8 steps). A fresh `/restart` does NOT guarantee a
  fast load. ~2 GB stays resident after a render. Don't attribute render-time
  swings to sampler-param changes — check `loaded partially ... offloaded` in the log.
- **Planting cost** — the in-context reference frames make the base stage the
  dominant cost; keep base steps low (8) and watch sequence length if adding guides.
- **Frame extraction past clip EOF** returns a misleading frame — always seek `< duration`.
- **Two supervisor copies** — `comfy_supervisor.py` (root) vs
  `host_supervisor/comfy_supervisor.py`; AGENTS.md says edit the host copy. The
  running service on `:8787` answers `/restart` and `/status`.
- **verify timeout** is 3600 s (planting renders can approach it under VRAM pressure).
- **cfg=1 → negative prompts are dead** (distilled-LoRA needs cfg=1, so CFG is off).
  Wardrobe/accessory drift cannot be fixed via the negative prompt — only via the
  reference image, positive prompt, or architecture.

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
