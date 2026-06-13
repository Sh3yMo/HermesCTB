# Perf: ~15 min/still on the 12GB RTX 3060 (Flux2-Klein 9B T2I)

Branch: `perf/still-vram-encoder`. Prepared during the Stage-P E2E test run while
the visual fixes were verified (portrait was perfect). NOT yet A/B-confirmed.

## Root cause (measured)

- GPU: **RTX 3060, 12 GB VRAM**.
- The still workflows (`Flux2 Klein 9B - T2I`, `F2K9B MCA`, `Flux2 Klein M-I Edit`)
  load the **Flux2-Klein 9B UNet (~9 GB)** + the **Qwen-8B-fp8 text encoder
  (~8.1 GB)** = **~17 GB ≫ 12 GB**, forcing partial offload.
- The T2I scheduler runs only **4 steps**, yet a still takes **~900 s** → ~3 min
  *per step* = classic offloaded sampling (the LTX crash log showed the same
  shape: 191 s/it). The first portrait: `[still] Flux2 Klein 9B - T2I: 899.7s`;
  the second still was still running at 807 s → systemic, not cold-start.
- At ~15 min/still and ~17 stills (portrait + 4 MCA views + 6 backgrounds +
  6 Flux2 segment frames) the image phase alone is **~4 h** before LTX.

This is pre-existing (see memory obs 2228, 2026-06-11) — independent of the
Stage-P visual changes.

## Levers (in likely-impact order)

1. **Smaller still text encoder (implemented here, flag-off by default).**
   `_swap_text_encoder_if_configured()` swaps any `qwen_3*` CLIP loader in the
   still workflows when `music_video.mca.t2i_text_encoder` is set. Candidates
   already present in `I:/ComfyUI/models/text_encoders/`:
   - `qwen_3_8b_fp4mixed.safetensors` (6.4 GB) — same model, lower precision, safe.
   - `qwen_3.5_2b_bf16.safetensors` (4.3 GB) — much smaller, bigger saving, verify
     compatibility/quality.
   Hypothesis: if ComfyUI keeps the encoder resident alongside the UNet, dropping
   it from 8.1 → 4.3 GB pulls 9 + 4.3 = 13.3 GB close enough to 12 GB to largely
   eliminate sampling offload. **Needs A/B** (measure `[still]` seconds + check
   the portrait still looks right).
   - Applied at: `_generate_still` (portrait + backgrounds) and both Flux2 funcs
     (segment frames + duet). **TODO:** also apply inside `_run_mca_variants`
     (the 4 MCA views) once the swap is confirmed to help.

2. **Lower still resolution.** Fewer activation bytes → less offload. The MSR
   sheet cells are 768 px anyway; backgrounds render at ~1 MP. A 0.6–0.75 MP cap
   for stills would cut VRAM during sampling.

3. **Stop freeing between same-model stills.** `_generate_still` and
   `_run_mca_variants` call `free_comfy()` before *every* still (Stage O7). The
   portrait, MCA views and backgrounds all use the same 9B model — keeping it
   resident across them avoids redundant reloads. Risk: re-introduces the
   fragmentation O7 was added to fix; gate behind a flag and measure.

4. **Hardware / smaller UNet quant.** Only `flux-2-klein-9b-Q8_0.gguf` (9.3 GB)
   is present — same size as fp8mixed, so no help. A Q5_K/Q4_K Klein GGUF
   (~6–7 GB) would fit with headroom; would need downloading.

## How to A/B (after the current run finishes)

1. Set in `config.json` → `music_video.mca.t2i_text_encoder` =
   `"qwen_3_8b_fp4mixed.safetensors"`.
2. Restart the API (`start_api.bat`).
3. Fire a short job; compare `[still] ...: Ns` in the log vs the ~900 s baseline,
   and eyeball the portrait quality/identity.
4. If good, try `qwen_3.5_2b_bf16.safetensors` for a bigger win; extend the swap
   to `_run_mca_variants`; consider levers 2–3.
