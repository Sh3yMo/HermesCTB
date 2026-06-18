# Hand-off: Fix 26 — Same-Gender Duet Support (female+female, male+male)

Copy this entire file as the first message in the new Claude Code session.

---

## Goal

Extend the music-video pipeline so a song with **two distinct same-gender singers** (e.g. an Asian woman + a Latina woman, or two men) is rendered correctly: a separate portrait per singer **plus** a duet portrait of both — for both `female + female` and `male + male` cases.

## Current Architecture Limit (verified read-only)

- Portrait routing in `_run_create_music_video` ([api.py:950-963](api.py:950)) iterates only `("male", "female")` → exactly one portrait per gender role.
- `_resolve_duet_portrait(male_image_path, female_image_path, ...)` ([api.py:642](api.py:642)) **requires** both `male_p AND female_p` — if one is missing, no duet portrait is generated and duet sections fall back to whatever single portrait exists.
- Role labels (`[Verse - male]`, `[Chorus - duet]`, …) are emitted by the LLM lyric author and parsed via `extract_section_role` + `partition_anchors_by_role` in `music_video_pipeline.py`.
- ACE-Step caption tags are built from the same labels via `_build_vocal_role_tags` in `audio_enhancer.py` (Fix 22, deduped on `(section, role)` pair — only `male|female|duet` recognised).

## Concrete Symptom Just Observed

A run with the brief
> *"An English-language reggae beach anthem performed by TWO FEMALE vocalists ONLY: an East-Asian woman and a Latina woman with a feminine curvy hourglass build, both in bikinis on a tropical beach. Alternate solo female verses and shared DUET choruses; no male singer."*

produced lyrics with 4× `[… - female]` + 3× `[… - duet]` (no `male`) → pipeline generated **one** female portrait (the LLM picked the Asian woman; the Latina is ignored) and **zero** duet portraits (no male reference for `_resolve_duet_portrait`). Run: `outputs/2026-05-30/` (sidecar + lyrics file present).

## Required Capability

1. **Two distinct singers per gender**: support a role scheme like `female1` / `female2` (and `male1` / `male2`) so the brief can describe two same-gender singers individually and each gets its own portrait.
2. **Same-gender duet portrait**: `_resolve_duet_portrait` (or a new helper) must work with any two single-singer portraits — not just `male+female`. The Flux2 Klein M-I Edit workflow has 3 LoadImage slots; current code duplicates the male into slot 36 (status-quo policy, Fix 24B). For same-gender, duplicate one of the two refs into slot 36 with the same intent.
3. **ACE-Step tags**: `_build_vocal_role_tags` must understand the new role labels and emit caption tags ACE-Step can interpret. ACE-Step only knows generic `male/female/duet` voice quality — it cannot distinguish two women — so the tags should keep `female` / `duet` as the **voice timbre**, while internal routing uses the per-singer label. (Honest scope note: voice timbre between two women may sound similar — this is a known ACE-Step 1.5 model limit, not a pipeline bug.)

## Design Decisions to Make in the New Session

- **Role label scheme** — backwards-compatible? Options: (a) extend the label parser to accept `[Verse - female1]` / `[Chorus - female2]`, (b) keep `female` but add a `voice_id`/`singer_id` JSON sidecar, (c) compound key `[Chorus - duet-ff]` for same-gender duet.
- **LLM steering** — how to make the lyric-author LLM produce per-singer labels when the brief asks for two same-gender singers. The prompt that controls this lives in `audio_enhancer.py` (search for `_build_vocal_prompt` or the system prompt that mentions `male/female/duet`). May need: "when the brief specifies two same-gender singers, label sections as `[Verse - female1]` / `[Verse - female2]` and choruses as `[Chorus - duet]`".
- **Duet-portrait prompt** — `build_duet_portrait_prompt(theme)` in `music_video_pipeline.py` is already identity-neutral and unchanged-friendly (Fix 24A); no edit expected.
- **Routing** — `partition_anchors_by_role`, `extract_section_role`, `_resolve_singer_portrait`'s `_ROLE_SEED_PREFIX` ([api.py:559-577](api.py:559)) need new role keys and seed prefixes (e.g. "Performer for sections sung by FEMALE SINGER 1 — visually distinct from female singer 2…").

## Critical Files

- `api.py`: `_resolve_singer_portrait` (line 580), `_resolve_duet_portrait` (642), portrait-routing block in `_run_create_music_video` (~940-970), `_ROLE_SEED_PREFIX` (559-577).
- `music_video_pipeline.py`: `extract_section_role`, `partition_anchors_by_role`, `enforce_performer_role`, `build_duet_portrait_prompt`, `MVPromptGenerator.plan_segments` (role-aware segment prompts).
- `audio_enhancer.py`: `_build_vocal_role_tags` (~917), `_build_vocal_prompt` (LLM lyric-author system prompt — controls which labels the LLM emits).
- `tests/test_vocal_gender_tags.py` + `tests/test_music_video_helpers.py`: extend with same-gender cases (TDD).

## Verification Approach

- **New unit tests**: `_build_vocal_role_tags` with `[Verse - female1]` / `[Verse - female2]` → emits both with the right voice timbre. `partition_anchors_by_role` with new keys. `build_duet_portrait_prompt` unchanged. Routing test: same-gender duet picks both single portraits, generates duet via `_resolve_duet_portrait`.
- **Full suite**: `py -3 -m pytest -q` — baseline currently 171 passed.
- **Live MV run** with brief *"two female singers, Asian + Latina, with shared duet choruses"* → confirm sidecar `outputs/<date>/segments_*.json` has two distinct portrait paths and a duet portrait. Compare visually.

## Recent Reference Commits on `master`

- `f2a50c5` Fix 22 — vocal-role caption `(section, role)` dedup (female no longer drops).
- `5062651` Fix 23 — `_call_openrouter` 429 retry + backoff + fallback_models; portrait empty-fallback no longer leaks raw lyrics; top-level `OPENROUTER_API_KEY` env fallback.
- `0c3f8a8` Fix 24A — deterministic identity-neutral duet portrait prompt (no LLM call).
- `57e7933` Fix 25 — api.py per-job `tmp_dir` cleanup via `try/finally + shutil.rmtree`.
- `0a136a1` config.backup feature + pre-commit hook (`scripts/backup_config.py` + `.githooks/pre-commit`).

## Environment Caveats Worth Knowing Upfront

- `config.json` is gitignored and currently lives only on disk (reconstructed earlier this session from `config.example.json` + observed values + the OpenRouter key). Bitdefender previously quarantined it because of the `sk-or-v1-…` string — the I: drive is now on the Bitdefender exclusion list. Be aware before editing config or moving the key around.
- ComfyUI Desktop runs natively on the Windows host (port 8188); the HermesCTB API runs in the `hermes-ctb-api-1` Docker container (port 8766) with `I:/HermesCTB → /app` bind mount + `uvicorn --reload`. Editing `api.py` triggers an immediate reload; do not edit `api.py` while a long-running job is in flight unless you intend to interrupt it.
- Python invocation on this Windows machine is `py -3` (Python 3.12.10). Plain `python` / `python3` fail on PATH.
- Docker WSL2 VHDX lives on `I:` (~103 GB sparse), so `/tmp` inside the container does NOT eat C:. Per-job temp leaks were fixed in Fix 25.

## Out-of-Scope Reminder

ACE-Step 1.5 cannot reliably distinguish two voices of the same gender from caption tags alone — the model only conditions on `male / female / duet` voice quality. The pipeline can plan and route per-singer (so the visual side is correct), but the rendered voice timbre between two women may sound similar. Document this honestly in the fix instead of trying to force ACE-Step to do something it cannot.
