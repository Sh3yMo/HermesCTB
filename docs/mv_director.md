# MV Producer-Style Director (Stage K)

Adds empirically researched **producer style profiles** to `plan_segments()` so
the LLM gets explicit per-segment shot directives instead of choosing freely
(which always defaulted to midframe and occasionally produced cross-genre
artefacts like a DJ pult in a rock song).

## Architecture

```
plan_segments(aligned_sections)
  └─► MVDirector
        ├─ classify_sub_genre_and_sentiment()   # 1 cheap LLM call via text_model
        ├─ select_producer_profile()            # seeded pick per song
        ├─ apply_sub_genre_modifiers()          # closeup_freq, edit tempo, forbidden_extra
        ├─ build_shot_plan()                    # per-segment shot/motion/light/mood
        └─ render_director_brief()              # numbered RULES block
              └─► appended to aligned_system prompt
```

Fully orthogonal to Stage F/G/H/I/J — director runs upstream and only modifies
the system prompt string. MCA, portrait resolution, role gating, and timeline
build remain untouched.

## Files

| File | Purpose |
|------|---------|
| `data/mv_producer_profiles.json` | 9 producer profiles + sub-genre modifiers (pure data) |
| `mv_director.py` | `MVDirector` class — selection, classification, shot plan |
| `music_video_pipeline.py` (`plan_segments`) | Wire-up at the aligned-mode branch |
| `tests/test_mv_director.py` | Unit tests (28) |
| `tests/test_mv_director_integration.py` | End-to-end injection verification |

## Producers (3 per genre)

- **Pop**: Dave Meyers (maximalist surreal tableau), Melina Matsoukas (documentary realism), Floria Sigismondi (gothic theatre)
- **Rock**: Anton Corbijn (monochrome portrait), Samuel Bayer (grunge handheld), Mark Romanek (gallery tableau)
- **Hip Hop**: Hype Williams (fisheye luxury), Hiro Murai (surreal allegory), Cole Bennett (cartoon overlay)

Each profile has ≥3 verified videos (Wikipedia/IMVDb-sourced).

## Sub-genre modifiers

`gangster_rap`, `drill`, `luxury_pop_rap`, `conscious`, `alternative_hip_hop`,
`trap`, `soundcloud_rap`, `mumble_rap`, `rnb`, `power_ballad`, `grunge`,
`post_punk`, `industrial_rock`, `dance_pop`, `dark_pop`.

Modifiers can shift: `closeup_freq_mult`, `closeup_freq_floor`,
`edit_tempo_override`, `film_grain_force`, `lighting_bias`, `forbid_extra`,
`force_settings`, `preferred_producers`.

## The DJ-pult fix

`universal_forbidden_by_genre.rock` includes:
`dj_booth, dj_pult, turntables, nightclub_dancefloor, choreographed_dance_routine,
money_throw, champagne_pour, luxury_car_pan, shiny_suit_cyc, 2D_animation_overlay,
freeze_frame_bounce, cartoon_overlay, strobe_rave_lighting`.

These get listed in the brief's FORBIDDEN block whenever the song's genre is rock.

## Variety guardrails (encoded in brief as numbered HARD RULES)

1. Use the exact shot type per segment in the directive table.
2. No two consecutive segments share shot type.
3. Close-up quota: ≥ `round(closeup_frequency × N)` segments use CU/ECU.
4. ≥1 establishing shot (WS or LS) in first 3 segments.
5. Sad/melancholic chorus segments MUST use CU/ECU on vocalist emotion.
6. Story segments MUST match producer archetypes.
7. FORBIDDEN list is hard — never depict listed elements.

## Failure safety

- Feature flag: `MV_DIRECTOR_ENABLED=0` env var disables the brief entirely (exact pre-K prompt).
- Per-profile `enabled: false` field rolls a single profile back without code change.
- `try/except` around the whole director block — any exception sets `director_brief=""` and the pipeline reverts to pre-K behavior, with a `[Stage K]` warning log.
- Per-profile `enabled` defaults to `true`.

## How to add a new profile

1. Pick the producer + 3 verifiable videos with distinct style markers.
2. Append a JSON entry to `data/mv_producer_profiles.json/producers[]` matching the schema (see existing entries).
3. Ensure `shot_distribution` values sum to ~1.0.
4. Add an entry to `universal_forbidden_by_genre` if the producer reveals a new cross-genre veto pattern.
5. Run `pytest tests/test_mv_director.py` — schema + sum + required-fields checks will catch malformed entries.

## How to add a new sub-genre modifier

Add a key under `sub_genre_modifiers` with any subset of:
`aggression_mult, intimacy_mult, closeup_freq_mult, closeup_freq_floor,
edit_tempo_override, film_grain_force, lighting_bias, forbid_extra,
force_settings, preferred_producers`.
