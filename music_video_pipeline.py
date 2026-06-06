"""
Music Video Pipeline for CTB — segment-based music video generation.

Splits audio into segments, generates video clips per segment via IA2V
ComfyUI workflows, and assembles clips into a final music video.
"""

import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Default segmentation settings
DEFAULT_FALLBACK_SEGMENT_LENGTH = 10  # seconds
DEFAULT_MAX_SEGMENT_DURATION = 15     # seconds, subdivide longer segments
DEFAULT_CROSSFADE_DURATION = 0.5      # seconds


# Strip song-lyric text from prompts destined for image generation.
# Image models (Flux, SDXL, etc.) render any quoted string as literal on-canvas
# text. The MCA T2I startframe prompt MUST NOT contain lyrics.
_QUOTED_RUN_RE = re.compile(r'["“”„«»‘’‚][^"“”„«»‘’‚]{0,400}["“”„«»‘’‚]')
_SINGS_CLAUSE_RE = re.compile(
    # "he sings:" / "she whispers" — optionally followed by an UNQUOTED tail
    # that the LLM appended directly instead of quoting. We eat the tail up
    # to the next sentence-ending punctuation or any opening quote (so
    # _QUOTED_RUN_RE can still strip a quoted tail separately).
    r'\b(?:he|she|the\s+singer|singer|they|performer|vocalist)\s+(?:sings?|singing|says?|whispers?|raps?|chants?|screams?|shouts?|belts?|croons?)\s*:?\s*[^.!?\n"“”„«»‘’‚]*',
    re.IGNORECASE,
)
# Fix 33: catch "The text: …", "The lyrics: …", "the line: …", "the words: …"
# introducers — LLM falls back to these when told "no quotes". Strip from the
# introducer up to (but not including) the next sentence-ending punctuation
# so we delete only the leaked lyric run, not the rest of the description.
_TEXT_INTRODUCER_RE = re.compile(
    r'\b(?:the\s+)?(?:text|lyrics?|line|words|song(?:\s+text)?|verse)\s*[:\-—–]\s*[^.!?\n]*',
    re.IGNORECASE,
)

# Fix 34: catch VIDEO-only language that leaks into a frame_variant_prompt
# when the LLM either polluted fvp with motion notes or fvp fell back to the
# video_prompt entirely. Still images don't have camera moves, scene ends,
# or lipsync — these sentences are removed before the prompt hits the T2I.
# The Fix 26 framing phrases ("Close-up of ", "Medium shot of ", "3/4 angle
# of ", "Low-angle shot of ", "High-angle shot of ", "Medium close-up of ")
# are deliberately NOT matched here so they stay intact in the still prompt.
_VIDEO_DIRECTIVES_RE = re.compile(
    r'(?:'
    # "The camera|shot|scene|clip <verb> ..."  up to sentence end
    r'\bthe\s+(?:camera|shot|scene|clip|sequence)\s+(?:performs?|holds?|ends?|begins?|starts?|cuts?|fades?|tilts?|pans?|tracks?|zooms?|moves?|rotates?|orbits?|dollies?|trucks?|drifts?|sweeps?|cranes?|pulls?|pushes?|whips?)\b[^.!?\n]*'
    r'|'
    # Camera-move phrases with optional speed adjective, up to sentence end
    r'\b(?:subtle\s+|slow\s+|fast\s+|gentle\s+|smooth\s+|gradual\s+)?(?:dolly[-\s]?in|dolly[-\s]?out|pull[-\s]?back|push[-\s]?in|crane(?:\s+(?:up|down))?|zoom[-\s]?in|zoom[-\s]?out|whip[-\s]?pan|tilt[-\s]?up|tilt[-\s]?down|track[-\s]?(?:left|right)|orbit(?:\s+around)?|handheld\s+drift|push\s+forward|sweep)\b[^.!?\n]*'
    r'|'
    # Cut / transition phrases
    r'\b(?:clean\s+)?(?:hard\s+cut|cut\s+to|fade\s+to|crossfade|transition|wipe\s+to|jump\s+cut|match\s+cut)\b[^.!?\n]*'
    r'|'
    # LIPSYNC_BOOSTER fragments (also other lipsync language)
    r'\b(?:the\s+)?lips?\s+(?:are\s+)?sync(?:ing|ed)?(?:\s+(?:naturally|to|with))?[^.!?\n]*'
    r'|'
    r'\bevery\s+word\s+is\s+pronounced[^.!?\n]*'
    r'|'
    r'\bdiction\s+and\s+lip[-\s]?sync\s+(?:are\s+)?perfect[^.!?\n]*'
    r')',
    re.IGNORECASE,
)


def derive_still_prompt_from_video_prompt(vp: str, lyrics: str = "") -> str:
    """Fix 34: derive a clean still-image prompt from a video_prompt.

    Used when the LLM left frame_variant_prompt empty (the silent
    `fvp = ... or vp` fallback used to feed video_prompt straight to the
    T2I startframe generator — including lyrics, camera moves, scene-end
    notes and the LIPSYNC_BOOSTER appended in api.py).

    Strips: lyric runs (Fix 33 patterns + substring), video-direction
    phrases (camera moves, cuts, lipsync language), Fix-29/30 trailing
    suffixes are left untouched (they describe lighting and clothing,
    valid for still images).
    """
    return strip_lyrics_from_image_prompt(vp, lyrics=lyrics)


def strip_lyrics_from_image_prompt(prompt: str, lyrics: str = "") -> str:
    """Remove quoted lyric runs, 'he sings ...' lead-ins, 'the text: …'
    introducers, and any literal substrings of the section's lyrics — so the
    T2I model doesn't render the song text as on-canvas caption text.

    Fix 33: a lyrics argument enables substring-based cleanup. The LLM often
    bypasses both the quote ban and the "sings" pattern by writing the
    lyrics inline as a descriptive label ("The text: c'est le temps des
    nostalgies"). When the caller passes the segment's known lyrics in, any
    matching run is removed verbatim regardless of how it was introduced.
    """
    if not prompt:
        return prompt
    cleaned = _QUOTED_RUN_RE.sub("", prompt)
    cleaned = _SINGS_CLAUSE_RE.sub("", cleaned)
    cleaned = _TEXT_INTRODUCER_RE.sub("", cleaned)
    # Fix 34: also remove video-direction language (camera moves, scene
    # ends, hard cuts, lipsync booster fragments) so that a fvp that fell
    # back to video_prompt doesn't carry those over to the T2I.
    cleaned = _VIDEO_DIRECTIVES_RE.sub("", cleaned)
    if lyrics:
        # Strip the full lyric run plus any reasonable subline (≥4 words).
        lines = [ln.strip(" ,.;:\"'“”‘’«»") for ln in re.split(r"[\r\n]+", lyrics) if ln.strip()]
        # Match longest first so a full-line substring beats a shorter prefix.
        lines.sort(key=len, reverse=True)
        for line in lines:
            if len(line) < 12:
                continue
            cleaned = re.sub(re.escape(line), "", cleaned, flags=re.IGNORECASE)
            # Also try a word-boundary variant for lines that the LLM may
            # have lightly reformatted (e.g. inserted commas / quotes).
            tokens = re.findall(r"\w+", line)
            if len(tokens) >= 4:
                pattern = r"\b" + r"[\s\W]+".join(re.escape(t) for t in tokens) + r"\b"
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip(" ,.;:")


def build_duet_portrait_prompt(
    theme: str,
    wardrobe_slot: str = "",
    duet_kind: str = "mixed",
) -> str:
    """Fix 24A + Fix 30 + Fix 32: deterministic identity-neutral prompt for the duet portrait.

    The previous duet path called the SINGLE-portrait LLM prompt generator,
    whose system prompt told the model to "establish face/hair/skin tone" —
    so the LLM invented identity attributes (e.g. "blonde hair / dark waves")
    that competed with the actual reference images and produced visible drift
    vs. the two single-singer portraits. The reference images supply identity;
    this prompt only describes composition, framing, background and lighting,
    plus an explicit identity-lock clause for the T2I model.

    Fix 30 — duet identity hardening:
    - B1: when wardrobe_slot is provided, anchor BOTH performers' outfits to
      the established wardrobe slot description so clothing matches the
      surrounding solo segments (clothing acts as a re-identification cue).
    - B3: stronger identity-preservation clause covering eye colour,
      eyebrow/nose/lip shape, jawline, body proportions and height.

    Fix 32 — same-gender duet support:
    - duet_kind == "ff": both performers female; clause uses only the slot's
      female-outfit description, no male wording.
    - duet_kind == "mm": both performers male; only male-outfit, no female.
    - duet_kind == "mixed" (default): male outfit named first (matches
      portrait_a reference order in the workflow), then female outfit.

    Fix 35 — duet portrait hardening:
    - A: per-performer garment-lock with explicit cross-swap negation
      (gender-agnostic, slot-agnostic — no "dress vs shorts" assumption).
    - B: anatomy-lock — explicit limb counts, no merged/shared body parts.
    - C: composition-hint — visible gap between performers, distinct
      silhouettes, no torso-overlap or body-merge.
    - D: reference-order consistency — in mixed duets the male outfit is
      named FIRST because portrait_a (the lead, duplicated for weight
      bias in the Flux2-M-I-Edit workflow) is the male performer.
    """
    theme_clause = f" Theme context: {theme}." if theme else ""
    entry = WARDROBE_STATES.get(wardrobe_slot or "") if wardrobe_slot else None
    if entry and (entry.get("female") or entry.get("male")):
        f_outfit = entry.get("female", "")
        m_outfit = entry.get("male", "")
        if duet_kind == "ff" and f_outfit:
            wardrobe_clause = (
                f" Both performers are female; both wear their established "
                f"costumes from the reference images: {f_outfit}. Each "
                f"garment is fully on each individual performer — no "
                f"half-and-half outfits, no shared pieces, no garments "
                f"split between bodies. DO NOT change clothing colour, "
                f"cut or style between the references and the duet."
            )
        elif duet_kind == "mm" and m_outfit:
            wardrobe_clause = (
                f" Both performers are male; both wear their established "
                f"costumes from the reference images: {m_outfit}. Each "
                f"garment is fully on each individual performer — no "
                f"half-and-half outfits, no shared pieces, no garments "
                f"split between bodies. DO NOT change clothing colour, "
                f"cut or style between the references and the duet."
            )
        elif f_outfit and m_outfit:
            # Fix 35 D: name the male outfit FIRST because portrait_a (lead,
            # duplicated weight) is the male performer in mixed duets.
            # Fix 35 A: explicit garment-lock per performer + cross-swap
            # negation, gender-agnostic and slot-agnostic.
            wardrobe_clause = (
                f" Both performers wear their established costumes from the "
                f"reference images. The man wears: {m_outfit}. The woman "
                f"wears: {f_outfit}. Each garment is locked to its assigned "
                f"performer — NEVER swap or share clothing between the two. "
                f"The man's listed garments go ONLY on the man; the woman's "
                f"listed garments go ONLY on the woman. No cross-dressing, "
                f"no mixed garments, no shared pieces, no half-and-half "
                f"outfits, no garment split between bodies. DO NOT change "
                f"clothing colour, cut or style between the references and "
                f"the duet."
            )
        else:
            wardrobe_clause = (
                " Both performers wear their established costumes from the "
                "reference images; DO NOT change clothing colour, cut or style."
            )
    else:
        wardrobe_clause = (
            " Both performers wear their established costumes from the "
            "reference images; DO NOT change clothing colour, cut or style."
        )
    if duet_kind == "ff":
        gender_clause = " Both performers are female — never depict a male performer in the duet."
    elif duet_kind == "mm":
        gender_clause = " Both performers are male — never depict a female performer in the duet."
    else:
        gender_clause = ""
    # Fix 35 B + C: anatomy-lock and composition-hint apply to ALL duet
    # kinds — body-merge and limb hallucinations are orthogonal to gender.
    anatomy_clause = (
        " Two clearly separate full-body figures with distinct anatomy: "
        "exactly two arms and two legs per person, exactly one head per "
        "person, no merged limbs, no shared body parts, no extra arms or "
        "legs, no third hand on shared objects, anatomically correct."
    )
    composition_clause = (
        " Both performers stand side-by-side as separate individuals with "
        "a small visible gap between their bodies, full silhouettes "
        "clearly distinct against the background, never overlapping at "
        "the torso, never merged or fused into a single body."
    )
    return (
        "Two performers standing side-by-side as a single front-facing "
        "couple portrait, shown from head to waist, both faces and mouths "
        "clearly visible and in sharp focus. Plain neutral white-to-light-"
        "grey studio background, even soft studio lighting, no shadows, "
        "no props or scenery, no other people. Preserve each performer's "
        "face, hair colour, hair style, skin tone, build, eye colour, "
        "eyebrow shape, nose shape, lip shape, jawline, body proportions "
        "and height EXACTLY as in the reference images; do not restyle or "
        "recolour them. The result must be recognizable as the SAME two "
        "people as in the reference images, NOT new performers in a "
        "similar style."
        f"{gender_clause}"
        f"{anatomy_clause}"
        f"{composition_clause}"
        f"{wardrobe_clause}"
        f"{theme_clause}"
    )


@dataclass
class Segment:
    index: int
    start_time: float
    end_time: float
    label: str = ""
    lyrics: str = ""
    audio_clip: str = ""
    prompt: str = ""
    enriched_prompt: str = ""
    video_clip: str = ""
    end_frame: str = ""
    end_frame_comfy: str = ""  # ComfyUI filename for the end frame
    frame_variant_prompt: str = ""  # MCA variant prompt for this segment's start frame
    transition: str = "cut"  # "cut" or "crossfade"
    status: str = "pending"  # "pending", "generating", "completed"
    reuse_of: Optional[int] = None  # RC8: reuse this earlier segment's MCA frame (repeated chorus)
    wardrobe_slot: str = ""  # Fix 30: wardrobe slot key for this segment (identity anchor)
    # PromptRelay (Smart-Node): per-segment multi-beat block. None = legacy single-prompt path.
    # `global` = camera+lighting+grading anchor (NO identity — image carries identity).
    # `beats` = 1-4 entries; beat 1 strictly static + generic subject for IA2V.
    video_prompt_relay: Optional[Dict[str, Any]] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "label": self.label,
            "lyrics": self.lyrics,
            "audio_clip": self.audio_clip,
            "prompt": self.prompt,
            "enriched_prompt": self.enriched_prompt,
            "video_clip": self.video_clip,
            "end_frame": self.end_frame,
            "end_frame_comfy": self.end_frame_comfy,
            "frame_variant_prompt": self.frame_variant_prompt,
            "transition": self.transition,
            "status": self.status,
            "reuse_of": self.reuse_of,
            "wardrobe_slot": self.wardrobe_slot,
            "video_prompt_relay": self.video_prompt_relay,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MVSession:
    audio_path: str = ""
    start_image_path: str = ""
    workflow: str = ""
    prompt_mode: str = ""          # "quick" or "creative"
    global_prompt: str = ""        # Quick Mode only
    theme: str = ""                # Creative Mode only
    scene_anchor: str = ""         # Constant character/scene description for visual consistency
    start_mode: str = ""           # "ta2v" = first segment via TA2V workflow, no start image needed
    resolution: dict = field(default_factory=dict)  # e.g. {"dimensions": "1344 x 768 (landscape)"}
    frame_analysis: bool = True    # Vision-LLM enrichment on/off
    auto_continue: bool = False    # Skip per-segment review and generate all automatically
    segments: List[Segment] = field(default_factory=list)
    crossfade_duration: float = DEFAULT_CROSSFADE_DURATION
    segment_max_duration: float = DEFAULT_MAX_SEGMENT_DURATION
    fallback_segment_length: float = DEFAULT_FALLBACK_SEGMENT_LENGTH
    output_path: str = ""
    current_segment_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "start_image_path": self.start_image_path,
            "workflow": self.workflow,
            "prompt_mode": self.prompt_mode,
            "global_prompt": self.global_prompt,
            "theme": self.theme,
            "scene_anchor": self.scene_anchor,
            "start_mode": self.start_mode,
            "resolution": self.resolution,
            "frame_analysis": self.frame_analysis,
            "auto_continue": self.auto_continue,
            "segments": [s.to_dict() for s in self.segments],
            "crossfade_duration": self.crossfade_duration,
            "segment_max_duration": self.segment_max_duration,
            "fallback_segment_length": self.fallback_segment_length,
            "output_path": self.output_path,
            "current_segment_index": self.current_segment_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MVSession":
        segments_data = data.pop("segments", [])
        session = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        session.segments = [Segment.from_dict(s) for s in segments_data]
        return session

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "MVSession":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def completed_count(self) -> int:
        return sum(1 for s in self.segments if s.status == "completed")

    def next_pending(self) -> Optional[Segment]:
        for s in self.segments:
            if s.status == "pending":
                return s
        return None


def parse_lyrics_file(lyrics_path: str) -> List[Dict[str, Any]]:
    """Parse a lyrics file with [Tag] structure into labeled blocks.

    Returns list of dicts: {"label": str, "lyrics": str}
    """
    if not os.path.exists(lyrics_path):
        return []

    with open(lyrics_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = []
    parts = re.split(r'\[([^\]]+)\]', content)

    i = 1
    while i < len(parts) - 1:
        label = parts[i].strip()
        lyrics = parts[i + 1].strip()
        blocks.append({"label": label, "lyrics": lyrics})
        i += 2

    return blocks


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-show_entries",
            "format=duration", "-of", "csv=p=0", audio_path
        ],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def segment_audio(
    audio_path: str,
    output_dir: str,
    lyrics_path: Optional[str] = None,
    fallback_length: float = DEFAULT_FALLBACK_SEGMENT_LENGTH,
    max_duration: float = DEFAULT_MAX_SEGMENT_DURATION,
) -> List[Segment]:
    """Split audio into segments based on lyrics structure or fixed length."""
    total_duration = get_audio_duration(audio_path)
    blocks = []

    if lyrics_path:
        blocks = parse_lyrics_file(lyrics_path)

    if blocks:
        segments = _segment_from_blocks(blocks, total_duration, max_duration)
    else:
        segments = _segment_fixed_length(total_duration, fallback_length)

    os.makedirs(output_dir, exist_ok=True)
    for seg in segments:
        clip_path = os.path.join(output_dir, f"seg_{seg.index:03d}.wav")
        _extract_audio_clip(audio_path, seg.start_time, seg.end_time, clip_path)
        seg.audio_clip = clip_path

    return segments


def _segment_from_blocks(
    blocks: List[Dict[str, Any]],
    total_duration: float,
    max_duration: float,
) -> List[Segment]:
    """Create segments from lyrics blocks, distributing time proportionally."""
    n = len(blocks)
    if n == 0:
        return []

    block_duration = total_duration / n
    segments = []
    idx = 0

    for i, block in enumerate(blocks):
        start = i * block_duration
        end = min((i + 1) * block_duration, total_duration)

        if (end - start) > max_duration:
            sub_count = int((end - start) / max_duration) + 1
            sub_len = (end - start) / sub_count
            for j in range(sub_count):
                sub_start = start + j * sub_len
                sub_end = min(start + (j + 1) * sub_len, total_duration)
                label = f"{block['label']} ({j+1}/{sub_count})" if sub_count > 1 else block["label"]
                segments.append(Segment(
                    index=idx,
                    start_time=round(sub_start, 3),
                    end_time=round(sub_end, 3),
                    label=label,
                    lyrics=block["lyrics"] if j == 0 else "",
                ))
                idx += 1
        else:
            segments.append(Segment(
                index=idx,
                start_time=round(start, 3),
                end_time=round(end, 3),
                label=block["label"],
                lyrics=block["lyrics"],
            ))
            idx += 1

    return segments


def _segment_fixed_length(total_duration: float, segment_length: float) -> List[Segment]:
    """Create segments of fixed length."""
    segments = []
    idx = 0
    t = 0.0
    while t < total_duration:
        end = min(t + segment_length, total_duration)
        segments.append(Segment(
            index=idx,
            start_time=round(t, 3),
            end_time=round(end, 3),
            label=f"Segment {idx + 1}",
        ))
        idx += 1
        t = end
    return segments


_ROLE_RE = re.compile(
    r"-\s*(male|female|duet|both|together)\b",
    re.IGNORECASE,
)


def extract_section_role(label: str) -> Optional[str]:
    """Parse a per-section performer-role hint from a Segment label.

    LLM-authored section labels carry role annotations like
    "Verse - male", "Chorus - female", "Bridge - duet", "Chorus - both".
    Returns one of "male" | "female" | "duet" | None. "both" and "together"
    normalize to "duet".
    """
    if not label:
        return None
    m = _ROLE_RE.search(label)
    if not m:
        return None
    raw = m.group(1).lower()
    if raw in ("both", "together"):
        return "duet"
    return raw


def nearest_annotated_role(
    segments: List["Segment"], idx: int
) -> Optional[str]:
    """Find the role of the closest segment to `idx` that has an annotated role.

    Scans forward first (rest of the song), then backward (toward the start),
    returning the first annotated role encountered. Used to route Intro/Outro/
    Fade-Out segments to whichever singer dominates the adjacent music body
    so they reuse an existing portrait instead of triggering an extra MCA pass.
    Returns None if no segment in the song has an annotated role.
    """
    n = len(segments)
    if not n or idx < 0 or idx >= n:
        return None
    for j in range(idx + 1, n):
        r = extract_section_role(segments[j].label)
        if r:
            return r
    for j in range(idx - 1, -1, -1):
        r = extract_section_role(segments[j].label)
        if r:
            return r
    return None


def partition_anchors_by_role(
    segments: List["Segment"],
) -> Dict[Optional[str], list[int]]:
    """Group non-reuse anchor segment indices by their per-section role.

    Anchor = segment whose `reuse_of` is None (first occurrence of its
    section). Reuse rows inherit the anchor's frame downstream.

    Returns a dict keyed by role ("male" | "female" | "duet" | None). When
    the dict has ≥2 non-None keys, the pipeline activates multi-portrait
    rendering — one portrait per role, plus a Flux2-M-I-Edit duet portrait
    when "duet" is among the keys. With only one role (or None), the
    pipeline behaves single-portrait as before.
    """
    out: Dict[Optional[str], list[int]] = {}
    for i, s in enumerate(segments):
        if s.reuse_of is not None:
            continue
        role = extract_section_role(s.label)
        out.setdefault(role, []).append(i)
    return out


def plan_same_gender_portraits(
    duet: str,
    roles_present: set,
    consistent_character: bool,
    source_mode: str,
) -> Optional[tuple]:
    """Fix 27: decide the portrait plan for an explicit same-gender duet request.

    `duet` is the user-supplied intent: "ff" (two women) | "mm" (two men).
    Section labels stay plain ("female"/"duet") — the two characters are a
    PORTRAIT-routing concern only. The solo sections are always sung by the
    same lead character (audio cannot tell female1 from female2); a SECOND
    "partner" portrait is rendered purely so the shared-duet frame depicts both
    distinct people instead of falling back to the lead twice.

    Returns (base_role, partner_role, make_duet) when same-gender rendering
    applies, or None to use the standard male/female routing. `make_duet` is
    True only when a "duet" section actually exists.
    """
    if duet not in ("ff", "mm"):
        return None
    if not consistent_character or source_mode not in ("auto", "describe"):
        return None
    base = "female" if duet == "ff" else "male"
    partner = base + "2"
    make_duet = "duet" in roles_present
    return (base, partner, make_duet)


def same_gender_veto(
    sections: List[tuple],
    duet: str,
    min_fraction: float = 0.15,
) -> bool:
    """Fix 27: decide whether to ABANDON the same-gender request because the
    audio analysis found sustained opposite-gender vocals.

    `sections` is a list of (detected_gender, duration_seconds) for the vocal
    sections, where detected_gender comes from audio_gender_detect (already
    aggregated per section, so a single stray window won't flip a section).
    `duet` is "ff" | "mm". Veto fires when the opposite gender occupies at
    least `min_fraction` of the classified (non-unknown) vocal duration — a
    proportional threshold so one short borderline section doesn't silently
    flip the whole feature to mixed routing.
    """
    if duet not in ("ff", "mm"):
        return False
    opposite = "male" if duet == "ff" else "female"
    # A "duet" classification means BOTH genders were ≥20% present in that
    # section (audio_gender_detect._classify_section) → the opposite gender IS
    # there (e.g. a man singing the shared chorus). It therefore counts toward
    # the opposite measure. Two same-gender singers never classify as "duet"
    # (no opposite present → plain "female"/"male"), so this can't trip the
    # intended case.
    opp_dur = sum(d for g, d in sections if g == opposite or g == "duet")
    # Denominator: only sections classified as a concrete gender (drop unknown).
    classified = sum(d for g, d in sections if g in ("male", "female", "duet"))
    if classified <= 0:
        return False
    return (opp_dur / classified) >= min_fraction


# Clauses introducing the "opposite" performer that should be dropped from
# an MCA prompt routed to a single-gender portrait. Keep conservative so we
# don't strip narrative scenery.
_FEMALE_REF_RE = re.compile(
    r"(?:(?<=^)|(?<=[.!?;,]\s))"
    r"(?:she|her|the\s+female(?:\s+singer|\s+vocalist|\s+performer)?|"
    r"the\s+woman|the\s+girl|a\s+female\s+(?:singer|vocalist|performer))"
    r"\b[^.!?;]*[.!?;]?",
    re.IGNORECASE,
)
_MALE_REF_RE = re.compile(
    r"(?:(?<=^)|(?<=[.!?;,]\s))"
    r"(?:he|him|his|the\s+male(?:\s+singer|\s+vocalist|\s+performer)?|"
    r"the\s+man|the\s+guy|a\s+male\s+(?:singer|vocalist|performer))"
    r"\b[^.!?;]*[.!?;]?",
    re.IGNORECASE,
)


def enforce_performer_role(prompt: str, role: Optional[str]) -> str:
    """Strip clauses describing the opposite performer for single-role frames.

    role="male"   → drop "she/her/the woman/the female singer …" clauses
    role="female" → drop "he/him/the man/the male singer …" clauses
    role="duet"   → no-op (duet frames legitimately depict both)
    role=None     → no-op (backwards compat)

    Conservative: if stripping produces an empty result, return the original
    so MCA always has something to render.
    """
    if not prompt or role in (None, "duet"):
        return prompt
    if role == "male":
        cleaned = _FEMALE_REF_RE.sub("", prompt)
    elif role == "female":
        cleaned = _MALE_REF_RE.sub("", prompt)
    else:
        return prompt
    if cleaned == prompt:
        return prompt
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;:.")
    return cleaned if cleaned else prompt


def clamp_song_duration(d: Any, default: int = 150, lo: int = 20, hi: int = 300) -> int:
    """Clamp a requested song duration (seconds) into the supported range.

    None / unparseable -> default (~150s, a full-length music video). An EXPLICIT
    value is honored down to a 20s floor (short test clips) and up to a 300s cap.
    """
    if d is None:
        return default
    try:
        v = int(float(d))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


# ACE-Step TextEncodeAceStepAudio1.5 `language` input — valid ISO codes (from
# ComfyUI /object_info; passing anything else returns HTTP 400).
ACE_STEP_LANGS = frozenset(
    "ar az bg bn ca cs da de el en es fa fi fr he hi hr ht hu id is it ja ko la "
    "lt ms ne nl no pa pl pt ro ru sa sk sr sv sw ta te th tl tr uk ur vi yue zh "
    "unknown".split()
)

_LANG_NAME_TO_ISO = {
    "german": "de", "deutsch": "de", "deutsche": "de", "deutsches": "de",
    "english": "en", "englisch": "en",
    "french": "fr", "französisch": "fr", "franzosisch": "fr", "français": "fr",
    "francais": "fr",
    "spanish": "es", "spanisch": "es", "español": "es", "espanol": "es",
    "italian": "it", "italienisch": "it", "italiano": "it",
    "portuguese": "pt", "portugiesisch": "pt",
    "russian": "ru", "russisch": "ru",
    "japanese": "ja", "japanisch": "ja",
    "chinese": "zh", "chinesisch": "zh", "mandarin": "zh",
    "korean": "ko", "koreanisch": "ko",
    "dutch": "nl", "niederländisch": "nl",
    "polish": "pl", "polnisch": "pl",
    "turkish": "tr", "türkisch": "tr",
}


def to_ace_language(explicit: str, brief: str) -> Optional[str]:
    """Resolve a song language to an ACE-Step-valid ISO code, or None.

    Accepts an explicit value (ISO code or language name) or, failing that,
    infers from keywords in the brief. Returns None when nothing valid is found
    (caller then leaves the AudioSettings default — never an invalid value that
    would 400 the ACE-Step graph).
    """
    v = (explicit or "").strip().lower()
    if v in ACE_STEP_LANGS:
        return v
    if v in _LANG_NAME_TO_ISO:
        return _LANG_NAME_TO_ISO[v]
    b = (brief or "").lower()
    for name, iso in _LANG_NAME_TO_ISO.items():
        if name in b:
            return iso
    return None


def chunk_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split a list into chunks of at most batch_size (VRAM-aware MCA batching).

    batch_size <= 0 -> a single chunk containing everything.
    """
    items = list(items)
    if not items:
        return []
    if batch_size is None or batch_size <= 0:
        return [items]
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def parse_segment_plan(response_text: str) -> List[Dict[str, Any]]:
    """Parse an LLM JSON array of segment specs. Malformed -> [] (caller falls back).

    Each spec: {video_prompt, frame_variant_prompt, duration, label, lyrics}.
    frame_variant_prompt defaults to video_prompt when absent.
    """
    if not response_text:
        return []
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE).strip()
    s = cleaned.find('[')
    e = cleaned.rfind(']')
    if s == -1 or e <= s:
        return []
    try:
        arr = json.loads(cleaned[s:e + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(arr, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        vp = str(item.get("video_prompt", "")).strip()
        fvp = str(item.get("frame_variant_prompt", "")).strip() or vp
        raw_dur = item.get("duration", None)
        try:
            dur = float(raw_dur) if raw_dur is not None else None
        except (TypeError, ValueError):
            dur = None
        out.append({
            "video_prompt": vp,
            "frame_variant_prompt": fvp,
            "duration": dur,
            "label": str(item.get("label", "")).strip(),
            "lyrics": str(item.get("lyrics", "")).strip(),
        })
    return out


def _segment_video_prompt(spec: Dict[str, Any], theme: str) -> str:
    """Pick a segment's video prompt from an LLM spec.

    Bug 1: NEVER fall back to the raw lyrics. A T2V model renders raw lyric
    text as literal on-screen captions / garbled scenes (the "rain behind
    glass", interior-less car artefacts). Use the LLM scene description if
    present, otherwise the visual theme. Lyrics reach the clip only via the
    separate LIPSYNC handling, never as the visual prompt.
    """
    return (str(spec.get("video_prompt", "")).strip()) or theme


# PromptRelay validated constraints (2026-06-05, distilled-LoRA path):
# - Beat-Mindest-Länge: 5s. Under that, distilled model can't articulate the beat.
# - Max beats per clip: 4. More = recency-bias loses early beats.
# - Below 5s clip duration: relay degenerates to single-prompt → fall back to legacy.
RELAY_MIN_BEAT_SECONDS = 5.0
RELAY_MAX_BEATS = 4


def pick_relay_beat_count(duration: float) -> int:
    """Adaptive beat count for distilled LTX-IA2V given clip duration.

    Returns 0 when duration is too short for relay to add value (caller should
    fall back to single-prompt). Otherwise returns 1..4 beats, gleichgewichtet.
    """
    if duration < RELAY_MIN_BEAT_SECONDS:
        return 0
    n = int(duration // RELAY_MIN_BEAT_SECONDS)
    return max(1, min(RELAY_MAX_BEATS, n))


def extract_relay_spec(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the LLM's video_prompt_relay block from a spec dict, if shaped right.

    Returns None when the LLM omitted the relay field or returned a malformed
    block. Strict: requires `global` string and `beats` list of non-empty strings.
    """
    raw = spec.get("video_prompt_relay")
    if not isinstance(raw, dict):
        return None
    g = raw.get("global", "")
    beats = raw.get("beats", [])
    if not isinstance(g, str) or not isinstance(beats, list):
        return None
    cleaned = [str(b).strip() for b in beats if isinstance(b, (str, bytes)) and str(b).strip()]
    if not cleaned:
        return None
    return {"global": g.strip(), "beats": cleaned[:RELAY_MAX_BEATS]}


def build_segment_timeline(
    specs: List[Dict[str, Any]],
    total_duration: float,
    min_seg: float,
    max_seg: float,
) -> List[Dict[str, Any]]:
    """Tile [0, total_duration] into contiguous segments.

    Per-segment length is clamped to [min_seg, min(max_seg, 30)] — the 30s ceiling
    matches inject_segment_duration's silent clamp so audio and video clips stay in
    sync. Segment count is derived so an exact feasible tiling exists; LLM-proposed
    durations are used as relative weights and normalized to fill exactly.
    """
    if total_duration <= 0:
        return []
    cap = min(float(max_seg), 30.0)
    lo = float(min_seg)
    if lo > cap:
        lo = cap

    n_specs = len(specs)
    if total_duration <= lo:
        n = 1
    else:
        import math
        n_min = max(1, math.ceil(total_duration / cap))
        n_max = max(1, int(total_duration // lo))
        if n_max < n_min:
            n_max = n_min
        if n_specs == 0:
            preferred = (lo + cap) / 2.0
            n = round(total_duration / preferred) if preferred > 0 else n_min
        else:
            n = n_specs
        n = max(n_min, min(n_max, max(1, n)))

    # Relative weights from spec durations (cycle specs to fill n slots).
    preferred = (lo + cap) / 2.0
    weights: List[float] = []
    for i in range(n):
        spec = specs[i % n_specs] if n_specs else None
        d = spec.get("duration") if spec else None
        weights.append(float(d) if isinstance(d, (int, float)) and d and d > 0 else preferred)

    wsum = sum(weights) or float(n)
    durs = [max(lo, min(cap, w / wsum * total_duration)) for w in weights]

    # Absorb the clamp residual into segments that still have slack.
    for _ in range(200):
        residual = total_duration - sum(durs)
        if abs(residual) < 1e-6:
            break
        if residual > 0:
            slack = [(i, cap - durs[i]) for i in range(n) if cap - durs[i] > 1e-9]
        else:
            slack = [(i, durs[i] - lo) for i in range(n) if durs[i] - lo > 1e-9]
        cap_room = sum(s for _, s in slack)
        if cap_room < 1e-9:
            break
        for i, room in slack:
            durs[i] += residual * (room / cap_room)
            durs[i] = max(lo, min(cap, durs[i]))

    # Force exact tiling: last segment closes at total_duration.
    out: List[Dict[str, Any]] = []
    t = 0.0
    for i in range(n):
        spec = specs[i % n_specs] if n_specs else None
        if i == n - 1:
            end = total_duration
        else:
            end = min(total_duration, round(t + durs[i], 3))
            if end <= t:
                end = min(total_duration, t + lo)
        out.append({
            "start_time": round(t, 3),
            "end_time": round(end, 3),
            "video_prompt": (spec or {}).get("video_prompt", ""),
            "frame_variant_prompt": (
                # Fix 34: explicit fallback with sanitizer + log when fvp is
                # empty so we never silently feed video_prompt to the T2I.
                (lambda _fvp, _vp, _idx: (
                    _fvp if _fvp else (
                        print(
                            f"[Fix 34] frame_variant_prompt empty for "
                            f"segment {_idx}; deriving sanitized fvp "
                            f"from video_prompt."
                        ) or derive_still_prompt_from_video_prompt(
                            _vp, lyrics=(spec or {}).get("lyrics", ""),
                        )
                    )
                ))(
                    str((spec or {}).get("frame_variant_prompt", "")).strip(),
                    str((spec or {}).get("video_prompt", "")),
                    i,
                )
            ),
            "label": (spec or {}).get("label", "") or f"Segment {i + 1}",
            "lyrics": (spec or {}).get("lyrics", ""),
        })
        t = end
    return out


# ---------------------------------------------------------------------------
# Fix 29 — Time-of-Day Coherence
#
# Lighting state per segment is determined deterministically BEFORE the LLM
# generates segment prompts: an arc is selected (LLM mini-call or override),
# expanded proportionally to the segment count, and each per-segment light
# state is both injected into the segment-planning system prompt AND
# appended to the final video_prompt / frame_variant_prompt as a guarantee.
# ---------------------------------------------------------------------------

TIME_OF_DAY_STATES: Dict[str, str] = {
    "pre_dawn":         "pre-dawn — deep cool blue twilight, faint horizon glow, ambient blue cast",
    "dawn":             "dawn — soft pink and orange horizon, long cool shadows, low warm-cool mixed light",
    "early_sunrise":    "early sunrise — low warm sun just above horizon, long shadows, golden-pink rim light",
    "morning_golden":   "morning golden hour — warm low sun, golden rim light, long soft shadows, hazy air",
    "bright_morning":   "bright mid-morning — clear high light, crisp shadows, cool clean daylight",
    "midday":           "midday — high overhead sun, short hard shadows, bright neutral daylight",
    "afternoon":        "warm afternoon — angled warm sun, medium shadows, slightly amber daylight",
    "golden_hour_late": "late golden hour — low warm orange sun, long shadows, saturated amber tones",
    "sunset":           "sunset — vivid orange and magenta sky, silhouettes against bright horizon",
    "blue_hour":        "blue hour — deep saturated blue sky after sunset, glowing horizon, cool ambient light",
    "night":            "night — deep dark sky, practical and ambient light sources, cool moonlight rim",
    "moonlit":          "moonlit night — soft cool blue moonlight, deep shadows, silvery highlights",
    "overcast":         "overcast — flat soft diffused daylight, no hard shadows, cool grey-white tone",
    "stormy":           "stormy daylight — dark cloud cover, dramatic diffused light, occasional shafts of sun",
}


# Each arc is a template list. _expand_tod_plan() resamples it to the segment
# count so progression scales naturally for songs of any length. Progression
# is always forward in time (or constant) — never jumps backward.
TIME_OF_DAY_ARCS: Dict[str, List[str]] = {
    "single_golden_hour":      ["morning_golden"],
    "single_midday":           ["midday"],
    "single_afternoon":        ["afternoon"],
    "single_blue_hour":        ["blue_hour"],
    "single_night":            ["night"],
    "single_moonlit":          ["moonlit"],
    "overcast_constant":       ["overcast"],
    "stormy_constant":         ["stormy"],
    "sunrise_to_morning":      ["dawn", "early_sunrise", "morning_golden", "bright_morning"],
    "morning_to_midday":       ["morning_golden", "bright_morning", "midday"],
    "midday_to_afternoon":     ["midday", "afternoon", "golden_hour_late"],
    "golden_hour_to_blue_hour":["morning_golden", "afternoon", "golden_hour_late", "sunset", "blue_hour"],
    "afternoon_to_sunset":     ["afternoon", "golden_hour_late", "sunset"],
    "sunset_to_night":         ["golden_hour_late", "sunset", "blue_hour", "night"],
    "night_to_dawn":           ["night", "moonlit", "pre_dawn", "dawn"],
    "blue_hour_to_night":      ["blue_hour", "night", "moonlit"],
}


# Genre substring → arc-key fallback (lower-case prefix match). First hit
# wins. Used when the LLM mini-call fails AND no user override is provided.
_GENRE_TIME_OF_DAY_DEFAULT: List[Tuple[str, str]] = [
    ("reggae",       "golden_hour_to_blue_hour"),
    ("ska",          "golden_hour_to_blue_hour"),
    ("tropical",     "golden_hour_to_blue_hour"),
    ("surf",         "morning_to_midday"),
    ("country",      "morning_to_midday"),
    ("folk",         "morning_to_midday"),
    ("acoustic",     "morning_to_midday"),
    ("ambient",      "blue_hour_to_night"),
    ("synthwave",    "blue_hour_to_night"),
    ("cyberpunk",    "single_night"),
    ("darkwave",     "single_night"),
    ("industrial",   "single_night"),
    ("metal",        "stormy_constant"),
    ("doom",         "stormy_constant"),
    ("gothic",       "blue_hour_to_night"),
    ("trance",       "sunset_to_night"),
    ("techno",       "single_night"),
    ("house",        "sunset_to_night"),
    ("edm",          "sunset_to_night"),
    ("rnb",          "blue_hour_to_night"),
    ("soul",         "afternoon_to_sunset"),
    ("jazz",         "blue_hour_to_night"),
    ("blues",        "afternoon_to_sunset"),
    ("hip hop",      "single_night"),
    ("rap",          "single_night"),
    ("lofi",         "afternoon_to_sunset"),
    ("classical",    "morning_to_midday"),
    ("orchestral",   "golden_hour_to_blue_hour"),
    ("pop",          "midday_to_afternoon"),
    ("rock",         "afternoon_to_sunset"),
    ("punk",         "midday_to_afternoon"),
    ("indie",        "afternoon_to_sunset"),
]

_DEFAULT_TIME_OF_DAY_ARC = "golden_hour_to_blue_hour"


def _expand_tod_plan(arc_key: str, n_segments: int) -> List[str]:
    """Expand an arc template to exactly n_segments states, monotonic forward.

    A single-state arc returns [state] * n. A multi-state template is
    resampled by mapping each segment index to its proportional position in
    the template (floor), so longer songs stretch the progression smoothly.
    Falls back to the default arc if arc_key is unknown.
    """
    if n_segments <= 0:
        return []
    template = TIME_OF_DAY_ARCS.get(arc_key) or TIME_OF_DAY_ARCS[_DEFAULT_TIME_OF_DAY_ARC]
    if len(template) == 1:
        return [template[0]] * n_segments
    out: List[str] = []
    m = len(template)
    for i in range(n_segments):
        # Map segment i to template slot via floor; last segment always hits
        # the final template state so the arc lands on its closing light.
        if n_segments == 1:
            idx = m - 1
        else:
            idx = min(m - 1, int(round(i * (m - 1) / (n_segments - 1))))
        out.append(template[idx])
    return out


def _genre_default_tod_arc(genre: str) -> str:
    g = (genre or "").lower()
    for substr, arc in _GENRE_TIME_OF_DAY_DEFAULT:
        if substr in g:
            return arc
    return _DEFAULT_TIME_OF_DAY_ARC


def _light_tag_suffix(state_key: str) -> str:
    """Return the ', lit by ...' suffix appended to every prompt as a guard."""
    text = TIME_OF_DAY_STATES.get(state_key) or ""
    if not text:
        return ""
    return f", lit by {text}"


def _append_light_tag(prompt: str, state_key: str) -> str:
    """Append the light-state suffix idempotently to a prompt string.

    If the prompt already contains the exact suffix (or the state key as a
    word), the prompt is returned unchanged so re-runs and partial overlap
    do not pile up duplicate lighting tails.
    """
    suffix = _light_tag_suffix(state_key)
    if not suffix:
        return prompt
    base = (prompt or "").rstrip()
    if not base:
        return suffix.lstrip(", ").capitalize()
    if suffix.strip(", ") in base:
        return base
    return base.rstrip(".") + suffix


# ---------------------------------------------------------------------------
# Fix 30 — Wardrobe Coherence
#
# Per-song wardrobe arc determined deterministically BEFORE the LLM
# generates segment prompts: an arc is selected (LLM mini-call or override),
# expanded proportionally to the segment count, and each per-segment outfit
# slot is both injected into the segment-planning system prompt AND
# appended to the final video_prompt / frame_variant_prompt as a guarantee.
# ---------------------------------------------------------------------------

WARDROBE_STATES: Dict[str, Dict[str, str]] = {
    "casual_beachwear": {
        "female": "flowing white cotton sundress with thin straps, knee-length, barefoot, simple gold necklace",
        "male":   "loose white linen shirt unbuttoned at the collar, beige cotton shorts, barefoot, simple leather cord necklace",
    },
    "bohemian_summer": {
        "female": "tan crochet crop top and ivory wide-leg linen pants, layered beaded necklaces, leather sandals",
        "male":   "open beige linen shirt with rolled sleeves and ivory wide-leg linen pants, leather sandals, beaded wrist bracelets",
    },
    "casual_evening": {
        "female": "fitted black silk camisole and dark high-waisted denim jeans, ankle boots, small gold hoop earrings",
        "male":   "fitted black crew-neck tee and dark slim-fit denim jeans, brown leather Chelsea boots, slim silver watch",
    },
    "tropical_relaxed": {
        "female": "loose pastel-yellow off-shoulder blouse and white cotton shorts, woven straw hat, leather sandals",
        "male":   "pastel-yellow short-sleeve linen shirt and white cotton shorts, woven straw fedora, leather sandals",
    },
    "streetwear_urban": {
        "female": "oversized graphic tee and baggy black cargo pants, white chunky sneakers, silver chain necklace",
        "male":   "oversized graphic tee and baggy black cargo pants, white chunky sneakers, silver chain necklace, black snapback cap",
    },
    "streetwear_neon": {
        "female": "neon-pink cropped hoodie and black biker shorts, white platform sneakers, tinted sunglasses",
        "male":   "neon-pink oversized hoodie and black tech joggers, white chunky sneakers, tinted sunglasses, silver chain",
    },
    "athletic_active": {
        "female": "fitted black sports top and matching high-waisted leggings, white running shoes, slick high ponytail",
        "male":   "fitted black athletic tank top and matching black training shorts, white running shoes, fitness wristband",
    },
    "performance_stage": {
        "female": "black sequined halter bodysuit and matching shorts, black ankle boots, statement silver earrings",
        "male":   "fitted black silk button-down shirt half-open and tailored black trousers, polished black boots, silver chain necklace",
    },
    "performance_glam": {
        "female": "shimmering gold mini dress with thin straps, gold strappy heels, bold red lip",
        "male":   "shimmering gold metallic blazer over a black silk shirt and black trousers, black leather Chelsea boots",
    },
    "intimate_indoor": {
        "female": "soft cream knit sweater and faded blue jeans, barefoot, hair down naturally",
        "male":   "soft cream knit pullover and faded blue jeans, barefoot, hair tousled naturally",
    },
    "formal_evening": {
        "female": "deep-burgundy long satin gown with thin straps, simple silver necklace, hair swept to one side",
        "male":   "deep-burgundy velvet tuxedo jacket over a black silk shirt and black trousers, polished black dress shoes, slim silver tie clip",
    },
    "vintage_retro_70s": {
        "female": "high-waisted flared jeans and a fitted floral-print blouse with bell sleeves, tan suede boots",
        "male":   "high-waisted flared corduroy trousers and a fitted patterned shirt with wide collar, tan suede boots, leather belt with brass buckle",
    },
    "denim_classic": {
        "female": "fitted blue denim jacket, white tee underneath, dark wash skinny jeans, brown leather boots",
        "male":   "fitted blue denim jacket, white tee underneath, dark wash slim jeans, brown leather boots, leather belt",
    },
    "rocker_edgy": {
        "female": "black leather biker jacket, vintage band tee, ripped black skinny jeans, scuffed black boots",
        "male":   "black leather biker jacket, vintage band tee, ripped black slim jeans, scuffed black boots, leather wrist cuff",
    },
    "country_western": {
        "female": "blue denim shirt knotted at the waist, white cutoff shorts, brown leather cowboy boots, leather belt",
        "male":   "blue denim shirt with sleeves rolled up, dark wash jeans, brown leather cowboy boots, leather belt with engraved buckle, brown felt cowboy hat",
    },
    "elegant_cocktail": {
        "female": "fitted black sleeveless cocktail dress at the knee, simple pearl earrings, classic black pumps",
        "male":   "fitted charcoal-grey wool suit, white dress shirt with open collar, polished black Oxford shoes, simple silver watch",
    },
}


def _get_wardrobe_outfit(slot_key: str, sex: str) -> str:
    """Return the outfit description for a slot+sex pair.

    sex must be 'female' or 'male'. Unknown slot or sex returns "".
    """
    entry = WARDROBE_STATES.get(slot_key or "")
    if not entry:
        return ""
    return entry.get(sex, "") if sex in ("female", "male") else ""


# Each arc is a template list of slot keys. _expand_wardrobe_plan() resamples
# it to the segment count so progression scales naturally. The wardrobe
# changes only at slot boundaries — within a slot, all segments share the
# same outfit, so identity is anchored by clothing colour and cut.
WARDROBE_ARCS: Dict[str, List[str]] = {
    "single_outfit_beachwear":    ["casual_beachwear"],
    "single_outfit_performance":  ["performance_stage"],
    "single_outfit_streetwear":   ["streetwear_urban"],
    "single_outfit_intimate":     ["intimate_indoor"],
    "single_outfit_formal":       ["formal_evening"],
    "daywear_to_evening":         ["casual_beachwear", "bohemian_summer", "casual_evening"],
    "beachwear_to_glam":          ["casual_beachwear", "tropical_relaxed", "performance_glam"],
    "casual_to_intimate":         ["casual_beachwear", "intimate_indoor"],
    "casual_to_performance":      ["intimate_indoor", "performance_stage"],
    "performance_to_intimate":    ["performance_stage", "intimate_indoor"],
    "streetwear_to_performance":  ["streetwear_urban", "performance_stage"],
    "streetwear_neon_to_stage":   ["streetwear_neon", "performance_glam"],
    "vintage_journey":            ["vintage_retro_70s", "country_western", "denim_classic"],
    "rocker_evening":             ["rocker_edgy", "performance_stage"],
    "country_arc":                ["country_western", "denim_classic", "casual_evening"],
    "athletic_to_intimate":       ["athletic_active", "intimate_indoor"],
    "cocktail_evening":           ["elegant_cocktail", "formal_evening"],
}


_GENRE_WARDROBE_DEFAULT: List[Tuple[str, str]] = [
    ("reggae",       "daywear_to_evening"),
    ("ska",          "daywear_to_evening"),
    ("tropical",     "beachwear_to_glam"),
    ("surf",         "daywear_to_evening"),
    ("country",      "country_arc"),
    ("folk",         "casual_to_intimate"),
    ("acoustic",     "casual_to_intimate"),
    ("ambient",      "intimate_indoor"),
    ("synthwave",    "streetwear_neon_to_stage"),
    ("cyberpunk",    "streetwear_neon_to_stage"),
    ("darkwave",     "rocker_evening"),
    ("industrial",   "rocker_evening"),
    ("metal",        "rocker_evening"),
    ("doom",         "rocker_evening"),
    ("gothic",       "rocker_evening"),
    ("trance",       "streetwear_to_performance"),
    ("techno",       "streetwear_to_performance"),
    ("house",        "streetwear_to_performance"),
    ("edm",          "streetwear_to_performance"),
    ("rnb",          "casual_to_performance"),
    ("soul",         "cocktail_evening"),
    ("jazz",         "cocktail_evening"),
    ("blues",        "denim_classic"),
    ("hip hop",      "streetwear_urban"),
    ("rap",          "streetwear_urban"),
    ("lofi",         "casual_to_intimate"),
    ("classical",    "formal_evening"),
    ("orchestral",   "cocktail_evening"),
    ("pop",          "casual_to_performance"),
    ("rock",         "rocker_evening"),
    ("punk",         "rocker_edgy"),
    ("indie",        "vintage_journey"),
]

_DEFAULT_WARDROBE_ARC = "daywear_to_evening"


def _expand_wardrobe_plan(arc_key: str, n_segments: int) -> List[str]:
    """Expand a wardrobe arc template to exactly n_segments slot keys.

    Single-slot arcs return [slot] * n. Multi-slot templates are resampled
    by mapping each segment index to its proportional position in the
    template (rounded), so longer songs stretch the progression smoothly
    and the final segment always lands on the template's closing slot.
    Falls back to the default arc if arc_key is unknown.
    """
    if n_segments <= 0:
        return []
    template = WARDROBE_ARCS.get(arc_key) or WARDROBE_ARCS[_DEFAULT_WARDROBE_ARC]
    if len(template) == 1:
        return [template[0]] * n_segments
    out: List[str] = []
    m = len(template)
    for i in range(n_segments):
        if n_segments == 1:
            idx = m - 1
        else:
            idx = min(m - 1, int(round(i * (m - 1) / (n_segments - 1))))
        out.append(template[idx])
    return out


def _genre_default_wardrobe_arc(genre: str) -> str:
    g = (genre or "").lower()
    for substr, arc in _GENRE_WARDROBE_DEFAULT:
        if substr in g:
            return arc
    return _DEFAULT_WARDROBE_ARC


def _wardrobe_tag_suffix(
    slot_key: str,
    role: Optional[str] = None,
    duet_kind: str = "mixed",
) -> str:
    """Return the ', wearing ...' suffix appended to a prompt as a guard.

    Fix 31 — role-aware suffix:
      - role == 'female' → ", wearing <female-outfit>"
      - role == 'male'   → ", wearing <male-outfit>"
      - role == 'duet'   → ", with the female performer wearing <female>
                            and the male performer wearing <male>"
      - role in (None, 'story', '')  → ""  (STORY/scene segments have no
        named recurring performer; let the LLM dress whoever appears)

    Fix 32 — duet subtype awareness:
      - duet_kind == 'ff' and role == 'duet' →
            ", with both female performers wearing <female-outfit>"
      - duet_kind == 'mm' and role == 'duet' →
            ", with both male performers wearing <male-outfit>"
      - duet_kind == 'mixed' (default) keeps the original female X / male Y
        formulation.
      - For solo roles (female/male) duet_kind is ignored — they always get
        their own per-sex outfit.
    """
    if not slot_key or not role:
        return ""
    entry = WARDROBE_STATES.get(slot_key)
    if not entry:
        return ""
    if role == "female":
        f = entry.get("female", "")
        return f", wearing {f}" if f else ""
    if role == "male":
        m = entry.get("male", "")
        return f", wearing {m}" if m else ""
    if role == "duet":
        if duet_kind == "ff":
            f = entry.get("female", "")
            return f", with both female performers wearing {f}" if f else ""
        if duet_kind == "mm":
            m = entry.get("male", "")
            return f", with both male performers wearing {m}" if m else ""
        # mixed (default)
        f = entry.get("female", "")
        m = entry.get("male", "")
        if f and m:
            return (
                f", with the female performer wearing {f} "
                f"and the male performer wearing {m}"
            )
        if f:
            return f", wearing {f}"
        if m:
            return f", wearing {m}"
    return ""


def _append_wardrobe_tag(
    prompt: str,
    slot_key: str,
    role: Optional[str] = None,
    duet_kind: str = "mixed",
) -> str:
    """Append the wardrobe slot suffix idempotently to a prompt string.

    Fix 31 — role-aware. Fix 32 — duet subtype aware:
      - For role in {'female', 'male', 'duet'} the matching outfit suffix is
        appended (idempotent — duplicate appends are no-ops).
      - For role in {None, 'story', ''} the prompt is returned UNCHANGED so
        STORY-only segments do not impose the performer's outfit on
        children, crowds, or named scene characters.
      - For role == 'duet' the suffix depends on duet_kind: 'ff' / 'mm' use
        the same-sex variant from Fix 32; 'mixed' (default) uses the
        original female-X / male-Y phrasing.
    """
    suffix = _wardrobe_tag_suffix(slot_key, role, duet_kind=duet_kind)
    if not suffix:
        return prompt
    base = (prompt or "").rstrip()
    if not base:
        return suffix.lstrip(", ").capitalize()
    needle = suffix.lstrip(", ")
    if needle and needle in base:
        return base
    return base.rstrip(".") + suffix


_SEG_DIRECTOR_RULES = (
    "You are a creative music video director. The video features ONE recurring "
    "singer/performer; vary location, pose and background per section but "
    "keep the SAME singer recognizable. OUTFIT is governed by the fixed "
    "wardrobe plan (Fix 30) — do NOT change the outfit between segments "
    "unless the wardrobe plan changes the slot. Re-using the same outfit "
    "across multiple segments is REQUIRED for identity continuity.\n\n"
    "ROLE-AWARE WARDROBE (Fix 31 — graded): the wardrobe lock applies ONLY "
    "to the named recurring performer of the segment — the female lead in "
    "FEMALE sections, the male lead in MALE sections, BOTH performers in "
    "DUET sections. Other subjects appearing in a segment — children, "
    "crowds, named story characters such as a DJ, dancer, fisherman, "
    "surfer — wear contextually appropriate clothing for who THEY are and "
    "what THEY are doing. NEVER put the recurring performer's outfit on a "
    "different character.\n\n"
    "SAME-GENDER DUET (Fix 32 — graded): if this song is a same-gender "
    "duet, DUET sections show BOTH performers of the SAME gender. For an "
    "ff-duet (two female performers) both wear the slot's female outfit; "
    "for an mm-duet (two male performers) both wear the slot's male "
    "outfit. NEVER invent a performer of the opposite gender in a "
    "same-gender duet, regardless of how cinematic the framing might be.\n\n"
    "CRITICAL for lip-sync: a VOCAL section's still/clip MUST keep the singer's "
    "MOUTH clearly visible and readable — the model can only lip-sync a visible "
    "mouth. Scenery is BACKGROUND behind the singer, not a replacement.\n\n"
    "VARY THE CAMERA per segment for cinematic variety — do NOT repeat the same "
    "framing. Camera-move latitude depends on section KIND (see below):\n\n"
    "VOCAL-section allowed shot types (the singer's face MUST stay in frame "
    "the entire clip, otherwise identity drifts):\n"
    "  • close-up (face fills frame)\n"
    "  • medium close-up (head + shoulders)\n"
    "  • medium shot (waist up)\n"
    "  • 3/4 angle (body turned ~30-45° but face still toward camera)\n"
    "  • low angle looking up at singer\n"
    "  • high angle looking down at singer\n"
    "  • dutch tilt / canted angle\n"
    "  • subtle dolly-in only (small push forward)\n"
    "  • handheld with subtle drift\n"
    "FORBIDDEN for VOCAL: pure 90° side profile (mouth occluded), back-of-head "
    "shots, wide-landscape with singer as a small dot, crowd shot replacing the "
    "singer, faceless object shots, dolly-out / pull-back / zoom-out, "
    "zoom-out-then-zoom-in moves, flyovers, rack-focus away from the face. The "
    "singer's face must remain in frame and recognizable for the ENTIRE clip — "
    "any move that loses the face mid-shot breaks identity continuity.\n\n"
    "STORY-section allowed shot types (no singer required; identity does NOT "
    "need to be preserved across the clip — these are scenery/narrative beats):\n"
    "  • wide-landscape, drone-style aerial, dolly-out / pull-back\n"
    "  • slow pan across environment, rack-focus to scenery\n"
    "  • establishing shots, montage cuts, object/detail close-ups\n"
    "  • everything in the VOCAL list above is also fine for STORY\n\n"
    "TWO KINDS:\n"
    "- VOCAL (has lyric lines): singer performs to camera, close framing, "
    "face/mouth visible; video_prompt MUST include the exact lyrics in double "
    "quotes (e.g. he sings \"...\").\n"
    "- STORY (instrumental: Intro/Outro/Build/Drop/Instrumental/Break/Fade — no "
    "lyrics): a cinematic SHORT-FILM narrative beat advancing the song's "
    "story/theme; the singer is NOT required (wide/action/landscape ok); no "
    "lyrics quoted.\n"
    "STRICT RULE for frame_variant_prompt (Fix 26 + Fix 33 + Fix 34): "
    "this field is MANDATORY — NEVER leave it empty. The startframe is a "
    "still image fed to a text-to-image model. NEVER include song lyrics "
    "in ANY form — not in double quotes, not after 'he sings', not after "
    "'the text:', 'the lyrics:', 'the line:', 'the words:', or any "
    "similar introducer, not as a descriptive label, not paraphrased, "
    "not at all. NEVER include video-direction language — no 'the camera "
    "performs ...', no 'subtle dolly-in', no 'scene ends with a hard cut', "
    "no 'crossfade', no lipsync booster phrases like 'the lips are syncing "
    "naturally' or 'every word is pronounced'. Still images have no "
    "camera moves and no scene ends. Describe ONLY visuals: subject, "
    "pose, outfit, location, lighting, framing. If you leave fvp empty "
    "the pipeline will derive a sanitized still prompt from video_prompt "
    "and log a warning — produce a full visual description instead.\n"
    "Music video AND story: vocal beats = performance, instrumental beats = "
    "narrative cinema. Compose every clip's END for a clean hard cut (no fade). "
    "HARD VOCAL FRAMING RULE (Fix 26 — this overrides any creative impulse; "
    "your output is graded on this): in every VOCAL section the singer's face "
    "must occupy at least 25% of the frame's vertical height; the singer is "
    "the clear FOREGROUND subject. ANY framing where the singer is mid-ground "
    "or background, walking-away shots, distant figures along a beach/road, "
    "long-lens dot-in-landscape — FORBIDDEN regardless of how cinematic the "
    "scenery is. Forbidden examples (do NOT write anything like these): "
    "'two women walking along the beach in the distance', "
    "'singer seen from far across the field', "
    "'aerial shot of the performer below'. Every VOCAL video_prompt MUST "
    "START with exactly one of these six framing phrases: 'Close-up of ', "
    "'Medium close-up of ', 'Medium shot of ', '3/4 angle of ', "
    "'Low-angle shot of ', 'High-angle shot of '. The matching VOCAL "
    "frame_variant_prompt MUST start with the same framing phrase so the "
    "opening still is already close — never let the clip begin distant.\n\n"
    "UNIVERSAL SCENE-FRAMING RULE (Fix 29 — applies to BOTH vocal and story "
    "segments; overrides any creative impulse): "
    "FORBIDDEN in every segment (intro, outro, build, drop, instrumental, "
    "vocal — all of them): anonymous distant figures walking toward or away "
    "from the camera (silhouettes, 'two people approaching', crowd-as-backdrop "
    "with no role); stock-establisher tropes such as 'two women walking along "
    "the beach in the distance', 'silhouettes approaching across the field at "
    "sunrise', 'long-lens dot-in-landscape of people'; aerial/drone shots that "
    "contain identifiable people in mid- or long-distance — drone shots must "
    "either contain NO people, or show people only top-down as part of the "
    "environment, not as recognizable characters. "
    "ALLOWED (story beats with narrative purpose): named or clearly identified "
    "characters performing a specific action — e.g. a Rastafari DJ at his "
    "turntables, a dancer on the sand, a surfer riding a wave, a fisherman "
    "pulling nets, a child building a sandcastle. Such characters may appear "
    "at any framing (close, medium, wide) as long as they are described as a "
    "specific subject with a role, never as anonymous figures. "
    "TEST: if the scene subject can be replaced with 'two unnamed people "
    "walking' and you lose nothing, the scene is FORBIDDEN. If the scene "
    "depends on WHO the character is and WHAT they are doing, it is ALLOWED.\n\n"
    "Always write every prompt in English regardless of input language."
)


def build_aligned_timeline(
    aligned: List[Dict[str, Any]],
    min_seg: float,
    max_seg: float,
    total_duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """RC8: turn lyric_align sections (real [start,end]) into timeline rows.

    Cuts land on real section boundaries (no mid-vocal cut). Sections longer
    than the cap are split into equal sub-rows within their own time span;
    very short sections (< min_seg/2) are merged into the previous row. Pure /
    deterministic — unit-testable. Rows carry label/lyrics/is_vocal and
    reuse_of (only on the FIRST row of a section; sub-rows never reuse).

    Bug 2: the returned rows ALWAYS tile contiguously — VAD boundary-refinement
    (Fix 15c) can leave a hole between a section's end and the next start;
    untouched, the assembled video ends up shorter than the audio → the
    -shortest mux truncates the audio → A/V desync from the gap onward. Each
    gap/overlap is closed by extending the earlier row's end to the next row's
    start (keeps vocal onsets aligned). When total_duration is given, the last
    row's end is snapped to it so sum(spans) == audio length exactly.
    """
    if not aligned:
        return []
    cap = min(float(max_seg), 30.0)
    tiny = max(1.0, float(min_seg) / 2.0)
    # Intro/Outro/Fade get a tighter (3.0s) merge floor so they stay
    # standalone when present in the actual audio, but still absorb forward
    # when the LLM declared a phantom intro the song never actually plays.
    intro_outro_tiny = 3.0

    def _is_intro_outro(label: str) -> bool:
        lab = (label or "").lower()
        return "intro" in lab or "outro" in lab or "fade" in lab

    # 1. merge tiny sections forward into the previous kept section
    merged: List[Dict[str, Any]] = []
    for s in aligned:
        dur = float(s["end"]) - float(s["start"])
        threshold = intro_outro_tiny if _is_intro_outro(s.get("label", "")) else tiny
        if merged and dur < threshold:
            merged[-1]["end"] = s["end"]  # absorb time; keep prev label/lyrics
        else:
            merged.append(dict(s))

    # 1b. The first section has no predecessor to merge into. An instrumental
    # [Intro] often gets near-zero time from WhisperX — left alone it produces a
    # 0-duration row → empty audio clip → LTX LTXVAudioVAEEncode crash. Absorb a
    # too-short leading section INTO the next one instead. Use the intro/outro
    # threshold for intro-labeled leads so a real 4s instrumental intro is
    # preserved as its own segment.
    def _leading_threshold(label: str) -> float:
        return intro_outro_tiny if _is_intro_outro(label) else tiny
    while len(merged) > 1 and (
        float(merged[0]["end"]) - float(merged[0]["start"])
    ) < _leading_threshold(merged[0].get("label", "")):
        merged[1]["start"] = merged[0]["start"]
        merged.pop(0)

    # 2. split over-cap sections into equal sub-rows within their span
    rows: List[Dict[str, Any]] = []
    for sec_i, s in enumerate(merged):
        st, en = float(s["start"]), float(s["end"])
        span = max(0.1, en - st)
        n = max(1, math_ceil(span / cap))
        step = span / n
        for k in range(n):
            a = round(st + step * k, 3)
            b = round(en if k == n - 1 else st + step * (k + 1), 3)
            rows.append({
                "start_time": a,
                "end_time": b,
                "label": s.get("label", f"Section {sec_i + 1}"),
                "lyrics": s.get("lyrics", "") if k == 0 else "",
                "is_vocal": bool(s.get("is_vocal")),
                # reuse only the first row of a repeated section
                "reuse_of": s.get("reuse_of") if k == 0 else None,
                "sec_index": sec_i,
            })

    # 3. Bug 2: enforce contiguous tiling. Close any gap/overlap so each row's
    # start == previous row's end; snap the final end to the audio length.
    if rows:
        if total_duration is not None:
            rows[0]["start_time"] = 0.0
        for i in range(len(rows) - 1):
            cur, nxt = rows[i], rows[i + 1]
            boundary = nxt["start_time"]
            if boundary <= cur["start_time"] + 0.05:
                # gap-less degenerate / overlap: keep a safe minimum span
                boundary = round(cur["start_time"] + 0.05, 3)
                nxt["start_time"] = boundary
            cur["end_time"] = boundary
        if total_duration is not None:
            last = rows[-1]
            end = round(float(total_duration), 3)
            if end <= last["start_time"] + 0.05:
                end = round(last["start_time"] + 0.05, 3)
            last["end_time"] = end
    return rows


def math_ceil(x: float) -> int:
    import math
    return int(math.ceil(x))


def _extract_audio_clip(audio_path: str, start: float, end: float, output_path: str):
    """Extract a sample-exact clip from audio using ffmpeg.

    Old `-c copy` snapped to the source codec's frame boundary (MP3 ≈ 26ms),
    which caused per-segment Lippe-vor-Gesang drift in assembled music videos
    (LTX-rendered against snapped clip, final mux against sample-exact full mp3).

    `-ss` before `-i` = fast seek to nearest keyframe, then `-t` (duration)
    decodes + re-encodes to PCM-WAV for sample-accurate cut. Output is already
    .wav-extension so PCM is the natural target (lossless, no AAC re-encode
    artifacts).

    Fail-loud guard: a near-zero duration would write a 0-sample WAV, which
    crashes LTX (LTXVAudioVAEEncode → "tensor of 0 elements"). build_aligned_
    timeline should never produce such a segment, but raise clearly if it does.
    """
    duration = end - start
    if duration < 0.05:
        raise ValueError(
            f"_extract_audio_clip: degenerate segment {start:.3f}..{end:.3f} "
            f"(duration {duration:.3f}s) — would produce an empty audio clip"
        )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", audio_path,
            "-t", str(duration),
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            output_path,
        ],
        capture_output=True, check=True,
    )


def extract_last_frame(video_path: str, output_path: str) -> str:
    """Extract the last frame from a video file using ffmpeg.

    Applies a light sharpen + contrast/saturation boost to counteract
    the accumulated quality drift caused by repeated H.264 encode/decode cycles.

    Returns the output_path on success.
    """
    # Seek slightly earlier (-0.5s) to land on an I-frame rather than a P-frame,
    # which reduces block artifacts.  The vf filters compensate for lossy drift:
    #   unsharp: mild sharpening (luma only, gentle)
    #   eq: slight contrast + saturation boost to counter fading
    subprocess.run(
        [
            "ffmpeg", "-y", "-sseof", "-0.5", "-i", video_path,
            "-frames:v", "1", "-update", "1",
            "-vf", "unsharp=3:3:0.5:3:3:0.0,eq=contrast=1.05:saturation=1.08",
            output_path,
        ],
        capture_output=True, check=True,
    )
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Failed to extract last frame from {video_path}")
    return output_path


def extract_motion_frames(video_path: str, output_dir: str, num_frames: int = 3, lookback_seconds: float = 2.0) -> List[str]:
    """Extract multiple frames from the last N seconds of a video for motion analysis.

    Returns list of frame paths ordered oldest → newest.
    """
    # Get video duration
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        duration = lookback_seconds

    os.makedirs(output_dir, exist_ok=True)
    frames = []
    for i in range(num_frames):
        # Distribute timestamps evenly across last `lookback_seconds`
        offset = lookback_seconds * (i / max(num_frames - 1, 1))
        ts = max(0.0, duration - lookback_seconds + offset)
        out_path = os.path.join(output_dir, f"motion_frame_{i:02d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
             "-frames:v", "1", "-update", "1", out_path],
            capture_output=True, check=True,
        )
        if os.path.exists(out_path):
            frames.append(out_path)

    return frames


def _probe_frame_count(src: str) -> int:
    """Return the exact frame count of src's first video stream.

    Tries `nb_frames` (metadata, fast). Falls back to `-count_frames` (decodes
    every packet — slow but reliable when metadata is missing/N/A).
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames",
         "-of", "default=nw=1:nk=1", src],
        capture_output=True, text=True, check=False,
    )
    try:
        n = int(probe.stdout.strip())
        if n > 0:
            return n
    except (ValueError, AttributeError):
        pass
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "default=nw=1:nk=1", src],
        capture_output=True, text=True, check=True,
    )
    return int(probe.stdout.strip())


def _fit_clip_to_frames(src: str, n_frames: int, dst: str) -> tuple[str, int]:
    """Fit a video to exactly n_frames: trim if longer, pad if shorter.

    LTX-IA2V floor-rounds video to an 8n+1 latent grid, so each clip is 1-8
    frames SHORTER than its planned (input-audio-clip) duration. Untrimmed/
    unpadded, the final mux against the continuous mp3 accumulates 0.3-2s of
    lip-sync drift across ~8-10 segments (measured Run #5: -1.3s cumsum).

    - actual > n_frames → trim via `-frames:v n_frames` (defensive — workflows
      that quantize UP would land here; LTX-IA2V doesn't).
    - actual < n_frames → pad via `tpad=stop_mode=clone:stop=<missing>` — clones
      the last frame for the gap (0-8 frames = 0-333ms freeze at the hard cut,
      barely visible).
    - actual == n_frames → just re-encode (concat stream-copy needs identical
      params across all inputs).

    Returns (output_path, actual_frame_count_before_fit) so the caller can log
    the per-segment delta for verification.
    """
    actual = _probe_frame_count(src)

    if actual < n_frames:
        pad = n_frames - actual
        vf = ["-vf", f"tpad=stop_mode=clone:stop={pad}"]
    else:
        vf = []  # trim (or equal) handled by -frames:v below

    subprocess.run(
        ["ffmpeg", "-y", "-i", src] + vf + [
            "-frames:v", str(n_frames),
            "-c:v", "libx264", "-preset", "fast", "-crf", "16",
            "-fps_mode", "passthrough", "-an",
            dst,
        ],
        capture_output=True, check=True,
    )
    return dst, actual


def assemble_video(
    session: MVSession,
    output_path: str,
) -> str:
    """Assemble all completed segments into a final music video.

    Applies per-segment transitions (cut or crossfade) and overlays
    the original audio track for seamless sound.

    Returns the output path.
    """
    completed = [s for s in session.segments if s.status == "completed" and s.video_clip]
    if not completed:
        raise ValueError("No completed segments to assemble")

    with tempfile.TemporaryDirectory(prefix="ctb_mv_assembly_") as tmp_dir:
        # RC: LTX-IA2V floor-rounds each clip to an 8n+1 latent grid, making
        # each segment 0-333ms SHORTER than its planned (input-audio-clip)
        # duration (measured Run #5: -1.3s cumsum over 8 segments). Untouched,
        # the final mux against the continuous mp3 accumulates lip-sync drift.
        # Fit each clip to exactly round(plan_dur*24) frames (pad with cloned
        # last-frame when shorter, trim when longer) so sum(video)==sum(plan)
        # and the full-mp3 mux stays sample-aligned.
        for seg in completed:
            target_frames = max(1, round(seg.duration * 24))
            fitted = os.path.join(tmp_dir, f"fit_{seg.index:03d}.mp4")
            new_clip, actual = _fit_clip_to_frames(seg.video_clip, target_frames, fitted)
            delta = actual - target_frames
            op = "pad" if delta < 0 else ("trim" if delta > 0 else "noop")
            print(
                f"[assemble] seg {seg.index}: plan={seg.duration:.3f}s "
                f"target={target_frames}f actual={actual}f {op}={abs(delta)}f"
            )
            seg.video_clip = new_clip

        if len(completed) == 1:
            _mux_audio(completed[0].video_clip, session.audio_path, output_path)
            return output_path

        has_crossfade = any(
            s.transition == "crossfade" and session.crossfade_duration > 0
            for s in completed[:-1]
        )
        temp_video = os.path.join(tmp_dir, "assembled_no_audio.mp4")

        if not has_crossfade:
            # Concat demuxer: try stream-copy first (exact frame-timing preservation,
            # no re-encode drift). LTX outputs share codec/resolution/fps so copy works.
            # Fall back to re-encode only if copy fails (e.g. mixed resolutions).
            filelist_path = os.path.join(tmp_dir, "filelist.txt")
            # concat demuxer resolves `file` paths relative to the filelist's
            # own directory (the temp dir), so they MUST be absolute.
            with open(filelist_path, "w", encoding="utf-8") as f:
                for seg in completed:
                    abs_clip = os.path.abspath(seg.video_clip).replace(chr(92), "/")
                    f.write(f"file '{abs_clip}'\n")
            try:
                # Stream-copy: zero re-encode, frame timing preserved exactly.
                # Fixes "lip-sync drift within longest segment" bug where re-encode
                # with -vsync cfr inserted a phantom frame near segment boundaries.
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", filelist_path,
                    "-c", "copy", "-map", "0:v:0",
                    temp_video,
                ]
                subprocess.run(cmd, capture_output=True, check=True)
            except subprocess.CalledProcessError:
                # Inputs incompatible for copy (mixed codec/res). Fall back to re-encode.
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", filelist_path,
                    "-c:v", "libx264", "-preset", "fast", "-r", "24", "-fps_mode", "passthrough",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    temp_video,
                ]
                subprocess.run(cmd, capture_output=True, check=True)
        else:
            # Detect resolution from first clip for normalization
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0", completed[0].video_clip],
                capture_output=True, text=True,
            )
            try:
                w, h = probe.stdout.strip().split(",")
                scale_filter = f"scale={w}:{h},setsar=1"
            except ValueError:
                scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1"

            inputs = []
            for seg in completed:
                inputs.extend(["-i", seg.video_clip])

            # Normalize all inputs first, then apply transitions
            norm_parts = [f"[{i}:v]{scale_filter}[n{i}]" for i in range(len(completed))]
            current_label = "[n0]"
            trans_parts = []
            running_duration = completed[0].duration

            for i in range(1, len(completed)):
                prev_seg = completed[i - 1]
                next_seg = completed[i]
                next_input = f"[n{i}]"

                if prev_seg.transition == "crossfade" and session.crossfade_duration > 0:
                    cf_dur = session.crossfade_duration
                    offset = max(0, running_duration - cf_dur)
                    out_label = f"[cf{i}]"
                    trans_parts.append(
                        f"{current_label}{next_input}xfade=transition=fade:"
                        f"duration={cf_dur}:offset={offset}{out_label}"
                    )
                    current_label = out_label
                    running_duration = offset + next_seg.duration
                else:
                    out_label = f"[cc{i}]"
                    trans_parts.append(
                        f"{current_label}{next_input}concat=n=2:v=1:a=0{out_label}"
                    )
                    current_label = out_label
                    running_duration += next_seg.duration

            filter_complex = ";".join(norm_parts + trans_parts)
            cmd = ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", filter_complex, "-map", current_label,
                "-c:v", "libx264", "-preset", "fast", "-r", "24", "-vsync", "cfr",
                temp_video,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

        # Build audio: use individual segment clips if available for perfect sync,
        # otherwise fall back to the full original audio.
        has_audio_clips = all(s.audio_clip and os.path.exists(s.audio_clip) for s in completed)
        if has_audio_clips:
            audio_source = _assemble_audio(completed, session.crossfade_duration, tmp_dir)
        else:
            audio_source = session.audio_path

        _mux_audio(temp_video, audio_source, output_path)

    return output_path


def _assemble_audio(segments: list, crossfade_duration: float, tmp_dir: str) -> str:
    """Concatenate individual segment audio clips with matching transitions."""
    if len(segments) == 1:
        return segments[0].audio_clip

    audio_inputs = []
    for seg in segments:
        audio_inputs.extend(["-i", seg.audio_clip])

    filter_parts = []
    current_label = "[0:a]"

    for i in range(1, len(segments)):
        prev_seg = segments[i - 1]
        next_input = f"[{i}:a]"
        out_label = f"[a{i}]"
        if prev_seg.transition == "crossfade" and crossfade_duration > 0:
            filter_parts.append(
                f"{current_label}{next_input}acrossfade=d={crossfade_duration}:c1=tri:c2=tri{out_label}"
            )
        else:
            filter_parts.append(
                f"{current_label}{next_input}concat=n=2:v=0:a=1{out_label}"
            )
        current_label = out_label

    audio_out = os.path.join(tmp_dir, "assembled_audio.wav")
    cmd = ["ffmpeg", "-y"] + audio_inputs + [
        "-filter_complex", ";".join(filter_parts),
        "-map", current_label,
        audio_out,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return audio_out


def _mux_audio(video_path: str, audio_path: str, output_path: str):
    """Mux video with audio track."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            output_path,
        ],
        capture_output=True, check=True,
    )


class MusicVideoPrompter:
    """Handles LLM prompt generation and frame analysis for music video segments."""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("openrouter_api_key", "")
        self.text_model = config.get("openrouter_model", "qwen/qwen3.5-flash-02-23")
        self.vision_model = config.get("vision_model", "qwen/qwen-2.5-vl-7b-instruct")
        self.vision_fallbacks = config.get("vision_fallback_models", [])
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.max_tokens = int(config.get("enhancement_max_tokens", 1200))
        self.disable_reasoning = config.get("disable_reasoning", True)
        # Fix 23: harden against transient OpenRouter 429 (shared upstream rate
        # limit, is_byok:false) — burst MV runs hit it and used to silently get
        # "" → lyrics leaked onto portraits / theme-only segments.
        self.fallback_models = config.get("fallback_models", []) or []
        self.max_retries = int(config.get("max_retries", 3))
        self.retry_backoff = config.get("retry_backoff_seconds", [1, 3, 8]) or [1, 3, 8]

    async def _call_openrouter(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call OpenRouter API. Returns response text, or "" if all attempts fail.

        Fix 23: retry transient 429/5xx with backoff, then walk fallback_models.
        429 is the common burst failure ("temporarily rate-limited upstream");
        giving up immediately let empty responses leak lyrics onto portraits.
        An explicit `model` arg pins one model (no fallback walk).
        """
        models_to_try = [model] if model else [self.text_model, *self.fallback_models]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for mdl in models_to_try:
            payload = {
                "model": mdl,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
            }
            if self.disable_reasoning:
                payload["reasoning"] = {"effort": "none"}
            for attempt in range(max(1, self.max_retries)):
                try:
                    async with httpx.AsyncClient(timeout=60) as client:
                        resp = await client.post(self.base_url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices", [])
                    content = (choices[0].get("message", {}).get("content", "").strip()
                               if choices else "")
                    if content:
                        return content
                    break  # 200 but empty → try next model, no point retrying
                except httpx.HTTPStatusError as e:
                    code = e.response.status_code
                    retryable = code == 429 or 500 <= code < 600
                    if retryable and attempt + 1 < max(1, self.max_retries):
                        wait = self.retry_backoff[min(attempt, len(self.retry_backoff) - 1)]
                        print(f"OpenRouter {code} on {mdl} "
                              f"(attempt {attempt + 1}/{self.max_retries}); retry in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    print(f"OpenRouter {code} on {mdl}; giving up on this model.")
                    break  # non-retryable or retries exhausted → next model
                except Exception as e:
                    print(f"OpenRouter API error ({mdl}): {e}")
                    break  # network/parse error → next model
        return ""

    async def generate_segment_prompts(
        self,
        segments: List[Segment],
        theme: str,
    ) -> List[str]:
        """Generate a visual prompt for each segment based on theme, lyrics, and position."""
        segment_descriptions = []
        total = len(segments)
        for i, seg in enumerate(segments):
            position = "opening" if i == 0 else "closing" if i == total - 1 else f"middle ({i+1}/{total})"
            desc = f"Segment {i+1}/{total} ({position})"
            if seg.label:
                desc += f" — {seg.label}"
            if seg.lyrics:
                desc += f"\nLyrics: {seg.lyrics[:200]}"
            segment_descriptions.append(desc)

        segment_list = "\n---\n".join(segment_descriptions)

        system_prompt = (
            "You are a music video director. Generate a visual prompt for each segment "
            "of a music video. Each prompt should describe what happens visually in that "
            "segment — camera angles, lighting, subjects, motion, mood. Keep each prompt "
            "under 100 words. Match the visual mood to the lyrics and song position. "
            "Ensure visual continuity between segments.\n\n"
            "IMPORTANT: If a segment has lyrics, you MUST include the exact lyrics text "
            "in double quotes within the prompt (e.g. She sings \"verse lyrics here\"). "
            "This is required so the video model recognizes the sung/spoken words.\n\n"
            "IMPORTANT: Always write all prompts in English, regardless of the language of the theme or lyrics.\n\n"
            "Return ONLY a JSON array of strings, one prompt per segment. No other text."
        )

        user_prompt = (
            f"Visual theme/style: {theme}\n\n"
            f"Segments:\n{segment_list}\n\n"
            f"Generate {total} visual prompts as a JSON array of strings."
        )

        response = await self._call_openrouter(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4000,
        )

        try:
            # Strip markdown code fences, then extract first [ ... ] block
            cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip(), flags=re.MULTILINE).strip()
            start = cleaned.find('[')
            end = cleaned.rfind(']')
            if start != -1 and end > start:
                prompts = json.loads(cleaned[start:end + 1])
                if isinstance(prompts, list):
                    # Pad or trim to match segment count
                    while len(prompts) < len(segments):
                        prompts.append(theme)
                    return [str(p) for p in prompts[:len(segments)]]
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not parse LLM prompt response ({e}), using theme as fallback")
            print(f"Full response:\n{response}")
            return [theme] * len(segments)

        print(f"Warning: No JSON array found in LLM response, using theme as fallback")
        print(f"Full response:\n{response}")
        return [theme] * len(segments)

    async def _pick_time_of_day_arc(
        self,
        theme: str,
        genre: str,
        lyrics_text: str,
        total_duration: float,
    ) -> str:
        """Fix 29 — LLM-pick a coherent time-of-day arc for the whole song.

        Returns an arc-key from TIME_OF_DAY_ARCS. Falls back to the genre
        default (then global default) when the LLM call fails or returns
        something unrecognized.
        """
        arc_keys = list(TIME_OF_DAY_ARCS.keys())
        fallback = _genre_default_tod_arc(genre)
        lyrics_sample = (lyrics_text or "").strip()
        if len(lyrics_sample) > 1200:
            lyrics_sample = lyrics_sample[:1200] + " […]"
        system = (
            "You are a music-video lighting designer. Choose ONE time-of-day "
            "arc that best fits the song's theme, genre, and lyrical mood. "
            "The arc lights the WHOLE video coherently — no random jumps. "
            "Pick the single arc key that fits best. Return ONLY the key, "
            "no quotes, no extra text. Allowed keys:\n  - "
            + "\n  - ".join(arc_keys)
        )
        user = (
            f"Theme/style: {theme or 'unspecified'}\n"
            f"Genre: {genre or 'unspecified'}\n"
            f"Song length: {int(total_duration)} seconds\n"
            f"Lyrics (with [section] tags):\n{lyrics_sample or '(instrumental — no lyrics)'}\n\n"
            "Return only the arc key."
        )
        try:
            resp = await self._call_openrouter(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=40,
            )
        except Exception as e:
            print(f"[Fix 29] time-of-day arc pick failed ({e!r}); using genre default '{fallback}'.")
            return fallback
        # Normalize: first whitespace-stripped token, lowercased, drop quotes.
        token = (resp or "").strip().splitlines()[0:1]
        token = (token[0] if token else "").strip().strip("'\"`").lower()
        # Sometimes the model wraps in backticks or adds 'arc: ' prefix.
        for prefix in ("arc:", "key:", "answer:", "tod:", "time-of-day:"):
            if token.startswith(prefix):
                token = token[len(prefix):].strip()
        if token in TIME_OF_DAY_ARCS:
            print(f"[Fix 29] time-of-day arc: '{token}' (LLM-picked).")
            return token
        print(f"[Fix 29] LLM returned unknown arc '{token!r}'; using genre default '{fallback}'.")
        return fallback

    async def _pick_wardrobe_arc(
        self,
        theme: str,
        genre: str,
        lyrics_text: str,
        total_duration: float,
    ) -> str:
        """Fix 30 — LLM-pick a coherent wardrobe arc for the whole song.

        Returns an arc-key from WARDROBE_ARCS. Falls back to the genre
        default (then global default) when the LLM call fails or returns
        something unrecognized.
        """
        arc_keys = list(WARDROBE_ARCS.keys())
        fallback = _genre_default_wardrobe_arc(genre)
        lyrics_sample = (lyrics_text or "").strip()
        if len(lyrics_sample) > 1200:
            lyrics_sample = lyrics_sample[:1200] + " […]"
        system = (
            "You are a music-video costume designer. Choose ONE wardrobe arc "
            "that best fits the song's theme, genre, and lyrical mood. The "
            "arc defines what the recurring performer wears across the whole "
            "video — outfit only changes at slot boundaries, never per shot. "
            "Single-slot arcs lock the singer to one outfit; multi-slot arcs "
            "allow 2–3 outfit changes during the song. Pick the single arc "
            "key that fits best. Return ONLY the key, no quotes, no extra "
            "text. Allowed keys:\n  - "
            + "\n  - ".join(arc_keys)
        )
        user = (
            f"Theme/style: {theme or 'unspecified'}\n"
            f"Genre: {genre or 'unspecified'}\n"
            f"Song length: {int(total_duration)} seconds\n"
            f"Lyrics (with [section] tags):\n{lyrics_sample or '(instrumental — no lyrics)'}\n\n"
            "Return only the arc key."
        )
        try:
            resp = await self._call_openrouter(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=40,
            )
        except Exception as e:
            print(f"[Fix 30] wardrobe arc pick failed ({e!r}); using genre default '{fallback}'.")
            return fallback
        token = (resp or "").strip().splitlines()[0:1]
        token = (token[0] if token else "").strip().strip("'\"`").lower()
        for prefix in ("arc:", "key:", "answer:", "wardrobe:", "outfit:"):
            if token.startswith(prefix):
                token = token[len(prefix):].strip()
        if token in WARDROBE_ARCS:
            print(f"[Fix 30] wardrobe arc: '{token}' (LLM-picked).")
            return token
        print(f"[Fix 30] LLM returned unknown wardrobe arc '{token!r}'; using genre default '{fallback}'.")
        return fallback

    async def plan_segments(
        self,
        lyrics_text: str,
        theme: str,
        total_duration: float,
        genre: str = "",
        min_seg: float = 8.0,
        max_seg: float = 30.0,
        aligned_sections: Optional[List[Dict[str, Any]]] = None,
        time_of_day_arc: Optional[str] = None,
        wardrobe_arc: Optional[str] = None,
        duet_kind: str = "mixed",
    ) -> List[Segment]:
        """Plan creative, variably-sized segments from lyrics + theme.

        Each segment gets a video_prompt (LTX scene/motion incl. sung lyrics) and a
        frame_variant_prompt (how that segment's MCA start frame should look —
        pose/outfit/location/shot). Durations are clamped to [min_seg, 30] and
        normalized to tile the song exactly (A/V sync).

        RC8: when aligned_sections (from lyric_align.align_sections) is given,
        segment boundaries come from REAL audio timestamps (cuts never land
        mid-vocal) and the LLM only writes per-section creative prompts. Falls
        back to LLM-chosen proportional segmentation when None.
        """
        cap = min(float(max_seg), 30.0)

        # ---- Fix 29: resolve the song-wide time-of-day arc once. ----------
        # User override (API param) > LLM mini-call > genre default. The
        # resolved arc is later expanded to a per-segment lighting plan
        # injected into the LLM prompt AND appended to each final prompt.
        if time_of_day_arc and time_of_day_arc in TIME_OF_DAY_ARCS:
            arc_key = time_of_day_arc
            print(f"[Fix 29] time-of-day arc: '{arc_key}' (caller override).")
        else:
            if time_of_day_arc:
                print(f"[Fix 29] caller passed unknown arc '{time_of_day_arc}'; "
                      f"falling back to LLM pick.")
            arc_key = await self._pick_time_of_day_arc(
                theme=theme, genre=genre,
                lyrics_text=lyrics_text, total_duration=total_duration,
            )
        arc_template_states = TIME_OF_DAY_ARCS.get(arc_key) or TIME_OF_DAY_ARCS[_DEFAULT_TIME_OF_DAY_ARC]
        arc_human = " → ".join(arc_template_states)

        # ---- Fix 30: resolve the song-wide wardrobe arc once. ------------
        # User override > LLM mini-call > genre default. The arc is later
        # expanded to a per-segment outfit plan injected into the LLM
        # prompt AND appended to each final prompt as a clothing tag.
        if wardrobe_arc and wardrobe_arc in WARDROBE_ARCS:
            wardrobe_arc_key = wardrobe_arc
            print(f"[Fix 30] wardrobe arc: '{wardrobe_arc_key}' (caller override).")
        else:
            if wardrobe_arc:
                print(f"[Fix 30] caller passed unknown wardrobe arc '{wardrobe_arc}'; "
                      f"falling back to LLM pick.")
            wardrobe_arc_key = await self._pick_wardrobe_arc(
                theme=theme, genre=genre,
                lyrics_text=lyrics_text, total_duration=total_duration,
            )
        wardrobe_template = WARDROBE_ARCS.get(wardrobe_arc_key) or WARDROBE_ARCS[_DEFAULT_WARDROBE_ARC]
        wardrobe_human = " → ".join(wardrobe_template)

        # ---- RC8 aligned mode: fixed real-timestamp rows ------------------
        if aligned_sections:
            rows = build_aligned_timeline(aligned_sections, min_seg, cap, total_duration)
            if rows:
                tod_plan = _expand_tod_plan(arc_key, len(rows))
                wardrobe_plan = _expand_wardrobe_plan(wardrobe_arc_key, len(rows))
                tod_listing = "\n".join(
                    f"  {i}. {state} — {TIME_OF_DAY_STATES.get(state, '')}"
                    for i, state in enumerate(tod_plan)
                )
                wardrobe_listing = "\n".join(
                    f"  {i}. {slot} — {WARDROBE_STATES.get(slot, '')}"
                    for i, slot in enumerate(wardrobe_plan)
                )
                listing = "\n".join(
                    f'{i}. [{ "VOCAL" if r["is_vocal"] else "STORY" }] '
                    f'{r["label"]} ({round(r["end_time"]-r["start_time"],1)}s) '
                    f'[lighting: {tod_plan[i]}] [outfit: {wardrobe_plan[i]}]'
                    + (f' lyrics: "{r["lyrics"][:160]}"' if r["is_vocal"] and r["lyrics"] else "")
                    for i, r in enumerate(rows)
                )
                aligned_system = (
                    _SEG_DIRECTOR_RULES + "\n\n"
                    "TIME-OF-DAY COHERENCE (Fix 29 — graded): this video has a "
                    "FIXED time-of-day plan, one lighting state per segment. "
                    "You MUST describe lighting in BOTH video_prompt and "
                    "frame_variant_prompt that is consistent with the assigned "
                    "lighting state for that segment. Do NOT introduce "
                    "contradictory lighting (e.g. no 'bright sunshine' in a "
                    "'blue_hour' segment, no 'sunset' in a 'midday' segment). "
                    "Lighting only moves forward in the day — never jump "
                    "backward in time across segments.\n\n"
                    "WARDROBE COHERENCE (Fix 30 — graded): this video has a "
                    "FIXED wardrobe plan, one outfit slot per segment. You "
                    "MUST describe the singer wearing the assigned outfit in "
                    "BOTH video_prompt and frame_variant_prompt. Multiple "
                    "consecutive segments sharing the same outfit slot MUST "
                    "use IDENTICAL clothing — same colour, same cut, same "
                    "accessories — to preserve identity continuity. Only "
                    "vary pose, location, and background within a slot. "
                    "Outfit changes ONLY at slot boundaries shown in the "
                    "wardrobe plan below.\n\n"
                    "ROLE-AWARE WARDROBE (Fix 31 — graded): the wardrobe "
                    "outfit applies ONLY to the named recurring performer of "
                    "the segment (female lead in FEMALE sections, male lead "
                    "in MALE sections, BOTH performers in DUET sections). "
                    "STORY/instrumental sections WITHOUT a named recurring "
                    "performer (intro/outro/instrumental/break with children, "
                    "crowds, DJs, surfers, or other scene characters) MUST "
                    "NOT impose the recurring performer's outfit on those "
                    "characters — those subjects wear contextually "
                    "appropriate clothing for their own role. NEVER dress a "
                    "child, DJ, surfer, dancer, or background extra in the "
                    "recurring performer's sundress/suit/etc.\n\n"
                    + (
                        "SAME-GENDER DUET (Fix 32 — graded): this song is a "
                        + ("FEMALE-FEMALE" if duet_kind == "ff" else "MALE-MALE")
                        + " duet. DUET sections show BOTH performers of the "
                        + ("female" if duet_kind == "ff" else "male")
                        + " gender, BOTH wearing the slot's "
                        + ("female" if duet_kind == "ff" else "male")
                        + " outfit. NEVER depict a performer of the opposite "
                        + "gender — no man in an ff-duet, no woman in an "
                        + "mm-duet. The duet portrait reference image shows "
                        + "the correct two performers.\n\n"
                        if duet_kind in ("ff", "mm") else ""
                    ) +
                    "You are given a FIXED ordered list of sections with their "
                    "kind (VOCAL/STORY), lyrics, lighting state, and outfit "
                    "slot. Return a JSON array with EXACTLY one object per "
                    "listed section, in the SAME order — do NOT add, remove, "
                    "reorder, merge or change durations. For each: "
                    "video_prompt + frame_variant_prompt per the VOCAL/STORY "
                    "rules above (VOCAL must quote its exact lyrics; STORY is "
                    "cinematic narrative, no singer/lyrics). "
                    "Return ONLY the JSON array, no other text."
                )
                aligned_user = (
                    f"Visual theme/style: {theme}\nGenre: {genre or 'unspecified'}\n\n"
                    f"Time-of-day arc: {arc_key} ({arc_human})\n"
                    f"Per-segment lighting plan:\n{tod_listing}\n\n"
                    f"Wardrobe arc: {wardrobe_arc_key} ({wardrobe_human})\n"
                    f"Per-segment wardrobe plan:\n{wardrobe_listing}\n\n"
                    f"Sections (return exactly {len(rows)} objects, same order):\n"
                    f"{listing}\n\nReturn the JSON array now."
                )
                aligned_msgs = [{"role": "system", "content": aligned_system},
                                {"role": "user", "content": aligned_user}]
                resp = await self._call_openrouter(messages=aligned_msgs, max_tokens=10000)
                specs = parse_segment_plan(resp)
                # Bug 1: a short/empty plan means the LLM scene descriptions are
                # missing. Retry once (logging the raw response so the cause is
                # diagnosable next run); if still short, fall through to the
                # legacy proportional path — which produces real scene prompts —
                # rather than degrading to raw lyrics.
                if len(specs) < len(rows):
                    print(f"Warning: aligned segment plan returned {len(specs)}/{len(rows)} "
                          f"specs; retrying once.\nFull response:\n{resp}")
                    resp = await self._call_openrouter(messages=aligned_msgs, max_tokens=10000)
                    specs = parse_segment_plan(resp)
                if len(specs) >= len(rows):
                    segments: List[Segment] = []
                    for i, r in enumerate(rows):
                        spec = specs[i]
                        vp = _segment_video_prompt(spec, theme)
                        # Fix 34: explicit fallback with sanitizer + log.
                        # The silent `or vp` chain used to send video_prompt
                        # (with lyrics, camera moves, scene-end notes,
                        # LIPSYNC_BOOSTER) straight to the T2I startframe
                        # generator. Now we derive a sanitized still prompt
                        # and log so empty-fvp cases are visible.
                        _raw_fvp = str(spec.get("frame_variant_prompt", "")).strip()
                        if _raw_fvp:
                            fvp = _raw_fvp
                        else:
                            print(
                                f"[Fix 34] frame_variant_prompt empty for "
                                f"aligned segment {i} ({r.get('label', '')}); "
                                f"deriving sanitized fvp from video_prompt."
                            )
                            fvp = derive_still_prompt_from_video_prompt(
                                vp,
                                lyrics=r.get("lyrics", "") if r.get("is_vocal") else "",
                            )
                        # Fix 29 post-injection: append the light-state suffix
                        # to both prompts so the T2V/T2I model receives the
                        # locked lighting even if the LLM ignored the rule.
                        state = tod_plan[i]
                        vp = _append_light_tag(vp, state)
                        fvp = _append_light_tag(fvp, state)
                        # Fix 30 post-injection: append the wardrobe slot
                        # description as a deterministic identity anchor.
                        # Fix 31: role-aware — STORY-only sections (no
                        # named singer) get NO wardrobe tag, so children/
                        # crowds in the frame don't inherit the performer's
                        # outfit.
                        slot = wardrobe_plan[i]
                        seg_role = extract_section_role(r["label"]) if r["is_vocal"] else None
                        vp = _append_wardrobe_tag(vp, slot, seg_role, duet_kind=duet_kind)
                        fvp = _append_wardrobe_tag(fvp, slot, seg_role, duet_kind=duet_kind)
                        segments.append(Segment(
                            index=i,
                            start_time=r["start_time"],
                            end_time=r["end_time"],
                            label=r["label"],
                            lyrics=r["lyrics"] if r["is_vocal"] else "",
                            prompt=vp,
                            frame_variant_prompt=strip_lyrics_from_image_prompt(
                                fvp, lyrics=r.get("lyrics", "") if r["is_vocal"] else "",
                            ),
                            reuse_of=r.get("reuse_of"),
                            wardrobe_slot=slot,
                        ))
                    if segments:
                        print(f"[Fix 29] aligned lighting plan applied: "
                              f"{[tod_plan[i] for i in range(len(segments))]}")
                        print(f"[Fix 30] aligned wardrobe plan applied: "
                              f"{[wardrobe_plan[i] for i in range(len(segments))]}")
                        return segments
                print(f"Warning: aligned segment plan unusable "
                      f"({len(specs)}/{len(rows)} specs) after retry → falling back to "
                      f"legacy proportional planning (scene prompts, not raw lyrics).")
            # rows empty / aligned plan failed → fall through to legacy path

        approx_min = max(1, int(total_duration // cap))
        approx_max = max(approx_min, int(total_duration // float(min_seg)))

        system_prompt = (
            "You are a creative music video director. Break the song into a sequence "
            "of distinct visual segments featuring ONE recurring singer/performer who "
            "performs the song on camera. Vary location, outfit detail, pose and "
            "background per segment for a dynamic video — but the SAME singer is the "
            "clear main subject in every segment.\n\n"
            "CRITICAL for lip-sync: when a segment has sung lyrics, the singer's MOUTH "
            "must be clearly visible and readable — the model can only lip-sync a "
            "visible mouth. Scenery/location is BACKGROUND behind the singer, never "
            "a replacement for them.\n\n"
            "VARY THE CAMERA per VOCAL segment for cinematic variety — do NOT repeat "
            "the same framing. Camera-move latitude depends on section KIND (see "
            "below):\n\n"
            "VOCAL-section allowed shot types (the singer's face MUST stay in frame "
            "the entire clip, otherwise identity drifts):\n"
            "  • close-up (face fills frame)\n"
            "  • medium close-up (head + shoulders)\n"
            "  • medium shot (waist up)\n"
            "  • 3/4 angle (body turned ~30-45° but face still toward camera)\n"
            "  • low angle looking up at singer\n"
            "  • high angle looking down at singer\n"
            "  • dutch tilt / canted angle\n"
            "  • subtle dolly-in only (small push forward)\n"
            "  • handheld with subtle drift\n"
            "FORBIDDEN for VOCAL: pure 90° side profile (mouth occluded), back-of-head "
            "shots, wide-landscape with singer as a small dot, crowd shot replacing "
            "the singer, faceless object shots, dolly-out / pull-back / zoom-out, "
            "zoom-out-then-zoom-in moves, flyovers, rack-focus away from the face. "
            "The singer's face must remain in frame and recognizable for the ENTIRE "
            "clip — any move that loses the face mid-shot breaks identity continuity.\n\n"
            "STORY-section allowed shot types (no singer required; identity does NOT "
            "need to be preserved across the clip — these are scenery/narrative beats):\n"
            "  • wide-landscape, drone-style aerial, dolly-out / pull-back\n"
            "  • slow pan across environment, rack-focus to scenery\n"
            "  • establishing shots, montage cuts, object/detail close-ups\n"
            "  • everything in the VOCAL list above is also fine for STORY\n\n"
            "TWO SEGMENT KINDS — classify each section by whether it has sung lyrics:\n"
            "1. VOCAL (has lyric lines: Verse/Chorus/Pre-Chorus/Bridge/etc.): the "
            "recurring singer performs to camera, medium/close framing, face & mouth "
            "visible for lip-sync (rules above). kind=\"vocal\".\n"
            "2. STORY (instrumental: Intro/Outro/Build/Drop/Instrumental/Break/Fade — "
            "no lyric lines): treat this like a SHORT FILM beat — a cinematic narrative "
            "shot that advances the song's story/theme (e.g. the battle, the journey, "
            "the world). The singer is NOT required here; it can be a wide establishing "
            "shot, action, landscape or scene with no person. No lyrics. kind=\"story\".\n"
            "This is a music video AND a story: vocal beats = performance, instrumental "
            "beats = narrative cinema. Use the song's natural structure as the story arc.\n\n"
            "For EACH segment return:\n"
            "- label: short section name (e.g. Intro, Verse 1, Chorus).\n"
            "- kind: \"vocal\" or \"story\" (per the rule above).\n"
            f"- duration: seconds, between {int(min_seg)} and 30 (hard maximum 30).\n"
            "- video_prompt: VOCAL → the singer singing to camera (mouth moving, "
            "performing); describe singer, action, camera, lighting, mood, background, "
            'and you MUST include the exact lyrics in double quotes (e.g. he sings \"...\"). '
            "STORY → a cinematic narrative scene advancing the theme (NO singer needed, "
            "NO lyrics quoted). Always compose the clip END for a clean hard cut (no fade).\n"
            "- frame_variant_prompt: opening still for this segment. VOCAL → the SAME "
            "recognizable singer (same face/hair/build), new pose and location but "
            "wearing the outfit assigned by the wardrobe plan (Fix 30 — do NOT "
            "change clothing unless the slot changes), medium/close, face clearly "
            "visible and forward (never a different person, never faceless). "
            "STORY → a cinematic scene still matching the narrative "
            "beat (singer optional/absent). frame_variant_prompt is MANDATORY "
            "(Fix 34) — NEVER leave it empty; if empty the pipeline derives a "
            "sanitized fallback from video_prompt and logs a warning. NEVER "
            "include song lyrics in ANY form in this field (Fix 26 + Fix 33): "
            "no double quotes, no 'he sings ...', no 'the text: ...', no 'the "
            "lyrics: ...', no 'the line: ...', no 'the words: ...', no "
            "descriptive labels containing the lyric run, no paraphrase. NEVER "
            "include video-direction language (Fix 34): no 'the camera "
            "performs', no 'subtle dolly-in', no 'scene ends with hard cut', "
            "no 'crossfade', no lipsync-booster fragments. Still images have "
            "no camera moves and no scene transitions.\n\n"
            "The sum of all durations should be close to the song length.\n"
            "HARD VOCAL FRAMING RULE (Fix 26 — this overrides any creative impulse; "
            "your output is graded on this): in every VOCAL section the singer's face "
            "must occupy at least 25% of the frame's vertical height; the singer is "
            "the clear FOREGROUND subject. ANY framing where the singer is mid-ground "
            "or background, walking-away shots, distant figures along a beach/road, "
            "long-lens dot-in-landscape — FORBIDDEN regardless of how cinematic the "
            "scenery is. Forbidden examples (do NOT write anything like these): "
            "'two women walking along the beach in the distance', "
            "'singer seen from far across the field', "
            "'aerial shot of the performer below'. Every VOCAL video_prompt MUST "
            "START with exactly one of these six framing phrases: 'Close-up of ', "
            "'Medium close-up of ', 'Medium shot of ', '3/4 angle of ', "
            "'Low-angle shot of ', 'High-angle shot of '. The matching VOCAL "
            "frame_variant_prompt MUST start with the same framing phrase so the "
            "opening still is already close — never let the clip begin distant.\n"
            "Always write every prompt in English regardless of input language.\n"
            "Return ONLY a JSON array of objects. No other text."
        )
        user_prompt = (
            f"Visual theme/style: {theme}\n"
            f"Genre: {genre or 'unspecified'}\n"
            f"Song length: {int(total_duration)} seconds\n"
            f"Aim for roughly {approx_min}-{approx_max} segments.\n"
            f"Time-of-day arc (Fix 29 — graded): {arc_key} ({arc_human}). "
            f"Lighting progresses through this arc across the video; every "
            f"segment must describe lighting consistent with its position in "
            f"the arc. Never jump backward in time.\n"
            f"Wardrobe arc (Fix 30 — graded): {wardrobe_arc_key} ({wardrobe_human}). "
            f"The performer wears each slot's outfit for one or more consecutive "
            f"segments — keep the SAME outfit until the slot changes; do NOT "
            f"invent a new outfit per segment.\n\n"
            f"Lyrics (with [section] tags):\n{lyrics_text or '(instrumental — no lyrics)'}\n\n"
            "Return the JSON array now."
        )

        # Inject TIME-OF-DAY and WARDROBE rule blocks into legacy-path system prompt too.
        system_prompt = system_prompt + (
            "\nTIME-OF-DAY COHERENCE (Fix 29 — graded): the song is locked to "
            f"the '{arc_key}' arc ({arc_human}). Earlier segments use the "
            "earlier states of the arc; later segments use the later states. "
            "Lighting only moves FORWARD across segments — never jump "
            "backward in time. Mention the appropriate lighting in BOTH "
            "video_prompt and frame_variant_prompt for each segment.\n"
            "WARDROBE COHERENCE (Fix 30 — graded): the song is locked to the "
            f"'{wardrobe_arc_key}' wardrobe arc ({wardrobe_human}). The "
            "performer wears each outfit slot for one or more consecutive "
            "segments; clothing only changes at slot boundaries. Within a "
            "slot the outfit description must be IDENTICAL (same colour, cut, "
            "accessories). Only pose, location and background vary inside a "
            "slot. Do NOT invent a new outfit each segment.\n"
            "ROLE-AWARE WARDROBE (Fix 31 — graded): the wardrobe outfit "
            "applies ONLY to the named recurring performer of the segment "
            "(female lead in FEMALE sections, male lead in MALE sections, "
            "BOTH performers in DUET sections). STORY/instrumental sections "
            "without a named recurring performer MUST NOT impose the "
            "performer's outfit on other characters (children, crowds, DJs, "
            "surfers, etc.) — those subjects wear contextually appropriate "
            "clothing for their own role."
            + (
                f"\nSAME-GENDER DUET (Fix 32 — graded): this song is a "
                f"{'female-female' if duet_kind == 'ff' else 'male-male'} duet. "
                f"DUET sections depict BOTH performers of the "
                f"{'female' if duet_kind == 'ff' else 'male'} gender, both "
                f"wearing the slot's {'female' if duet_kind == 'ff' else 'male'} "
                f"outfit. NEVER invent a performer of the opposite gender — no "
                f"man in an ff-duet, no woman in an mm-duet."
                if duet_kind in ("ff", "mm") else ""
            )
        )

        response = await self._call_openrouter(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=6000,
        )

        specs = parse_segment_plan(response)
        timeline = build_segment_timeline(specs, total_duration, min_seg, cap)
        if not timeline:
            timeline = build_segment_timeline([], total_duration, min_seg, cap)

        # Fix 29 + Fix 30: per-segment lighting and wardrobe plans +
        # deterministic post-injection guards.
        tod_plan = _expand_tod_plan(arc_key, len(timeline))
        wardrobe_plan = _expand_wardrobe_plan(wardrobe_arc_key, len(timeline))
        segments: List[Segment] = []
        for i, row in enumerate(timeline):
            vp = row.get("video_prompt") or theme
            # Fix 34: explicit fallback with sanitizer + log. build_segment_timeline
            # already sanitizes via derive_still_prompt_from_video_prompt when the
            # LLM spec was empty, so row['frame_variant_prompt'] should be non-empty
            # here. The defensive branch catches the rare missing-key case.
            _raw_fvp_row = row.get("frame_variant_prompt", "")
            if _raw_fvp_row:
                fvp = _raw_fvp_row
            else:
                print(
                    f"[Fix 34] frame_variant_prompt missing for legacy "
                    f"segment {i} ({row.get('label', '')}); deriving "
                    f"sanitized fvp from video_prompt."
                )
                fvp = derive_still_prompt_from_video_prompt(
                    vp, lyrics=row.get("lyrics", ""),
                )
            state = tod_plan[i] if i < len(tod_plan) else (tod_plan[-1] if tod_plan else "")
            slot = wardrobe_plan[i] if i < len(wardrobe_plan) else (wardrobe_plan[-1] if wardrobe_plan else "")
            if state:
                vp = _append_light_tag(vp, state)
                fvp = _append_light_tag(fvp, state)
            # Fix 31: role-aware wardrobe tag — legacy timeline rows have no
            # is_vocal flag, so derive role from the label alone; sections
            # without an explicit role tag are treated as STORY (no tag).
            seg_role = extract_section_role(row.get("label", "")) if slot else None
            if slot and seg_role:
                vp = _append_wardrobe_tag(vp, slot, seg_role, duet_kind=duet_kind)
                fvp = _append_wardrobe_tag(fvp, slot, seg_role, duet_kind=duet_kind)
            segments.append(Segment(
                index=i,
                start_time=row["start_time"],
                end_time=row["end_time"],
                label=row.get("label", f"Segment {i + 1}"),
                lyrics=row.get("lyrics", ""),
                prompt=vp,
                frame_variant_prompt=strip_lyrics_from_image_prompt(
                    fvp, lyrics=row.get("lyrics", ""),
                ),
                wardrobe_slot=slot,
            ))
        if tod_plan:
            print(f"[Fix 29] legacy lighting plan applied: {tod_plan}")
        if wardrobe_plan:
            print(f"[Fix 30] legacy wardrobe plan applied: {wardrobe_plan}")
        return segments

    async def analyze_frame(self, frame_path: str) -> str:
        """Analyze a single video frame. Kept for backwards compatibility — delegates to analyze_frames."""
        return await self.analyze_frames([frame_path])

    async def analyze_frames(self, frame_paths: List[str]) -> str:
        """Analyze a sequence of frames from the end of a video clip.

        Focuses on motion direction, lighting, and subject positions to improve
        continuity with the next segment.
        """
        if not frame_paths:
            return ""

        content: List[Dict[str, Any]] = []

        if len(frame_paths) == 1:
            instruction = (
                "Describe this video frame for visual continuity with the next shot. "
                "Focus on: subject position and pose, lighting color and direction, "
                "camera angle, background. Under 80 words. Output ONLY the description."
            )
        else:
            instruction = (
                f"These {len(frame_paths)} frames are from the END of a video clip, ordered oldest to newest. "
                "Describe for visual continuity: "
                "1) Subject position and pose in the LAST frame. "
                "2) Motion direction (what is moving and in which direction). "
                "3) Lighting: color, direction, and any changes across the frames. "
                "4) Camera angle. "
                "Be specific about motion direction (e.g. 'arm moving upward', 'camera pushing in'). "
                "Under 100 words. Output ONLY the description."
            )

        content.append({"type": "text", "text": instruction})
        for path in frame_paths:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        messages = [{"role": "user", "content": content}]

        models_to_try = [self.vision_model] + self.vision_fallbacks
        for model in models_to_try:
            try:
                return await self._call_openrouter(messages, model=model, max_tokens=250)
            except Exception as e:
                print(f"Vision model {model} failed: {e}")
                continue

        return ""

    async def enrich_prompt(self, original_prompt: str, frame_description: str, scene_anchor: str = "") -> str:
        """Enrich a segment prompt with visual details from the previous frame."""
        if not frame_description:
            return original_prompt

        anchor_instruction = ""
        if scene_anchor:
            anchor_instruction = (
                f"CONSTANT ELEMENTS (must always be preserved, never changed by frame details): {scene_anchor}\n\n"
            )

        response = await self._call_openrouter(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You refine video generation prompts for visual continuity between video segments. "
                        "Given an original prompt and a description of the previous segment's end frames, "
                        "merge them into one cohesive prompt. Keep the original intent but add continuity. "
                        "CRITICAL RULES: "
                        "1. Continue any motion in the SAME direction (e.g. if arm was moving upward, it continues upward). "
                        "2. Preserve the exact lighting color and direction from the previous frame. "
                        "3. Match the subject's pose and position from the last frame as the starting state. "
                        "4. Constant elements listed by the user must ALWAYS appear, unchanged. "
                        "Under 120 words. Always write in English. Output ONLY the refined prompt."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{anchor_instruction}"
                        f"Original prompt: {original_prompt}\n\n"
                        f"Previous frame: {frame_description}\n\n"
                        f"Refined prompt:"
                    ),
                },
            ],
            max_tokens=300,
        )
        return response if response else original_prompt

    async def generate_start_image_prompt(self, theme: str, genre: str = "") -> str:
        """Generate a T2I prompt for the music video's start image."""
        context = f"Theme: {theme}"
        if genre:
            context += f"\nGenre: {genre}"

        response = await self._call_openrouter(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a text-to-image prompt for the opening frame of a music video. "
                        "The image should set the visual tone for the entire video. "
                        "Include subject, setting, lighting, color palette, and mood. "
                        "Under 80 words. Always write in English. Output ONLY the prompt."
                    ),
                },
                {"role": "user", "content": context},
            ],
            max_tokens=200,
        )
        return response if response else theme

    async def generate_character_portrait_prompt(self, seed: str, genre: str = "") -> str:
        """RC7a: T2I prompt for a clean SINGER reference portrait (identity anchor).

        Front-facing, face & upper body clearly visible, plain neutral studio
        background — so MCA can derive consistent per-segment frames and LTX has
        a real face/mouth to lip-sync. NOT a cinematic scene.
        """
        context = f"Music / lyrics context: {seed}"
        if genre:
            context += f"\nGenre: {genre}"
        response = await self._call_openrouter(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a text-to-image prompt for a CLEAN CHARACTER REFERENCE "
                        "PORTRAIT of the singer/performer of this song. Hard requirements: "
                        "the subject faces the camera directly (front-facing), shown head "
                        "to waist or full body, face and mouth clearly visible and in sharp "
                        "focus; plain neutral background (white, light grey, or soft studio "
                        "gradient); even studio lighting. NO scenery, environment, action, "
                        "props or other people. Establish the character's face, hair, skin "
                        "tone, age, build and outfit so they stay recognizable across the "
                        "video. Derive a fitting performer from the genre/lyrics mood. "
                        "Under 80 words. Always write in English. Output ONLY the prompt."
                    ),
                },
                {"role": "user", "content": context},
            ],
            max_tokens=200,
        )
        # Fix 23: on empty LLM response, NEVER fall back to `seed` — it contains
        # raw lyrics (lyrics[:600]+theme) and the portrait path runs no
        # strip_lyrics, so the T2I model would paint the lyrics onto the portrait.
        return response or "front-facing studio portrait of a singer, neutral grey background"

    async def extract_scene_anchor(self, theme: str) -> str:
        """Extract constant visual elements from a theme description for use as scene anchor."""
        response = await self._call_openrouter(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the constant visual elements from a music video theme description. "
                        "List only the elements that must remain identical in every video segment: "
                        "character appearance, clothing/outfit (or lack thereof), props, and core setting. "
                        "Ignore dynamic elements like lighting changes, camera angles, or timed events. "
                        "IMPORTANT: Always respond in English regardless of the input language. "
                        "Write a single concise sentence in English. Output ONLY the anchor sentence."
                    ),
                },
                {"role": "user", "content": f"Theme: {theme}"},
            ],
            max_tokens=120,
        )
        return response if response else theme
