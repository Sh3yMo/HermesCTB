"""MSR (Multiple Subject Reference) reference-asset helpers.

Builds the reference images consumed by the "LTX2.3 - IA2V-PromptRelay-MSR"
workflow's LiconMSR node: per-role character sheets (or raw multi-view lists),
per-section background stills, and the compact reference description block the
MSR LoRA needs in the prompt (model card: concise but accurate descriptions of
the references; over- or under-description degrades consistency).

Slot contract (LiconMSR): subjects occupy slots 1-4, slot 5 is the mandatory
background. Subject order = characters first, then props.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# Multi-view prompts (run through the F2K9B MCA workflow, portrait as source)
# ---------------------------------------------------------------------------

# Stage M closed-mouth default applies: stills must never show an open mouth,
# otherwise IA2V lip-sync starts mid-phoneme ("stroke effect").
_VIEW_COMMON = (
    "same person, identical face, identical hairstyle, identical outfit, "
    "mouth closed, relaxed neutral expression, plain neutral grey studio "
    "background, even soft studio lighting"
)

# Stage O3: the full-body FRONT view comes straight from the T2I portrait
# (the portrait IS a full-body frontal now), so MCA only renders the four
# missing views. The face appears front AND side for stronger identity
# anchoring; the full-body side view keeps the 3D body-shape information
# (build/silhouette) that front+back alone cannot convey.
MSR_VIEW_PROMPTS: List[str] = [
    f"full body shot, standing, seen from behind, back view showing the back of the head and outfit, {_VIEW_COMMON}",
    f"full body shot, standing, strict side profile view facing left, {_VIEW_COMMON}",
    f"close-up portrait of the face, head and shoulders, facing the camera, {_VIEW_COMMON}",
    f"close-up portrait of the face in strict side profile view, head and shoulders, {_VIEW_COMMON}",
]

# Labels align with the sheet/refs ordering: [portrait] + MCA views.
MSR_VIEW_LABELS: List[str] = [
    "full body front", "back", "side", "face front", "face side",
]


# ---------------------------------------------------------------------------
# Character sheet composition (msr_ref_mode="sheet")
# ---------------------------------------------------------------------------

def compose_character_sheet(
    view_paths: List[str],
    out_path: str,
    cell_size: int = 768,
    border: int = 16,
) -> str:
    """Compose up to 6 view stills into one character-sheet grid.

    1 view -> 1x1, 2-4 views -> 2 columns, 5-6 views -> 3 columns (Stage O3:
    full-body front + back + side + face front + face side = 5 cells).
    Views are letterboxed (aspect preserved) into square cells on a white
    canvas with a white separating border — one single reference image that
    occupies one MSR subject slot. Returns out_path.
    """
    if not view_paths:
        raise ValueError("compose_character_sheet needs at least one view image")
    paths = view_paths[:6]
    cols = 1 if len(paths) == 1 else (3 if len(paths) >= 5 else 2)
    rows = (len(paths) + cols - 1) // cols
    width = cols * cell_size + (cols + 1) * border
    height = rows * cell_size + (rows + 1) * border
    sheet = Image.new("RGB", (width, height), "white")
    for idx, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        img.thumbnail((cell_size, cell_size), Image.LANCZOS)
        col = idx % cols
        row = idx // cols
        x = border + col * (cell_size + border) + (cell_size - img.width) // 2
        y = border + row * (cell_size + border) + (cell_size - img.height) // 2
        sheet.paste(img, (x, y))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path, "PNG")
    return out_path


# ---------------------------------------------------------------------------
# Background fallback prompt (when the LLM omitted background_prompt)
# ---------------------------------------------------------------------------

# Stage O1: the old fallback concatenated "scenery only:" with the FULL
# segment prompt. Segment prompts describe the character, and that detailed
# description overrides the soft "no people" prefix — Flux rendered a person
# into every "background" (job ffd3b9a6), which then bled wardrobe/tattoos
# into the MSR composite. The fallback now NEVER copies segment text; it only
# mines coarse scene/lighting keywords and maps them to neutral phrases.

_BG_HINTS: List[Tuple[str, str]] = [
    (r"neon", "neon signs reflecting on wet surfaces"),
    (r"rain|wet ", "rain-slick ground with soft reflections"),
    (r"moonlight|moonlit|moon", "cool moonlight under a deep dark sky"),
    (r"night|midnight|dark sky", "night exterior with sparse practical lights"),
    (r"sunset|golden hour|dusk", "warm golden-hour sky"),
    (r"sunrise|dawn", "soft dawn light on the horizon"),
    (r"desert", "open desert landscape"),
    (r"beach|ocean|shore|sea", "empty shoreline with distant waves"),
    (r"rooftop", "empty rooftop overlooking the city"),
    (r"warehouse|industrial", "empty industrial warehouse interior"),
    (r"stage|concert|club", "empty concert stage with haze and rig lights"),
    (r"forest|woods", "misty forest clearing"),
    (r"studio", "empty studio space with seamless backdrop"),
    (r"city|urban|street|alley|downtown", "empty urban street"),
]

_BG_DEFAULT_SCENE = (
    "atmospheric empty urban exterior at night, soft practical lighting"
)

# Negated exclusions ("no people") must not trip the person guard.
_BG_NEGATED_RE = re.compile(
    r"\bno\s+(?:people|persons?|characters?|faces?|humans?)\b[,;]?\s*",
    re.IGNORECASE,
)
_BG_PERSON_RE = re.compile(
    r"\b(?:person|people|human|man|men|woman|women|male|female|boy|girl|guy|"
    r"lady|singer|vocalist|performer|artist|dancer|musician|character|crowd|"
    r"band|face|faces|portrait|wearing|outfit|wardrobe|hoodie|jacket|dress|"
    r"shirt|jeans|joggers|shorts|sneakers|heels|sunglasses|tattoo|tattoos|"
    r"hair|hairstyle|she|he|her|his)\b",
    re.IGNORECASE,
)


def background_prompt_mentions_people(bg_prompt: str) -> bool:
    """True when a background prompt contains person/wardrobe vocabulary.

    Defense-in-depth guard before the background T2I: a prompt that mentions
    a person produces a person, which the MSR background slot must never
    contain. Negated phrases ("no people") are stripped before matching.
    """
    text = _BG_NEGATED_RE.sub("", bg_prompt or "")
    return bool(_BG_PERSON_RE.search(text))


def derive_background_prompt(segment_prompt: str) -> str:
    """Build a people-free scene prompt WITHOUT copying segment text.

    Deterministic, no LLM round-trip: only coarse scene/lighting keywords are
    mined from the segment prompt and mapped to neutral scenery phrases (max
    3). The segment text itself never reaches the T2I model.
    """
    text = (segment_prompt or "").lower()
    hints = [phrase for pat, phrase in _BG_HINTS if re.search(pat, text)]
    scene = ", ".join(hints[:3]) if hints else _BG_DEFAULT_SCENE
    return (
        "empty location scene, no people, no characters, no faces, "
        f"scenery only: {scene}"
    )


# ---------------------------------------------------------------------------
# Slot allocation + reference description block
# ---------------------------------------------------------------------------

def allocate_msr_subjects(
    role: Optional[str],
    char_refs: Dict[str, List[str]],
    char_descs: Dict[str, str],
    prop_refs: Optional[List[Tuple[str, str]]] = None,
    max_subjects: int = 4,
) -> Tuple[List[str], List[str]]:
    """Pick subject slot images + matching short descriptions for a segment.

    `char_refs` maps role -> reference image list (one sheet, or up to 4
    views). `char_descs` maps role -> one-line character description.
    `prop_refs` is a list of (image_path, short_description) tuples filling
    the remaining slots. Duet segments get BOTH characters' refs; when both
    use multi-view lists that would overflow 4 slots, each character is
    reduced to its first (front) view.

    Returns (subject_paths, subject_descs), both <= max_subjects entries.
    """
    paths: List[str] = []
    descs: List[str] = []

    def _add(imgs: List[str], desc: str) -> None:
        for k, img in enumerate(imgs):
            if len(paths) >= max_subjects:
                return
            paths.append(img)
            if len(imgs) > 1:
                descs.append(f"{desc} ({MSR_VIEW_LABELS[k] if k < len(MSR_VIEW_LABELS) else 'view'} view)")
            else:
                descs.append(desc)

    if role == "duet":
        # Mixed duet -> male+female; same-gender duet (Fix 27) -> base+"<base>2".
        members = [r for r in ("male", "male2", "female", "female2") if r in char_refs]
        if not members:
            members = list(char_refs.keys())
        members = members[:2]
        overflow = sum(len(char_refs[m]) for m in members) > max_subjects
        for m in members:
            imgs = char_refs[m]
            if overflow and len(imgs) > 1:
                imgs = imgs[:1]  # front view only — two multi-view sets don't fit
            _add(imgs, char_descs.get(m, "the performer"))
    elif role in char_refs:
        _add(char_refs[role], char_descs.get(role, "the performer"))
    elif char_refs:
        # Story/unknown role: anchor to the first available character so the
        # recurring performer stays consistent in narrative segments too.
        first = next(iter(char_refs))
        _add(char_refs[first], char_descs.get(first, "the performer"))

    for img, desc in (prop_refs or []):
        if len(paths) >= max_subjects:
            break
        paths.append(img)
        descs.append(desc)

    return paths, descs


def build_msr_reference_block(subject_descs: List[str], background_desc: str) -> str:
    """Compact reference description appended to the (global) prompt.

    The MSR model card requires concise, accurate descriptions of the
    reference images; numbering matches the LiconMSR slot order.
    """
    if not subject_descs:
        return ""
    lines = ["References:"]
    for i, d in enumerate(subject_descs, start=1):
        lines.append(f"[{i}] {d}.")
    if background_desc:
        lines.append(f"Background reference: {background_desc.rstrip('.')}.")
    return " ".join(lines)
