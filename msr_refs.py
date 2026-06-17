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

from PIL import Image, ImageOps

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

# Stage O3 / Stage Q: the full-body FRONT view comes straight from the T2I
# portrait (the portrait IS a full-body frontal now), so MCA only renders the
# three missing views. Portrait + back + side + face-front = 4 cells -> a
# seamless 2x2 sheet. (The face-side close-up was dropped so all four cells are
# equal-size portrait tiles that join into one rectangle.)
MSR_VIEW_PROMPTS: List[str] = [
    f"full body shot, standing, seen from behind, back view showing the back of the head and outfit, {_VIEW_COMMON}",
    f"full body shot, standing, strict side profile view facing left, {_VIEW_COMMON}",
    f"close-up portrait of the face, head and shoulders, facing the camera, {_VIEW_COMMON}",
]

# Labels align with the sheet/refs ordering: [portrait] + MCA views.
MSR_VIEW_LABELS: List[str] = [
    "full body front", "back", "side", "face front",
]


def build_view_prompts(style_descriptor: str = "", appearance_desc: str = "") -> List[str]:
    """The 3 MCA reference-view prompts, pinned to a concrete appearance + medium.

    The static MSR_VIEW_PROMPTS only say "identical hairstyle, identical outfit"
    — too vague, so the model invents per-angle details that disagree (phantom
    bun in the side view, recoloured garment). `appearance_desc` (the portrait's
    own concrete appearance text) and `style_descriptor` (the producer visual
    medium) are injected so all reference cells depict the SAME person in the
    SAME look. Both empty -> identical to the static prompts (back-compat).
    """
    extras: List[str] = []
    if appearance_desc:
        extras.append(
            "the SAME person with this exact appearance, do not alter or invent "
            "any detail: " + appearance_desc.strip().rstrip(".")
        )
    if style_descriptor:
        extras.append(
            "rendered in this exact visual medium: "
            + style_descriptor.strip().rstrip(".")
        )
    common = _VIEW_COMMON + ((", " + ", ".join(extras)) if extras else "")
    return [
        f"full body shot, standing, seen from behind, back view showing the back of the head and outfit, {common}",
        f"full body shot, standing, strict side profile view facing left, {common}",
        f"close-up portrait of the face, head and shoulders, facing the camera, {common}",
    ]


# ---------------------------------------------------------------------------
# Character sheet composition (msr_ref_mode="sheet")
# ---------------------------------------------------------------------------

def compose_character_sheet(
    view_paths: List[str],
    out_path: str,
    cell_w: int = 512,
    cell_h: int = 768,
    **_legacy,
) -> str:
    """Compose up to 4 portrait view stills into one seamless 2x2 sheet.

    Stage Q: the sheet is exactly 4 equal-size 2:3 portrait cells
    (portrait + back + side + face-front) joined edge-to-edge into a single
    2:3 rectangle — NO white border, NO letterbox. Each view is cover-cropped
    to the cell (`ImageOps.fit`, aspect preserved, centred, edges trimmed) so a
    landscape source (e.g. the legacy `lead` source frame) is forced to
    portrait too — fixing the "first cell landscape, rest portrait" mismatch.

    Fewer than 4 views still tile into a 2-column grid (trailing cells black).
    `cell_w`/`cell_h` set the per-cell pixel size (default 512x768 = 2:3).
    Legacy `cell_size`/`border` kwargs are accepted and ignored (back-compat).
    Returns out_path.
    """
    if not view_paths:
        raise ValueError("compose_character_sheet needs at least one view image")
    paths = view_paths[:4]
    cols = 1 if len(paths) == 1 else 2
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "black")
    for idx, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        img = ImageOps.fit(img, (cell_w, cell_h), Image.LANCZOS, centering=(0.5, 0.5))
        col = idx % cols
        row = idx // cols
        sheet.paste(img, (col * cell_w, row * cell_h))
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
    (r"beach bar|tiki bar|shore bar", "empty beach bar with warm practical lights"),
    (r"beach|ocean|shore|sea", "empty shoreline with distant waves"),
    (r"jungle|rainforest|tropical forest", "dense tropical jungle vegetation"),
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
    elif role:
        if char_refs:
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
    ordinals = ["first", "second", "third", "fourth"]
    lines = [
        "Use the subject references only for identity, body, face, hair and locked clothing; "
        "do not use their plain studio backgrounds as the scene."
    ]
    for i, d in enumerate(subject_descs, start=1):
        label = ordinals[i - 1] if i <= len(ordinals) else f"subject {i}"
        lines.append(f"Use the {label} subject reference as {d.rstrip('.')}.")
    if background_desc:
        lines.append(
            f"Use the background reference as the real location: {background_desc.rstrip('.')}. "
            "The performer should look physically present there, with natural contact "
            "shadows and matching light."
        )
    return " ".join(lines)
