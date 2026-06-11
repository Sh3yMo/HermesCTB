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

MSR_VIEW_PROMPTS: List[str] = [
    f"full body shot, standing, facing the camera directly, frontal view, {_VIEW_COMMON}",
    f"full body shot, standing, seen from behind, back view showing the back of the head and outfit, {_VIEW_COMMON}",
    f"full body shot, standing, strict side profile view facing left, {_VIEW_COMMON}",
    f"close-up portrait of the face, head and shoulders, facing the camera, {_VIEW_COMMON}",
]

MSR_VIEW_LABELS: List[str] = ["front", "back", "side", "face"]


# ---------------------------------------------------------------------------
# Character sheet composition (msr_ref_mode="sheet")
# ---------------------------------------------------------------------------

def compose_character_sheet(
    view_paths: List[str],
    out_path: str,
    cell_size: int = 768,
    border: int = 16,
) -> str:
    """Compose up to 4 view stills into one 2x2 character-sheet grid.

    Views are letterboxed (aspect preserved) into square cells on a white
    canvas with a white separating border — one single reference image that
    occupies one MSR subject slot. Returns out_path.
    """
    if not view_paths:
        raise ValueError("compose_character_sheet needs at least one view image")
    paths = view_paths[:4]
    cols = 1 if len(paths) == 1 else 2
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

def derive_background_prompt(segment_prompt: str) -> str:
    """Derive a people-free scene prompt from a segment's still/video prompt.

    Deterministic, no LLM round-trip: prefix an exclusion instruction so the
    T2I model renders the location only. Good enough as a fallback — the LLM
    path emits a proper background_prompt per section.
    """
    base = (segment_prompt or "").strip()
    if not base:
        return "empty atmospheric stage with soft lighting, no people"
    return (
        "empty location scene, no people, no characters, no faces, "
        f"scenery only: {base}"
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
