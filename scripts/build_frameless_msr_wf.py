"""Build the frameless MSR video workflow.

MSR is incompatible with an I2V start frame: the reference grid bleeds into the
first frames (drift / character swap). The fix is to run MSR *frameless* — no
start-frame input; the MSR guide + audio drive the clip (T2V/TA2V character).

Rather than hand-deleting the start-frame subgraph from the IA2V-MSR workflow
(error-prone), we re-derive the graph cleanly:

    frameless-MSR = TA2V backbone  +  (PromptRelay + MSR) delta

The delta is computed as the difference between "LTX2.3 - IA2V.json" (legacy,
no relay, no MSR) and "LTX2.3 - IA2V-PromptRelay-MSR.json": that difference is
EXACTLY the PromptRelay + MSR additions/rewrites and contains NO start-frame
wiring (both IA2V files share the same start-frame chain, so it cancels out).
Applying that delta onto TA2V — which already has the correct frameless
empty-latent wiring + audio — yields a frameless MSR graph: the MSR guide
(node 2008) takes its latent base from TA2V's empty latent (759:1360) instead
of a start frame.

A reference guard skips any transplanted wiring that points at a node absent
from the final graph (defensive — should never trigger for genuine PR/MSR
changes). The result is validated (no dangling refs, MSR + relay nodes present,
no "First Frame" LoadImage) before it overwrites the workflow (old file backed
up to .bak3).

Run:  py -3 scripts/build_frameless_msr_wf.py [--dry]
"""
import copy
import json
import sys
from pathlib import Path

WF_DIR = Path(__file__).resolve().parent.parent / "Workflows"
F_TA2V = WF_DIR / "LTX2.3 - TA2V.json"
F_IA2V = WF_DIR / "LTX2.3 - IA2V.json"
OUT = WF_DIR / "LTX2.3 - IA2V-PromptRelay-MSR.json"  # target
# Source of the PR+MSR delta = the ORIGINAL (start-frame) MSR workflow. After
# the first build OUT is the frameless result, so always diff against the
# pristine backup, never against OUT itself.
F_MSR = WF_DIR / "LTX2.3 - IA2V-PromptRelay-MSR.json.bak3"


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _refs(inputs: dict):
    """Yield (key, src_node_id) for every [node_id, slot] link in an inputs dict."""
    for k, v in (inputs or {}).items():
        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)):
            yield k, str(v[0])


def build() -> dict:
    ta2v = _load(F_TA2V)
    ia2v = _load(F_IA2V)
    msr = _load(F_MSR)

    final = copy.deepcopy(ta2v)

    # 1) New nodes introduced by PromptRelay + MSR (present in msr, absent in ia2v).
    added = [nid for nid in msr if nid not in ia2v]
    for nid in added:
        final[nid] = copy.deepcopy(msr[nid])
    print(f"[build] new nodes added: {sorted(added)}")

    final_ids = set(final)  # base TA2V nodes + added PR/MSR nodes

    # 2) Input rewrites caused by PR + MSR: nodes shared by ia2v & msr whose
    #    inputs changed. (Start-frame wiring is identical in both IA2V files, so
    #    it never appears here.) Apply onto the matching TA2V node — guarded so
    #    we never import a link to a node that doesn't exist in the final graph.
    rewired, skipped = [], []
    for nid in msr:
        if nid in added or nid not in ia2v or nid not in final:
            continue
        if msr[nid] == ia2v[nid]:
            continue  # unchanged by PR/MSR
        # Copy the WHOLE node from msr (class_type + inputs + _meta): PR/MSR may
        # change a node's CLASS, not just its links — e.g. the sampler scheduler
        # is ManualSigmas in T2V/TA2V but LTXVScheduler / Sigmas Rescale in MSR.
        # Copying inputs alone would leave a class/schema mismatch that ComfyUI
        # prunes (sampler chain dropped -> no video output).
        new_node = copy.deepcopy(msr[nid])
        bad = [src for _k, src in _refs(new_node.get("inputs", {})) if src not in final_ids]
        if bad:
            skipped.append((nid, bad))
            continue
        final[nid] = new_node
        rewired.append(nid)
    print(f"[build] inputs rewired on shared nodes: {sorted(rewired)}")
    if skipped:
        print(f"[build] WARNING skipped (ref to absent node): {skipped}")

    # 3) reasoning_I2V LoRA für MSR-Renders aktivieren. Der Referenz-MSR-WF
    #    (distill-lora-API) fährt diese LoRA an @0.5 — sie stabilisiert die
    #    Referenz-Adhärenz. TA2V liefert den Power-Lora-Loader mit der LoRA
    #    'off' aus; für MSR-Identität schalten wir sie hier scoped ein (die
    #    TA2V-Quelle bleibt unberührt -> keine Nebenwirkung auf Nicht-MSR).
    enabled = []
    for nid, node in final.items():
        if node.get("class_type") != "Power Lora Loader (rgthree)":
            continue
        for slot, v in node.get("inputs", {}).items():
            if isinstance(v, dict) and "reasoning_I2V" in str(v.get("lora", "")):
                v["on"] = True
                v["strength"] = 0.5
                enabled.append(f"{nid}.{slot}")
    print(f"[build] reasoning_I2V LoRA enabled @0.5 on: {enabled or 'NONE (!)'} ")

    # 4) Sheet-Planting ins Basis-Latent (LTXVAddGuideMulti) — Referenz-Parität.
    #    Der IC-LoRA-Guide (2008) liefert NUR Cross-Attn-Konditionierung; ohne ein
    #    Bild im Basis-Latent dominiert der Prompt und die Identität driftet komplett
    #    (fremde Person). Der distill-lora-Referenz-WF (Node 85) pflanzt das Sheet
    #    (+Background) bei Frame 0 strength 1 in das leere Basis-Latent — exakt das
    #    replizieren wir. Basis-Stufe (759:1054) liest dann aus dem geplanteten Latent
    #    statt aus 2008:2. Slots: MSR-Subject-1 = 2001 (Sheet), MSR-Background = 2005.
    PLANT_ID = "2009"
    final[PLANT_ID] = {
        "class_type": "LTXVAddGuideMulti",
        "_meta": {"title": "MSR Sheet Guide-Planting [[P:MSR]]"},
        "inputs": {
            "num_guides": "2",
            "num_guides.frame_idx_1": 0, "num_guides.strength_1": 1,
            "num_guides.frame_idx_2": 0, "num_guides.strength_2": 1,
            "positive": ["759:1067", 0],
            "negative": ["759:1067", 1],
            "vae": ["174", 0],
            "latent": ["759:1360", 0],
            "num_guides.image_1": ["2001", 0],
            "num_guides.image_2": ["2005", 0],
        },
    }
    final["759:1055"]["inputs"]["video_latent"] = [PLANT_ID, 2]
    print(f"[build] sheet guide-planting node {PLANT_ID} (LTXVAddGuideMulti) -> base latent 759:1055")

    return final


def validate(wf: dict) -> list[str]:
    errs: list[str] = []
    ids = set(wf)
    cls = {n.get("class_type") for n in wf.values() if isinstance(n, dict)}

    # a) every link resolves
    for nid, node in wf.items():
        for k, src in _refs(node.get("inputs", {})):
            if src not in ids:
                errs.append(f"node {nid}.{k} -> missing node {src}")

    # b) required feature nodes present
    if "PromptRelaySmartEncode" not in cls:
        errs.append("missing PromptRelaySmartEncode (relay)")
    if "LiconMSR" not in cls:
        errs.append("missing LiconMSR (MSR)")
    if "LTXAddVideoICLoRAGuide" not in cls:
        errs.append("missing LTXAddVideoICLoRAGuide (MSR guide)")
    if "LoadAudio" not in cls:
        errs.append("missing LoadAudio (TA2V audio)")

    # c) NO start frame: every LoadImage must be an MSR reference slot
    #    ([[P:MSR]]). Any other LoadImage = a leftover start-frame input.
    #    (TA2V keeps the inplace 'image' wired to an EMPTY Any-Switch — that is
    #    its native frameless pattern, not a start frame, so we don't flag it.)
    for nid, node in wf.items():
        if node.get("class_type") == "LoadImage":
            title = node.get("_meta", {}).get("title", "")
            if "[[P:MSR]]" not in title:
                errs.append(f"non-MSR LoadImage present (start-frame leftover): {nid} ({title!r})")

    # d) MSR guide must draw its latent from a node that exists (frameless = TA2V empty path)
    guide = next((n for n in wf.values() if n.get("class_type") == "LTXAddVideoICLoRAGuide"), None)
    if guide:
        lat = guide.get("inputs", {}).get("latent")
        if isinstance(lat, list) and str(lat[0]) not in ids:
            errs.append(f"MSR guide latent -> missing node {lat[0]}")
    return errs


def main() -> int:
    dry = "--dry" in sys.argv
    wf = build()
    errs = validate(wf)
    print(f"[build] final node count: {len(wf)}")
    if errs:
        print("[build] VALIDATION FAILED:")
        for e in errs:
            print("   -", e)
        return 1
    print("[build] validation OK (links resolve, relay+MSR+audio present, frameless)")

    if dry:
        tmp = OUT.with_suffix(".frameless.preview.json")
        tmp.write_text(json.dumps(wf, indent=2), encoding="utf-8")
        print(f"[build] DRY RUN -> wrote preview {tmp} (workflow NOT overwritten)")
        return 0

    bak = OUT.with_suffix(".json.bak3")
    if OUT.exists() and not bak.exists():
        bak.write_text(OUT.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[build] backup -> {bak}")
    OUT.write_text(json.dumps(wf, indent=2), encoding="utf-8")
    print(f"[build] wrote frameless MSR workflow -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
