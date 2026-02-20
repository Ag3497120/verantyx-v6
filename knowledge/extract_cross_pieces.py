#!/usr/bin/env python3
"""
Cross DB → Verantyx Pieces Extractor
Converts cross_db_full.json (SVD features of 15K experts) into piece_db entries.

Strategy:
  - Load cross_db.json (8.7GB, streams in ~2min with 720GB RAM available)
  - Group experts by layer depth (Z coordinate)
  - Find "specialized" experts via high top_singular_ratio + low sparsity
  - Map expert clusters to HLE domains
  - Output as pieces_600b.jsonl
"""

import json
import sys
import math
from pathlib import Path
from collections import defaultdict

# ── Config ─────────────────────────────────────────────
import os as _os
CROSS_DB_PATH  = _os.environ.get("CROSS_DB_PATH",  "/Users/motonishikoudai/Downloads/cross_db_full.json")
OUTPUT_PATH    = _os.environ.get("OUTPUT_PATH",    "/Users/motonishikoudai/Downloads/pieces_600b.jsonl")
NUM_LAYERS     = 61
TOP_K_PIECES   = 200   # how many pieces to write

# Map (depth_band, abstraction_band) → Verantyx domain
REGION_DOMAIN_MAP = {
    (0, 0): "math",          # early + concrete
    (0, 1): "logic",         # early + abstract
    (1, 0): "chemistry",     # mid + concrete
    (1, 1): "math",          # mid + abstract
    (2, 0): "physics",       # late + concrete
    (2, 1): "math",          # late + abstract
    (3, 0): "biology",       # final + concrete
    (3, 1): "math",          # final + abstract
}

DOMAIN_EXECUTOR = {
    "math":      "executors.knowledge.lookup",
    "logic":     "executors.logic.evaluate",
    "chemistry": "executors.knowledge.lookup",
    "physics":   "executors.knowledge.lookup",
    "biology":   "executors.knowledge.lookup",
}

# ── Helpers ─────────────────────────────────────────────

def depth_band(z):
    if z < 0.25: return 0
    if z < 0.50: return 1
    if z < 0.75: return 2
    return 3

def abstraction_band(x):
    return 1 if x > 0.5 else 0

def specialization_score(entry):
    """Higher = more 'focused' expert = more likely to represent specific knowledge."""
    feat = entry.get("features", {})
    tsr = feat.get("top_singular_ratio", 0.0)
    eff = feat.get("effective_rank", 1.0)
    spa = feat.get("sparsity", 0.0)
    ent = feat.get("spectral_entropy", 5.0)
    score = tsr / (1 + eff * 0.05) / (1 + ent * 0.1) / (1 + spa)
    return score


def make_piece(rank, domain, entry, key):
    layer  = entry.get("layer",  -1)
    expert = entry.get("expert_id", -1)
    coords = entry.get("cross_coords", [0.0, 0.0, 0.0])
    x, y, z = coords[0], coords[1], coords[2]
    feat   = entry.get("features", {})
    tsr    = feat.get("top_singular_ratio", 0.0)
    conf   = min(0.95, 0.60 + tsr * 0.03)

    pid = f"cross600b_{domain}_L{layer}_E{expert}"
    return {
        "piece_id":    pid,
        "name":        f"600B Expert L{layer}/E{expert} ({domain})",
        "description": (
            f"Knowledge extracted from DeepSeek V3 671B expert "
            f"(layer={layer}, expert={expert}, depth={z:.2f}, abstraction={x:.2f}). "
            f"Specialization score: {tsr:.3f}."
        ),
        "in": {
            "requires": [f"domain:{domain}"],
            "slots": []
        },
        "out": {
            "produces": ["knowledge"],
            "schema":   "knowledge"
        },
        "executor":   DOMAIN_EXECUTOR.get(domain, "executors.knowledge.lookup"),
        "confidence": round(conf, 3),
        "tags":       ["600b_extracted", "cross_structure", domain, f"layer_{layer}"],
        "source":     "600b_weight_extraction",
        "knowledge": {
            "domain":    domain,
            "layer":     layer,
            "expert":    expert,
            "cross_xyz": [round(x, 4), round(entry.get("cross_y", 0.0), 4), round(z, 4)],
            "tsr":       round(tsr, 4),
        }
    }


# ── Main ────────────────────────────────────────────────

def main():
    print(f"[LOAD] Reading {CROSS_DB_PATH} ...")
    print("       (8.7GB – may take ~2 minutes)")

    with open(CROSS_DB_PATH) as f:
        raw = json.load(f)

    # Navigate to experts dict (top-level keys: "metadata" + "experts")
    if "experts" in raw:
        cross_db = raw["experts"]
        meta = raw.get("metadata", {})
        print(f"[INFO] metadata: {json.dumps(meta, indent=2)[:300]}")
    else:
        cross_db = raw  # fallback

    print(f"[INFO] Loaded {len(cross_db)} experts")

    # ── Inspect first entry ──
    first_key = next(iter(cross_db))
    first_val = cross_db[first_key]
    print(f"[INFO] First key: {first_key}, keys: {list(first_val.keys())}")
    print(json.dumps(first_val, indent=2)[:400])

    # ── Score every expert ──
    print("[SCORE] Computing specialization scores ...")
    scored = []
    for key, entry in cross_db.items():
        score = specialization_score(entry)
        scored.append((score, key, entry))

    scored.sort(key=lambda t: t[0], reverse=True)
    print(f"[INFO] Top score: {scored[0][0]:.4f} | Bottom: {scored[-1][0]:.4f}")

    # ── Pick top-K and assign domains ──
    pieces = []
    seen_domain_count = defaultdict(int)

    for rank, (score, key, entry) in enumerate(scored[:TOP_K_PIECES * 3]):
        coords = entry.get("cross_coords", [0.0, 0.0, 0.0])
        x, y, z = coords[0], coords[1], coords[2]
        db = depth_band(z)
        ab = abstraction_band(x)
        domain = REGION_DOMAIN_MAP.get((db, ab), "math")

        # Limit per domain to keep diversity
        if seen_domain_count[domain] >= 60:
            continue

        piece = make_piece(rank, domain, entry, key)
        pieces.append(piece)
        seen_domain_count[domain] += 1

        if len(pieces) >= TOP_K_PIECES:
            break

    print(f"\n[PIECES] Created {len(pieces)} pieces")
    print(f"  Domain distribution: {dict(seen_domain_count)}")

    # ── Write output ──
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for p in pieces:
            f.write(json.dumps(p) + "\n")

    print(f"\n[DONE] Written: {OUTPUT_PATH}")
    print(f"       {Path(OUTPUT_PATH).stat().st_size // 1024} KB, {len(pieces)} pieces")

    # ── Stats summary ──
    print("\n=== Cross Structure Statistics ===")
    zvals = [v["cross_coords"][2] for v in cross_db.values()]
    xvals = [v["cross_coords"][0] for v in cross_db.values()]
    yvals = [v["cross_coords"][1] for v in cross_db.values()]
    tsrs  = [v.get("features", {}).get("top_singular_ratio", 0) for v in cross_db.values()]

    def stats(vals, name):
        mean = sum(vals) / len(vals)
        mn   = min(vals)
        mx   = max(vals)
        print(f"  {name:25s}: mean={mean:.3f}, min={mn:.3f}, max={mx:.3f}")

    stats(zvals, "cross_z (depth)")
    stats(xvals, "cross_x (abstraction)")
    stats(yvals, "cross_y (application)")
    stats(tsrs,  "top_singular_ratio")
    print(f"\n[NOTE] tsr range: {min(tsrs):.4f} – {max(tsrs):.4f}")

    print("\n=== Layer Depth Distribution ===")
    band_counts = defaultdict(int)
    for v in cross_db.values():
        band_counts[depth_band(v.get("cross_z", 0))] += 1
    for band, cnt in sorted(band_counts.items()):
        label = ["early (0-25%)", "mid (25-50%)", "late (50-75%)", "final (75-100%)"][band]
        print(f"  Band {band} {label}: {cnt} experts")


if __name__ == "__main__":
    main()
