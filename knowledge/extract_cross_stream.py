#!/usr/bin/env python3
"""
Streaming Cross DB → Verantyx Pieces Extractor
Uses ijson to avoid loading 8.7GB into memory.
Outputs top-200 specialized experts as pieces_600b.jsonl.
"""
import json
import ijson
import os
import math
from pathlib import Path
from collections import defaultdict

CROSS_DB_PATH = os.environ.get("CROSS_DB_PATH", "/Users/motonishikoudai/Downloads/cross_db_full.json")
OUTPUT_PATH   = os.environ.get("OUTPUT_PATH",   "/Users/motonishikoudai/Downloads/pieces_600b.jsonl")
TOP_K         = 200

REGION_DOMAIN_MAP = {
    (0, 0): "math",      (0, 1): "logic",
    (1, 0): "chemistry", (1, 1): "math",
    (2, 0): "physics",   (2, 1): "math",
    (3, 0): "biology",   (3, 1): "math",
}
DOMAIN_EXECUTOR = {
    "math":      "executors.knowledge.lookup",
    "logic":     "executors.logic.evaluate",
    "chemistry": "executors.knowledge.lookup",
    "physics":   "executors.knowledge.lookup",
    "biology":   "executors.knowledge.lookup",
}

def depth_band(z):
    if z < 0.25: return 0
    if z < 0.50: return 1
    if z < 0.75: return 2
    return 3

def abstraction_band(x):
    return 1 if x > 0.5 else 0

def spec_score(feat, coords):
    tsr = feat.get("top_singular_ratio", 0.0)
    eff = feat.get("effective_rank", 1.0)
    ent = feat.get("spectral_entropy", 5.0)
    spa = feat.get("sparsity", 0.0)
    return tsr / (1 + eff * 0.05) / (1 + ent * 0.1) / (1 + spa)

def make_piece(rank, domain, layer, expert_id, feat, coords):
    x, y, z = coords
    tsr  = feat.get("top_singular_ratio", 0.0)
    conf = min(0.95, 0.60 + tsr * 0.03)
    return {
        "piece_id":    f"cross600b_{domain}_L{layer}_E{expert_id}",
        "name":        f"600B Expert L{layer}/E{expert_id} ({domain})",
        "description": (
            f"Knowledge from DeepSeek V3 671B expert "
            f"(layer={layer}, expert={expert_id}, "
            f"depth={z:.2f}, abstraction={x:.2f}, tsr={tsr:.3f})."
        ),
        "in":  {"requires": [f"domain:{domain}"], "slots": []},
        "out": {"produces": ["knowledge"], "schema": "knowledge"},
        "executor":   DOMAIN_EXECUTOR.get(domain, "executors.knowledge.lookup"),
        "confidence": round(conf, 3),
        "tags":       ["600b_extracted", "cross_structure", domain, f"layer_{layer}"],
        "source":     "600b_weight_extraction",
        "knowledge":  {
            "domain": domain, "layer": layer, "expert_id": expert_id,
            "cross_xyz": [round(v, 4) for v in coords],
            "tsr": round(tsr, 4),
        }
    }

def main():
    print(f"[STREAM] Reading {CROSS_DB_PATH} with ijson (streaming)...")
    fsize = Path(CROSS_DB_PATH).stat().st_size
    print(f"         File size: {fsize/1e9:.1f} GB")

    experts = []  # (score, layer, expert_id, feat, coords)
    count = 0

    with open(CROSS_DB_PATH, "rb") as f:
        # Stream each expert entry under "experts.*"
        for key, entry in ijson.kvitems(f, "experts"):
            layer     = entry.get("layer", -1)
            expert_id = entry.get("expert_id", -1)
            feat      = {k: float(v) for k, v in entry.get("features", {}).items() if not isinstance(v, (list, dict))}
            coords    = [float(v) for v in entry.get("cross_coords", [0.0, 0.0, 0.0])]

            # Skip concept_directions (large, not needed)
            score = spec_score(feat, coords)
            experts.append((score, layer, expert_id, feat, coords))
            count += 1
            if count % 1000 == 0:
                print(f"  processed {count} experts...", end="\r")

    print(f"\n[INFO] Total experts streamed: {count}")

    # Sort by specialization score
    experts.sort(key=lambda t: t[0], reverse=True)
    print(f"[INFO] Score range: {experts[0][0]:.4f} – {experts[-1][0]:.4f}")

    # Pick top-K with domain diversity
    pieces = []
    seen = defaultdict(int)
    for score, layer, expert_id, feat, coords in experts:
        x, y, z = coords
        db = depth_band(z)
        ab = abstraction_band(x)
        domain = REGION_DOMAIN_MAP.get((db, ab), "math")

        if seen[domain] >= 50:
            continue

        pieces.append(make_piece(len(pieces), domain, layer, expert_id, feat, coords))
        seen[domain] += 1
        if len(pieces) >= TOP_K:
            break

    print(f"\n[PIECES] {len(pieces)} pieces | distribution: {dict(seen)}")

    # Write
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for p in pieces:
            f.write(json.dumps(p) + "\n")
    print(f"[DONE] {OUTPUT_PATH} ({Path(OUTPUT_PATH).stat().st_size//1024} KB)")

if __name__ == "__main__":
    main()
