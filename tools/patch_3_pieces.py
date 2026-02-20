#!/usr/bin/env python3
"""
A+ Step 2: nt_factorial / arithmetic_power / combinatorics_permutation に
verify + worldgen を追加するパッチスクリプト。
"""

import json, pathlib, sys

DB_PATH = pathlib.Path(__file__).parent.parent / "pieces" / "piece_db.jsonl"

PATCHES = {
    "nt_factorial": {
        "verify": {
            "kind": "cross_check",
            "method": "double_eval",
            "params": {
                "description": "math.factorial(n) で独立検証",
                "type_check": "integer",
                "range": {"lo": 1, "hi": 1e15}  # 15! = 1.307e12 まで安全
            }
        },
        "worldgen": {
            "domain": "number",
            "params": {
                "description": "n in [0..12] でランダムテスト（n=13! は 6 billion で爆発回避）",
                "n": {"type": "int", "min": 0, "max": 12}
            }
        }
    },
    "arithmetic_power": {
        "verify": {
            "kind": "cross_check",
            "method": "double_eval",
            "params": {
                "description": "pow(base, exponent) で独立検証",
                "type_check": "integer",
                "range": {"lo": -1e18, "hi": 1e18}
            }
        },
        "worldgen": {
            "domain": "number",
            "params": {
                "description": "base in [-9..9], exponent in [0..8]（0^0 は除外）",
                "base": {"type": "int", "min": -9, "max": 9},
                "exponent": {"type": "int", "min": 0, "max": 8},
                "exclude": {"base": 0, "exponent": 0}
            }
        }
    },
    "combinatorics_permutation": {
        "verify": {
            "kind": "cross_check",
            "method": "double_eval",
            "params": {
                "description": "math.factorial(n)//math.factorial(n-r) で独立検証",
                "type_check": "integer",
                "range": {"lo": 1, "hi": 1e15},
                "constraints": ["0 <= r <= n"]
            }
        },
        "worldgen": {
            "domain": "number",
            "params": {
                "description": "n in [1..12], r in [0..n]",
                "n": {"type": "int", "min": 1, "max": 12},
                "r": {"type": "int", "min": 0, "max": "n"}  # r <= n は実行時に保証
            }
        }
    }
}

def main():
    lines = DB_PATH.read_text().splitlines()
    patched = 0
    new_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            piece = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue

        pid = piece.get("piece_id", "")
        if pid in PATCHES:
            patch = PATCHES[pid]
            piece["verify"]   = patch["verify"]
            piece["worldgen"] = patch["worldgen"]
            patched += 1
            print(f"✅ Patched: {pid}")

        new_lines.append(json.dumps(piece, ensure_ascii=False))

    DB_PATH.write_text("\n".join(new_lines) + "\n")
    print(f"\nTotal patched: {patched}/{len(PATCHES)}")
    if patched < len(PATCHES):
        missing = set(PATCHES) - {json.loads(l).get("piece_id","") for l in new_lines if l.strip()}
        print(f"⚠️  Missing piece_ids: {missing}")

if __name__ == "__main__":
    main()
