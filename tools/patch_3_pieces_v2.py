#!/usr/bin/env python3
"""
3ピースの verify/worldgen を新 Verifier API (cross_check_fn) 対応に更新
"""
import json, pathlib

DB_PATH = pathlib.Path(__file__).parent.parent / "pieces" / "piece_db.jsonl"

PATCHES = {
    "nt_factorial": {
        "verify": {
            "kind": "cross_check",
            "method": "double_eval",
            "params": {
                "type_check": "integer",
                "range": {"lo": 1, "hi": 1e15}
            }
        },
        "worldgen": {
            "domain": "number",
            "params": {
                "n": {"type": "int", "min": 0, "max": 12},
                "cross_check_fn": "factorial(n)"
            }
        }
    },
    "nt_factorial_compute": {
        "verify": {
            "kind": "cross_check",
            "method": "double_eval",
            "params": {
                "type_check": "integer",
                "range": {"lo": 1, "hi": 1e15}
            }
        },
        "worldgen": {
            "domain": "number",
            "params": {
                "n": {"type": "int", "min": 0, "max": 12},
                "cross_check_fn": "factorial(n)"
            }
        }
    },
    "arithmetic_power": {
        "verify": {
            "kind": "cross_check",
            "method": "double_eval",
            "params": {
                "type_check": "integer",
                "range": {"lo": -1e18, "hi": 1e18}
            }
        },
        "worldgen": {
            "domain": "number",
            "params": {
                "base": {"type": "int", "min": -9, "max": 9},
                "exponent": {"type": "int", "min": 0, "max": 8},
                "cross_check_fn": "pow(base, exponent)"
            }
        }
    },
    "combinatorics_permutation": {
        "verify": {
            "kind": "cross_check",
            "method": "double_eval",
            "params": {
                "type_check": "integer",
                "range": {"lo": 1, "hi": 1e15},
                "constraints": ["0 <= r <= n"]
            }
        },
        "worldgen": {
            "domain": "number",
            "params": {
                "n": {"type": "int", "min": 1, "max": 12},
                "r": {"type": "int", "min": 0, "max": "n"},
                "cross_check_fn": "factorial(n) // factorial(n - r)"
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
            piece["verify"]   = PATCHES[pid]["verify"]
            piece["worldgen"] = PATCHES[pid]["worldgen"]
            patched += 1
            print(f"✅ Patched: {pid}")
        new_lines.append(json.dumps(piece, ensure_ascii=False))
    DB_PATH.write_text("\n".join(new_lines) + "\n")
    print(f"\nTotal patched: {patched}/{len(PATCHES)}")

if __name__ == "__main__":
    main()
