"""
patch_piece_verify_worldgen.py

上位5ピースに verify + worldgen スペックを付与する。
piece_db.jsonl を in-place で更新する。

対象:
  1. arithmetic_eval / arithmetic_eval_integer / arithmetic_eval_decimal
  2. nt_gcd_compute / number_theory_gcd
  3. number_theory_prime
  4. algebra_evaluate_poly

各ピースの worldgen は CEGISLoop の WorldGenerator.generate() に直接渡す。
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "pieces", "piece_db.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# verify / worldgen スペック定義
# ─────────────────────────────────────────────────────────────────────────────

PATCH_MAP = {
    # ── arithmetic ─────────────────────────────────────────────────────────
    "arithmetic_eval": {
        "verify": {
            "kind": "cross_check",
            "method": "double_eval",
            "params": {
                "description": "eval(expr) を2回実行して一致確認。型整合性・範囲チェック付き",
                "type_check": "integer",
                "range": {"lo": -1e12, "hi": 1e12},
            },
        },
        "worldgen": {
            "domain": "number",
            "params": {"lo": -20, "hi": 20},
            "constraints": ["integer"],
        },
    },
    "arithmetic_eval_integer": {
        "verify": {
            "kind": "cross_check",
            "method": "double_eval",
            "params": {"type_check": "integer", "range": {"lo": -1e12, "hi": 1e12}},
        },
        "worldgen": {
            "domain": "number",
            "params": {"lo": -20, "hi": 20},
            "constraints": ["integer"],
        },
    },
    "arithmetic_eval_decimal": {
        "verify": {
            "kind": "interval",
            "method": "interval_arithmetic",
            "params": {
                "description": "区間演算で誤差境界を計算して一致確認",
                "tolerance": 1e-9,
            },
        },
        "worldgen": {
            "domain": "number",
            "params": {"lo": -100, "hi": 100},
            "constraints": [],
        },
    },

    # ── number theory: GCD ─────────────────────────────────────────────────
    "nt_gcd_compute": {
        "verify": {
            "kind": "substitution",
            "method": "gcd_divisibility",
            "params": {
                "description": "g=GCD(a,b) ならば g|a かつ g|b かつ g が最大であることを確認",
                "checks": ["divides_a", "divides_b", "maximality"],
            },
        },
        "worldgen": {
            "domain": "number",
            "params": {
                "lo": 1, "hi": 100,
                "description": "a=p*u, b=p*v で共通因子 p を埋め込んだサンプル生成",
            },
            "constraints": ["positive", "integer"],
        },
    },
    "number_theory_gcd": {
        "verify": {
            "kind": "substitution",
            "method": "gcd_divisibility",
            "params": {
                "description": "g=GCD(a,b) ならば g|a かつ g|b かつ g が最大であることを確認",
                "checks": ["divides_a", "divides_b", "maximality"],
            },
        },
        "worldgen": {
            "domain": "number",
            "params": {"lo": 1, "hi": 100},
            "constraints": ["positive", "integer"],
        },
    },

    # ── number theory: prime ───────────────────────────────────────────────
    "number_theory_prime": {
        "verify": {
            "kind": "small_world",
            "method": "miller_rabin",
            "params": {
                "description": "決定論的 Miller–Rabin (64bit 対応 witness セット)",
                "witnesses": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37],
                "schema_mapping": {"True": "Yes", "False": "No"},
            },
        },
        "worldgen": {
            "domain": "number",
            "params": {
                "lo": 2, "hi": 200,
                "description": "素数表から素数 / p*q 形式の合成数を生成",
            },
            "constraints": ["positive", "integer"],
        },
    },

    # ── algebra: polynomial evaluation ────────────────────────────────────
    "algebra_evaluate_poly": {
        "verify": {
            "kind": "cross_check",
            "method": "sympy_eval",
            "params": {
                "description": "SymPy で多項式を評価して候補と比較",
                "library": "sympy",
                "fallback": "eval_string",
            },
        },
        "worldgen": {
            "domain": "polynomial",
            "params": {
                "max_deg": 3,
                "coeff_range": [-3, 3],
                "description": "ランダム係数多項式 + 代入点 → 期待値内部計算",
            },
            "constraints": [],
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# piece_db.jsonl を読み込んでパッチを適用
# ─────────────────────────────────────────────────────────────────────────────

def patch_db(db_path: str) -> None:
    # 読み込み
    pieces = []
    with open(db_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pieces.append(json.loads(line))

    print(f"Loaded {len(pieces)} pieces from {db_path}")

    patched = 0
    for piece in pieces:
        pid = piece.get("piece_id", "")
        if pid in PATCH_MAP:
            spec = PATCH_MAP[pid]
            old_verify   = piece.get("verify",   {})
            old_worldgen = piece.get("worldgen", {})
            piece["verify"]   = spec["verify"]
            piece["worldgen"] = spec["worldgen"]
            print(f"  PATCHED: {pid}")
            print(f"    verify:   {old_verify!r} → {spec['verify']['kind']}")
            print(f"    worldgen: {old_worldgen!r} → {spec['worldgen']['domain']}")
            patched += 1

    # 上書き保存
    with open(db_path, "w", encoding="utf-8") as f:
        for piece in pieces:
            f.write(json.dumps(piece, ensure_ascii=False) + "\n")

    print(f"\nPatched {patched} pieces, saved to {db_path}")


if __name__ == "__main__":
    patch_db(DB_PATH)
