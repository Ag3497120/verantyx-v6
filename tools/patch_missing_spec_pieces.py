"""
patch_missing_spec_pieces.py
============================
Guard1 でブロックされている 4 ピースに verify/worldgen を追加する。

対象:
  - solve_multiple_choice  (10回 / missing_spec)
  - string_length          ( 3回)
  - palindrome_check       ( 1回)
  - prob_expected_value    ( 1回)

戦略:
  verify  → simulation: 小世界で実際に executor を動かして答えが一意に定まるか確認
  worldgen → 各ピース固有の世界生成（MCQ:既知答え問題、文字列:ランダム文字列等）
"""
import json
import shutil
from pathlib import Path

PIECE_DB = Path(__file__).parent.parent / "pieces/piece_db.jsonl"
BACKUP   = PIECE_DB.with_suffix(".jsonl.bak3")

# ── パッチ定義 ─────────────────────────────────────────────────────────────
PATCHES = {
    "solve_multiple_choice": {
        "verify": {
            "kind": "simulation",
            "method": "mcq_option_check",
            "params": {
                "description": "選択されたオプションを計算で検証。他オプションと比較して一意性を確認",
                "strategy": "computational_elimination",
                "min_confidence": 0.75,
            },
        },
        "worldgen": {
            "domain": "multiple_choice",
            "params": {
                "templates": [
                    # 算術MCQ（答え確定可能）
                    {"stem": "What is {a}*{b}?",
                     "choices": ["A:{w1}", "B:{correct}", "C:{w2}", "D:{w3}"],
                     "gen": "arithmetic_multiply"},
                    # 素数判定MCQ
                    {"stem": "Which of the following is a prime number?",
                     "choices": ["A:{composite1}", "B:{composite2}", "C:{prime}", "D:{composite3}"],
                     "gen": "prime_check"},
                    # 階乗MCQ
                    {"stem": "What is {n}! (factorial)?",
                     "choices": ["A:{w1}", "B:{w2}", "C:{correct}", "D:{w3}"],
                     "gen": "factorial"},
                ],
                "num_worlds": 5,
            },
            "constraints": ["answer_is_option_label", "computational_verifiable"],
        },
    },

    "string_length": {
        "verify": {
            "kind": "simulation",
            "method": "recompute",
            "params": {
                "description": "len(text) を再計算して一致確認",
                "type_check": "integer",
                "range": {"lo": 0, "hi": 100000},
            },
        },
        "worldgen": {
            "domain": "string",
            "params": {
                "lo": 0,
                "hi": 200,
                "charsets": ["ascii_lower", "ascii_upper", "digits", "mixed"],
                "num_worlds": 10,
            },
            "constraints": ["integer_output", "deterministic"],
        },
    },

    "palindrome_check": {
        "verify": {
            "kind": "simulation",
            "method": "recompute",
            "params": {
                "description": "text == text[::-1] を再計算して確認",
                "type_check": "boolean",
            },
        },
        "worldgen": {
            "domain": "string",
            "params": {
                "palindromes": ["racecar", "level", "madam", "noon", "aba", "amanaplanacanalpanama"],
                "non_palindromes": ["hello", "world", "python", "verantyx", "compute"],
                "num_worlds": 8,
            },
            "constraints": ["boolean_output", "deterministic"],
        },
    },

    "prob_expected_value": {
        "verify": {
            "kind": "simulation",
            "method": "recompute",
            "params": {
                "description": "sum(x*p for x,p in zip(values,probs)) を再計算して一致確認",
                "type_check": "decimal",
                "prob_sum_check": True,  # sum(probs) ≈ 1.0 を確認
                "tolerance": 1e-9,
            },
        },
        "worldgen": {
            "domain": "probability",
            "params": {
                "templates": [
                    {"values": [1, 2, 3, 4, 5, 6], "probs": [1/6]*6,
                     "expected": 3.5, "desc": "fair_die"},
                    {"values": [0, 1], "probs": [0.5, 0.5],
                     "expected": 0.5, "desc": "fair_coin"},
                    {"values": [1, 2], "probs": [0.3, 0.7],
                     "expected": 1.7, "desc": "biased_binary"},
                    {"values": [0, 1, 2], "probs": [0.25, 0.5, 0.25],
                     "expected": 1.0, "desc": "symmetric_ternary"},
                ],
                "num_worlds": 6,
            },
            "constraints": ["decimal_output", "probs_sum_to_one"],
        },
    },
}


def patch():
    # バックアップ
    shutil.copy(PIECE_DB, BACKUP)
    print(f"Backup: {BACKUP}")

    patched = 0
    lines_out = []
    with open(PIECE_DB) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            pid = p.get("piece_id")
            if pid in PATCHES:
                p["verify"]   = PATCHES[pid]["verify"]
                p["worldgen"] = PATCHES[pid]["worldgen"]
                patched += 1
                print(f"  Patched: {pid}")
            lines_out.append(json.dumps(p, ensure_ascii=False))

    with open(PIECE_DB, "w") as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"\nDone: {patched} pieces patched → {PIECE_DB}")


if __name__ == "__main__":
    patch()
