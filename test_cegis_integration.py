"""
test_cegis_integration.py

CEGIS ループがパイプライン内で 1 問でも最後まで回ることを確認する。
スコア目的ではなく「計測目的」のテスト:

  - executor_result → Candidate → CEGISLoop.run() が動くか
  - CEGIS 診断ログが出るか
  - hallucination reject が動くか
  - フォールバックが既存パスに戻るか
"""

import sys
import json
import os

sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced

pipeline = VerantyxV6Enhanced(
    piece_db_path="pieces/piece_db.jsonl",
    use_beam_search=True,
    use_simulation=True,
)

print("=" * 60)
print("CEGIS Integration Test (計測目的)")
print("=" * 60)

TEST_CASES = [
    # (問題文, 期待答え, 説明)
    ("What is 3 + 5?",                  "8",    "arithmetic: basic add"),
    ("What is 6!?",                     "720",  "combinatorics: factorial"),
    ("What is the GCD of 12 and 18?",   "6",    "number_theory: gcd"),
    ("Is 17 a prime number? (Yes/No)",  "Yes",  "decide: prime"),
    ("What is 7 * 8?",                  "56",   "arithmetic: multiply"),
    ("What is 2^10?",                   "1024", "arithmetic: power"),
    ("How many permutations of 4 elements?", "24", "combinatorics: permutation"),
    ("What is sin(0)?",                 "0",    "math: trig"),
    # MCQ 形式
    (
        "Which is prime?\n(A) 4\n(B) 6\n(C) 7\n(D) 9",
        "C",
        "mcq: prime choice"
    ),
    # フォールバック系
    (
        "What is the capital of France?",
        "Paris",
        "knowledge: geography (CEGIS fallback expected)"
    ),
]

results = []
for i, (q, expected, desc) in enumerate(TEST_CASES):
    print(f"\n[{i+1:02d}] {desc}")
    print(f"  Q: {q[:80]}")
    try:
        r = pipeline.solve(q, expected_answer=expected)
        answer = r.get("answer", "—")
        status = r.get("status", "?")
        conf   = r.get("confidence", 0.0)
        method = r.get("method", "?")

        # CEGIS が走ったか確認
        trace = r.get("trace", [])
        cegis_lines = [t for t in trace if "cegis" in t.lower()]

        ok = (status in ("VERIFIED", "SOLVED"))
        results.append(ok)

        print(f"  A: {answer!r} (expected={expected!r})")
        print(f"  Status: {status}  Conf: {conf:.2f}  Method: {method}")
        if cegis_lines:
            print(f"  CEGIS trace ({len(cegis_lines)} lines):")
            for line in cegis_lines[:5]:
                print(f"    {line}")
            if len(cegis_lines) > 5:
                print(f"    ... (+{len(cegis_lines)-5} more)")
        else:
            print(f"  CEGIS: (no CEGIS trace - early exit before Step 6)")

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.append(False)

# ── 統計出力 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for r in results if r)
print(f"Solved: {passed}/{len(results)}")
print()
pipeline.print_stats()

# ── 診断サマリー ─────────────────────────────────────────────────────────────
print("\n── CEGIS 診断サマリー ─────────────────────────────────")
stats = pipeline.stats
cegis_ran = stats.get("cegis_ran", 0)
if cegis_ran > 0:
    proved_pct   = 100 * stats["cegis_proved"] / cegis_ran
    highconf_pct = 100 * stats["cegis_high_confidence"] / cegis_ran
    fallback_pct = 100 * (stats["cegis_timeout"] + stats["cegis_fallback"]) / cegis_ran
    print(f"  CEGISLoop が回った問題数: {cegis_ran}")
    print(f"  proved:          {stats['cegis_proved']} ({proved_pct:.0f}%)")
    print(f"  high_confidence: {stats['cegis_high_confidence']} ({highconf_pct:.0f}%)")
    print(f"  fallback:        {stats['cegis_timeout']+stats['cegis_fallback']} ({fallback_pct:.0f}%)")
    print()
    print(f"  Hallucination rejected: {stats['cegis_rejected_hallucination']}")
    print(f"  Missing verify specs:   {stats['cegis_missing_verify']} piece-calls")
    print(f"    → 補強が必要な上位ピース候補がここに表示されます")
    print(f"  Missing worldgen specs: {stats['cegis_missing_worldgen']} piece-calls")
    print(f"    → worldgen が無いピースは反例テストをスキップしています")
else:
    print("  CEGISLoop は 1 回も実行されませんでした")
    print("  → MCQ / ExactDetector / Crystal が全問を先取りしている可能性大")
    print("  → Step 6（Execute）まで到達する問題で再テストしてください")

print("\n✅ Integration test complete")
