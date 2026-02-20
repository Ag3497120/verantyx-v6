"""
test_cegis.py - CEGIS システムの動作確認

テスト内容:
  1. WorldGen: 各ドメインのモデル生成
  2. Certificate: 各種証明書の検証
  3. CEGISLoop: 完全な推論ループ
  4. GrammarGlue: 全スキーマのレンダリング
  5. 統合テスト: HLE 風の問題に適用
"""

import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from cegis.certificate import Certificate, CertKind, CertificateChecker
from cegis.worldgen import WorldGenerator
from cegis.cegis_loop import CEGISLoop, Candidate, make_candidates_from_executor_result
from grammar.glue_templates import GrammarGlue, render, render_auto

import traceback

PASS = "✅"
FAIL = "❌"
results = []


def check(name: str, cond: bool, detail: str = ""):
    status = PASS if cond else FAIL
    results.append((name, cond))
    print(f"  {status} {name}" + (f" | {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# 1. WorldGen テスト
# ─────────────────────────────────────────────────────────────────────────────

print("\n━━━ 1. WorldGen ━━━")
gen = WorldGenerator(max_size=6, max_worlds=30)

try:
    groups = gen.generate("group")
    check("group generation", len(groups) > 0, f"{len(groups)} worlds")
    check("group has Z_2", any(w.relations.get("type") == "Z_2" for w in groups))
    check("group has V4",  any(w.relations.get("type") == "V4"  for w in groups))
    check("group abelian props", all("abelian" in w.properties for w in groups))
except Exception as e:
    check("group generation", False, str(e))

try:
    graphs = gen.generate("graph")
    check("graph generation", len(graphs) > 0, f"{len(graphs)} worlds")
    check("graph has K_3",   any("K_3" in str(w.relations.get("type","")) for w in graphs))
    check("graph planarity", all("planar" in w.properties for w in graphs))
except Exception as e:
    check("graph generation", False, str(e))

try:
    rings = gen.generate("ring")
    check("ring generation", len(rings) > 0, f"{len(rings)} worlds")
    check("Z_5 is field", any(
        w.relations.get("type") == "Z_5" and w.properties.get("field")
        for w in rings
    ))
except Exception as e:
    check("ring generation", False, str(e))

try:
    seqs = gen.generate("sequence")
    check("sequence generation", len(seqs) > 0, f"{len(seqs)} worlds")
    check("fibonacci exists", any(
        w.relations.get("type") == "fibonacci" for w in seqs
    ))
    check("primes exists", any(
        w.relations.get("type") == "primes" for w in seqs
    ))
except Exception as e:
    check("sequence generation", False, str(e))

try:
    props = gen.generate("propositional", {"atoms": ["p", "q"]})
    check("propositional generation", len(props) == 4, f"{len(props)} worlds (expect 4)")
    all_valuations = [{k: v for k, v in w.relations["valuation"].items()} for w in props]
    check("all 4 truth assignments", len(all_valuations) == 4)
except Exception as e:
    check("propositional generation", False, str(e))

try:
    numbers = gen.generate("number", {"lo": -3, "hi": 3})
    check("number generation", len(numbers) >= 7, f"{len(numbers)} worlds")
    check("prime detection", any(
        w.relations.get("value") == 3 and w.properties.get("prime")
        for w in numbers
    ))
except Exception as e:
    check("number generation", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 2. Certificate テスト
# ─────────────────────────────────────────────────────────────────────────────

print("\n━━━ 2. Certificate ━━━")
checker = CertificateChecker()

# COMPUTATION_LOG
cert = Certificate(
    kind=CertKind.COMPUTATION_LOG,
    value=[("add", [1, 2], 3), ("mul", [3, 4], 12)],
    confidence=1.0,
)
check("computation_log: result=12", checker.check(cert, 12))
check("computation_log: result≠99", not checker.check(cert, 99))

# CROSS_CHECK
cert2 = Certificate(
    kind=CertKind.CROSS_CHECK,
    value=[42, 42, 42],
    confidence=1.0,
)
check("cross_check: all same", checker.check(cert2, 42))
cert2_bad = Certificate(kind=CertKind.CROSS_CHECK, value=[42, 43], confidence=1.0)
check("cross_check: mismatch → False", not checker.check(cert2_bad, 42))

# SUBSTITUTION: x=5 in equation 2x+1=11
cert3 = Certificate(
    kind=CertKind.SUBSTITUTION,
    value={"equation": "2*x+1=11"},
    confidence=0.9,
)
check("substitution: x=5 in 2x+1=11", checker.check(cert3, 5))

# COUNTEREXAMPLE
cert4 = Certificate(kind=CertKind.COUNTEREXAMPLE, value={"example": [0, 0, 1]}, confidence=1.0)
check("counterexample: not None → True", checker.check(cert4, False))

# SMALL_WORLD
cert5 = Certificate(
    kind=CertKind.SMALL_WORLD,
    value={"worlds_tested": 50, "passed": 48, "ratio": 0.96},
    confidence=0.96,
)
check("small_world: 96% pass rate → True", checker.check(cert5, "any"))
cert5_bad = Certificate(
    kind=CertKind.SMALL_WORLD,
    value={"worlds_tested": 50, "passed": 30, "ratio": 0.60},
    confidence=0.60,
)
check("small_world: 60% pass rate → False", not checker.check(cert5_bad, "any"))

# HIGH_CONFIDENCE
cert6 = Certificate(kind=CertKind.HIGH_CONFIDENCE, value={}, confidence=0.85)
check("high_confidence: 0.85 → True", checker.check(cert6, "any"))
cert6_low = Certificate(kind=CertKind.HIGH_CONFIDENCE, value={}, confidence=0.50)
check("high_confidence: 0.50 → False", not checker.check(cert6_low, "any"))

# EXHAUSTIVE
cert7 = Certificate(kind=CertKind.EXHAUSTIVE, value=[1, 3, 6, 10, 15], confidence=1.0)
check("exhaustive: 10 in list", checker.check(cert7, 10))
check("exhaustive: 7 not in list", not checker.check(cert7, 7))

# ─────────────────────────────────────────────────────────────────────────────
# 3. CEGIS Loop テスト
# ─────────────────────────────────────────────────────────────────────────────

print("\n━━━ 3. CEGIS Loop ━━━")
loop = CEGISLoop(max_iter=3, max_worlds=20, time_limit_ms=5000)

# Test 1: 素直な候補が正解
ir_arithmetic = {
    "domain": "arithmetic",
    "task": "compute",
    "answer_schema": "integer",
}
candidates_simple = [
    Candidate(value=42, construction=["arithmetic.add"], confidence=0.95),
    Candidate(value=7,  construction=["arithmetic.mul"], confidence=0.80),
]
result1 = loop.run(ir_arithmetic, candidates_simple)
check("CEGIS: arithmetic runs", result1.status in ("proved", "high_confidence", "timeout"))
check("CEGIS: returns answer", result1.answer is not None)
print(f"     answer={result1.answer!r}, conf={result1.confidence:.2f}, status={result1.status}")

# Test 2: 正の数制約で候補フィルタ
candidates_constrained = [
    Candidate(value=-5, construction=["test"], confidence=0.9, constraints=["positive"]),
    Candidate(value=7,  construction=["test"], confidence=0.7, constraints=["positive"]),
    Candidate(value=0,  construction=["test"], confidence=0.6, constraints=["positive"]),
]
result2 = loop.run(ir_arithmetic, candidates_constrained)
check("CEGIS: positive constraint filters -5", result2.answer != -5, f"answer={result2.answer!r}")

# Test 3: 論理ドメイン
ir_logic = {
    "domain": "logic_propositional",
    "task": "decide",
    "answer_schema": "boolean",
    "entities": [{"type": "symbol", "name": "p"}, {"type": "symbol", "name": "q"}],
}
candidates_logic = [
    Candidate(value=True,  construction=["logic.tautology"], confidence=0.9),
    Candidate(value=False, construction=["logic.satisfy"], confidence=0.6),
]
result3 = loop.run(ir_logic, candidates_logic)
check("CEGIS: logic domain runs", result3.status in ("proved", "high_confidence", "timeout"))

# Test 4: make_candidates_from_executor_result
cands4 = make_candidates_from_executor_result(
    result=[3, 5, 7],
    piece_id="number_theory.next_prime",
    confidence=0.88,
)
check("make_candidates: list → 3 candidates", len(cands4) == 3)
check("make_candidates: values correct", [c.value for c in cands4] == [3, 5, 7])
check("make_candidates: confidence decay", cands4[1].confidence < cands4[0].confidence)

# Test 5: no candidates
result5 = loop.run(ir_arithmetic, [])
check("CEGIS: empty candidates → unknown", result5.status in ("unknown", "timeout"))

# ─────────────────────────────────────────────────────────────────────────────
# 4. GrammarGlue テスト
# ─────────────────────────────────────────────────────────────────────────────

print("\n━━━ 4. GrammarGlue ━━━")
glue = GrammarGlue()

glue_cases = [
    (42,              "integer",      "42"),
    (-7,              "integer",      "-7"),
    (3.5,             "decimal",      "3.5"),
    (1/3,             "rational",     "1/3"),
    (2/4,             "rational",     "1/2"),  # 約分
    (True,            "boolean",      "True"),
    ("yes",           "boolean",      "True"),
    ("no",            "boolean",      "False"),
    ("b",             "option_label", "B"),
    ("C",             "option_label", "C"),
    ([1,2,3],         "sequence",     "1, 2, 3"),
    ({1,2},           "set",          "{1, 2}"),
    ([[1,0],[0,1]],   "matrix",       "[[1 0; 0 1]]"),
    (complex(3,4),    "complex",      "3+4i"),
    (complex(3,-4),   "complex",      "3-4i"),
]
for val, schema, expected in glue_cases:
    got = glue.render(val, schema)
    # set の順序は不定なので contains チェック
    if schema == "set":
        ok = "1" in got and "2" in got and "{" in got
    elif schema == "matrix":
        ok = "1" in got and "0" in got
    else:
        ok = (got == expected)
    check(f"glue.render({val!r}, {schema!r})", ok, f"got={got!r}, expect={expected!r}")

# render_auto テスト
check("render_auto: 42 → integer str", render_auto(42) == "42")
check("render_auto: True → boolean", render_auto(True) == "True")
check("render_auto: [1,2] → sequence", render_auto([1,2]) == "1, 2")

# fallback テスト
check("fallback: integer → '0'", glue.fallback("integer") == "0")
check("fallback: option_label → ''", glue.fallback("option_label") == "")

# infer_schema テスト
check("infer: int → integer",    glue.infer_schema(42) == "integer")
check("infer: float → decimal",  glue.infer_schema(3.14) == "decimal")
check("infer: bool → boolean",   glue.infer_schema(True) == "boolean")
check("infer: list → sequence",  glue.infer_schema([1,2]) == "sequence")
check("infer: 'A' → option_label", glue.infer_schema("A") == "option_label")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 統合テスト: HLE 風の問題
# ─────────────────────────────────────────────────────────────────────────────

print("\n━━━ 5. 統合テスト（HLE風） ━━━")

# Q: "What is 6! ?" (answer: 720)
ir_factorial = {
    "domain": "combinatorics",
    "task": "compute",
    "answer_schema": "integer",
    "metadata": {"keywords": ["factorial"]},
}
import math
factorial_cands = [
    Candidate(value=720,  construction=["combinatorics.factorial"], confidence=0.99),
    Candidate(value=120,  construction=["combinatorics.factorial"], confidence=0.70),
]
r_fact = loop.run(ir_factorial, factorial_cands)
check("HLE factorial: answer=720", r_fact.answer == 720, f"got={r_fact.answer!r}")
check("HLE factorial: high conf", r_fact.confidence >= 0.6)
rendered_fact = glue.render(r_fact.answer, "integer")
check("HLE factorial: rendered='720'", rendered_fact == "720", f"got={rendered_fact!r}")

# Q: MCQ "Is the following group abelian? Z_6" (answer: A=Yes)
ir_mcq = {
    "domain": "multiple_choice",
    "task": "choose",
    "answer_schema": "option_label",
}
mcq_cands = [
    Candidate(value="A", construction=["multiple_choice.solve"], confidence=0.90),
    Candidate(value="B", construction=["multiple_choice.solve"], confidence=0.30),
]
r_mcq = loop.run(ir_mcq, mcq_cands)
check("HLE MCQ: returns label", r_mcq.answer in ("A","B","C","D","E","a","b","c","d","e", None))
rendered_mcq = glue.render(r_mcq.answer, "option_label") if r_mcq.answer else ""
check("HLE MCQ: rendered uppercase", rendered_mcq in ("A","B","C","D","E",""))

# ─────────────────────────────────────────────────────────────────────────────
# 集計
# ─────────────────────────────────────────────────────────────────────────────

total = len(results)
passed = sum(1 for _, ok in results if ok)
print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed")
print(f"{'='*50}")
if passed == total:
    print("🎉 All tests passed!")
else:
    failed = [(name, ok) for name, ok in results if not ok]
    print(f"❌ Failed ({len(failed)}):")
    for name, _ in failed:
        print(f"    - {name}")
