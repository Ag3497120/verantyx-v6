"""
eval_hle_50.py — HLE 50問評価（Guard 1+2 付き）

Guard 1: strict_spec_mode=True
  → verify/worldgen 未補強ピース（nt_factorial / arithmetic_power / combinatorics_permutation）
    が選択されたら CEGIS をブロックし missing_spec タグで記録する

Guard 2: 5指標に固定したログ
  1. step6_reach_rate     — Executor まで到達した割合
  2. cegis_runs           — CEGIS が起動した問題数
  3. proved_rate          — CEGIS が proved/high_confidence を返した割合
  4. missing_spec_count   — Guard 1 でブロックされた問題数
  5. fail_tags_histogram  — 失敗タイプの内訳

使い方:
  python3 eval_hle_50.py                  # 先頭50問
  python3 eval_hle_50.py --offset 100     # 100問目から50問
  python3 eval_hle_50.py --n 100          # 先頭100問
  python3 eval_hle_50.py --category Math  # Math カテゴリのみ
"""

import sys
import json
import time
import re
import argparse
from collections import defaultdict
from typing import Optional

sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match


# ─────────────────────────────────────────────────────────────────────────────
# 引数解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",        type=int, default=50,   help="評価問題数")
    p.add_argument("--offset",   type=int, default=0,    help="開始インデックス")
    p.add_argument("--category", type=str, default=None, help="絞り込みカテゴリ")
    p.add_argument("--dataset",  type=str, default="hle_2500_eval.jsonl")
    p.add_argument("--strict",   action="store_true", default=True,
                   help="Guard 1: strict_spec_mode を有効化（デフォルト ON）")
    p.add_argument("--no-strict", dest="strict", action="store_false",
                   help="Guard 1 を無効化")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 失敗タグ分類器（Guard 2: fail_tags_histogram 用）
# ─────────────────────────────────────────────────────────────────────────────

def classify_fail(result: dict, question: dict) -> str:
    """失敗した問題に1タグを付ける"""
    trace   = result.get("trace", [])
    method  = result.get("method", "")
    answer  = result.get("answer")
    q_text  = question.get("question", "")

    # 明示的な missing_spec タグ
    if any("missing_spec" in t for t in trace):
        return "missing_spec"

    # MCQ 関連
    if any("mcq" in t or "MCQ" in t for t in trace):
        return "mcq_solver_fail"
    if re.search(r'\([A-E]\)', q_text):
        return "mcq_no_route"

    # CEGIS 関連
    if any("cegis:GUARD1_BLOCK" in t for t in trace):
        return "missing_spec"
    if any("cegis:B2_MCQ_GATE" in t for t in trace):
        return "mcq_gate_blocked"
    if any("cegis:no_candidates" in t for t in trace):
        return "extract_fail"

    # Step 6 未到達
    if answer is None or str(answer).strip() in ("", "—", "None"):
        if "step:execute" not in " ".join(trace):
            return "step6_not_reached"

    # Executor 失敗
    if "execution_failed" in " ".join(trace) or "execute_error" in " ".join(trace):
        return "executor_fail"

    # Compose 失敗
    if "compose_failed" in " ".join(trace) or "compose_error" in " ".join(trace):
        return "compose_fail"

    # 知識問題
    category = question.get("category", "")
    if category in ("Biology", "Medicine", "History", "Law", "Philosophy",
                    "Economics", "Literature", "Music", "Art", "Geography"):
        return "knowledge_qa"

    # デフォルト
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# メイン評価ループ
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── データセット読み込み ──────────────────────────────────────────────────
    print(f"Loading {args.dataset}...")
    questions = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    # フィルタ
    if args.category:
        questions = [q for q in questions if q.get("category", "") == args.category]
        print(f"Category filter '{args.category}': {len(questions)} questions")

    questions = questions[args.offset : args.offset + args.n]
    total = len(questions)
    print(f"Evaluating {total} questions (offset={args.offset})\n")

    # ── パイプライン初期化 ────────────────────────────────────────────────────
    pipeline = VerantyxV6Enhanced(
        piece_db_path="pieces/piece_db.jsonl",
        use_beam_search=True,
        use_simulation=True,
    )
    pipeline.strict_spec_mode = args.strict
    print(f"Guard 1 strict_spec_mode: {pipeline.strict_spec_mode}")
    print("=" * 60)

    # ── Guard 2: 5指標トラッカー ──────────────────────────────────────────────
    metrics = {
        "step6_reached":    0,   # Step 6（Execute）到達数
        "cegis_runs":       0,
        "cegis_proved":     0,
        "missing_spec_count": 0,
        "fail_tags":        defaultdict(int),
    }

    correct   = 0
    fail_log  = []   # 失敗詳細（後ほど集計）

    start = time.time()

    for i, q in enumerate(questions):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1:3d}/{total}]  correct={correct}  "
                  f"({100*correct/(i+1):.1f}%)  {elapsed:.1f}s")

        expected = q.get("answer", "")
        try:
            r = pipeline.solve(q["question"])
        except Exception as e:
            r = {"status": "FAILED", "answer": None, "trace": [f"exception:{e}"]}

        answer = r.get("answer")
        trace  = r.get("trace", [])

        # ── Guard 2: 5指標を更新 ─────────────────────────────────────────────
        # 1. step6_reach_rate
        if "step:execute" in " ".join(trace):
            metrics["step6_reached"] += 1

        # 2. cegis_runs (パイプライン統計から)
        # (後で pipeline.stats から取得)

        # 4. missing_spec_count
        if any("missing_spec" in t or "GUARD1_BLOCK" in t for t in trace):
            metrics["missing_spec_count"] += 1

        # 正解チェック
        if answer is not None and expected:
            if flexible_match(str(answer), str(expected), tolerance=1e-4):
                correct += 1
                continue  # 正解はスキップ

        # 失敗タグ付け
        tag = classify_fail(r, q)
        metrics["fail_tags"][tag] += 1
        fail_log.append({
            "idx":      i,
            "q":        q["question"][:80],
            "expected": expected[:40] if expected else "",
            "answer":   str(answer)[:40] if answer else "—",
            "tag":      tag,
            "method":   r.get("method", "?"),
        })

    elapsed_total = time.time() - start

    # ── Guard 2: 最終集計 ─────────────────────────────────────────────────────
    stats = pipeline.stats
    cegis_ran          = stats.get("cegis_ran", 0)
    cegis_loop_started = stats.get("cegis_loop_started", 0)
    cegis_iters        = stats.get("cegis_iters", 0)
    cegis_proved       = stats.get("cegis_proved", 0) + stats.get("cegis_high_confidence", 0)
    cegis_no_cands     = stats.get("cegis_no_candidates", 0)
    cegis_registry     = stats.get("cegis_registry_hits", 0)

    metrics["cegis_runs"]   = cegis_ran
    metrics["cegis_proved"] = cegis_proved

    print(f"\n{'='*60}")
    print(f"RESULT: {correct}/{total} ({100*correct/total:.2f}%)")
    print(f"Elapsed: {elapsed_total:.1f}s  ({elapsed_total/total*1000:.0f}ms/q)")
    print(f"{'='*60}")

    print("\n── Guard 2: 5指標 ────────────────────────────────────────")
    step6_rate   = metrics["step6_reached"] / total
    proved_rate  = cegis_proved / max(1, cegis_loop_started)
    print(f"  1. step6_reach_rate:  {metrics['step6_reached']}/{total} = {step6_rate:.1%}")
    print(f"  2. cegis_entered:     {cegis_ran}  (関数入口)")
    print(f"     cegis_loop_start:  {cegis_loop_started}  (実ループ起動)")
    print(f"     cegis_no_cands:    {cegis_no_cands}  (候補空で止まった数)")
    print(f"     cegis_registry:    {cegis_registry}  (WORLDGEN_REGISTRY hits)")
    print(f"     cegis_iters:       {cegis_iters}  (総ループ回転数)")
    print(f"  3. proved_rate:       {cegis_proved}/{cegis_loop_started} = {proved_rate:.1%}")
    print(f"  4. missing_spec_cnt:  {metrics['missing_spec_count']}")
    print(f"  5. fail_tags:")
    for tag, cnt in sorted(metrics["fail_tags"].items(), key=lambda x: -x[1]):
        print(f"       {tag:30s} {cnt:4d}")

    # パイプライン詳細統計
    print(f"\n── Pipeline stats ────────────────────────────────────────")
    n = max(1, total)
    print(f"  IR extracted:    {stats['ir_extracted']}/{total} ({100*stats['ir_extracted']/n:.0f}%)")
    print(f"  Pieces found:    {stats['pieces_found']}/{total} ({100*stats['pieces_found']/n:.0f}%)")
    print(f"  Executed:        {stats['executed']}/{total} ({100*stats['executed']/n:.0f}%)")
    print(f"  Crystal hits:    {stats['crystal_hit']}")
    print(f"  Guard1 blocked:  {stats['cegis_missing_spec_blocked']}")
    print(f"  Halluc rejected: {stats['cegis_rejected_hallucination']}")

    # カテゴリ別
    print(f"\n── カテゴリ別正答数 ──────────────────────────────────────")
    cat_stats: dict = defaultdict(lambda: {"total": 0, "correct": 0})
    for i2, q in enumerate(questions):
        cat = q.get("category", "Unknown")
        cat_stats[cat]["total"] += 1
    # 再計算（正解は上ループで continue したので fail_log から逆算）
    fail_idxs = {f["idx"] for f in fail_log}
    for i2, q in enumerate(questions):
        cat = q.get("category", "Unknown")
        if i2 not in fail_idxs:
            cat_stats[cat]["correct"] += 1
    for cat, s in sorted(cat_stats.items(), key=lambda x: -x[1]["total"]):
        t, c = s["total"], s["correct"]
        pct = 100 * c / t if t > 0 else 0
        print(f"  {cat:30s} {c:3d}/{t:3d} ({pct:.0f}%)")

    # 失敗サンプル（各タグ1件）
    print(f"\n── 失敗サンプル（タグ別1件） ─────────────────────────────")
    shown_tags: set = set()
    for f in fail_log:
        if f["tag"] not in shown_tags:
            shown_tags.add(f["tag"])
            print(f"  [{f['tag']}]")
            print(f"    Q: {f['q']}")
            print(f"    expected={f['expected']!r}  got={f['answer']!r}  method={f['method']}")

    print(f"\n✅ Done — {correct}/{total} ({100*correct/total:.2f}%)")
    return correct, total


if __name__ == "__main__":
    main()
