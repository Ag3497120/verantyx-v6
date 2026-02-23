"""
HLE 2500 並列評価 — ProcessPool版

各ワーカーが独立したpipelineインスタンスを持つ。
チェックポイントはメインプロセスが管理。
"""
import sys, os, json, time, signal
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ['DISABLE_PATTERN_DETECTORS'] = '1'
# os.environ['DISABLE_CONCEPT_BOOST'] = '1'

sys.path.insert(0, os.path.dirname(__file__))

NUM_WORKERS = int(os.environ.get('EVAL_WORKERS', '4'))
CHECKPOINT_FILE = 'hle_2500_parallel_checkpoint.json'
RESULT_FILE = 'hle_2500_parallel_eval.json'

# ── Worker function (runs in subprocess) ──

_worker_pipeline = None
_worker_matcher = None
_worker_extractor = None

def _init_worker():
    """Initialize pipeline once per worker process."""
    global _worker_pipeline, _worker_matcher, _worker_extractor
    import os, sys
    os.environ['DISABLE_PATTERN_DETECTORS'] = '1'
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import knowledge.knowledge_pipeline as kp_mod
    from knowledge.knowledge_pipeline_v2 import KnowledgePipelineV2
    kp_mod.KnowledgePipeline = KnowledgePipelineV2

    from pipeline_enhanced import VerantyxV6Enhanced
    from core.answer_matcher import flexible_match
    from decomposer.concept_extractor_v2 import extract_concepts_v2

    _worker_pipeline = VerantyxV6Enhanced(
        piece_db_path="pieces/piece_db_empty.jsonl",
        use_llm_decomposer=False,
    )
    _worker_matcher = flexible_match
    _worker_extractor = extract_concepts_v2


def _solve_one(args):
    """Solve a single question in a subprocess."""
    global _worker_pipeline, _worker_matcher, _worker_extractor
    idx, q_json = args
    import json, types

    from knowledge.knowledge_pipeline_v2 import KnowledgePipelineV2

    pipeline = _worker_pipeline
    flexible_match = _worker_matcher
    extract_concepts_v2 = _worker_extractor

    q = json.loads(q_json)
    text = q.get('question', '')
    expected = q.get('answer', '')

    try:
        extracted = extract_concepts_v2(text)
        high_conf = [ec for ec in extracted if ec.confidence >= 0.5]

        kp = pipeline._knowledge_pipeline
        if kp:
            # BUG FIX: クロージャで high_conf を捕捉するのではなく、
            # 問題ごとに fresh な concepts を渡す
            _original_run = KnowledgePipelineV2.run
            def _wrapped_run(self_kp, ir, extra_concepts=None, _hc=high_conf):
                merged = list(_hc)
                if extra_concepts:
                    merged.extend(extra_concepts)
                return _original_run(self_kp, ir, extra_concepts=merged)
            kp.run = types.MethodType(_wrapped_run, kp)

        result = pipeline.solve(text, expected_answer=expected)
        pred = result.get('answer')
        method = result.get('method', '')
        trace = result.get('trace', [])
        is_correct = flexible_match(pred, expected) if pred and expected else False

    except Exception as e:
        pred = None
        method = f"error:{e}"
        trace = []
        is_correct = False

    return {
        'idx': idx,
        'pred': pred,
        'expected': expected,
        'is_correct': is_correct,
        'method': method,
        'wiki_hit': any('knowledge:accepted' in t or 'wiki_hits=' in t for t in trace),
        'concepts': len([ec for ec in extract_concepts_v2(text) if ec.confidence >= 0.5]) if text else 0,
    }


def main():
    print(f"Loading HLE 2500 (workers={NUM_WORKERS})...")
    questions = []
    with open("hle_2500_eval.jsonl", 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    total = len(questions)
    print(f"Loaded {total} questions")

    # Checkpoint
    done_set = set()
    per_problem = []
    correct = 0
    stats = {"wiki_hits": 0, "knowledge_match": 0, "elimination": 0,
             "sympy": 0, "concepts": 0, "direct": 0, "xdec": 0}

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            cp = json.load(f)
        per_problem = cp.get('per_problem', [])
        correct = cp.get('correct', 0)
        stats = cp.get('stats', stats)
        done_set = {p['idx'] for p in per_problem}
        print(f"[Resume] {len(done_set)} done, {correct} correct")

    # Build work items (skip already done)
    work = []
    for i, q in enumerate(questions):
        if i not in done_set:
            work.append((i, json.dumps(q, ensure_ascii=False)))

    print(f"Submitting {len(work)} problems to {NUM_WORKERS} workers...")
    start_time = time.time()

    checkpoint_interval = 10  # save every N completions
    completed_since_save = 0

    def save_checkpoint():
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({
                'correct': correct,
                'next_idx': len(per_problem),
                'per_problem': per_problem,
                'stats': stats,
            }, f)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=_init_worker) as pool:
        futures = {pool.submit(_solve_one, w): w[0] for w in work}

        for future in as_completed(futures):
            try:
                r = future.result(timeout=120)
            except Exception as e:
                idx = futures[future]
                r = {'idx': idx, 'pred': None, 'expected': '', 'is_correct': False,
                     'method': f'timeout:{e}', 'wiki_hit': False, 'concepts': 0}

            idx = r['idx']
            is_correct = r['is_correct']
            pred = r['pred']
            expected = r['expected']
            method = r['method']

            if is_correct:
                correct += 1

            # Stats
            if r['wiki_hit']:
                stats['wiki_hits'] += 1
            stats['concepts'] += r.get('concepts', 0)
            if 'km_v2' in method or 'knowledge_match' in method or 'exact_assembler' in method:
                stats['knowledge_match'] += 1
            if 'elimination' in method:
                stats['elimination'] += 1
            if 'sympy' in method:
                stats['sympy'] += 1
            if 'mcq_direct' in method:
                stats['direct'] += 1
            if 'cross_decompose' in method or 'exact_from_atoms' in method:
                stats['xdec'] += 1

            per_problem.append({
                'idx': idx, 'pred': pred, 'expected': expected,
                'correct': is_correct, 'method': method,
            })

            done = len(per_problem)
            elapsed = time.time() - start_time
            spq = elapsed / done if done else 0
            remaining = (total - done - len(done_set)) * spq
            mark = "✓" if is_correct else "✗"
            print(f"  [{done+len(done_set)}/{total}] {mark} {spq:.1f}s/q pred={pred} gold={expected[:30]}")

            completed_since_save += 1
            if completed_since_save >= checkpoint_interval:
                save_checkpoint()
                completed_since_save = 0
                done_total = done + len(done_set)
                rate = correct / done_total * 100 if done_total else 0
                eta_min = remaining / 60
                print(f"[{done_total}/{total}] {correct}/{done_total} ({rate:.1f}%)  ETA {eta_min:.0f}min  ({spq:.1f}s/q)  km={stats['knowledge_match']} el={stats['elimination']} sy={stats['sympy']} dir={stats['direct']} xd={stats['xdec']}")

    save_checkpoint()

    # Final report
    done_total = len(per_problem) + len(done_set)
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"RESULT: {correct}/{done_total} ({correct/done_total*100:.1f}%) [parallel, {NUM_WORKERS} workers]")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Stats: {stats}")
    print(f"{'='*60}")

    with open(RESULT_FILE, 'w') as f:
        json.dump({
            'score': correct / total if total > 0 else 0,
            'correct': correct, 'total': total,
            'done': done_total, 'elapsed_s': elapsed,
            'stats': stats, 'per_problem': per_problem,
        }, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
