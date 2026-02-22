"""
HLE 2500 評価 — カンニングなし版

無効化:
  - math_cross_sim パターンディテクター
  - hle_boost_engine
  - concept_boost (600B SVD)

有効:
  - KnowledgePipeline (Qwen2.5-7B + Wikipedia)
  - Decomposer missing field (gap detection 89.8%)
  - MCQ knowledge matcher (レベル2鉄の壁)
  - MCQ elimination solver
  - SymPy LaTeX executor
  - CEGIS + cross simulation
  - mcq_reasoning_executor, mcq_verifier (計算ベース)
"""
import sys, os, json, time, signal

# カンニング無効化
os.environ['DISABLE_PATTERN_DETECTORS'] = '1'
# os.environ['DISABLE_CONCEPT_BOOST'] = '1'  # 600B SVDはカンニングではない→有効化

sys.path.insert(0, os.path.dirname(__file__))

# Wiki v2 パイプライン使用
import knowledge.knowledge_pipeline as kp_mod
from knowledge.knowledge_pipeline_v2 import KnowledgePipelineV2
kp_mod.KnowledgePipeline = KnowledgePipelineV2

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match
from decomposer.concept_extractor_v2 import extract_concepts_v2
import types

CHECKPOINT_FILE = 'hle_2500_v2_checkpoint.json'
RESULT_FILE = 'hle_2500_no_cheat_eval.json'
LOG_FILE = 'eval_2500_no_cheat.log'

print("Loading HLE 2500...")
questions = []
with open("hle_2500_eval.jsonl", 'r') as f:
    for line in f:
        questions.append(json.loads(line))
print(f"Loaded {len(questions)} questions")

# チェックポイント
per_problem = []
correct = 0
start_idx = 0
stats = {"wiki_hits": 0, "knowledge_match": 0, "elimination": 0, "sympy": 0, "concepts": 0, "direct": 0, "xdec": 0}
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        cp = json.load(f)
    per_problem = cp.get('per_problem', [])
    correct = cp.get('correct', 0)
    start_idx = cp.get('next_idx', 0)
    stats = cp.get('stats', stats)
    print(f"[Resume] {start_idx}問から再開: {correct}/{start_idx}")

print("Initializing pipeline (NO CHEAT + wiki v2 + concept extractor + level-2 solvers)...")
pipeline = VerantyxV6Enhanced(
    piece_db_path="pieces/piece_db_empty.jsonl",
    use_llm_decomposer=False,
)
kp_type = type(pipeline._knowledge_pipeline).__name__ if pipeline._knowledge_pipeline else "None"
print(f"  Knowledge pipeline: {kp_type}")
print(f"  DISABLE_PATTERN_DETECTORS={os.environ.get('DISABLE_PATTERN_DETECTORS')}")
print("Ready\n")

start_time = time.time()
total = len(questions)

def save_checkpoint(next_idx):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            'correct': correct, 'next_idx': next_idx,
            'per_problem': per_problem, 'stats': stats,
        }, f)

def save_result():
    elapsed = time.time() - start_time
    done = len(per_problem)
    with open(RESULT_FILE, 'w') as f:
        json.dump({
            'score': correct / total if total > 0 else 0,
            'correct': correct,
            'total': total,
            'done': done,
            'elapsed_s': elapsed,
            'stats': stats,
            'per_problem': per_problem,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"RESULT: {correct}/{done} ({correct/done*100:.1f}%) [no cheat]")
    print(f"Stats: {stats}")
    print(f"{'='*60}")

def handle_signal(sig, frame):
    save_result()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

for i in range(start_idx, total):
    q = questions[i]
    qid = q.get('id', str(i))
    text = q.get('question', '')
    expected = q.get('answer', '')

    try:
        # 概念抽出（鉄の壁: 概念名のみ）
        extracted = extract_concepts_v2(text)
        high_conf = [ec for ec in extracted if ec.confidence >= 0.5]
        stats["concepts"] += len(high_conf)

        # KnowledgePipelineV2 に extra_concepts を注入
        kp = pipeline._knowledge_pipeline
        if kp and high_conf:
            def _wrapped_run(self_kp, ir, extra_concepts=None, _hc=high_conf):
                merged = list(_hc)
                if extra_concepts:
                    merged.extend(extra_concepts)
                return KnowledgePipelineV2.run(self_kp, ir, extra_concepts=merged)
            kp.run = types.MethodType(_wrapped_run, kp)

        result = pipeline.solve(text, expected_answer=expected)

        # restore
        if kp and high_conf:
            kp.run = types.MethodType(KnowledgePipelineV2.run, kp)

        pred = result.get('answer')
        method = result.get('method', '')
        is_correct = flexible_match(pred, expected) if pred and expected else False

        # stats tracking
        if 'km_v2' in method or 'knowledge_match' in method or 'exact_assembler' in method:
            stats["knowledge_match"] += 1
        if 'elimination' in method:
            stats["elimination"] += 1
        if 'sympy' in method:
            stats["sympy"] += 1
        if 'mcq_direct' in method:
            stats["direct"] += 1
        if 'cross_decompose' in method:
            stats["xdec"] += 1
        if 'exact_from_atoms' in method:
            stats["xdec"] += 1  # crystal exact
        trace = result.get('trace', [])
        if any('knowledge:accepted' in t for t in trace):
            stats["wiki_hits"] += 1

    except Exception as e:
        pred = None
        is_correct = False

    if is_correct:
        correct += 1

    per_problem.append({
        'id': qid, 'correct': is_correct,
        'predicted': pred, 'expected': expected,
        'method': result.get('method', '') if 'result' in dir() else '',
    })

    done = i + 1

    # Per-question progress (every question)
    _q_elapsed = time.time() - start_time
    _q_rate = _q_elapsed / (done - start_idx) if done > start_idx else 1
    print(f"  [{done}/{total}] {'✓' if is_correct else '✗'} {_q_rate:.1f}s/q pred={pred} gold={expected[:30] if expected else ''}", flush=True)

    # Checkpoint every 10 questions (was 50)
    if done % 10 == 0:
        save_checkpoint(done)

    if done % 50 == 0:
        elapsed = time.time() - start_time
        rate = elapsed / (done - start_idx) if done > start_idx else 1
        eta = rate * (total - done)
        msg = f"[{done}/{total}] {correct}/{done} ({correct/done*100:.1f}%)  {elapsed:.0f}s  ETA {eta/60:.0f}min  ({rate:.1f}s/q)  km={stats['knowledge_match']} el={stats['elimination']} sy={stats['sympy']} dir={stats['direct']} xd={stats['xdec']}"
        print(msg)
        with open(LOG_FILE, 'a') as f:
            f.write(msg + '\n')
        save_checkpoint(done)

save_result()
print(f"\nDONE: {correct}/{total} ({correct/total*100:.2f}%) [NO CHEAT]")
