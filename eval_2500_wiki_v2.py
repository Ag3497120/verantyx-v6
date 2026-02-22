"""
HLE 2500 評価 — Wikipedia v2 強化版（monkey-patch不使用）

改善点:
  - WikiKnowledgeFetcherV2（Summary API、lang-aware）
  - concept_extractor_v2 で抽出した概念を IR.missing に注入
  - monkey-patch不使用: decomposerをwrapして missing を追加
"""
import sys, os, json, time, signal, types
from collections import Counter

os.environ['DISABLE_CONCEPT_BOOST'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# KnowledgePipeline → v2 差し替え
import knowledge.knowledge_pipeline as kp_mod
from knowledge.knowledge_pipeline_v2 import KnowledgePipelineV2
kp_mod.KnowledgePipeline = KnowledgePipelineV2

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match
from decomposer.concept_extractor_v2 import extract_concepts_v2
from decomposer.decomposer import RuleBasedDecomposer

CHECKPOINT_FILE = 'hle_2500_wiki_v2_checkpoint.json'
RESULT_FILE = 'hle_2500_wiki_v2_eval.json'
LOG_FILE = 'eval_2500_wiki_v2.log'

print("Loading HLE 2500...")
questions = []
with open("hle_2500_eval.jsonl", 'r') as f:
    for line in f:
        questions.append(json.loads(line))
print(f"Loaded {len(questions)} questions")

per_problem = []
correct = 0
start_idx = 0
wiki_stats = {"concepts_extracted": 0, "extra_missing_added": 0}

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        cp = json.load(f)
    per_problem = cp.get('per_problem', [])
    correct = cp.get('correct', 0)
    start_idx = cp.get('next_idx', 0)
    wiki_stats = cp.get('wiki_stats', wiki_stats)
    print(f"[Resume] {start_idx}問からの再開: {correct}/{start_idx}")

print("Initializing pipeline (wiki v2)...")
pipeline = VerantyxV6Enhanced(
    piece_db_path="pieces/piece_db_empty.jsonl",
    use_llm_decomposer=False,
)
print(f"  Knowledge pipeline: {type(pipeline._knowledge_pipeline).__name__}")

# Decomposer を直接wrapして missing に extra concepts を注入
_orig_decompose = pipeline.decomposer.decompose

def patched_decompose(text):
    ir = _orig_decompose(text)
    # concept_extractor_v2 で追加概念を抽出
    try:
        extras = extract_concepts_v2(text)
        high_conf = [ec for ec in extras if ec.confidence >= 0.7]
        if high_conf:
            extra_missing = [
                {
                    "concept": ec.name.replace(" ", "_").lower(),
                    "kind": ec.kind,
                    "domain": ec.domain_hint,
                    "scope": "concise",
                }
                for ec in high_conf
            ]
            # 重複を除いて ir.missing に追加
            existing_concepts = {m.get("concept", "") for m in (ir.missing or [])}
            new_missing = [m for m in extra_missing if m["concept"] not in existing_concepts]
            if not hasattr(ir, 'missing') or ir.missing is None:
                ir.missing = []
            ir.missing = list(ir.missing) + new_missing
            wiki_stats["extra_missing_added"] += len(new_missing)
    except Exception:
        pass
    return ir

pipeline.decomposer.decompose = patched_decompose
print("  Decomposer patched with concept_extractor_v2")
print("Ready\n")

start_time = time.time()
total = len(questions)


def save_checkpoint(next_idx):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            'correct': correct, 'next_idx': next_idx,
            'per_problem': per_problem, 'wiki_stats': wiki_stats,
        }, f)


def save_result():
    elapsed = time.time() - start_time
    done = len(per_problem)
    with open(RESULT_FILE, 'w') as f:
        json.dump({
            'score': correct / total if total > 0 else 0,
            'correct': correct, 'total': total, 'done': done,
            'elapsed_s': elapsed, 'wiki_stats': wiki_stats,
            'per_problem': per_problem,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  結果保存: {correct}/{done} ({correct/done*100:.1f}% if done else 0.0%)")
    print(f"  Wiki stats: {wiki_stats}")


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
        wiki_stats["concepts_extracted"] += len(extract_concepts_v2(text))
        result = pipeline.solve(text, expected_answer=expected)
        pred = result.get('answer')
        is_correct = flexible_match(pred, expected) if pred and expected else False
    except Exception:
        pred = None
        is_correct = False

    if is_correct:
        correct += 1

    per_problem.append({
        'id': qid, 'correct': is_correct,
        'predicted': pred, 'expected': expected,
    })

    done = i + 1
    if done % 50 == 0:
        elapsed = time.time() - start_time
        rate = elapsed / (done - start_idx) if done > start_idx else 1
        eta = rate * (total - done)
        msg = (f"[{done}/{total}] {correct}/{done} ({correct/done*100:.1f}%)"
               f"  {elapsed:.0f}s elapsed  ETA {eta/60:.0f}min  ({rate:.1f}s/q)"
               f"  extra_missing={wiki_stats['extra_missing_added']}")
        print(msg)
        with open(LOG_FILE, 'a') as lf:
            lf.write(msg + '\n')
        save_checkpoint(done)

save_result()
print(f"\n✅ 完了: {correct}/{total} ({correct/total*100:.2f}%)")
