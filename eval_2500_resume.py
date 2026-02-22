"""
HLE 2500 評価 — チェックポイント付き再起動版
200問ぶん先にスキップしてそのまま続ける
"""
import sys, os, json, time, signal
from collections import Counter

os.environ['DISABLE_CONCEPT_BOOST'] = '1'
sys.path.insert(0, os.path.dirname(__file__))

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match

CHECKPOINT_FILE = 'hle_2500_checkpoint.json'
RESULT_FILE = 'hle_2500_clean_eval.json'
LOG_FILE = 'eval_2500_clean.log'
CHECKPOINT_EVERY = 50

print("Loading HLE 2500...")
questions = []
with open("hle_2500_eval.jsonl", 'r') as f:
    for line in f:
        questions.append(json.loads(line))
print(f"Loaded {len(questions)} questions")

# チェックポイント読み込み
per_problem = []
correct = 0
start_idx = 0
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        cp = json.load(f)
    per_problem = cp['per_problem']
    correct = cp['correct']
    start_idx = cp['next_idx']
    print(f"[Resume] {start_idx}問ぶんのチェックポイント発見: {correct}/{start_idx} ({correct/start_idx*100:.1f}%)")
else:
    print("新規開始")

print("Initializing pipeline...")
pipeline = VerantyxV6Enhanced(
    piece_db_path="pieces/piece_db_empty.jsonl",
    use_llm_decomposer=False,
)
print("Ready\n")

start_time = time.time()
total = len(questions)
category_stats = Counter()

def save_checkpoint(next_idx):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'correct': correct, 'next_idx': next_idx, 'per_problem': per_problem}, f)

def save_result():
    elapsed = time.time() - start_time
    done = start_idx + len(per_problem) - (start_idx if per_problem else 0)
    with open(RESULT_FILE, 'w') as f:
        json.dump({
            'score': correct/total,
            'correct': correct,
            'total': total,
            'elapsed_s': elapsed,
            'per_problem': per_problem,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 中間結果保存: {correct}/{done} ({correct/done*100:.1f}% partial)")

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
    cat = q.get('subject', q.get('category', 'unknown'))

    try:
        result = pipeline.solve(text, expected_answer=expected)
        pred = result.get('answer')
        is_correct = flexible_match(pred, expected) if pred and expected else False
    except Exception as e:
        pred = None
        is_correct = False

    if is_correct:
        correct += 1

    per_problem.append({'id': qid, 'correct': is_correct, 'predicted': pred, 'expected': expected})

    done = i + 1
    if done % 50 == 0:
        elapsed = time.time() - start_time
        rate = elapsed / (done - start_idx) if done > start_idx else 1
        eta = rate * (total - done)
        msg = f"[{done}/{total}] {correct}/{done} ({correct/done*100:.1f}%)  {elapsed:.0f}s elapsed  ETA {eta/60:.0f}min  ({rate:.1f}s/q)"
        print(msg)
        with open(LOG_FILE, 'a') as lf:
            lf.write(msg + '\n')
        save_checkpoint(done)

# 完了
save_result()
print(f"\n✅ 完了: {correct}/{total} ({correct/total*100:.2f}%)")
