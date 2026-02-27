"""Test DeepSeek on ver=0 tasks."""
import os, json, re, time, sys
from arc.grid import grid_eq, grid_shape
from arc.llm_deepseek import solve_task_deepseek

data_dir = "/tmp/arc-agi-2/data/training"

ver0 = []
with open("arc_v61_full.log") as f:
    for l in f:
        m = re.match(r'\s*\[(\d+)/1000\]\s+✗\s+[\d.]+s/t\s+(\w+)\s+ver=0', l)
        if m:
            ver0.append(m.group(2))

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-chat"
print(f"Testing {limit} ver=0 tasks with DeepSeek ({model})...")
print()

solved = 0
size_correct = 0
attempted = 0

for i, tid in enumerate(ver0[:limit]):
    path = os.path.join(data_dir, f"{tid}.json")
    with open(path) as f:
        data = json.load(f)
    
    train = [(ex['input'], ex['output']) for ex in data['train']]
    test_in = [ex['input'] for ex in data['test']]
    test_out = [ex.get('output') for ex in data['test']]
    
    t0 = time.time()
    preds = solve_task_deepseek(train, test_in, model=model)
    elapsed = time.time() - t0
    
    correct = False
    pred_shape = ""
    if preds and preds[0]:
        attempted += 1
        pred = preds[0][0]
        ph, pw = len(pred), len(pred[0]) if pred else 0
        eh, ew = grid_shape(test_out[0]) if test_out[0] else (0, 0)
        pred_shape = f"pred={ph}x{pw}"
        if ph == eh and pw == ew:
            size_correct += 1
        if test_out[0] and grid_eq(pred, test_out[0]):
            correct = True
            solved += 1
    
    h_in, w_in = grid_shape(train[0][0])
    h_out, w_out = grid_shape(train[0][1])
    status = "✓" if correct else "✗"
    
    print(f"  [{i+1}/{limit}] {status} {elapsed:.1f}s {tid} {h_in}x{w_in}→{h_out}x{w_out} {pred_shape}")

print()
print(f"{'='*60}")
print(f"Results: {solved}/{limit} ({solved/limit*100:.1f}%)")
print(f"Size correct: {size_correct}/{attempted}")
print(f"Attempted: {attempted}/{limit}")
