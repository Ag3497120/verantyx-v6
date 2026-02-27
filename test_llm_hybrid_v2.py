"""Test hybrid LLM solver (direct + hypothesis) on ver=0 tasks."""
import os, json, re, time, sys
from arc.grid import grid_eq, grid_shape
from arc.llm_direct import solve_hybrid

data_dir = "/tmp/arc-agi-2/data/training"

# Get ver=0 IDs
ver0 = []
with open("arc_v61_full.log") as f:
    for l in f:
        m = re.match(r'\s*\[(\d+)/1000\]\s+✗\s+[\d.]+s/t\s+(\w+)\s+ver=0', l)
        if m:
            ver0.append(m.group(2))

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
print(f"Testing {limit} ver=0 tasks with hybrid LLM solver...")
print()

solved = 0
direct_correct = 0
hyp_correct = 0

for i, tid in enumerate(ver0[:limit]):
    path = os.path.join(data_dir, f"{tid}.json")
    with open(path) as f:
        data = json.load(f)
    
    train = [(ex['input'], ex['output']) for ex in data['train']]
    test_in = [ex['input'] for ex in data['test']]
    test_out = [ex.get('output') for ex in data['test']]
    
    t0 = time.time()
    preds, method = solve_hybrid(train, test_in)
    elapsed = time.time() - t0
    
    correct = False
    if preds and preds[0]:
        for pred in preds[0]:
            if test_out[0] and grid_eq(pred, test_out[0]):
                correct = True
                break
    
    if correct:
        solved += 1
        if 'direct' in method:
            direct_correct += 1
        else:
            hyp_correct += 1
    
    status = "✓" if correct else "✗"
    
    # Show size info
    h_in, w_in = grid_shape(train[0][0])
    h_out, w_out = grid_shape(train[0][1])
    pred_shape = ""
    if preds and preds[0]:
        ph, pw = grid_shape(preds[0][0])
        pred_shape = f"pred={ph}x{pw}"
    
    print(f"  [{i+1}/{limit}] {status} {elapsed:.1f}s {tid} {h_in}x{w_in}→{h_out}x{w_out} method={method} {pred_shape}")

print()
print(f"{'='*60}")
print(f"Results: {solved}/{limit} ({solved/limit*100:.1f}%)")
print(f"  Direct LLM correct: {direct_correct}")
print(f"  Hypothesis correct: {hyp_correct}")
