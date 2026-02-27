"""Test LLM router: classify ver=0 tasks and analyze distribution."""
import os, json, re, sys, time
from collections import Counter
from arc.llm_router import classify_task, route_to_phases
from arc.grid import grid_shape

data_dir = "/tmp/arc-agi-2/data/training"

ver0 = []
with open("arc_v61_full.log") as f:
    for l in f:
        m = re.match(r'\s*\[(\d+)/1000\]\s+✗\s+[\d.]+s/t\s+(\w+)\s+ver=0', l)
        if m:
            ver0.append(m.group(2))

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
print(f"Classifying {limit} ver=0 tasks...")
print()

primary_counts = Counter()
secondary_counts = Counter()
results = {}

for i, tid in enumerate(ver0[:limit]):
    path = os.path.join(data_dir, f"{tid}.json")
    with open(path) as f:
        data = json.load(f)
    
    train = [(ex['input'], ex['output']) for ex in data['train']]
    h_in, w_in = grid_shape(train[0][0])
    h_out, w_out = grid_shape(train[0][1])
    
    t0 = time.time()
    cls = classify_task(train)
    elapsed = time.time() - t0
    
    if cls:
        p = cls.get('primary', '?')
        s = cls.get('secondary', '')
        conf = cls.get('confidence', 0)
        reason = cls.get('reasoning', '')[:60]
        primary_counts[p] += 1
        if s:
            secondary_counts[str(s)] += 1
        results[tid] = cls
        phases = route_to_phases(cls)
        print(f"  [{i+1}/{limit}] {tid} {h_in}x{w_in}→{h_out}x{w_out} | {p} (conf={conf}) → {phases[:2]}")
    else:
        print(f"  [{i+1}/{limit}] {tid} {h_in}x{w_in}→{h_out}x{w_out} | FAILED")

print()
print("=" * 60)
print("Primary classification distribution:")
for k, v in primary_counts.most_common():
    print(f"  {k:30s} {v:3d} ({v/limit*100:.0f}%)")
print()
print("This tells us which Verantyx primitives need strengthening")
print("to solve ver=0 tasks.")

# Save
with open("llm_classifications_sample.json", "w") as f:
    json.dump(results, f, indent=2)
