#!/usr/bin/env python3
"""Quick demo: solve 50 ARC-AGI-2 tasks live."""
import json, os, time, sys

# Star banner
print("\033[1;33m" + "=" * 60)
print("  ✨ Verantyx-v6 — ARC-AGI-2 Solver")
print("  🎯 Current Score: 196/1000 (19.6%)")
print("  📦 No cheats • No bias • No hardcode")
print("  🔗 github.com/Ag3497120/verantyx-v6")
print("=" * 60 + "\033[0m\n")

time.sleep(1)

from arc.cross_engine import solve_cross_engine
from arc.grid import grid_eq

data_dir = '/tmp/arc-agi-2/data/training'
tasks = sorted(os.listdir(data_dir))[:50]

solved = 0
total = 0
t0 = time.time()

for fname in tasks:
    tid = fname.replace('.json', '')
    with open(f'{data_dir}/{fname}') as f:
        task = json.load(f)
    
    train = [(p['input'], p['output']) for p in task['train']]
    test_inp = [p['input'] for p in task['test']]
    test_out = [p['output'] for p in task['test']]
    
    total += 1
    result, verified = solve_cross_engine(train, test_inp)
    
    ok = False
    if result:
        ok = all(any(grid_eq(c, test_out[i]) for c in result[i]) for i in range(len(test_out)))
    
    elapsed = time.time() - t0
    
    if ok:
        solved += 1
        method = verified[0][1].name if verified and hasattr(verified[0][1], 'name') else '?'
        print(f"\033[1;32m  [{total:3d}/50] ✓ {tid} — {method}\033[0m")
    else:
        print(f"\033[90m  [{total:3d}/50] ✗ {tid}\033[0m")
    
    sys.stdout.flush()

elapsed = time.time() - t0
print(f"\n\033[1;36m{'=' * 60}")
print(f"  Results: {solved}/{total} ({solved/total*100:.1f}%)")
print(f"  Time: {elapsed:.1f}s ({elapsed/total:.2f}s/task)")
print(f"{'=' * 60}\033[0m")
