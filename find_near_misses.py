#!/usr/bin/env python3
"""Find near-miss tasks where engine output is close to correct."""
import json, os, sys, numpy as np

def get_failed_ids(log_path):
    failed = []
    with open(log_path) as f:
        for line in f:
            if '✗' in line:
                parts = line.split()
                for p in parts:
                    if len(p) == 8 and all(c in '0123456789abcdef' for c in p):
                        failed.append(p)
                        break
    return failed

def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'arc_v77.log'
    data_dir = '/tmp/arc-agi-2/data/training'
    failed = get_failed_ids(log_path)
    print(f"Scanning {len(failed)} failed tasks...")

    sys.path.insert(0, '.')
    from arc.cross_engine import solve_cross_engine

    near_misses = []
    for i, tid in enumerate(failed):
        path = os.path.join(data_dir, tid + '.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            task = json.load(f)

        train = [(ex['input'], ex['output']) for ex in task['train']]
        test_inputs = [ex['input'] for ex in task['test']]

        try:
            predictions, verified = solve_cross_engine(train, test_inputs)
        except Exception:
            continue

        if not any(p for p in predictions):
            continue

        total_diff = 0
        total_cells = 0
        for ti, test in enumerate(task.get('test', [])):
            expected = np.array(test['output'])
            if ti < len(predictions) and predictions[ti]:
                # Take best prediction
                best_diff = 9999
                for pred in predictions[ti]:
                    pa = np.array(pred)
                    if pa.shape == expected.shape:
                        d = int(np.sum(pa != expected))
                        best_diff = min(best_diff, d)
                total_diff += best_diff
                total_cells += expected.size
            else:
                total_diff += 9999
                total_cells += expected.size

        if 0 < total_diff <= 15:
            near_misses.append((tid, total_diff, total_cells))
            print(f"  NEAR-MISS: {tid} diff={total_diff}/{total_cells}", flush=True)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(failed)}] near-misses so far: {len(near_misses)}", flush=True)

    print(f"\n=== {len(near_misses)} near-misses (diff <= 15) ===")
    for tid, diff, cells in sorted(near_misses, key=lambda x: x[1]):
        print(f"  {tid}: diff={diff}/{cells}")

if __name__ == '__main__':
    main()
