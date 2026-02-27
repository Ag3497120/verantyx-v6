"""
Test LLM hybrid solver on ver=0 tasks.
"""
import os
import sys
import json
import re
import time

# Get ver=0 task IDs from log
def get_ver0_ids(log_path="arc_v61_full.log"):
    ids = []
    with open(log_path) as f:
        for l in f:
            m = re.match(r'\s*\[(\d+)/1000\]\s+✗\s+[\d.]+s/t\s+(\w+)\s+ver=0', l)
            if m:
                ids.append(m.group(2))
    return ids

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    data_dir = "/tmp/arc-agi-2/data/training"
    
    ver0_ids = get_ver0_ids()
    print(f"Total ver=0 tasks: {len(ver0_ids)}")
    print(f"Testing first {limit}...")
    print()
    
    from arc.llm_hypothesis import solve_with_llm_hypothesis, generate_hypothesis
    from arc.grid import grid_eq
    
    solved = 0
    hypothesis_generated = 0
    
    for i, tid in enumerate(ver0_ids[:limit]):
        task_path = os.path.join(data_dir, f"{tid}.json")
        if not os.path.exists(task_path):
            continue
        
        with open(task_path) as f:
            data = json.load(f)
        
        train_pairs = [(ex['input'], ex['output']) for ex in data['train']]
        test_inputs = [ex['input'] for ex in data['test']]
        test_outputs = [ex.get('output') for ex in data['test']]
        
        t0 = time.time()
        predictions, verified, hypothesis = solve_with_llm_hypothesis(
            train_pairs, test_inputs)
        elapsed = time.time() - t0
        
        # Check correctness
        correct = False
        if predictions and predictions[0]:
            for pred in predictions[0]:
                if test_outputs[0] and grid_eq(pred, test_outputs[0]):
                    correct = True
                    break
        
        status = "✓" if correct else "✗"
        if correct:
            solved += 1
        
        cat = hypothesis.get('category', '?') if hypothesis else 'NO_HYPOTHESIS'
        desc = hypothesis.get('description', '')[:80] if hypothesis else ''
        n_verified = len(verified) if verified else 0
        
        if hypothesis:
            hypothesis_generated += 1
        
        print(f"  [{i+1}/{limit}] {status} {elapsed:.1f}s {tid} cat={cat} verified={n_verified}")
        if desc:
            print(f"           {desc}")
        if correct and verified:
            print(f"           RULE: {verified[0][1].name}")
        print()
    
    print(f"=" * 60)
    print(f"Results: {solved}/{limit} solved ({solved/limit*100:.1f}%)")
    print(f"Hypotheses generated: {hypothesis_generated}/{limit}")


if __name__ == "__main__":
    main()
