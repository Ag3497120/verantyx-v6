#!/usr/bin/env python3
"""Vote-based verification: run multiple transform candidates on test input, pick majority."""
import json, os, sys, importlib.util
from collections import Counter

def grid_to_tuple(grid):
    return tuple(tuple(row) for row in grid)

def run_transform(code_path, grid):
    try:
        spec = importlib.util.spec_from_file_location("mod", code_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = mod.transform(grid)
        return result
    except:
        return None

def vote_on_task(task_path, candidate_paths):
    """Run all candidates on test inputs, return majority vote output."""
    with open(task_path) as f:
        task = json.load(f)
    
    results = []
    for test_ex in task['test']:
        outputs = []
        for cp in candidate_paths:
            out = run_transform(cp, test_ex['input'])
            if out is not None:
                outputs.append(grid_to_tuple(out))
        
        if not outputs:
            results.append(None)
            continue
        
        counter = Counter(outputs)
        best, count = counter.most_common(1)[0]
        # Only accept if majority (>1 vote for same output)
        results.append({
            'output': [list(row) for row in best],
            'votes': count,
            'total': len(outputs),
            'consensus': count / len(outputs)
        })
    
    return results

if __name__ == '__main__':
    eval_dir = '/tmp/arc-agi-2/data/evaluation'
    base_dir = os.path.expanduser('~/verantyx_v6/eval_synth_multi')
    
    tasks = sorted([f.replace('.json','') for f in os.listdir(eval_dir) if f.endswith('.json')])
    
    correct = 0
    total = 0
    high_consensus = 0
    
    for tid in tasks:
        task_path = f'{eval_dir}/{tid}.json'
        # Find all candidates: eval_synth_multi/{tid}_v{0,1,2}.py
        candidates = []
        for v in range(5):
            cp = f'{base_dir}/{tid}_v{v}.py'
            if os.path.exists(cp):
                candidates.append(cp)
        # Also check single version
        single = os.path.expanduser(f'~/verantyx_v6/eval_synth_results/{tid}.py')
        if os.path.exists(single):
            candidates.append(single)
        
        if not candidates:
            continue
        
        total += 1
        with open(task_path) as f:
            task = json.load(f)
        
        results = vote_on_task(task_path, candidates)
        
        all_correct = True
        consensus_sum = 0
        for i, (res, test_ex) in enumerate(zip(results, task['test'])):
            if res is None:
                all_correct = False
                break
            if res['output'] != test_ex['output']:
                all_correct = False
            consensus_sum += res['consensus']
        
        avg_consensus = consensus_sum / len(results) if results else 0
        if avg_consensus > 0.5:
            high_consensus += 1
        
        if all_correct:
            correct += 1
            print(f'✓ {tid} (consensus: {avg_consensus:.0%})')
        else:
            print(f'✗ {tid} (consensus: {avg_consensus:.0%}, candidates: {len(candidates)})')
    
    print(f'\nResults: {correct}/{total} correct ({100*correct/max(1,total):.1f}%)')
    print(f'High consensus (>50%): {high_consensus}/{total}')
    print(f'Score: {correct}/120 ({100*correct/120:.1f}%)')
