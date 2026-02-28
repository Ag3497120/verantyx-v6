#!/usr/bin/env python3
"""
eval_v3_synth.py — 2-stage ARC-AGI-2 evaluation
Stage 1: Existing cross_engine (fast, ~244/1000)
Stage 2: DeepSeek V3 program synthesis on remaining tasks

V3 generates Python transform functions, verified against train examples.
LLM never directly generates answers.
"""

import json
import os
import sys
import time
import re
import signal
import numpy as np
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = "/tmp/arc-agi-2/data/training"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-1c9551e705dd4fbfbdcab991cc924526")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# Load stage 1 results
def load_stage1_results(log_path="arc_v80.log"):
    solved = set()
    with open(log_path) as f:
        for line in f:
            if '✓' in line:
                for p in line.split():
                    if len(p) == 8 and all(c in '0123456789abcdef' for c in p):
                        solved.add(p)
                        break
    return solved


def grid_to_text(grid):
    return '\n'.join(' '.join(str(c) for c in row) for row in grid)


def call_v3(prompt, temperature=0.0, max_tokens=8192, timeout=120):
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You solve ARC puzzles by writing Python. Output ONLY a ```python block with def transform(grid)."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    
    req = urllib.request.Request(DEEPSEEK_URL, data=payload, headers={
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    })
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return None


def extract_code(response):
    if not response:
        return None
    match = re.search(r'```python\s*\n(.*?)```', response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if 'def transform' in code:
            return code
    # Try without python tag
    match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if 'def transform' in code:
            return code
    return None


def execute_transform(code, input_grid, timeout_sec=5):
    ns = {'np': np, 'numpy': np, 'Counter': Counter, 'defaultdict': defaultdict}
    try:
        import scipy
        ns['scipy'] = scipy
        from scipy.ndimage import label
        ns['label'] = label
    except:
        pass
    
    try:
        exec(code, ns)
    except:
        return None
    
    if 'transform' not in ns:
        return None
    
    import threading
    result_box = [None]
    def run():
        try:
            result_box[0] = ns['transform'](input_grid)
        except:
            pass
    
    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        return None  # timed out
    
    if result_box[0] is None:
        return None
    try:
        return [[int(c) for c in row] for row in result_box[0]]
    except:
        return None


def solve_task_v3(task, n_attempts=3):
    """Try to solve task using V3 program synthesis. Returns test outputs or None."""
    train = task['train']
    
    examples = ''
    for i, ex in enumerate(train):
        examples += f'Example {i+1}:\nInput:\n{grid_to_text(ex["input"])}\nOutput:\n{grid_to_text(ex["output"])}\n\n'
    
    base_prompt = f"""Analyze these ARC puzzle input→output examples and find the transformation pattern.

{examples}Write def transform(grid): in a ```python block.
- grid is list[list[int]] (0-9 colors). Return list[list[int]].
- Find the GENERAL rule that works for ALL examples.
- Do NOT hardcode outputs.
- You may use numpy, scipy, collections."""
    
    for attempt in range(n_attempts):
        temp = attempt * 0.4
        
        if attempt > 0:
            prompt = base_prompt + f"\n\nPrevious attempt failed verification. Try a DIFFERENT approach."
        else:
            prompt = base_prompt
        
        response = call_v3(prompt, temperature=temp)
        code = extract_code(response)
        if not code:
            continue
        
        # Verify on ALL train examples
        all_ok = True
        for ex in train:
            pred = execute_transform(code, ex['input'])
            if pred is None or not np.array_equal(np.array(pred), np.array(ex['output'])):
                all_ok = False
                break
        
        if not all_ok:
            continue
        
        # Apply to test
        test_outputs = []
        for tex in task['test']:
            pred = execute_transform(code, tex['input'])
            if pred is None:
                all_ok = False
                break
            test_outputs.append(pred)
        
        if all_ok:
            return test_outputs
    
    return None


def process_one_task(args):
    """Process a single task (for parallel execution)."""
    tid, task = args
    t0 = time.time()
    result = solve_task_v3(task)
    dt = time.time() - t0
    
    if result:
        test_correct = True
        for j, tex in enumerate(task['test']):
            if not np.array_equal(np.array(result[j]), np.array(tex['output'])):
                test_correct = False
                break
        return tid, test_correct, dt
    return tid, None, dt


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    task_files = sorted(Path(DATA_DIR).glob("*.json"))
    all_tasks = {}
    for tf in task_files:
        tid = tf.stem
        with open(tf) as f:
            all_tasks[tid] = json.load(f)
    
    print(f"Total tasks: {len(all_tasks)}")
    
    stage1_log = "arc_v80.log"
    if os.path.exists(stage1_log):
        stage1_solved = load_stage1_results(stage1_log)
        print(f"Stage 1 (cross_engine): {len(stage1_solved)} solved")
    else:
        stage1_solved = set()
        print("No stage 1 results found")
    
    remaining = sorted(set(all_tasks.keys()) - stage1_solved)
    print(f"Stage 2 (V3 synth): {len(remaining)} tasks to attempt")
    print(f"Parallel workers: 4", flush=True)
    
    v3_solved = 0
    v3_attempted = 0
    v3_results = {}
    lock = threading.Lock()
    
    work = [(tid, all_tasks[tid]) for tid in remaining]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_one_task, w): w[0] for w in work}
        
        for future in as_completed(futures):
            tid, test_correct, dt = future.result()
            with lock:
                v3_attempted += 1
                if test_correct is True:
                    v3_solved += 1
                    tag = "✓"
                    v3_results[tid] = True
                elif test_correct is False:
                    tag = "✗_test"
                    v3_results[tid] = False
                else:
                    tag = "✗_synth"
                    v3_results[tid] = False
                
                total = len(stage1_solved) + v3_solved
                print(f"[{v3_attempted}/{len(remaining)}] {tag} {dt:.1f}s {tid} | V3: {v3_solved}/{v3_attempted} | Total: {total}/1000 ({total/10:.1f}%)", flush=True)
                
                # Save periodically
                if v3_attempted % 20 == 0:
                    with open("v3_synth_results.json", "w") as f:
                        json.dump({"stage1": list(stage1_solved), "v3_results": v3_results, "total": total}, f)
    
    total = len(stage1_solved) + v3_solved
    print(f"\n{'='*60}")
    print(f"Stage 1 (cross_engine): {len(stage1_solved)}")
    print(f"Stage 2 (V3 synth):     {v3_solved}/{v3_attempted}")
    print(f"TOTAL:                  {total}/1000 ({total/10:.1f}%)")
    print(f"{'='*60}")
    
    with open("v3_synth_results.json", "w") as f:
        json.dump({"stage1": list(stage1_solved), "v3_results": v3_results, "total": total}, f)


if __name__ == "__main__":
    main()
