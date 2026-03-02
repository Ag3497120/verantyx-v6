#!/usr/bin/env python3
"""
A/B Test: DeepSeek V3 synth WITH hints vs WITHOUT hints
"""
import json, os, sys, time, re, urllib.request, importlib.util, tempfile, signal
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from arc.hint_generator import generate_hints
from arc.grid import grid_eq

API_KEY = "sk-1c9551e705dd4fbfbdcab991cc924526"
API_URL = "https://api.deepseek.com/chat/completions"
DATA_DIR = Path("/tmp/arc-agi-2/data/training")


def call_deepseek(prompt, temp=0.0, max_tokens=4096, timeout=120):
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You solve ARC-AGI puzzles by writing Python. Output ONLY a ```python block with def transform(grid: list[list[int]]) -> list[list[int]]. No numpy. grid is list of lists of ints 0-9."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temp,
    }).encode()
    
    req = urllib.request.Request(API_URL, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    })
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return None


def extract_fn(text):
    if not text:
        return None
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    code = m.group(1) if m else text
    if 'def transform' not in code:
        return None
    try:
        ns = {}
        exec(code, ns)
        return ns.get('transform')
    except:
        return None


def grid_str(g):
    return '\n'.join(' '.join(str(c) for c in row) for row in g)


def build_prompt(task, hints=""):
    parts = ["Solve this ARC-AGI puzzle. Find the pattern from training examples and write a Python function.\n"]
    
    for i, ex in enumerate(task['train']):
        parts.append(f"Training {i+1} Input ({len(ex['input'])}x{len(ex['input'][0])}):")
        parts.append(grid_str(ex['input']))
        parts.append(f"Training {i+1} Output ({len(ex['output'])}x{len(ex['output'][0])}):")
        parts.append(grid_str(ex['output']))
        parts.append("")
    
    if hints:
        parts.append("=== ANALYSIS HINTS (from structural analysis) ===")
        parts.append(hints)
        parts.append("=== END HINTS ===")
        parts.append("Use these hints to guide your solution.\n")
    
    test_inp = task['test'][0]['input']
    parts.append(f"Test Input ({len(test_inp)}x{len(test_inp[0])}):")
    parts.append(grid_str(test_inp))
    
    return '\n'.join(parts)


def test_fn(fn, examples):
    """Verify function passes all training examples"""
    import copy
    for ex in examples:
        try:
            result = fn(copy.deepcopy(ex['input']))
            if not grid_eq(result, ex['output']):
                return False
        except:
            return False
    return True


def to_list(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, list):
        return [to_list(x) for x in obj]
    return obj


# Find unsolved tasks
solved = set()
log_path = Path(os.path.expanduser("~/verantyx_v6/arc_cross_engine_v9.log"))
with open(log_path) as f:
    for line in f:
        m = re.search(r'✓.*?([0-9a-f]{8})', line)
        if m:
            solved.add(m.group(1))

synth_dir = Path(os.path.expanduser("~/verantyx_v6/synth_results"))
if synth_dir.exists():
    for f in synth_dir.glob("*.py"):
        solved.add(f.stem)

# Pick 20 unsolved tasks
unsolved = []
for f in sorted(DATA_DIR.glob("*.json")):
    tid = f.stem
    if tid not in solved:
        unsolved.append(tid)
    if len(unsolved) >= 20:
        break

print(f"Testing {len(unsolved)} unsolved tasks with DeepSeek V3")
print(f"Format: WITH hints vs WITHOUT hints")
print("=" * 60)

with_h = 0
without_h = 0
total = 0

for tid in unsolved:
    with open(DATA_DIR / f"{tid}.json") as f:
        task = json.load(f)
    
    test_output = task['test'][0].get('output')
    if test_output is None:
        continue
    
    total += 1
    
    # Generate hints
    t0 = time.time()
    hints = generate_hints(task, include_partial=True)
    hint_time = time.time() - t0
    
    # WITH hints
    prompt_h = build_prompt(task, hints=hints)
    resp_h = call_deepseek(prompt_h)
    fn_h = extract_fn(resp_h)
    pass_h = False
    if fn_h and test_fn(fn_h, task['train']):
        try:
            pred = to_list(fn_h(task['test'][0]['input']))
            if grid_eq(pred, test_output):
                pass_h = True
                with_h += 1
        except:
            pass
    
    # WITHOUT hints
    prompt_n = build_prompt(task, hints="")
    resp_n = call_deepseek(prompt_n)
    fn_n = extract_fn(resp_n)
    pass_n = False
    if fn_n and test_fn(fn_n, task['train']):
        try:
            pred = to_list(fn_n(task['test'][0]['input']))
            if grid_eq(pred, test_output):
                pass_n = True
                without_h += 1
        except:
            pass
    
    h_mark = "✓" if pass_h else "✗"
    n_mark = "✓" if pass_n else "✗"
    delta = ""
    if pass_h and not pass_n:
        delta = " ← HINT WIN"
    elif pass_n and not pass_h:
        delta = " ← NO-HINT WIN"
    
    print(f"  [{total:2d}] {tid}  hints={h_mark}  no-hints={n_mark}  ({hint_time:.1f}s){delta}")

print("=" * 60)
print(f"WITH hints:    {with_h}/{total} ({with_h/max(total,1)*100:.0f}%)")
print(f"WITHOUT hints: {without_h}/{total} ({without_h/max(total,1)*100:.0f}%)")
print(f"Delta:         {with_h - without_h:+d}")
