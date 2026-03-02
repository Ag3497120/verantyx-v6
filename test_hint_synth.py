"""
Test: ヒント付きsynth vs ヒントなしsynth の比較
未解決10問で試す
"""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arc.hint_generator import generate_hints
from arc.opus_simulator_pipeline import (
    format_task_prompt, call_opus, extract_code, load_transform,
    check_invariants, score_candidate, to_list, grid_eq
)

DATA_DIR = Path("/tmp/arc-agi-2/data/training")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

if not API_KEY:
    print("Set ANTHROPIC_API_KEY")
    sys.exit(1)

# Find unsolved tasks (ver=0 in log)
import re
solved = set()
with open(os.path.expanduser("~/verantyx_v6/arc_cross_engine_v9.log")) as f:
    for line in f:
        m = re.search(r'✓.*?([0-9a-f]{8})', line)
        if m:
            solved.add(m.group(1))

# Also exclude synth-solved
synth_dir = Path(os.path.expanduser("~/verantyx_v6/synth_results"))
if synth_dir.exists():
    for f in synth_dir.glob("*.py"):
        solved.add(f.stem)

# Pick 10 unsolved tasks
unsolved = []
for f in sorted(DATA_DIR.glob("*.json")):
    tid = f.stem
    if tid not in solved:
        unsolved.append(tid)
    if len(unsolved) >= 10:
        break

print(f"Testing {len(unsolved)} unsolved tasks")
print(f"Using model: claude-sonnet-4-5-20250514")
print("="*60)

results = {"with_hints": 0, "without_hints": 0, "total": 0}

for tid in unsolved:
    with open(DATA_DIR / f"{tid}.json") as f:
        task = json.load(f)
    
    train_pairs = [(ex['input'], ex['output']) for ex in task['train']]
    test_input = task['test'][0]['input']
    test_output = task['test'][0].get('output')
    
    if test_output is None:
        continue
    
    results["total"] += 1
    
    # Generate hints
    hints = generate_hints(task, include_partial=True)
    
    # Try WITH hints
    prompt_h = format_task_prompt(train_pairs, test_input, "default", hints=hints)
    try:
        resp_h = call_opus(prompt_h, API_KEY, model="claude-sonnet-4-5-20250514")
        code_h = extract_code(resp_h)
        fn_h = load_transform(code_h)
        if fn_h:
            inv = check_invariants(train_pairs, fn_h)
            if inv['train_pass']:
                pred = to_list(fn_h(test_input))
                if grid_eq(pred, test_output):
                    results["with_hints"] += 1
                    print(f"  ✓ {tid} WITH hints")
                else:
                    print(f"  ✗ {tid} WITH hints (train pass, test fail)")
            else:
                print(f"  ✗ {tid} WITH hints (train fail)")
        else:
            print(f"  ✗ {tid} WITH hints (code error)")
    except Exception as e:
        print(f"  ✗ {tid} WITH hints (error: {e})")
    
    # Try WITHOUT hints
    prompt_n = format_task_prompt(train_pairs, test_input, "default", hints="")
    try:
        resp_n = call_opus(prompt_n, API_KEY, model="claude-sonnet-4-5-20250514")
        code_n = extract_code(resp_n)
        fn_n = load_transform(code_n)
        if fn_n:
            inv = check_invariants(train_pairs, fn_n)
            if inv['train_pass']:
                pred = to_list(fn_n(test_input))
                if grid_eq(pred, test_output):
                    results["without_hints"] += 1
                    print(f"  ✓ {tid} WITHOUT hints")
                else:
                    print(f"  ✗ {tid} WITHOUT hints (train pass, test fail)")
            else:
                print(f"  ✗ {tid} WITHOUT hints (train fail)")
        else:
            print(f"  ✗ {tid} WITHOUT hints (code error)")
    except Exception as e:
        print(f"  ✗ {tid} WITHOUT hints (error: {e})")
    
    print()

print("="*60)
print(f"WITH hints:    {results['with_hints']}/{results['total']}")
print(f"WITHOUT hints: {results['without_hints']}/{results['total']}")
