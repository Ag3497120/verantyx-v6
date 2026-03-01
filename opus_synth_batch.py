#!/usr/bin/env python3
"""
Opus program synthesis for ARC-AGI2 unsolved tasks.
Calls Anthropic API directly with claude-opus-4-6.
Runs 4 workers in parallel.
"""
import json, os, re, sys, subprocess, time
import anthropic
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path("/tmp/arc-agi-2/data/training")
RESULTS_DIR = Path(os.path.expanduser("~/verantyx_v6/synth_results"))
VERIFY_SCRIPT = Path(os.path.expanduser("~/verantyx_v6/verify_transform.py"))
MODEL = "claude-opus-4-6"
MAX_ATTEMPTS = 3
WORKERS = 4

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

def get_unsolved():
    solved_ce = set()
    log_path = os.path.expanduser("~/verantyx_v6/arc_v82.log")
    with open(log_path) as f:
        for line in f:
            m = re.search(r'✓.*?([0-9a-f]{8})\s+ver=', line)
            if m:
                solved_ce.add(m.group(1))
    synth = set(f.replace('.py','') for f in os.listdir(RESULTS_DIR) if f.endswith('.py'))
    all_tasks = set(f.replace('.json','') for f in os.listdir(DATA_DIR) if f.endswith('.json'))
    return sorted(all_tasks - solved_ce - synth)

def load_task(tid):
    with open(DATA_DIR / f"{tid}.json") as f:
        return json.load(f)

def format_examples(task):
    lines = []
    for i, ex in enumerate(task['train']):
        lines.append(f"Train {i}:")
        lines.append(f"  Input ({len(ex['input'])}x{len(ex['input'][0])}):")
        for row in ex['input']:
            lines.append(f"    {row}")
        lines.append(f"  Output ({len(ex['output'])}x{len(ex['output'][0])}):")
        for row in ex['output']:
            lines.append(f"    {row}")
    return '\n'.join(lines)

def extract_code(response_text):
    # Extract python code from response
    patterns = [
        r'```python\n(.*?)```',
        r'```\n(.*?)```',
    ]
    for pat in patterns:
        m = re.search(pat, response_text, re.DOTALL)
        if m:
            return m.group(1).strip()
    # If no code blocks, look for def transform
    m = re.search(r'(def transform\(.*?\n(?:.*\n)*?)(?=\n\S|\Z)', response_text)
    if m:
        return m.group(1).strip()
    return None

def verify(tid, code_path):
    try:
        r = subprocess.run(
            ['python3', str(VERIFY_SCRIPT), str(DATA_DIR / f"{tid}.json"), str(code_path)],
            capture_output=True, timeout=10
        )
        return r.returncode == 0
    except:
        return False

def solve_task(tid):
    task = load_task(tid)
    examples = format_examples(task)
    
    prompt = f"""You are solving an ARC-AGI2 puzzle. Study the training examples and write a Python function `transform(grid)` that converts input grids to output grids.

CRITICAL RULES:
- Do NOT hardcode specific coordinates, grid sizes, or color values
- Your function must generalize to ANY input following the same pattern
- Think about what changes between input and output across ALL examples
- Use numpy, scipy, collections if helpful
- Return list[list[int]]

{examples}

Write ONLY the Python code with `def transform(grid):` that implements the transformation rule. Include all necessary imports inside or before the function."""

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            code = extract_code(response.content[0].text)
            if not code:
                continue
            
            code_path = RESULTS_DIR / f"{tid}.py"
            with open(code_path, 'w') as f:
                f.write(code + '\n')
            
            if verify(tid, code_path):
                return tid, True, attempt + 1
            else:
                os.remove(code_path)
                # Add feedback for retry
                if attempt < MAX_ATTEMPTS - 1:
                    prompt += f"\n\nYour previous attempt failed verification. Try a DIFFERENT approach. Think more carefully about the pattern."
        except Exception as e:
            print(f"  [{tid}] Error attempt {attempt+1}: {e}", file=sys.stderr)
            time.sleep(2)
    
    return tid, False, MAX_ATTEMPTS

def main():
    unsolved = get_unsolved()
    print(f"Unsolved tasks: {len(unsolved)}")
    print(f"Model: {MODEL}")
    print(f"Workers: {WORKERS}")
    print(f"Max attempts per task: {MAX_ATTEMPTS}")
    print()
    
    solved = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(solve_task, tid): tid for tid in unsolved}
        
        for future in as_completed(futures):
            tid, success, attempts = future.result()
            if success:
                solved += 1
                print(f"✓ {tid} (attempt {attempts}) [{solved}/{solved+failed} of {len(unsolved)}]")
            else:
                failed += 1
                print(f"✗ {tid} [{solved}/{solved+failed} of {len(unsolved)}]")
    
    print(f"\nDone: {solved} solved, {failed} failed out of {len(unsolved)}")
    print(f"New total: 244 + previous_synth + {solved} = check with eval")

if __name__ == '__main__':
    main()
