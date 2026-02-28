import json, os, sys, subprocess, importlib.util
from pathlib import Path

sys.path.insert(0, str(Path.home() / 'verantyx_v6'))
from solutions_batch6 import *

TASKS = "6e02f1e3,6e19193c,6ecd11f4,6f473927,6ffe8f07,712bf12e,72207abc,72322fa7,72ca375d,73c3b0d8,73ccf9c2,7447852a,753ea09b,758abdf0,759f3fd3,75b8110e,760b3cac,762cd429,770cc55f,776ffc46,77fdfe62,780d0b14,782b5218,7837ac64,78e78cff,79369cc6,794b24be,79cce52d,7acdf6d3,7b6016b9,7bb29440,7c008303,7c8af763,7c9b52a0,7d18a6fb,7d419a02,7d7772cc,7ddcd7ec,7df24a62,7e02026e,7e0986d6,7e2bad24,7e4d4f7c,7e576d6e,7ec998c9,7ee1c6ea,7f4411dc,80214e03,80af3007,817e6c09".split(',')

SOLUTIONS = {
    '6e02f1e3': solve_6e02f1e3,
    '6e19193c': solve_6e19193c,
    '72322fa7': solve_72322fa7,
    '72ca375d': solve_72ca375d,
    '73c3b0d8': solve_73c3b0d8,
    '758abdf0': solve_758abdf0,
    '762cd429': solve_762cd429,
    '770cc55f': solve_770cc55f,
    '78e78cff': solve_78e78cff,
    '780d0b14': solve_780d0b14,
    '782b5218': solve_782b5218,
    '7acdf6d3': solve_7acdf6d3,
    '7b6016b9': solve_7b6016b9,
    '7c9b52a0': solve_7c9b52a0,
    '7d18a6fb': solve_7d18a6fb,
    '7ddcd7ec': solve_7ddcd7ec,
    '7e4d4f7c': solve_7e4d4f7c,
    '7ec998c9': solve_7ec998c9,
    '7ee1c6ea': solve_7ee1c6ea,
    '7f4411dc': solve_7f4411dc,
}

# Load previously working solutions
PREV_LOG = Path.home() / 'verantyx_v6/synth_results/batch6_log.json'
prev_log = {}
if PREV_LOG.exists():
    prev_log = json.loads(PREV_LOG.read_text())

results_dir = Path.home() / 'verantyx_v6/synth_results'
results_dir.mkdir(exist_ok=True)

log = dict(prev_log)

def write_solution(tid, fn_body):
    path = results_dir / f'{tid}.py'
    path.write_text(f'''def transform(grid):
{fn_body}
''')

def verify(tid):
    data_path = f'/tmp/arc-agi-2/data/training/{tid}.json'
    sol_path = str(results_dir / f'{tid}.py')
    result = subprocess.run(
        ['python3', str(Path.home() / 'verantyx_v6/verify_transform.py'), data_path, sol_path],
        capture_output=True, text=True, timeout=30
    )
    return 'correct' in result.stdout.lower(), result.stdout

for tid in TASKS:
    if log.get(tid, {}).get('status') == 'correct':
        print(f'{tid}: already correct, skipping')
        continue
    
    if tid not in SOLUTIONS:
        log[tid] = {'status': 'skipped', 'reason': 'no solution'}
        continue
    
    sol_fn = SOLUTIONS[tid]
    
    # Generate solution file
    import inspect
    src = inspect.getsource(sol_fn)
    # Extract body
    lines = src.split('\n')
    body_lines = []
    in_body = False
    for line in lines:
        if in_body:
            if line.startswith('def ') and not line.startswith('def transform'):
                break
            body_lines.append('    ' + line if line.strip() else line)
        elif line.startswith('def solve_'):
            in_body = True
    
    # Write directly using the function
    sol_path = results_dir / f'{tid}.py'
    sol_path.write_text(f'def transform(grid):\n    return _solve(grid)\n\n{src}\n\n_solve = solve_{tid.replace("-", "_")}\n')
    
    ok, out = verify(tid)
    if ok:
        log[tid] = {'status': 'correct'}
        print(f'{tid}: CORRECT')
    else:
        log[tid] = {'status': 'failed', 'output': out[:200]}
        print(f'{tid}: FAILED - {out[:100]}')

PREV_LOG.write_text(json.dumps(log, indent=2))
print(f"\nDone. Results: {sum(1 for v in log.values() if v.get('status')=='correct')} correct")
