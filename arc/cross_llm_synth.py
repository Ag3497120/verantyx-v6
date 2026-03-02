"""
arc/cross_llm_synth.py — LLM-guided Program Synthesis in Cross Loop

L0 failure report → LLM generates Python code → verify → evolve → recognize

Flow:
  1. L0 tries all hypotheses → failure report (over/under cells, features)
  2. Failure report + train examples → LLM prompt
  3. LLM generates transform function
  4. Python reviewer: exec in sandbox, verify against train
  5. If wrong: error feedback → LLM retry (max 3 rounds)
  6. If all train pass → apply to test
"""

import numpy as np
import json
import re
import os
import traceback
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any

Grid = List[List[int]]


def _bg(g):
    return int(Counter(np.array(g).flatten()).most_common(1)[0][0])


def grid_eq(a, b):
    a, b = np.array(a), np.array(b)
    return a.shape == b.shape and np.array_equal(a, b)


def _grid_to_str(g):
    """Grid to compact string for LLM prompt"""
    return '\n'.join(' '.join(str(int(v)) for v in row) for row in g)


def _diff_report(ga, go):
    """Generate diff report between input and output"""
    ga, go = np.array(ga), np.array(go)
    if ga.shape != go.shape:
        return f"Size change: {ga.shape} → {go.shape}"
    
    changes = []
    for r in range(ga.shape[0]):
        for c in range(ga.shape[1]):
            if ga[r, c] != go[r, c]:
                changes.append((r, c, int(ga[r, c]), int(go[r, c])))
    
    if not changes:
        return "No changes"
    
    # Summarize
    color_map = Counter((ov, nv) for _, _, ov, nv in changes)
    return {
        'n_changed': len(changes),
        'color_transitions': dict(color_map),
        'changed_cells': changes[:20],  # limit
        'bg': _bg(ga),
        'grid_size': ga.shape,
    }


def _l0_failure_report(train_pairs, l0_results=None):
    """Generate L0 failure report for LLM"""
    reports = []
    for i, (inp, out) in enumerate(train_pairs):
        diff = _diff_report(inp, out)
        reports.append({
            'pair_index': i,
            'input_size': np.array(inp).shape,
            'output_size': np.array(out).shape,
            'diff': diff,
        })
    return reports


def _build_prompt(train_pairs, failure_report, attempt=0, prev_code=None, prev_error=None):
    """Build LLM prompt from failure report + train examples"""
    
    prompt = """You are solving an ARC-AGI puzzle. Given input grids, produce output grids.

## Training Examples
"""
    for i, (inp, out) in enumerate(train_pairs):
        prompt += f"\n### Train {i}\nInput ({np.array(inp).shape[0]}x{np.array(inp).shape[1]}):\n"
        prompt += _grid_to_str(inp)
        prompt += f"\n\nOutput ({np.array(out).shape[0]}x{np.array(out).shape[1]}):\n"
        prompt += _grid_to_str(out)
        prompt += "\n"
    
    prompt += f"""
## Diff Analysis
Background color: {_bg(np.array(train_pairs[0][0]))}
"""
    for r in failure_report:
        prompt += f"\nTrain {r['pair_index']}: {r['input_size']} → {r['output_size']}"
        if isinstance(r['diff'], dict):
            prompt += f", {r['diff']['n_changed']} cells changed"
            prompt += f", transitions: {r['diff']['color_transitions']}"
    
    if attempt > 0 and prev_code and prev_error:
        prompt += f"""

## Previous Attempt (FAILED)
```python
{prev_code}
```
Error: {prev_error}

Fix the code based on this error.
"""
    
    prompt += """

## Task
Write a Python function `transform(grid: list[list[int]]) -> list[list[int]]` that transforms input to output.

Rules:
- grid is a list of lists of ints (0-9)
- Return a new grid (don't modify input)
- Use only standard library + numpy
- Be precise — must match ALL training outputs exactly
- Look for the pattern, don't hardcode coordinates

```python
import numpy as np
from collections import Counter

def transform(grid):
    g = np.array(grid)
    # ... your code ...
    return result.tolist()
```

Return ONLY the Python code block. No explanation."""
    
    return prompt


def _extract_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response"""
    # Try ```python ... ``` blocks
    matches = re.findall(r'```python\s*\n(.*?)```', response, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Try ``` ... ``` blocks  
    matches = re.findall(r'```\s*\n(.*?)```', response, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Try raw code (starts with import or def)
    lines = response.strip().split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.startswith(('import ', 'from ', 'def ')):
            in_code = True
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return None


def _verify_code(code: str, train_pairs) -> Tuple[bool, str, Optional[callable]]:
    """Execute code in sandbox and verify against train pairs"""
    # Create isolated namespace
    namespace = {
        'np': np,
        'numpy': np,
        'Counter': Counter,
        'collections': __import__('collections'),
        'itertools': __import__('itertools'),
        'functools': __import__('functools'),
        'math': __import__('math'),
    }
    
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"Syntax/Import error: {e}", None
    
    if 'transform' not in namespace:
        return False, "No 'transform' function defined", None
    
    fn = namespace['transform']
    
    # Verify against all train pairs
    errors = []
    for i, (inp, out) in enumerate(train_pairs):
        try:
            pred = fn(inp)
            if pred is None:
                errors.append(f"Train {i}: returned None")
                continue
            if not grid_eq(pred, out):
                pa = np.array(pred)
                oa = np.array(out)
                if pa.shape != oa.shape:
                    errors.append(f"Train {i}: shape {pa.shape} != expected {oa.shape}")
                else:
                    n_wrong = int(np.sum(pa != oa))
                    # Show first few wrong cells
                    wrong_cells = []
                    for r in range(pa.shape[0]):
                        for c in range(pa.shape[1]):
                            if pa[r, c] != oa[r, c]:
                                wrong_cells.append(f"({r},{c}): got {pa[r,c]} expected {oa[r,c]}")
                                if len(wrong_cells) >= 5:
                                    break
                        if len(wrong_cells) >= 5:
                            break
                    errors.append(f"Train {i}: {n_wrong} wrong cells. First: {'; '.join(wrong_cells)}")
        except Exception as e:
            errors.append(f"Train {i}: Runtime error: {e}")
    
    if errors:
        return False, '\n'.join(errors), fn
    
    return True, "All train pairs correct", fn


def _call_llm(prompt: str, model: str = None) -> Optional[str]:
    """Call LLM API to generate code"""
    import subprocess
    
    # Use openclaw's LLM routing via a simple subprocess call
    # Or use direct API if available
    
    # Try openai-compatible API first
    api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
    
    # Use litellm if available
    try:
        import litellm
        if model is None:
            model = os.environ.get('CROSS_LLM_MODEL', 'deepseek/deepseek-chat')
        
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except ImportError:
        pass
    
    # Fallback: use subprocess to call an LLM CLI
    try:
        result = subprocess.run(
            ['python3', '-c', f'''
import json, sys
try:
    import openai
    client = openai.OpenAI(
        base_url="{os.environ.get('OPENAI_BASE_URL', 'https://api.deepseek.com')}",
        api_key="{os.environ.get('DEEPSEEK_API_KEY', os.environ.get('OPENAI_API_KEY', ''))}"
    )
    resp = client.chat.completions.create(
        model="{model or 'deepseek-chat'}",
        messages=[{{"role": "user", "content": json.loads(sys.stdin.read())}}],
        temperature=0.3,
        max_tokens=4096,
    )
    print(resp.choices[0].message.content)
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
'''],
            input=json.dumps(prompt),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    
    return None


def llm_synthesize(train_pairs: List[Tuple[Grid, Grid]], 
                   max_attempts: int = 3,
                   model: str = None) -> Optional[Tuple[str, callable]]:
    """
    LLM-guided program synthesis with verification loop.
    
    Returns (code_str, transform_fn) or None
    """
    failure_report = _l0_failure_report(train_pairs)
    
    prev_code = None
    prev_error = None
    
    for attempt in range(max_attempts):
        prompt = _build_prompt(train_pairs, failure_report, attempt, prev_code, prev_error)
        
        response = _call_llm(prompt, model)
        if response is None:
            continue
        
        code = _extract_code(response)
        if code is None:
            prev_error = "Could not extract Python code from response"
            continue
        
        ok, error_msg, fn = _verify_code(code, train_pairs)
        
        if ok:
            return code, fn
        
        prev_code = code
        prev_error = error_msg
    
    return None


def cross_llm_solve(train_pairs: List[Tuple[Grid, Grid]],
                    test_inputs: List[Grid],
                    l0_failures: Dict = None,
                    model: str = None,
                    max_attempts: int = 3) -> Optional[List[Grid]]:
    """
    Cross Loop L0.5: LLM synthesis between L0 and L1
    
    Args:
        train_pairs: list of (input, output) pairs
        test_inputs: list of test inputs
        l0_failures: failure report from L0 (optional, enriches prompt)
        model: LLM model to use
        max_attempts: max retry rounds
    
    Returns:
        list of predictions for test_inputs, or None
    """
    result = llm_synthesize(train_pairs, max_attempts, model)
    if result is None:
        return None
    
    code, fn = result
    
    predictions = []
    for ti in test_inputs:
        try:
            pred = fn(ti)
            if pred is not None:
                predictions.append(pred)
            else:
                return None
        except Exception:
            return None
    
    return predictions


# === Integration point for cross_puzzle_engine ===

def cross_puzzle_solve_with_llm(train_pairs, l0_hypotheses=None, model=None):
    """
    Returns list of (name, apply_fn) pieces that can be added to verified candidates.
    Called from cross_puzzle_engine or cross_engine.
    """
    result = llm_synthesize(train_pairs, max_attempts=3, model=model)
    if result is None:
        return []
    
    code, fn = result
    return [('llm_synth', fn)]


if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    data_dir = Path('/tmp/arc-agi-2/data/training')
    
    # Test on a specific task
    tid = sys.argv[1] if len(sys.argv) > 1 else '045e512c'
    
    with open(data_dir / f'{tid}.json') as f:
        task = json.load(f)
    
    train = [(e['input'], e['output']) for e in task['train']]
    test_input = task['test'][0]['input']
    test_output = task['test'][0].get('output')
    
    print(f"Task: {tid}")
    print(f"Train pairs: {len(train)}")
    
    result = llm_synthesize(train, max_attempts=3)
    if result:
        code, fn = result
        print(f"\n✓ Synthesis succeeded!")
        print(f"Code:\n{code}")
        
        # Test
        pred = fn(test_input)
        if test_output and grid_eq(pred, test_output):
            print(f"\n✓ Test CORRECT!")
        else:
            print(f"\n✗ Test wrong")
    else:
        print(f"\n✗ Synthesis failed")
