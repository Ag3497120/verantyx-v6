"""
arc/llm_solver.py — LLM-based per-task solver

Uses a local LLM (Qwen 7B via ollama) to generate Python transformation
functions for unsolved ARC tasks. Each task's train pairs are shown to
the LLM, which writes a transform(input) -> output function.
The function is verified on all train pairs before being accepted.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import json
import subprocess
import numpy as np
import re
import hashlib


def _format_grid(grid):
    """Format grid for LLM prompt."""
    return '\n'.join(str(row) for row in grid)


def _make_prompt(train_pairs):
    """Create prompt for LLM."""
    examples = []
    for i, (inp, out) in enumerate(train_pairs):
        examples.append(f"Example {i+1}:")
        examples.append(f"Input ({len(inp)}x{len(inp[0])}):")
        examples.append(_format_grid(inp))
        examples.append(f"Output ({len(out)}x{len(out[0])}):")
        examples.append(_format_grid(out))
        examples.append("")
    
    prompt = f"""Analyze these input→output grid transformations and write a Python function.

{chr(10).join(examples)}

Write a Python function `transform(grid)` that takes a 2D list of integers (the input grid) and returns a 2D list of integers (the output grid). The function should implement the transformation pattern shown in the examples.

Rules:
- Only use standard Python (no imports except copy if needed)
- The function must work for any valid input following the same pattern
- Return the output as a list of lists of integers

```python
def transform(grid):
```"""
    
    return prompt


def _call_ollama(prompt, model="qwen2.5:7b-instruct", timeout=60):
    """Call ollama and get response."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout
    except (subprocess.TimeoutExpired, Exception):
        return None


def _extract_function(response):
    """Extract Python function from LLM response."""
    if not response:
        return None
    
    # Try to find code block
    patterns = [
        r'```python\s*(def transform.*?)```',
        r'```\s*(def transform.*?)```',
        r'(def transform\(grid\).*?)(?=\n\n|\Z)',
    ]
    
    for pattern in patterns:
        m = re.search(pattern, response, re.DOTALL)
        if m:
            code = m.group(1).strip()
            return code
    
    # Fallback: find def transform anywhere
    lines = response.split('\n')
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('def transform'):
            start = i
            break
    
    if start is not None:
        # Collect function body (indented lines after def)
        func_lines = [lines[start]]
        for i in range(start + 1, len(lines)):
            if lines[i].strip() and not lines[i].startswith(' ') and not lines[i].startswith('\t'):
                break
            func_lines.append(lines[i])
        return '\n'.join(func_lines)
    
    return None


def _verify_function(code, train_pairs):
    """Compile and verify function on train pairs."""
    try:
        namespace = {}
        exec(code, namespace)
        transform = namespace.get('transform')
        if transform is None:
            return False
        
        for inp, out in train_pairs:
            result = transform([list(row) for row in inp])
            if result is None:
                return False
            # Convert to comparable format
            result_list = [[int(x) for x in row] for row in result]
            out_list = [[int(x) for x in row] for row in out]
            if result_list != out_list:
                return False
        
        return True
    except Exception:
        return False


def solve_with_llm(train_pairs, model="qwen2.5:7b-instruct", max_attempts=3):
    """Try to solve a task using LLM-generated code.
    
    Returns: (transform_function, code_string) or (None, None)
    """
    prompt = _make_prompt(train_pairs)
    
    for attempt in range(max_attempts):
        response = _call_ollama(prompt, model)
        if not response:
            continue
        
        code = _extract_function(response)
        if not code:
            continue
        
        if _verify_function(code, train_pairs):
            namespace = {}
            exec(code, namespace)
            return namespace['transform'], code
    
    return None, None


def generate_llm_pieces(train_pairs, model="qwen2.5:7b-instruct"):
    """Generate CrossPiece from LLM-solved task."""
    from arc.cross_engine import CrossPiece
    
    transform, code = solve_with_llm(train_pairs, model, max_attempts=2)
    if transform is None:
        return []
    
    def apply_fn(inp_grid):
        return transform([list(row) for row in inp_grid])
    
    # Hash the code for unique naming
    code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
    return [CrossPiece(name=f"llm_qwen_{code_hash}", apply_fn=apply_fn, version=1)]


if __name__ == '__main__':
    import sys
    
    data_dir = '/tmp/arc-agi-2/data/training/'
    
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
    else:
        task_id = '9f5f939b'
    
    d = json.load(open(f'{data_dir}/{task_id}.json'))
    pairs = [(t['input'], t['output']) for t in d['train']]
    
    print(f"Solving {task_id}...")
    transform, code = solve_with_llm(pairs)
    
    if transform:
        print(f"SUCCESS! Code:")
        print(code)
    else:
        print("FAILED")
