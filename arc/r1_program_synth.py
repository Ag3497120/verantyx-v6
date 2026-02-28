"""
arc/r1_program_synth.py — DeepSeek R1 Program Synthesis for ARC-AGI-2

R1 generates Python transform functions, NOT grid answers.
Functions are verified against train examples before applying to test.

Flow:
1. Format task as train examples
2. Ask R1 to write a Python transform function
3. Execute function in sandbox
4. Verify against ALL train examples
5. If verified, apply to test inputs
"""

import json
import os
import re
import traceback
import numpy as np
from typing import List, Tuple, Optional
from arc.grid import Grid, grid_shape, grid_eq

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-1c9551e705dd4fbfbdcab991cc924526")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

SYSTEM_PROMPT = """You solve ARC-AGI-2 puzzles by writing Python code.
Output ONLY a ```python code block containing def transform(grid): that returns the output grid.
No explanation outside the code block. Use numpy if needed."""


def grid_to_text(grid: Grid) -> str:
    return '\n'.join(' '.join(str(c) for c in row) for row in grid)


def _call_r1(messages: list, temperature: float = 0.0, 
             max_tokens: int = 8192, timeout: int = 120) -> Optional[str]:
    """Call DeepSeek V3 (chat) API."""
    import urllib.request
    
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    
    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            msg = data["choices"][0]["message"]
            content = msg.get("content", "")
            # R1 sometimes puts useful info in reasoning_content too
            if not content and msg.get("reasoning_content"):
                content = msg["reasoning_content"]
            return content
    except Exception as e:
        return None


def _extract_function(response: str) -> Optional[str]:
    """Extract Python function from R1 response."""
    # Find code block
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        # Try without python tag
        pattern = r'```\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
    
    if not matches:
        return None
    
    code = matches[0].strip()
    
    # Ensure it contains a transform function
    if 'def transform' not in code:
        return None
    
    return code


def _execute_transform(code: str, input_grid: Grid, timeout_sec: float = 5.0) -> Optional[Grid]:
    """Execute transform function in restricted environment."""
    import signal
    
    namespace = {
        'np': np,
        'numpy': np,
    }
    
    # Add safe imports
    try:
        from scipy.ndimage import label as scipy_label
        namespace['scipy_label'] = scipy_label
        import scipy
        namespace['scipy'] = scipy
    except:
        pass
    
    from collections import Counter, defaultdict
    namespace['Counter'] = Counter
    namespace['defaultdict'] = defaultdict
    
    try:
        exec(code, namespace)
    except Exception:
        return None
    
    if 'transform' not in namespace:
        return None
    
    transform_fn = namespace['transform']
    
    # Execute with timeout
    result = [None]
    error = [None]
    
    def handler(signum, frame):
        raise TimeoutError("Transform execution timed out")
    
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(timeout_sec) + 1)
    
    try:
        result[0] = transform_fn(input_grid)
    except Exception as e:
        error[0] = str(e)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    
    if error[0] or result[0] is None:
        return None
    
    # Validate output format
    if not isinstance(result[0], (list, np.ndarray)):
        return None
    
    try:
        grid = [[int(c) for c in row] for row in result[0]]
        return grid
    except:
        return None


def _verify_on_train(code: str, train_pairs: List[Tuple[Grid, Grid]]) -> bool:
    """Verify generated transform on ALL training examples."""
    for inp, expected_out in train_pairs:
        predicted = _execute_transform(code, inp)
        if predicted is None:
            return False
        if not grid_eq(predicted, expected_out):
            return False
    return True


def solve_with_r1_synth(train_pairs: List[Tuple[Grid, Grid]],
                        test_inputs: List[Grid],
                        n_attempts: int = 3,
                        temperature: float = 0.0) -> Optional[List[Grid]]:
    """
    Use R1 to synthesize a transform function, verify on train, apply to test.
    
    Returns list of test output grids, or None if synthesis fails.
    """
    
    # Build the prompt
    parts = []
    for i, (inp, out) in enumerate(train_pairs):
        h_in, w_in = grid_shape(inp)
        h_out, w_out = grid_shape(out)
        parts.append(f"Example {i+1}:")
        parts.append(f"Input ({h_in}×{w_in}):")
        parts.append(grid_to_text(inp))
        parts.append(f"Output ({h_out}×{w_out}):")
        parts.append(grid_to_text(out))
        parts.append("")
    
    # Add test input info
    for i, test_inp in enumerate(test_inputs):
        h, w = grid_shape(test_inp)
        parts.append(f"Test Input {i+1} ({h}×{w}):")
        parts.append(grid_to_text(test_inp))
        parts.append("")
    
    task_description = '\n'.join(parts)
    
    user_msg = f"""Given these ARC puzzle input→output examples, write a Python function to implement the transformation.

{task_description}

INSTRUCTIONS:
- Write `def transform(grid):` that takes a list[list[int]] and returns list[list[int]]
- Find the GENERAL rule (don't hardcode)
- You may import numpy, scipy, collections
- Put your code in a ```python code block```
- The function must work for ALL examples above"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    
    for attempt in range(n_attempts):
        temp = temperature + attempt * 0.3  # Increase diversity on retries
        
        response = _call_r1(messages, temperature=min(temp, 1.0))
        if response is None:
            continue
        
        code = _extract_function(response)
        if code is None:
            continue
        
        # Verify on ALL training examples
        if _verify_on_train(code, train_pairs):
            # Apply to test inputs
            test_outputs = []
            all_ok = True
            for test_inp in test_inputs:
                test_out = _execute_transform(code, test_inp)
                if test_out is None:
                    all_ok = False
                    break
                test_outputs.append(test_out)
            
            if all_ok:
                return test_outputs
    
    return None


def solve_task_r1_program(task_path: str) -> Optional[dict]:
    """Solve a single ARC task using R1 program synthesis."""
    with open(task_path) as f:
        data = json.load(f)
    
    train = [(ex['input'], ex['output']) for ex in data['train']]
    test_inputs = [ex['input'] for ex in data['test']]
    
    result = solve_with_r1_synth(train, test_inputs, n_attempts=3)
    
    if result is None:
        return None
    
    return {
        'test_output': result,
        'method': 'r1_program_synth',
    }
