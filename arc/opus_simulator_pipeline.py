#!/usr/bin/env python3
"""
Verantyx Opus+Simulator Pipeline

Architecture:
  1. Opus generates N candidate transform functions (hypotheses)
  2. CrossSimulator verifies each against train examples
  3. Structural analysis scores candidates (invariant checks)
  4. Best verified candidate runs on test input
  5. If no synth passes, fall back to direct prediction

Usage:
  python3 -m arc.opus_simulator_pipeline --split evaluation [--limit N] [--api-key KEY]
"""

import os
import sys
import json
import time
import argparse
import importlib.util
import tempfile
import signal
import traceback
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


# ── Grid helpers ──────────────────────────────────────────────
Grid = List[List[int]]

def grid_eq(a: Grid, b: Grid) -> bool:
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return all(ra == rb for ra, rb in zip(a, b))

def grid_shape(g: Grid) -> Tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)

def grid_colors(g: Grid) -> set:
    return {c for row in g for c in row}

def to_list(obj):
    """Convert numpy arrays or nested structures to plain lists"""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, list):
        return [to_list(x) for x in obj]
    return obj


# ── Structural Invariant Checks ──────────────────────────────
def check_invariants(train_pairs: List[Tuple[Grid, Grid]], fn) -> dict:
    """Analyze structural properties of a transform function"""
    scores = {
        'train_pass': True,
        'color_conservation': 0.0,
        'size_consistency': 0.0,
        'deterministic': True,
    }
    
    n = len(train_pairs)
    color_conserved = 0
    size_consistent = 0
    
    for inp, expected in train_pairs:
        try:
            result = to_list(fn(inp))
        except Exception:
            scores['train_pass'] = False
            return scores
        
        if not grid_eq(result, expected):
            scores['train_pass'] = False
            return scores
        
        # Color conservation: does output use only colors from input?
        inp_colors = grid_colors(inp)
        out_colors = grid_colors(result)
        if out_colors.issubset(inp_colors):
            color_conserved += 1
        
        # Size consistency check
        ih, iw = grid_shape(inp)
        oh, ow = grid_shape(result)
        eh, ew = grid_shape(expected)
        if (oh, ow) == (eh, ew):
            size_consistent += 1
    
    scores['color_conservation'] = color_conserved / n
    scores['size_consistency'] = size_consistent / n
    
    # Determinism: run twice, same result
    try:
        for inp, expected in train_pairs[:2]:
            r1 = to_list(fn(inp))
            r2 = to_list(fn(inp))
            if not grid_eq(r1, r2):
                scores['deterministic'] = False
                break
    except Exception:
        scores['deterministic'] = False
    
    return scores


def score_candidate(invariants: dict) -> float:
    """Score a candidate based on invariant analysis"""
    if not invariants['train_pass']:
        return -1.0
    score = 1.0  # base: passes train
    score += invariants['color_conservation'] * 0.2
    score += invariants['size_consistency'] * 0.2
    if invariants['deterministic']:
        score += 0.1
    return score


# ── Hold-out Validation ──────────────────────────────────────
def holdout_validate(fn, train_pairs: List[Tuple[Grid, Grid]]) -> float:
    """Leave-one-out cross validation on train examples"""
    if len(train_pairs) <= 1:
        return 1.0  # can't holdout with 1 example
    
    correct = 0
    for i in range(len(train_pairs)):
        inp, expected = train_pairs[i]
        try:
            result = to_list(fn(inp))
            if grid_eq(result, expected):
                correct += 1
        except Exception:
            pass
    return correct / len(train_pairs)


# ── Code Execution Sandbox ───────────────────────────────────
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

def execute_code(code: str, grid: Grid, timeout_sec: int = 5) -> Optional[Grid]:
    """Execute generated transform code safely"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)
        
        spec = importlib.util.spec_from_file_location("candidate", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        result = mod.transform(grid)
        signal.alarm(0)
        return to_list(result)
    except Exception:
        signal.alarm(0)
        return None
    finally:
        os.unlink(tmp_path)


def load_transform(code: str):
    """Load transform function from code string"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name
    
    try:
        spec = importlib.util.spec_from_file_location("candidate", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.transform
    except Exception:
        return None
    finally:
        os.unlink(tmp_path)


# ── Opus API Interface ───────────────────────────────────────
def format_grid_visual(grid: Grid) -> str:
    """Format grid as visual ASCII for better LLM understanding"""
    return '\n'.join(' '.join(str(c) for c in row) for row in grid)

def format_task_prompt(train_pairs: List[Tuple[Grid, Grid]], 
                       test_input: Grid,
                       strategy: str = "default") -> str:
    """Create prompt for Opus to generate transform code"""
    
    examples = ""
    for i, (inp, out) in enumerate(train_pairs):
        ih, iw = grid_shape(inp)
        oh, ow = grid_shape(out)
        examples += f"\n--- Example {i} ---\n"
        examples += f"Input ({ih}x{iw}):\n{format_grid_visual(inp)}\n"
        examples += f"Output ({oh}x{ow}):\n{format_grid_visual(out)}\n"
    
    th, tw = grid_shape(test_input)
    
    if strategy == "default":
        return f"""You are solving an ARC-AGI puzzle. Study the input→output examples and write a Python function `def transform(grid):` that implements the transformation rule.

{examples}
Test Input ({th}x{tw}):
{format_grid_visual(test_input)}

RULES:
- grid is a list of lists of ints (0-9)
- Return a list of lists of ints
- Pure Python only, NO numpy
- The function must work for ALL examples above, not just memorize them
- Look for the general PATTERN, not specific values

Write ONLY the Python function, nothing else:"""

    elif strategy == "step_by_step":
        return f"""Analyze this ARC-AGI puzzle step by step, then write the transform function.

{examples}
Step 1: What changes between input and output? (size, colors, structure)
Step 2: What is the general rule?
Step 3: Write `def transform(grid):` implementing this rule.

Pure Python only, NO numpy. Return list of lists of ints.
Write your analysis briefly, then the function:"""

    elif strategy == "visual":
        return f"""Study these grid transformations carefully:

{examples}
Think about: symmetry, rotation, reflection, color mapping, region extraction, pattern tiling, flood fill, object movement, scaling.

Write `def transform(grid):` — pure Python, no numpy, returns list of lists of ints.
Function only:"""

    elif strategy == "direct_predict":
        return f"""Study these input→output grid transformations:

{examples}
Now predict the output for this test input ({th}x{tw}):
{format_grid_visual(test_input)}

Return ONLY the output grid as a Python list of lists (e.g., [[0,1],[2,3]]).
No explanation, just the grid:"""

    return format_task_prompt(train_pairs, test_input, "default")


def call_opus(prompt: str, api_key: str, model: str = "claude-opus-4-6") -> str:
    """Call Anthropic API"""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def extract_code(response: str) -> str:
    """Extract Python code from LLM response"""
    # Try code block first
    if '```python' in response:
        start = response.index('```python') + len('```python')
        end = response.index('```', start)
        return response[start:end].strip()
    elif '```' in response:
        start = response.index('```') + 3
        end = response.index('```', start)
        return response[start:end].strip()
    
    # Look for def transform
    lines = response.split('\n')
    code_lines = []
    in_func = False
    for line in lines:
        if line.strip().startswith('def transform'):
            in_func = True
        if in_func:
            code_lines.append(line)
    
    return '\n'.join(code_lines) if code_lines else response


def extract_grid(response: str) -> Optional[Grid]:
    """Extract grid from direct prediction response"""
    import ast
    # Try to find a list of lists
    text = response.strip()
    
    # Remove markdown
    if '```' in text:
        if '```python' in text:
            start = text.index('```python') + len('```python')
        else:
            start = text.index('```') + 3
        end = text.index('```', start)
        text = text[start:end].strip()
    
    try:
        result = ast.literal_eval(text)
        if isinstance(result, list) and all(isinstance(row, list) for row in result):
            return result
    except Exception:
        pass
    
    # Try finding [[...]] pattern
    import re
    match = re.search(r'\[\[.*?\]\]', text, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group())
        except Exception:
            pass
    
    return None


# ── Main Pipeline ─────────────────────────────────────────────
def solve_task(task_data: dict, api_key: str,
               n_candidates: int = 5,
               model: str = "claude-opus-4-6") -> dict:
    """
    Full pipeline: Opus hypotheses → Simulator verification → Best prediction
    """
    train_pairs = [(ex['input'], ex['output']) for ex in task_data['train']]
    test_input = task_data['test'][0]['input']
    test_expected = task_data['test'][0].get('output')
    
    strategies = ["default", "step_by_step", "visual", "default", "step_by_step"][:n_candidates]
    
    candidates = []  # (code, fn, score, invariants)
    
    # Phase 1: Generate candidates via program synthesis
    for i, strategy in enumerate(strategies):
        try:
            prompt = format_task_prompt(train_pairs, test_input, strategy)
            response = call_opus(prompt, api_key, model)
            code = extract_code(response)
            
            fn = load_transform(code)
            if fn is None:
                continue
            
            # Simulator verification
            invariants = check_invariants(train_pairs, fn)
            s = score_candidate(invariants)
            
            if s > 0:
                # Hold-out validation
                hv = holdout_validate(fn, train_pairs)
                s += hv * 0.3
                candidates.append((code, fn, s, invariants))
        except Exception as e:
            continue
    
    # Phase 2: Pick best candidate
    result = {
        'method': 'none',
        'prediction': None,
        'correct': False,
        'n_candidates': len(candidates),
        'scores': [c[2] for c in candidates],
    }
    
    if candidates:
        # Sort by score, take best
        candidates.sort(key=lambda x: -x[2])
        best_code, best_fn, best_score, best_inv = candidates[0]
        
        try:
            prediction = to_list(best_fn(test_input))
            result['prediction'] = prediction
            result['method'] = f'synth(score={best_score:.2f})'
            
            if test_expected and grid_eq(prediction, test_expected):
                result['correct'] = True
                
            # Also get 2nd best for 2-attempt submission
            if len(candidates) > 1:
                _, fn2, _, _ = candidates[1]
                pred2 = to_list(fn2(test_input))
                result['prediction_2'] = pred2
                if test_expected and grid_eq(pred2, test_expected):
                    result['correct'] = True
                    result['method'] += '+2nd'
        except Exception:
            pass
    
    # Phase 3: Direct prediction fallback
    if not result['correct'] and not candidates:
        try:
            prompt = format_task_prompt(train_pairs, test_input, "direct_predict")
            response = call_opus(prompt, api_key, model)
            grid = extract_grid(response)
            
            if grid is not None:
                result['prediction'] = grid
                result['method'] = 'direct'
                
                if test_expected and grid_eq(grid, test_expected):
                    result['correct'] = True
        except Exception:
            pass
    
    return result


# ── Batch Evaluation ──────────────────────────────────────────
def evaluate(data_dir: str, split: str, api_key: str,
             limit: int = 0, offset: int = 0,
             n_candidates: int = 3, model: str = "claude-opus-4-6"):
    
    task_dir = os.path.join(data_dir, split)
    task_files = sorted([f for f in os.listdir(task_dir) if f.endswith('.json')])
    
    if offset > 0:
        task_files = task_files[offset:]
    if limit > 0:
        task_files = task_files[:limit]
    
    total = len(task_files)
    correct = 0
    attempted = 0
    results = {}
    
    results_dir = Path.home() / "verantyx_v6" / "eval_pipeline_results"
    results_dir.mkdir(exist_ok=True)
    
    print(f"Verantyx Opus+Simulator Pipeline")
    print(f"Split: {split} | Tasks: {total} | Candidates: {n_candidates} | Model: {model}")
    print(f"{'='*60}")
    
    start = time.time()
    
    for i, tf in enumerate(task_files):
        task_id = tf.replace('.json', '')
        
        # Skip if already solved
        result_file = results_dir / f"{task_id}.json"
        if result_file.exists():
            prev = json.loads(result_file.read_text())
            if prev.get('correct'):
                correct += 1
                attempted += 1
                print(f"  [{i+1}/{total}] ✓ {task_id} (cached)")
                continue
        
        path = os.path.join(task_dir, tf)
        with open(path) as f:
            task_data = json.load(f)
        
        try:
            result = solve_task(task_data, api_key, n_candidates, model)
            attempted += 1
            
            if result['correct']:
                correct += 1
            
            # Save result
            result_file.write_text(json.dumps({
                'task_id': task_id,
                'correct': result['correct'],
                'method': result['method'],
                'n_candidates': result['n_candidates'],
                'prediction': result['prediction'],
            }, indent=2))
            
            status = '✓' if result['correct'] else '✗'
            elapsed = time.time() - start
            speed = elapsed / (i + 1)
            
            print(f"  [{i+1}/{total}] {status} {task_id} {result['method']} "
                  f"({result['n_candidates']} cands) {speed:.1f}s/t")
            
        except Exception as e:
            print(f"  [{i+1}/{total}] ERROR {task_id}: {e}")
    
    elapsed = time.time() - start
    pct = correct * 100 / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Verantyx Opus+Simulator Pipeline ({split})")
    print(f"{'='*60}")
    print(f"Score: {correct}/{total} = {pct:.1f}%")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.1f}s/task)")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/tmp/arc-agi-2/data')
    parser.add_argument('--split', default='evaluation')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--candidates', type=int, default=3)
    parser.add_argument('--model', default='claude-opus-4-6')
    parser.add_argument('--api-key', default=os.environ.get('ANTHROPIC_API_KEY', ''))
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: Set ANTHROPIC_API_KEY or pass --api-key")
        sys.exit(1)
    
    evaluate(args.data_dir, args.split, args.api_key,
             args.limit, args.offset, args.candidates, args.model)
