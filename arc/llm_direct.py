"""
arc/llm_direct.py — LLM Direct Grid Generation for ARC-AGI-2

Strategy: Ask LLM to directly generate the test output grid.
Then verify against training pairs using the symbolic engine
to check if a consistent rule can be found.

This is a "generate and verify" approach:
- LLM generates candidate output (may be imprecise)
- If output matches, we take it
- Combined with hypothesis approach for best results
"""

import json
import subprocess
import re
from typing import List, Tuple, Optional
from arc.grid import Grid, grid_shape, grid_eq


DIRECT_SYSTEM = """You are solving ARC-AGI-2 puzzles. Given input-output examples, predict the test output.

Rules:
1. Output ONLY the grid as rows of digits, one row per line
2. Each digit is a color (0-9)
3. No extra text, no explanation, just the grid
4. Match the exact dimensions required

Example response for a 3x3 grid:
012
345
678"""


def grid_to_text(grid: Grid) -> str:
    return '\n'.join(''.join(str(c) for c in row) for row in grid)


def text_to_grid(text: str) -> Optional[Grid]:
    """Parse text into a grid."""
    lines = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Extract only digit characters
        digits = re.findall(r'[0-9]', line)
        if digits:
            lines.append([int(d) for d in digits])
    
    if not lines:
        return None
    
    # Ensure all rows have same width
    max_w = max(len(row) for row in lines)
    for row in lines:
        while len(row) < max_w:
            row.append(0)
    
    return lines


def query_ollama_direct(prompt: str, model: str = "qwen2.5:7b-instruct",
                        timeout: int = 60) -> Optional[str]:
    """Query Ollama for direct grid output."""
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": DIRECT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 2048,
            }
        }
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", "http://localhost:11434/api/chat",
             "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return None
        resp = json.loads(result.stdout)
        return resp.get("message", {}).get("content", "")
    except Exception:
        return None


def solve_direct_llm(train_pairs: List[Tuple[Grid, Grid]],
                     test_inputs: List[Grid],
                     model: str = "qwen2.5:7b-instruct",
                     n_attempts: int = 1) -> Optional[List[List[Grid]]]:
    """
    Ask LLM to directly generate test output.
    Returns predictions if successful.
    """
    # Build prompt
    parts = []
    for i, (inp, out) in enumerate(train_pairs):
        h_in, w_in = grid_shape(inp)
        h_out, w_out = grid_shape(out)
        parts.append(f"Example {i+1}:")
        parts.append(f"Input ({h_in}x{w_in}):")
        parts.append(grid_to_text(inp))
        parts.append(f"Output ({h_out}x{w_out}):")
        parts.append(grid_to_text(out))
        parts.append("")
    
    all_predictions = []
    
    for test_inp in test_inputs:
        h, w = grid_shape(test_inp)
        
        # Estimate output size from training pairs
        h_out_est = None
        w_out_est = None
        sizes = set()
        for inp, out in train_pairs:
            hi, wi = grid_shape(inp)
            ho, wo = grid_shape(out)
            sizes.add((ho, wo))
            # Check if output size is constant
            if h_out_est is None:
                h_out_est, w_out_est = ho, wo
            elif (h_out_est, w_out_est) != (ho, wo):
                # Variable size — try to infer ratio
                h_out_est = None
                break
        
        prompt = '\n'.join(parts)
        prompt += f"\nTest Input ({h}x{w}):\n{grid_to_text(test_inp)}\n"
        if h_out_est and w_out_est:
            prompt += f"\nPredict the output ({h_out_est}x{w_out_est}):"
        else:
            prompt += f"\nPredict the output:"
        
        preds = []
        for attempt in range(n_attempts):
            response = query_ollama_direct(prompt, model=model)
            if response:
                grid = text_to_grid(response)
                if grid:
                    preds.append(grid)
        
        all_predictions.append(preds)
    
    if any(p for p in all_predictions):
        return all_predictions
    return None


def solve_hybrid(train_pairs: List[Tuple[Grid, Grid]],
                 test_inputs: List[Grid],
                 model: str = "qwen2.5:7b-instruct") -> Tuple[Optional[List[List[Grid]]], str]:
    """
    Combined approach:
    1. Try direct LLM output
    2. Try hypothesis-guided symbolic solving
    3. Return best result with method tag
    """
    from arc.llm_hypothesis import solve_with_llm_hypothesis
    
    # Strategy 1: Direct output
    direct_preds = solve_direct_llm(train_pairs, test_inputs, model=model)
    
    # Strategy 2: Hypothesis-guided
    hyp_preds, hyp_verified, hyp = solve_with_llm_hypothesis(
        train_pairs, test_inputs, model=model)
    
    # Prefer hypothesis-verified (symbolic guarantee)
    if hyp_verified:
        return hyp_preds, f"hypothesis:{hyp_verified[0][1].name}"
    
    # Fall back to direct output
    if direct_preds and direct_preds[0]:
        return direct_preds, "direct_llm"
    
    return None, "failed"
