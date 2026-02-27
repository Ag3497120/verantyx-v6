"""
arc/llm_deepseek.py — DeepSeek API integration for ARC-AGI-2

Uses DeepSeek-Chat (V3) or DeepSeek-Reasoner (R1) for:
1. Direct grid output generation
2. Hypothesis generation for symbolic engine
"""

import json
import os
import re
import urllib.request
from typing import List, Tuple, Optional
from arc.grid import Grid, grid_shape, grid_eq


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-1c9551e705dd4fbfbdcab991cc924526")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


SYSTEM_DIRECT = """You are solving ARC-AGI-2 puzzles. Given input-output grid examples, predict the test output.

CRITICAL RULES:
1. Output ONLY the grid as rows of digits, one row per line
2. Each digit is a color (0-9)  
3. NO explanation, NO markdown, NO extra text — JUST the grid digits
4. Match the EXACT dimensions specified
5. Study ALL examples carefully to find the pattern before answering

Colors: 0=black 1=blue 2=red 3=green 4=yellow 5=gray 6=magenta 7=orange 8=cyan 9=maroon"""


def grid_to_text(grid: Grid) -> str:
    return '\n'.join(''.join(str(c) for c in row) for row in grid)


def text_to_grid(text: str) -> Optional[Grid]:
    """Parse text into a grid, robustly."""
    lines = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Skip non-grid lines
        if any(c.isalpha() for c in line.replace('x', '')):
            continue
        digits = [int(c) for c in line if c.isdigit()]
        if digits:
            lines.append(digits)
    
    if not lines:
        return None
    
    # Ensure consistent width
    widths = [len(r) for r in lines]
    if len(set(widths)) > 1:
        # Use most common width
        from collections import Counter
        target_w = Counter(widths).most_common(1)[0][0]
        lines = [r for r in lines if len(r) == target_w]
    
    return lines if lines else None


def call_deepseek(messages: list, model: str = "deepseek-chat",
                  temperature: float = 0.0, max_tokens: int = 2048,
                  timeout: int = 60) -> Optional[str]:
    """Call DeepSeek API."""
    payload = json.dumps({
        "model": model,
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
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return None


def solve_task_deepseek(train_pairs: List[Tuple[Grid, Grid]],
                        test_inputs: List[Grid],
                        model: str = "deepseek-chat",
                        n_attempts: int = 1) -> Optional[List[List[Grid]]]:
    """Solve ARC task using DeepSeek direct grid generation."""
    
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
        
        # Estimate output size
        out_sizes = [(grid_shape(out)) for _, out in train_pairs]
        in_sizes = [(grid_shape(inp)) for inp, _ in train_pairs]
        
        # Check if same-size transform
        same_size = all(si == so for si, so in zip(in_sizes, out_sizes))
        # Check if constant output size
        constant_out = len(set(out_sizes)) == 1
        
        if same_size:
            expected_h, expected_w = h, w
        elif constant_out:
            expected_h, expected_w = out_sizes[0]
        else:
            # Try to infer ratio
            ratios = set()
            for (hi, wi), (ho, wo) in zip(in_sizes, out_sizes):
                if hi > 0 and wi > 0:
                    ratios.add((ho/hi, wo/wi))
            if len(ratios) == 1:
                rh, rw = ratios.pop()
                expected_h = int(h * rh)
                expected_w = int(w * rw)
            else:
                expected_h, expected_w = None, None
        
        prompt = '\n'.join(parts)
        prompt += f"\nTest Input ({h}x{w}):\n{grid_to_text(test_inp)}\n"
        if expected_h and expected_w:
            prompt += f"\nOutput the {expected_h}x{expected_w} grid (exactly {expected_h} rows of {expected_w} digits):"
        else:
            prompt += "\nOutput the grid:"
        
        messages = [
            {"role": "system", "content": SYSTEM_DIRECT},
            {"role": "user", "content": prompt},
        ]
        
        preds = []
        for _ in range(n_attempts):
            response = call_deepseek(messages, model=model)
            if response:
                grid = text_to_grid(response)
                if grid:
                    # Validate dimensions if known
                    if expected_h and expected_w:
                        gh, gw = len(grid), len(grid[0]) if grid else 0
                        if gh == expected_h and gw == expected_w:
                            preds.append(grid)
                        else:
                            # Still keep it as a candidate
                            preds.append(grid)
                    else:
                        preds.append(grid)
        
        all_predictions.append(preds)
    
    return all_predictions if any(p for p in all_predictions) else None
