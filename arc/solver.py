"""
arc/solver.py — ARC-AGI-2 Task Solver

Strategy:
  1. Detect common transforms across training examples
  2. Apply best transform to test input
  3. Verify consistency (if applied to training inputs, do we get training outputs?)
  4. Return up to 2 attempts (ARC rules: pass@2)
"""

from __future__ import annotations
import json
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from arc.grid import (
    Grid, grid_shape, grid_eq,
    rotate_90, rotate_180, rotate_270,
    flip_h, flip_v, transpose,
    tile, tile_with_flip, tile_checkerboard,
    recolor, extract_subgrid,
    flood_fill_regions, most_common_color, analyze,
)
from arc.pattern_atoms import (
    TransformAtom, detect_transforms, find_common_transforms,
)


@dataclass
class ArcTask:
    task_id: str
    train: List[Tuple[Grid, Grid]]  # (input, output) pairs
    test_inputs: List[Grid]
    test_outputs: Optional[List[Grid]] = None  # ground truth (if available)


@dataclass 
class SolveResult:
    task_id: str
    predictions: List[List[Grid]]  # per test input, up to 2 attempts
    method: str
    confidence: float
    transforms_used: List[TransformAtom]


def load_task(path: str) -> ArcTask:
    """Load ARC task from JSON file"""
    with open(path) as f:
        data = json.load(f)
    
    task_id = path.split('/')[-1].replace('.json', '')
    train = [(ex['input'], ex['output']) for ex in data['train']]
    test_inputs = [ex['input'] for ex in data['test']]
    test_outputs = [ex.get('output') for ex in data['test']] if 'output' in data['test'][0] else None
    
    return ArcTask(task_id=task_id, train=train, test_inputs=test_inputs, test_outputs=test_outputs)


def _detect_all_transforms(inp: Grid, out: Grid) -> List[TransformAtom]:
    """Run all detectors (basic + advanced)"""
    atoms = detect_transforms(inp, out)
    
    from arc.advanced_detectors import ALL_DETECTORS
    for detect_fn, _ in ALL_DETECTORS:
        try:
            atoms.extend(detect_fn(inp, out))
        except Exception:
            continue
    
    atoms.sort(key=lambda a: a.confidence, reverse=True)
    return atoms


def apply_transform(atom: TransformAtom, inp: Grid) -> Optional[Grid]:
    """Apply a detected transform atom to a new input grid"""
    op = atom.operation
    params = atom.params
    
    if op == 'identity':
        return [row[:] for row in inp]
    
    elif op == 'flip_h':
        return flip_h(inp)
    elif op == 'flip_v':
        return flip_v(inp)
    elif op == 'rotate_90':
        return rotate_90(inp)
    elif op == 'rotate_180':
        return rotate_180(inp)
    elif op == 'rotate_270':
        return rotate_270(inp)
    elif op == 'transpose':
        return transpose(inp)
    
    elif op == 'tile':
        return tile(inp, params['repeat_h'], params['repeat_w'])
    elif op == 'tile_flip':
        return tile_with_flip(inp, params['repeat_h'], params['repeat_w'])
    elif op == 'tile_checkerboard':
        return tile_checkerboard(inp, params['repeat_h'], params['repeat_w'])
    
    elif op == 'recolor':
        return recolor(inp, params['map'])
    elif op == 'color_swap':
        cmap = {}
        for a, b in params['swaps']:
            cmap[a] = b
            cmap[b] = a
        return recolor(inp, cmap)
    
    elif op == 'scale':
        factor = params['factor']
        h, w = grid_shape(inp)
        result = []
        for r in range(h):
            for _ in range(factor):
                row = []
                for c in range(w):
                    row.extend([inp[r][c]] * factor)
                result.append(row)
        return result
    
    elif op == 'extract':
        r, c, eh, ew = params['r'], params['c'], params['h'], params['w']
        h, w = grid_shape(inp)
        if r + eh <= h and c + ew <= w:
            return extract_subgrid(inp, r, c, r + eh - 1, c + ew - 1)
        return None
    
    elif op == 'extract_region':
        color = params['color']
        regions = flood_fill_regions(inp)
        for reg in regions:
            if reg['color'] == color:
                r1, c1, r2, c2 = reg['bbox']
                return extract_subgrid(inp, r1, c1, r2, c2)
        return None
    
    # Try advanced detectors
    from arc.advanced_detectors import ALL_DETECTORS
    for _, apply_fn in ALL_DETECTORS:
        try:
            result = apply_fn(atom, inp)
            if result is not None:
                return result
        except Exception:
            continue
    
    return None


def verify_transform(atom: TransformAtom, train_pairs: List[Tuple[Grid, Grid]]) -> bool:
    """Verify transform produces correct output for ALL training examples"""
    for inp, expected_out in train_pairs:
        result = apply_transform(atom, inp)
        if result is None or not grid_eq(result, expected_out):
            return False
    return True


def solve_task(task: ArcTask) -> SolveResult:
    """
    Solve an ARC-AGI-2 task.
    
    Strategy:
      1. Find common transforms across training examples
      2. Verify each against all training pairs
      3. Apply verified transforms to test inputs
      4. Return up to 2 attempts per test input
    """
    # Step 1: Detect all transforms per pair
    per_pair = [_detect_all_transforms(inp, out) for inp, out in task.train]
    
    # Step 2: Find common operations across all pairs
    if per_pair:
        first_ops = {(a.category, a.operation) for a in per_pair[0]}
        common_ops = first_ops
        for pair_atoms in per_pair[1:]:
            common_ops &= {(a.category, a.operation) for a in pair_atoms}
        
        # Collect common atoms
        common = []
        for cat, op in common_ops:
            atoms = [a for a in per_pair[0] if a.category == cat and a.operation == op]
            common.extend(atoms)
        common.sort(key=lambda a: a.confidence, reverse=True)
    else:
        common = []
    
    # Step 3: Verify each against all training pairs
    verified = []
    for atom in common:
        if verify_transform(atom, task.train):
            verified.append(atom)
    
    # Step 4: If no common verified, try individual pair atoms
    if not verified:
        for pair_atoms in per_pair:
            for atom in pair_atoms:
                if atom.confidence >= 0.7 and verify_transform(atom, task.train):
                    if atom not in verified:
                        verified.append(atom)
            if verified:
                break
    
    # Step 4: Apply to test inputs
    predictions = []
    for test_inp in task.test_inputs:
        attempts = []
        for atom in verified[:2]:  # Up to 2 attempts
            result = apply_transform(atom, test_inp)
            if result is not None:
                attempts.append(result)
        predictions.append(attempts)
    
    method = 'none'
    confidence = 0.0
    if verified:
        method = f"{verified[0].category}:{verified[0].operation}"
        confidence = verified[0].confidence
    
    return SolveResult(
        task_id=task.task_id,
        predictions=predictions,
        method=method,
        confidence=confidence,
        transforms_used=verified[:2],
    )
