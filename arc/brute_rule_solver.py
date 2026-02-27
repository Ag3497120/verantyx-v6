"""
arc/brute_rule_solver.py — Brute-force rule discovery

Tries a wide range of simple transformations on same-size tasks:
1. Rotate 90/180/270
2. Flip horizontal/vertical
3. Transpose
4. Replace one color with another
5. Remove one color (→ bg)
6. Gravity in each direction
7. Sort rows/cols by some criterion
"""

import numpy as np
from typing import List, Tuple
from collections import Counter
from arc.cross_engine import CrossPiece

Grid = List[List[int]]

def _g(g): return np.array(g, dtype=int)
def _l(a): return a.tolist()
def _bg(g): return int(Counter(g.flatten()).most_common(1)[0][0])


def _try_transforms(train_pairs):
    """Try all basic transforms, return list of (name, fn) that work on all train"""
    results = []
    
    transforms = [
        ('rot90', lambda g: np.rot90(g, 1)),
        ('rot180', lambda g: np.rot90(g, 2)),
        ('rot270', lambda g: np.rot90(g, 3)),
        ('flip_h', lambda g: g[:, ::-1]),
        ('flip_v', lambda g: g[::-1, :]),
        ('transpose', lambda g: g.T),
        ('flip_h_rot90', lambda g: np.rot90(g[:, ::-1], 1)),
        ('flip_v_rot90', lambda g: np.rot90(g[::-1, :], 1)),
    ]
    
    for name, fn in transforms:
        all_match = True
        for inp_g, out_g in train_pairs:
            inp, out = _g(inp_g), _g(out_g)
            try:
                pred = fn(inp)
                if not np.array_equal(pred, out):
                    all_match = False
                    break
            except:
                all_match = False
                break
        
        if all_match:
            results.append((name, lambda g, f=fn: _l(f(_g(g)))))
    
    return results


def _try_color_replace(train_pairs):
    """Try replacing one color with another"""
    results = []
    
    for old_c in range(10):
        for new_c in range(10):
            if old_c == new_c:
                continue
            
            all_match = True
            for inp_g, out_g in train_pairs:
                inp, out = _g(inp_g), _g(out_g)
                if inp.shape != out.shape:
                    all_match = False
                    break
                pred = inp.copy()
                pred[pred == old_c] = new_c
                if not np.array_equal(pred, out):
                    all_match = False
                    break
            
            if all_match:
                def _apply(g, oc=old_c, nc=new_c):
                    a = _g(g)
                    a[a == oc] = nc
                    return _l(a)
                results.append((f'replace_{old_c}_with_{new_c}', _apply))
    
    return results


def _try_gravity(train_pairs):
    """Try gravity (push non-bg cells in a direction)"""
    results = []
    
    for direction in ['down', 'up', 'left', 'right']:
        all_match = True
        for inp_g, out_g in train_pairs:
            inp, out = _g(inp_g), _g(out_g)
            if inp.shape != out.shape:
                all_match = False
                break
            
            bg = _bg(inp)
            pred = _apply_gravity(inp, bg, direction)
            if not np.array_equal(pred, out):
                all_match = False
                break
        
        if all_match:
            def _apply(g, d=direction):
                a = _g(g)
                bg = _bg(a)
                return _l(_apply_gravity(a, bg, d))
            results.append((f'gravity_{direction}', _apply))
    
    return results


def _apply_gravity(grid, bg, direction):
    """Apply gravity: push non-bg cells in given direction"""
    H, W = grid.shape
    out = np.full_like(grid, bg)
    
    if direction == 'down':
        for c in range(W):
            col = [int(grid[r, c]) for r in range(H) if grid[r, c] != bg]
            for i, v in enumerate(col):
                out[H - len(col) + i, c] = v
    elif direction == 'up':
        for c in range(W):
            col = [int(grid[r, c]) for r in range(H) if grid[r, c] != bg]
            for i, v in enumerate(col):
                out[i, c] = v
    elif direction == 'right':
        for r in range(H):
            row = [int(grid[r, c]) for c in range(W) if grid[r, c] != bg]
            for i, v in enumerate(row):
                out[r, W - len(row) + i] = v
    elif direction == 'left':
        for r in range(H):
            row = [int(grid[r, c]) for c in range(W) if grid[r, c] != bg]
            for i, v in enumerate(row):
                out[r, i] = v
    
    return out


def _try_sort_rows(train_pairs):
    """Try sorting rows by some criterion"""
    results = []
    
    for criterion in ['nonbg_count_asc', 'nonbg_count_desc', 'color_sum_asc', 'color_sum_desc']:
        all_match = True
        for inp_g, out_g in train_pairs:
            inp, out = _g(inp_g), _g(out_g)
            if inp.shape != out.shape:
                all_match = False
                break
            
            bg = _bg(inp)
            rows = list(range(inp.shape[0]))
            
            if criterion == 'nonbg_count_asc':
                rows.sort(key=lambda r: sum(1 for c in range(inp.shape[1]) if inp[r,c] != bg))
            elif criterion == 'nonbg_count_desc':
                rows.sort(key=lambda r: -sum(1 for c in range(inp.shape[1]) if inp[r,c] != bg))
            elif criterion == 'color_sum_asc':
                rows.sort(key=lambda r: sum(int(inp[r,c]) for c in range(inp.shape[1])))
            elif criterion == 'color_sum_desc':
                rows.sort(key=lambda r: -sum(int(inp[r,c]) for c in range(inp.shape[1])))
            
            pred = inp[rows, :]
            if not np.array_equal(pred, out):
                all_match = False
                break
        
        if all_match:
            def _apply(g, crit=criterion):
                a = _g(g)
                bg = _bg(a)
                rows = list(range(a.shape[0]))
                if crit == 'nonbg_count_asc':
                    rows.sort(key=lambda r: sum(1 for c in range(a.shape[1]) if a[r,c] != bg))
                elif crit == 'nonbg_count_desc':
                    rows.sort(key=lambda r: -sum(1 for c in range(a.shape[1]) if a[r,c] != bg))
                elif crit == 'color_sum_asc':
                    rows.sort(key=lambda r: sum(int(a[r,c]) for c in range(a.shape[1])))
                elif crit == 'color_sum_desc':
                    rows.sort(key=lambda r: -sum(int(a[r,c]) for c in range(a.shape[1])))
                return _l(a[rows, :])
            results.append((f'sort_rows_{criterion}', _apply))
    
    return results


def _verify(fn, train_pairs):
    for inp, out in train_pairs:
        pred = fn(inp)
        if pred is None or not np.array_equal(_g(pred), _g(out)):
            return False
    return True


def generate_brute_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    pieces = []
    if not train_pairs:
        return pieces
    
    # Try all strategies
    for try_fn in [_try_transforms, _try_color_replace, _try_gravity, _try_sort_rows]:
        try:
            found = try_fn(train_pairs)
            for name, fn in found:
                if _verify(fn, train_pairs):
                    pieces.append(CrossPiece(name=f"brute:{name}", apply_fn=fn))
                    return pieces
        except Exception:
            continue
    
    return pieces
