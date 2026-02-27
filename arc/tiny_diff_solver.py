"""
arc/tiny_diff_solver.py — Solver for tasks with tiny input→output differences

Targets: same-size tasks where output differs from input by <5% of cells.
Strategy: learn what changes and why, then apply the same rule to test.

Patterns detected:
1. Color stamp at special positions (marker color → add surrounding pattern)
2. Single-cell recolor based on local context
3. Fill specific cells based on global property (e.g., intersection point)
4. Conditional recolor (if neighbor count = N, change color)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
from arc.cross_engine import CrossPiece

Grid = List[List[int]]


def _grid_np(g):
    return np.array(g, dtype=int)


def _np_grid(a):
    return a.tolist()


def _diff_cells(inp, out):
    """Return list of (r, c, inp_color, out_color) for differing cells"""
    diffs = []
    H, W = inp.shape
    for r in range(H):
        for c in range(W):
            if inp[r, c] != out[r, c]:
                diffs.append((r, c, int(inp[r, c]), int(out[r, c])))
    return diffs


def _neighborhood(grid, r, c, radius=1):
    """Get neighborhood values"""
    H, W = grid.shape
    nb = {}
    for dr in range(-radius, radius+1):
        for dc in range(-radius, radius+1):
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W:
                nb[(dr, dc)] = int(grid[nr, nc])
    return nb


# === Strategy 1: Color-based stamp ===
# Each input color triggers a specific pattern addition around it

def learn_color_stamp(train_pairs):
    """Learn: for each color, what pattern is added around it?"""
    stamp_rules = {}  # color -> list of (dr, dc, new_color)
    
    for inp_g, out_g in train_pairs:
        inp = _grid_np(inp_g)
        out = _grid_np(out_g)
        if inp.shape != out.shape:
            return None
        
        diffs = _diff_cells(inp, out)
        if not diffs:
            continue
        
        # For each diff cell, find nearest non-bg input cell
        H, W = inp.shape
        for r, c, old_c, new_c in diffs:
            # Search for source cell (non-zero in input, within radius 3)
            best_src = None
            best_dist = 999
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and inp[nr, nc] != 0:
                        dist = abs(dr) + abs(dc)
                        if 0 < dist < best_dist:
                            best_dist = dist
                            best_src = (nr, nc, int(inp[nr, nc]), -dr, -dc)
            
            if best_src:
                src_r, src_c, src_color, off_r, off_c = best_src
                key = src_color
                if key not in stamp_rules:
                    stamp_rules[key] = set()
                stamp_rules[key].add((off_r, off_c, new_c))
    
    if not stamp_rules:
        return None
    
    # Convert to sorted lists
    stamp_rules = {k: sorted(v) for k, v in stamp_rules.items()}
    return stamp_rules


def apply_color_stamp(inp_g, stamp_rules):
    """Apply stamp rules: for each colored cell, add the pattern"""
    inp = _grid_np(inp_g)
    out = inp.copy()
    H, W = inp.shape
    
    for r in range(H):
        for c in range(W):
            color = int(inp[r, c])
            if color in stamp_rules:
                for dr, dc, new_c in stamp_rules[color]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and out[nr, nc] == 0:
                        out[nr, nc] = new_c
    
    return _np_grid(out)


# === Strategy 2: Intersection/overlap recolor ===
# Find cells where multiple objects' influence overlaps

def learn_intersection_fill(train_pairs):
    """Learn: fill intersection points of row/col projections"""
    rules = []
    
    for inp_g, out_g in train_pairs:
        inp = _grid_np(inp_g)
        out = _grid_np(out_g)
        if inp.shape != out.shape:
            return None
        
        diffs = _diff_cells(inp, out)
        if not diffs:
            continue
        
        H, W = inp.shape
        
        # For each diff, check if it's at intersection of row/col with same-colored objects
        for r, c, old_c, new_c in diffs:
            # Find colors in same row and same col
            row_colors = set(int(inp[r, cc]) for cc in range(W) if cc != c and inp[r, cc] != 0)
            col_colors = set(int(inp[rr, c]) for rr in range(H) if rr != r and inp[rr, c] != 0)
            
            intersect = row_colors & col_colors
            if new_c in intersect:
                rules.append({
                    'type': 'row_col_intersect',
                    'fill_color': new_c,
                })
    
    if not rules:
        return None
    
    # Check consistency
    fill_colors = set(r['fill_color'] for r in rules)
    if len(fill_colors) > 3:  # too many different colors, probably not this pattern
        return None
    
    return {'type': 'row_col_intersect', 'fill_colors': list(fill_colors)}


def apply_intersection_fill(inp_g, rule):
    """Fill cells at row/col intersections of colored objects"""
    inp = _grid_np(inp_g)
    out = inp.copy()
    H, W = inp.shape
    
    for r in range(H):
        for c in range(W):
            if inp[r, c] != 0:
                continue
            
            row_colors = set(int(inp[r, cc]) for cc in range(W) if cc != c and inp[r, cc] != 0)
            col_colors = set(int(inp[rr, c]) for rr in range(H) if rr != r and inp[rr, c] != 0)
            
            intersect = row_colors & col_colors
            for fc in rule.get('fill_colors', []):
                if fc in intersect:
                    out[r, c] = fc
                    break
    
    return _np_grid(out)


# === Strategy 3: Conditional neighbor recolor ===

def learn_neighbor_recolor(train_pairs):
    """Learn: recolor cells based on their neighbor count/pattern"""
    rules = []
    
    for inp_g, out_g in train_pairs:
        inp = _grid_np(inp_g)
        out = _grid_np(out_g)
        if inp.shape != out.shape:
            return None
        
        diffs = _diff_cells(inp, out)
        if not diffs:
            continue
        
        H, W = inp.shape
        bg = int(Counter(inp.flatten()).most_common(1)[0][0])
        
        for r, c, old_c, new_c in diffs:
            # Count non-bg neighbors
            nb_count = 0
            nb_colors = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and int(inp[nr, nc]) != bg:
                    nb_count += 1
                    nb_colors.append(int(inp[nr, nc]))
            
            rules.append({
                'old_color': old_c,
                'new_color': new_c,
                'nb_count': nb_count,
                'nb_colors': sorted(nb_colors),
            })
    
    if not rules:
        return None
    
    # Find consistent rule
    # Group by (old_color, new_color, nb_count)
    grouped = Counter((r['old_color'], r['new_color'], r['nb_count']) for r in rules)
    if len(grouped) == 1:
        (old_c, new_c, nb_cnt), _ = grouped.most_common(1)[0]
        return {'old_color': old_c, 'new_color': new_c, 'nb_count': nb_cnt}
    
    return None


def apply_neighbor_recolor(inp_g, rule):
    inp = _grid_np(inp_g)
    out = inp.copy()
    H, W = inp.shape
    bg = int(Counter(inp.flatten()).most_common(1)[0][0])
    
    for r in range(H):
        for c in range(W):
            if int(inp[r, c]) != rule['old_color']:
                continue
            nb_count = 0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and int(inp[nr, nc]) != bg:
                    nb_count += 1
            if nb_count == rule['nb_count']:
                out[r, c] = rule['new_color']
    
    return _np_grid(out)


# === Strategy 4: Diagonal stamp (X-pattern) per color ===

def learn_diagonal_stamp(train_pairs):
    """Learn: specific colors get diagonal(X) or cross(+) stamps"""
    color_patterns = {}  # src_color -> {(dr,dc): stamp_color}
    
    for inp_g, out_g in train_pairs:
        inp = _grid_np(inp_g)
        out = _grid_np(out_g)
        if inp.shape != out.shape:
            return None
        
        H, W = inp.shape
        diffs = _diff_cells(inp, out)
        
        # For each non-zero input cell, check what was added around it
        for r in range(H):
            for c in range(W):
                src_c = int(inp[r, c])
                if src_c == 0:
                    continue
                
                pattern = {}
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if int(out[nr, nc]) != int(inp[nr, nc]) and int(inp[nr, nc]) == 0:
                                pattern[(dr, dc)] = int(out[nr, nc])
                
                if pattern:
                    if src_c not in color_patterns:
                        color_patterns[src_c] = pattern
                    else:
                        # Verify consistency
                        if color_patterns[src_c] != pattern:
                            # Allow subset (some cells might be blocked by boundaries)
                            pass
    
    if not color_patterns:
        return None
    
    return color_patterns


def apply_diagonal_stamp(inp_g, color_patterns):
    inp = _grid_np(inp_g)
    out = inp.copy()
    H, W = inp.shape
    
    for r in range(H):
        for c in range(W):
            src_c = int(inp[r, c])
            if src_c in color_patterns:
                for (dr, dc), stamp_c in color_patterns[src_c].items():
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and out[nr, nc] == 0:
                        out[nr, nc] = stamp_c
    
    return _np_grid(out)


# === Strategy 5: Row/col projection fill ===

def learn_projection_fill(train_pairs):
    """Fill bg cells that are in the same row AND col as non-bg cells of specific colors"""
    for inp_g, out_g in train_pairs:
        inp = _grid_np(inp_g)
        out = _grid_np(out_g)
        if inp.shape != out.shape:
            return None
        
        H, W = inp.shape
        diffs = _diff_cells(inp, out)
        if not diffs:
            continue
        
        # Check: each diff cell gets the color that appears in both its row and col
        all_match = True
        for r, c, old_c, new_c in diffs:
            row_cs = set(int(inp[r, cc]) for cc in range(W)) - {0}
            col_cs = set(int(inp[rr, c]) for rr in range(H)) - {0}
            if new_c not in (row_cs & col_cs):
                all_match = False
                break
        
        if not all_match:
            return None
    
    return {'type': 'projection_fill'}


def apply_projection_fill(inp_g, rule):
    inp = _grid_np(inp_g)
    out = inp.copy()
    H, W = inp.shape
    
    for r in range(H):
        for c in range(W):
            if inp[r, c] != 0:
                continue
            row_cs = set(int(inp[r, cc]) for cc in range(W)) - {0}
            col_cs = set(int(inp[rr, c]) for rr in range(H)) - {0}
            inter = row_cs & col_cs
            if len(inter) == 1:
                out[r, c] = inter.pop()
    
    return _np_grid(out)


# === Main piece generator ===

def _verify(fn, train_pairs):
    for inp, out in train_pairs:
        pred = fn(inp)
        if pred is None or not np.array_equal(_grid_np(pred), _grid_np(out)):
            return False
    return True


def generate_tiny_diff_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """Generate pieces for tiny-diff tasks"""
    pieces = []
    if not train_pairs:
        return pieces
    
    # Check if this is a tiny-diff task
    for inp, out in train_pairs:
        inp_np = _grid_np(inp)
        out_np = _grid_np(out)
        if inp_np.shape != out_np.shape:
            return pieces
        diff_pct = np.sum(inp_np != out_np) / inp_np.size
        if diff_pct > 0.3:  # not a tiny-diff task
            return pieces
    
    # Try each strategy
    strategies = [
        ('color_stamp', learn_color_stamp, apply_color_stamp),
        ('diagonal_stamp', learn_diagonal_stamp, apply_diagonal_stamp),
        ('intersection_fill', learn_intersection_fill, apply_intersection_fill),
        ('neighbor_recolor', learn_neighbor_recolor, apply_neighbor_recolor),
        ('projection_fill', learn_projection_fill, apply_projection_fill),
    ]
    
    for name, learn_fn, apply_fn in strategies:
        try:
            rule = learn_fn(train_pairs)
            if rule is None:
                continue
            
            fn = lambda inp, r=rule, af=apply_fn: af(inp, r)
            if _verify(fn, train_pairs):
                pieces.append(CrossPiece(
                    name=f"tiny_diff:{name}",
                    apply_fn=fn,
                ))
                return pieces  # one is enough
        except Exception:
            continue
    
    return pieces
