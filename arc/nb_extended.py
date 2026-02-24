"""
Extended Neighborhood Rules for ARC-AGI2

These go beyond exact neighborhood matching to capture more abstract patterns:
1. Count-based rules: "if N+ non-bg neighbors → change"  
2. Directional rules: "non-bg in direction X → fill"
3. Multi-pass rules: apply until convergence
4. Large radius rules: radius 3-4
"""

from typing import List, Tuple, Optional, Dict
from collections import Counter
from arc.grid import Grid, grid_eq, grid_shape, most_common_color


def _count_nb_features(grid: Grid, r: int, c: int, bg: int) -> Dict:
    """Extract neighborhood feature vector for a cell."""
    h, w = grid_shape(grid)
    center = grid[r][c]
    
    # 4-directional counts
    dirs4 = [(0,1), (0,-1), (1,0), (-1,0)]
    # 8-directional counts
    dirs8 = dirs4 + [(1,1), (1,-1), (-1,1), (-1,-1)]
    
    n_nonbg_4 = sum(1 for dr, dc in dirs4 
                    if 0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr][c+dc] != bg)
    n_nonbg_8 = sum(1 for dr, dc in dirs8
                    if 0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr][c+dc] != bg)
    n_same_4 = sum(1 for dr, dc in dirs4
                   if 0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr][c+dc] == center)
    n_same_8 = sum(1 for dr, dc in dirs8
                   if 0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr][c+dc] == center)
    
    # Directional: is there non-bg in each of 4 directions?
    has_up = any(grid[r2][c] != bg for r2 in range(r) if 0 <= r2 < h)
    has_down = any(grid[r2][c] != bg for r2 in range(r+1, h))
    has_left = any(grid[r][c2] != bg for c2 in range(c))
    has_right = any(grid[r][c2] != bg for c2 in range(c+1, w))
    
    # On border?
    on_border = (r == 0 or r == h-1 or c == 0 or c == w-1)
    
    return {
        'center_is_bg': center == bg,
        'n_nonbg_4': n_nonbg_4,
        'n_nonbg_8': n_nonbg_8,
        'n_same_4': n_same_4,
        'n_same_8': n_same_8,
        'has_up': has_up,
        'has_down': has_down,
        'has_left': has_left,
        'has_right': has_right,
        'on_border': on_border,
        'h_between': has_left and has_right,  # between non-bg horizontally
        'v_between': has_up and has_down,      # between non-bg vertically
    }


def learn_count_nb_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn a rule based on neighborhood counts.
    
    For each cell that changes, find what count-based condition triggers it.
    Try: "bg cells with N+ non-bg 4-neighbors become color X"
    """
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    
    # Collect: for bg→non-bg transitions, what count threshold works?
    for threshold in range(1, 5):
        for use_8 in [False, True]:
            ok = True
            fill_color = None
            
            for inp, out in train_pairs:
                h, w = grid_shape(inp)
                
                for r in range(h):
                    for c in range(w):
                        feats = _count_nb_features(inp, r, c, bg)
                        count = feats['n_nonbg_8'] if use_8 else feats['n_nonbg_4']
                        
                        if inp[r][c] == bg:
                            if count >= threshold:
                                # Should change
                                if out[r][c] == bg:
                                    ok = False; break
                                if fill_color is None:
                                    fill_color = out[r][c]
                                elif fill_color != out[r][c]:
                                    ok = False; break
                            else:
                                # Should stay bg
                                if out[r][c] != bg:
                                    ok = False; break
                        else:
                            # Non-bg should stay
                            if out[r][c] != inp[r][c]:
                                ok = False; break
                    if not ok: break
                if not ok: break
            
            if ok and fill_color is not None:
                return {
                    'type': 'count_threshold',
                    'threshold': threshold,
                    'use_8': use_8,
                    'fill_color': fill_color,
                    'bg': bg,
                }
    
    return None


def apply_count_nb_rule(inp: Grid, rule: Dict) -> Grid:
    h, w = grid_shape(inp)
    bg = rule['bg']
    threshold = rule['threshold']
    use_8 = rule['use_8']
    fill_color = rule['fill_color']
    
    result = [row[:] for row in inp]
    for r in range(h):
        for c in range(w):
            if inp[r][c] == bg:
                feats = _count_nb_features(inp, r, c, bg)
                count = feats['n_nonbg_8'] if use_8 else feats['n_nonbg_4']
                if count >= threshold:
                    result[r][c] = fill_color
    return result


def learn_between_fill_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: fill bg cells that are 'between' non-bg cells in same row/col.
    
    Variations:
    - h_only: between horizontally (same row)
    - v_only: between vertically (same col)
    - hv: between in either direction
    - hv_and: between in BOTH directions
    """
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    
    for mode in ['h_only', 'v_only', 'hv', 'hv_and']:
        ok = True
        fill_color = None
        
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            
            for r in range(h):
                for c in range(w):
                    if inp[r][c] != bg:
                        if out[r][c] != inp[r][c]:
                            ok = False; break
                        continue
                    
                    feats = _count_nb_features(inp, r, c, bg)
                    
                    if mode == 'h_only':
                        should_fill = feats['h_between']
                    elif mode == 'v_only':
                        should_fill = feats['v_between']
                    elif mode == 'hv':
                        should_fill = feats['h_between'] or feats['v_between']
                    else:  # hv_and
                        should_fill = feats['h_between'] and feats['v_between']
                    
                    if should_fill:
                        if out[r][c] == bg:
                            ok = False; break
                        if fill_color is None:
                            fill_color = out[r][c]
                        elif fill_color != out[r][c]:
                            # Allow per-cell color (nearest non-bg)
                            fill_color = 'nearest'
                    else:
                        if out[r][c] != bg:
                            ok = False; break
                if not ok: break
            if not ok: break
        
        if ok and fill_color is not None:
            return {
                'type': 'between_fill',
                'mode': mode,
                'fill_color': fill_color,
                'bg': bg,
            }
    
    return None


def apply_between_fill_rule(inp: Grid, rule: Dict) -> Grid:
    h, w = grid_shape(inp)
    bg = rule['bg']
    mode = rule['mode']
    fill_color = rule['fill_color']
    
    result = [row[:] for row in inp]
    for r in range(h):
        for c in range(w):
            if inp[r][c] != bg:
                continue
            
            feats = _count_nb_features(inp, r, c, bg)
            
            if mode == 'h_only':
                should = feats['h_between']
            elif mode == 'v_only':
                should = feats['v_between']
            elif mode == 'hv':
                should = feats['h_between'] or feats['v_between']
            else:
                should = feats['h_between'] and feats['v_between']
            
            if should:
                if fill_color == 'nearest':
                    # Find nearest non-bg color
                    best_d = float('inf')
                    best_c = bg
                    for r2 in range(h):
                        for c2 in range(w):
                            if inp[r2][c2] != bg:
                                d = abs(r-r2) + abs(c-c2)
                                if d < best_d:
                                    best_d = d
                                    best_c = inp[r2][c2]
                    result[r][c] = best_c
                else:
                    result[r][c] = fill_color
    return result


def learn_multipass_nb_rule(train_pairs: List[Tuple[Grid, Grid]], max_passes: int = 5) -> Optional[Dict]:
    """Learn a simple NB rule applied multiple times until convergence.
    
    Rule: if bg cell has >= threshold non-bg neighbors, fill with color.
    Repeat until no more changes.
    """
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    
    for threshold in range(1, 4):
        for use_8 in [False, True]:
            ok = True
            fill_color = None
            
            for inp, out in train_pairs:
                h, w = grid_shape(inp)
                
                # Determine fill color from output
                for r in range(h):
                    for c in range(w):
                        if inp[r][c] == bg and out[r][c] != bg:
                            if fill_color is None:
                                fill_color = out[r][c]
                            elif fill_color != out[r][c]:
                                # Try 'copy_neighbor' mode
                                fill_color = 'copy'
                            break
                    if fill_color is not None: break
                
                if fill_color is None:
                    ok = False; break
                if fill_color == 'copy':
                    ok = False; break  # too complex for now
                
                # Simulate multi-pass
                current = [row[:] for row in inp]
                for _ in range(max_passes):
                    changed = False
                    new = [row[:] for row in current]
                    for r in range(h):
                        for c in range(w):
                            if current[r][c] == bg:
                                dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] if use_8 else [(-1,0),(1,0),(0,-1),(0,1)]
                                n = sum(1 for dr, dc in dirs 
                                       if 0 <= r+dr < h and 0 <= c+dc < w and current[r+dr][c+dc] != bg)
                                if n >= threshold:
                                    new[r][c] = fill_color
                                    changed = True
                    current = new
                    if not changed: break
                
                if not grid_eq(current, out):
                    ok = False; break
            
            if ok and fill_color is not None:
                return {
                    'type': 'multipass',
                    'threshold': threshold,
                    'use_8': use_8,
                    'fill_color': fill_color,
                    'bg': bg,
                    'max_passes': max_passes,
                }
    
    return None


def apply_multipass_nb_rule(inp: Grid, rule: Dict) -> Grid:
    h, w = grid_shape(inp)
    bg = rule['bg']
    threshold = rule['threshold']
    use_8 = rule['use_8']
    fill_color = rule['fill_color']
    max_passes = rule.get('max_passes', 10)
    
    current = [row[:] for row in inp]
    for _ in range(max_passes):
        changed = False
        new = [row[:] for row in current]
        for r in range(h):
            for c in range(w):
                if current[r][c] == bg:
                    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] if use_8 else [(-1,0),(1,0),(0,-1),(0,1)]
                    n = sum(1 for dr, dc in dirs
                           if 0 <= r+dr < h and 0 <= c+dc < w and current[r+dr][c+dc] != bg)
                    if n >= threshold:
                        new[r][c] = fill_color
                        changed = True
        current = new
        if not changed: break
    return current


def learn_multipass_copy_rule(train_pairs: List[Tuple[Grid, Grid]], max_passes: int = 10) -> Optional[Dict]:
    """Multi-pass rule where bg cells copy the color of adjacent non-bg cells.
    Like flood fill or cellular automata growth.
    """
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    
    for threshold in range(1, 4):
        for use_8 in [False, True]:
            ok = True
            
            for inp, out in train_pairs:
                h, w = grid_shape(inp)
                
                current = [row[:] for row in inp]
                for _ in range(max_passes):
                    changed = False
                    new = [row[:] for row in current]
                    for r in range(h):
                        for c in range(w):
                            if current[r][c] == bg:
                                dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] if use_8 else [(-1,0),(1,0),(0,-1),(0,1)]
                                nb_colors = [current[r+dr][c+dc] for dr, dc in dirs
                                            if 0 <= r+dr < h and 0 <= c+dc < w and current[r+dr][c+dc] != bg]
                                if len(nb_colors) >= threshold:
                                    new[r][c] = Counter(nb_colors).most_common(1)[0][0]
                                    changed = True
                    current = new
                    if not changed: break
                
                if not grid_eq(current, out):
                    ok = False; break
            
            if ok:
                return {
                    'type': 'multipass_copy',
                    'threshold': threshold,
                    'use_8': use_8,
                    'bg': bg,
                    'max_passes': max_passes,
                }
    
    return None


def apply_multipass_copy_rule(inp: Grid, rule: Dict) -> Grid:
    h, w = grid_shape(inp)
    bg = rule['bg']
    threshold = rule['threshold']
    use_8 = rule['use_8']
    max_passes = rule.get('max_passes', 10)
    
    current = [row[:] for row in inp]
    for _ in range(max_passes):
        changed = False
        new = [row[:] for row in current]
        for r in range(h):
            for c in range(w):
                if current[r][c] == bg:
                    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] if use_8 else [(-1,0),(1,0),(0,-1),(0,1)]
                    nb_colors = [current[r+dr][c+dc] for dr, dc in dirs
                                if 0 <= r+dr < h and 0 <= c+dc < w and current[r+dr][c+dc] != bg]
                    if len(nb_colors) >= threshold:
                        new[r][c] = Counter(nb_colors).most_common(1)[0][0]
                        changed = True
        current = new
        if not changed: break
    return current


# All extended NB learners
ALL_EXTENDED_NB = [
    ('count_nb', learn_count_nb_rule, apply_count_nb_rule),
    ('between_fill', learn_between_fill_rule, apply_between_fill_rule),
    ('multipass_nb', learn_multipass_nb_rule, apply_multipass_nb_rule),
    ('multipass_copy', learn_multipass_copy_rule, apply_multipass_copy_rule),
]
