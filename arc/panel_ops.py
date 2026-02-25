"""
arc/panel_ops.py — Panel split + reduction operations for ARC-AGI-2

Handles tasks where a grid is split into panels and combined via operations:
- Split by separator rows/cols
- Split by equal partitions
- Operations: XOR, OR, AND, majority, diff, overlay, select-unique
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from arc.grid import Grid, grid_shape, grid_eq, most_common_color


# ============================================================
# Panel splitting
# ============================================================

def _find_separator_rows(g: Grid, bg: int) -> List[int]:
    """Find rows that are entirely one non-bg color (separator lines)"""
    h, w = grid_shape(g)
    seps = []
    for r in range(h):
        vals = set(g[r])
        if len(vals) == 1 and vals.pop() != bg:
            seps.append(r)
    return seps


def _find_separator_cols(g: Grid, bg: int) -> List[int]:
    """Find cols that are entirely one non-bg color (separator lines)"""
    h, w = grid_shape(g)
    seps = []
    for c in range(w):
        vals = set(g[r][c] for r in range(h))
        if len(vals) == 1 and vals.pop() != bg:
            seps.append(c)
    return seps


def _find_bg_separator_rows(g: Grid, bg: int) -> List[int]:
    """Find rows that are entirely background color"""
    h, w = grid_shape(g)
    seps = []
    for r in range(h):
        if all(g[r][c] == bg for c in range(w)):
            seps.append(r)
    return seps


def _find_bg_separator_cols(g: Grid, bg: int) -> List[int]:
    """Find cols that are entirely background color"""
    h, w = grid_shape(g)
    seps = []
    for c in range(w):
        if all(g[r][c] == bg for r in range(h)):
            seps.append(c)
    return seps


def _split_by_separator_rows(g: Grid, seps: List[int]) -> List[Grid]:
    """Split grid into panels by separator rows"""
    h, w = grid_shape(g)
    panels = []
    # Group consecutive separators
    boundaries = [-1] + seps + [h]
    for i in range(len(boundaries) - 1):
        r_start = boundaries[i] + 1
        r_end = boundaries[i + 1]
        if r_start < r_end:
            panel = [g[r][:] for r in range(r_start, r_end)]
            panels.append(panel)
    return panels


def _split_by_separator_cols(g: Grid, seps: List[int]) -> List[Grid]:
    """Split grid into panels by separator cols"""
    h, w = grid_shape(g)
    panels = []
    boundaries = [-1] + seps + [w]
    for i in range(len(boundaries) - 1):
        c_start = boundaries[i] + 1
        c_end = boundaries[i + 1]
        if c_start < c_end:
            panel = [g[r][c_start:c_end] for r in range(h)]
            panels.append(panel)
    return panels


def split_into_panels(g: Grid, bg: int = 0) -> Tuple[Optional[str], List[Grid]]:
    """Auto-detect panels via separators or equal partitions.
    Returns (split_mode, panels) or (None, []) if no split found.
    """
    h, w = grid_shape(g)
    
    # Try separator rows (non-bg color)
    sep_rows = _find_separator_rows(g, bg)
    if sep_rows:
        # Check if separators are evenly spaced
        panels = _split_by_separator_rows(g, sep_rows)
        if len(panels) >= 2:
            sizes = [(len(p), len(p[0])) for p in panels]
            if len(set(sizes)) == 1:
                return ('sep_row', panels)
    
    # Try separator cols (non-bg color)
    sep_cols = _find_separator_cols(g, bg)
    if sep_cols:
        panels = _split_by_separator_cols(g, sep_cols)
        if len(panels) >= 2:
            sizes = [(len(p), len(p[0])) for p in panels]
            if len(set(sizes)) == 1:
                return ('sep_col', panels)
    
    # Try bg separator rows
    bg_sep_rows = _find_bg_separator_rows(g, bg)
    if bg_sep_rows:
        panels = _split_by_separator_rows(g, bg_sep_rows)
        if len(panels) >= 2:
            sizes = [(len(p), len(p[0])) for p in panels]
            if len(set(sizes)) == 1:
                return ('bg_sep_row', panels)
    
    # Try bg separator cols
    bg_sep_cols = _find_bg_separator_cols(g, bg)
    if bg_sep_cols:
        panels = _split_by_separator_cols(g, bg_sep_cols)
        if len(panels) >= 2:
            sizes = [(len(p), len(p[0])) for p in panels]
            if len(set(sizes)) == 1:
                return ('bg_sep_col', panels)
    
    # Try equal row partitions (2, 3, 4)
    for n in [2, 3, 4]:
        if h % n == 0:
            ph = h // n
            panels = [[g[r + i * ph][c] for c in range(w)] for i in range(n) for r in range(ph)]
            # Reshape
            panels_2d = []
            for i in range(n):
                panel = [g[r][:] for r in range(i * ph, (i + 1) * ph)]
                panels_2d.append(panel)
            return (f'equal_row_{n}', panels_2d)
    
    # Try equal col partitions (2, 3, 4)
    for n in [2, 3, 4]:
        if w % n == 0:
            pw = w // n
            panels_2d = []
            for i in range(n):
                panel = [g[r][i * pw:(i + 1) * pw] for r in range(h)]
                panels_2d.append(panel)
            return (f'equal_col_{n}', panels_2d)
    
    return (None, [])


# ============================================================
# Panel reduction operations  
# ============================================================

def _panel_op_xor(panels: List[Grid], bg: int) -> Grid:
    """XOR: cell is non-bg if exactly one panel has non-bg there"""
    if not panels:
        return []
    h, w = len(panels[0]), len(panels[0][0])
    result = [[bg] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            non_bg = [p[r][c] for p in panels if p[r][c] != bg]
            if len(non_bg) == 1:
                result[r][c] = non_bg[0]
    return result


def _panel_op_or(panels: List[Grid], bg: int) -> Grid:
    """OR: cell is non-bg if ANY panel has non-bg there"""
    if not panels:
        return []
    h, w = len(panels[0]), len(panels[0][0])
    result = [[bg] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            for p in panels:
                if p[r][c] != bg:
                    result[r][c] = p[r][c]
                    break
    return result


def _panel_op_and(panels: List[Grid], bg: int) -> Grid:
    """AND: cell is non-bg if ALL panels have non-bg there"""
    if not panels:
        return []
    h, w = len(panels[0]), len(panels[0][0])
    result = [[bg] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            vals = [p[r][c] for p in panels]
            if all(v != bg for v in vals):
                result[r][c] = vals[0]
    return result


def _panel_op_majority(panels: List[Grid], bg: int) -> Grid:
    """Majority: most common non-bg value wins"""
    from collections import Counter
    if not panels:
        return []
    h, w = len(panels[0]), len(panels[0][0])
    result = [[bg] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            vals = [p[r][c] for p in panels if p[r][c] != bg]
            if vals:
                cnt = Counter(vals)
                result[r][c] = cnt.most_common(1)[0][0]
    return result


def _panel_op_diff(panels: List[Grid], bg: int) -> Grid:
    """Diff: cells where panels differ"""
    if len(panels) < 2:
        return panels[0] if panels else []
    h, w = len(panels[0]), len(panels[0][0])
    result = [[bg] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            vals = [p[r][c] for p in panels]
            if len(set(vals)) > 1:
                # Return the non-bg value that's different from the first panel
                for v in vals:
                    if v != vals[0]:
                        result[r][c] = v
                        break
    return result


def _panel_op_overlay_priority(panels: List[Grid], bg: int) -> Grid:
    """Overlay: later panels overwrite earlier ones (non-bg cells)"""
    if not panels:
        return []
    h, w = len(panels[0]), len(panels[0][0])
    result = [[bg] * w for _ in range(h)]
    for p in panels:
        for r in range(h):
            for c in range(w):
                if p[r][c] != bg:
                    result[r][c] = p[r][c]
    return result


def _panel_op_first_nonbg(panels: List[Grid], bg: int) -> Grid:
    """First non-bg value encountered across panels"""
    return _panel_op_or(panels, bg)


def _panel_op_overlay_with_remap(panels: List[Grid], bg: int, remap: Dict[int, int] = None) -> Grid:
    """Overlay panels, optionally remapping colors"""
    if not panels:
        return []
    h, w = len(panels[0]), len(panels[0][0])
    result = [[bg] * w for _ in range(h)]
    for p in panels:
        for r in range(h):
            for c in range(w):
                v = p[r][c]
                if v != bg:
                    if remap and v in remap:
                        result[r][c] = remap[v]
                    else:
                        result[r][c] = v
    return result


def _panel_op_normalize(panels: List[Grid], bg: int) -> Grid:
    """Normalize: replace all non-bg colors with a single color, then overlay"""
    if not panels:
        return []
    h, w = len(panels[0]), len(panels[0][0])
    # Count per-cell: how many panels have non-bg
    result = [[bg] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            count = sum(1 for p in panels if p[r][c] != bg)
            vals = [p[r][c] for p in panels if p[r][c] != bg]
            if count == 1:
                result[r][c] = vals[0]
            elif count > 1:
                # All same -> keep; different -> special
                if len(set(vals)) == 1:
                    result[r][c] = vals[0]
                else:
                    result[r][c] = vals[-1]  # last wins
    return result


# ============================================================
# Panel selection (pick one panel based on property)
# ============================================================

def _panel_select_unique(panels: List[Grid], bg: int) -> Optional[Grid]:
    """Select the panel that is unique (different from all others)"""
    n = len(panels)
    if n < 3:
        return None
    # Convert to hashable
    def panel_key(p):
        return tuple(tuple(row) for row in p)
    
    keys = [panel_key(p) for p in panels]
    from collections import Counter
    cnt = Counter(keys)
    
    for i, k in enumerate(keys):
        if cnt[k] == 1 and len(cnt) > 1:
            # Check this is truly unique (others share a common pattern)
            return panels[i]
    return None


def _panel_select_most_colors(panels: List[Grid], bg: int) -> Optional[Grid]:
    """Select panel with most non-bg colors"""
    if not panels:
        return None
    def n_colors(p):
        colors = set()
        for row in p:
            for v in row:
                if v != bg:
                    colors.add(v)
        return len(colors)
    return max(panels, key=n_colors)


def _panel_select_most_filled(panels: List[Grid], bg: int) -> Optional[Grid]:
    """Select panel with most non-bg cells"""
    if not panels:
        return None
    def n_filled(p):
        return sum(1 for row in p for v in row if v != bg)
    return max(panels, key=n_filled)


def _panel_select_least_filled(panels: List[Grid], bg: int) -> Optional[Grid]:
    """Select panel with fewest non-bg cells"""
    if not panels:
        return None
    def n_filled(p):
        return sum(1 for row in p for v in row if v != bg)
    return min(panels, key=n_filled)


# ============================================================
# Learning interface
# ============================================================

ALL_OPS = {
    'xor': _panel_op_xor,
    'or': _panel_op_or,
    'and': _panel_op_and,
    'majority': _panel_op_majority,
    'diff': _panel_op_diff,
    'overlay': _panel_op_overlay_priority,
    'normalize': _panel_op_normalize,
}

ALL_SELECTS = {
    'unique': _panel_select_unique,
    'most_colors': _panel_select_most_colors,
    'most_filled': _panel_select_most_filled,
    'least_filled': _panel_select_least_filled,
}


def _all_splits(g: Grid, bg: int) -> List[Tuple[str, List[Grid]]]:
    """Generate ALL possible panel splits (not just first match)"""
    h, w = grid_shape(g)
    results = []
    
    # Separator-based splits
    for find_fn, split_fn, label in [
        (_find_separator_rows, _split_by_separator_rows, 'sep_row'),
        (_find_separator_cols, _split_by_separator_cols, 'sep_col'),
        (_find_bg_separator_rows, _split_by_separator_rows, 'bg_sep_row'),
        (_find_bg_separator_cols, _split_by_separator_cols, 'bg_sep_col'),
    ]:
        seps = find_fn(g, bg)
        if seps:
            panels = split_fn(g, seps)
            if len(panels) >= 2:
                sizes = [(len(p), len(p[0])) for p in panels]
                if len(set(sizes)) == 1:
                    results.append((label, panels))
    
    # Equal partitions
    for n in [2, 3, 4, 5, 6]:
        if h % n == 0:
            ph = h // n
            panels = [g[i*ph:(i+1)*ph] for i in range(n)]
            # deep copy rows
            panels = [[row[:] for row in p] for p in panels]
            results.append((f'equal_row_{n}', panels))
        if w % n == 0:
            pw = w // n
            panels = [[g[r][i*pw:(i+1)*pw] for r in range(h)] for i in range(n)]
            results.append((f'equal_col_{n}', panels))
    
    return results


def _candidate_bgs(train_pairs: List[Tuple[Grid, Grid]]) -> List[int]:
    """Get candidate background colors, sorted by frequency"""
    from collections import Counter
    cnt = Counter()
    for inp, _ in train_pairs:
        for row in inp:
            cnt.update(row)
    # Return top 2 most common
    return [c for c, _ in cnt.most_common(2)]


def learn_panel_reduce(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn panel split + reduction rule from training pairs.
    Returns dict with {split_mode, op_name, bg} or None.
    """
    for bg in _candidate_bgs(train_pairs):
        # Get output size from first pair
        out0 = train_pairs[0][1]
        oh, ow = len(out0), len(out0[0])
        
        # Try all splits on first input
        inp0 = train_pairs[0][0]
        for split_mode, panels in _all_splits(inp0, bg):
            if len(panels) < 2:
                continue
            ph, pw = len(panels[0]), len(panels[0][0])
            if (ph, pw) != (oh, ow):
                continue
            
            # Try all operations
            for op_name, op_fn in ALL_OPS.items():
                result = op_fn(panels, bg)
                if grid_eq(result, out0):
                    ok = True
                    for inp2, out2 in train_pairs[1:]:
                        all_sp = _all_splits(inp2, bg)
                        matched = False
                        for sm2, ps2 in all_sp:
                            if len(ps2) >= 2 and len(ps2[0]) == ph and len(ps2[0][0]) == pw:
                                r2 = op_fn(ps2, bg)
                                if grid_eq(r2, out2):
                                    matched = True
                                    break
                        if not matched:
                            ok = False
                            break
                    if ok:
                        return {'split_mode': split_mode, 'op': op_name, 'bg': bg}
            
            # Try selection
            for sel_name, sel_fn in ALL_SELECTS.items():
                result = sel_fn(panels, bg)
                if result is not None and grid_eq(result, out0):
                    ok = True
                    for inp2, out2 in train_pairs[1:]:
                        all_sp = _all_splits(inp2, bg)
                        matched = False
                        for sm2, ps2 in all_sp:
                            if len(ps2) >= 2:
                                r2 = sel_fn(ps2, bg)
                                if r2 is not None and grid_eq(r2, out2):
                                    matched = True
                                    break
                        if not matched:
                            ok = False
                            break
                    if ok:
                        return {'split_mode': split_mode, 'op': f'select_{sel_name}', 'bg': bg}
    
    return None


def apply_panel_reduce(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply learned panel reduce rule"""
    bg = params['bg']
    op = params['op']
    target_mode = params.get('split_mode', '')
    
    for split_mode, panels in _all_splits(inp, bg):
        if len(panels) < 2:
            continue
        # Prefer matching split mode, but try all if needed
        if target_mode and split_mode != target_mode:
            continue
        
        if op.startswith('select_'):
            sel_name = op[7:]
            if sel_name in ALL_SELECTS:
                result = ALL_SELECTS[sel_name](panels, bg)
                if result is not None:
                    return result
        elif op in ALL_OPS:
            return ALL_OPS[op](panels, bg)
    
    return None


# ============================================================
# Bbox extraction (crop to non-bg bounding box)
# ============================================================

def learn_crop_to_objects(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: output is the bounding box of non-bg cells in input"""
    for bg in [0, most_common_color(train_pairs[0][0])]:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            oh, ow = grid_shape(out)
            # Find bbox of non-bg
            rows = [r for r in range(h) for c in range(w) if inp[r][c] != bg]
            cols = [c for r in range(h) for c in range(w) if inp[r][c] != bg]
            if not rows:
                ok = False; break
            r1, r2 = min(rows), max(rows)
            c1, c2 = min(cols), max(cols)
            cropped = [inp[r][c1:c2+1] for r in range(r1, r2+1)]
            if not grid_eq(cropped, out):
                ok = False; break
        if ok:
            return {'bg': bg}
    return None


def apply_crop_to_objects(inp: Grid, params: Dict) -> Optional[Grid]:
    bg = params['bg']
    h, w = grid_shape(inp)
    rows = [r for r in range(h) for c in range(w) if inp[r][c] != bg]
    cols = [c for r in range(h) for c in range(w) if inp[r][c] != bg]
    if not rows:
        return None
    r1, r2 = min(rows), max(rows)
    c1, c2 = min(cols), max(cols)
    return [inp[r][c1:c2+1] for r in range(r1, r2+1)]


# ============================================================
# Subgrid extraction by marker/frame
# ============================================================

def learn_extract_by_frame(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: find rectangular frame in input, extract contents"""
    from arc.objects import detect_objects
    
    for bg in [0]:
        ok = True
        frame_color = None
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            oh, ow = grid_shape(out)
            
            # Try each non-bg color as potential frame
            colors = set()
            for r in range(h):
                for c in range(w):
                    if inp[r][c] != bg:
                        colors.add(inp[r][c])
            
            found = False
            for fc in colors:
                # Find bounding box of frame color
                fc_cells = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == fc]
                if not fc_cells:
                    continue
                r1 = min(r for r, c in fc_cells)
                c1 = min(c for r, c in fc_cells)
                r2 = max(r for r, c in fc_cells)
                c2 = max(c for r, c in fc_cells)
                
                # Check if frame forms a rectangle border
                inner_h = r2 - r1 - 1
                inner_w = c2 - c1 - 1
                if inner_h <= 0 or inner_w <= 0:
                    continue
                if inner_h != oh or inner_w != ow:
                    continue
                
                # Extract inner
                inner = [inp[r][c1+1:c2] for r in range(r1+1, r2)]
                # May need to replace bg with something
                if grid_eq(inner, out):
                    found = True
                    if frame_color is None:
                        frame_color = fc
                    elif frame_color != fc:
                        ok = False
                    break
            
            if not found:
                ok = False
                break
        
        if ok and frame_color is not None:
            return {'bg': bg, 'frame_color': frame_color}
    
    return None


def apply_extract_by_frame(inp: Grid, params: Dict) -> Optional[Grid]:
    bg = params['bg']
    fc = params['frame_color']
    h, w = grid_shape(inp)
    
    fc_cells = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == fc]
    if not fc_cells:
        return None
    r1 = min(r for r, c in fc_cells)
    c1 = min(c for r, c in fc_cells)
    r2 = max(r for r, c in fc_cells)
    c2 = max(c for r, c in fc_cells)
    
    if r2 - r1 < 2 or c2 - c1 < 2:
        return None
    
    return [inp[r][c1+1:c2] for r in range(r1+1, r2)]


# ============================================================
# Gravity operations
# ============================================================

def learn_gravity(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn gravity direction: cells fall in a direction until hitting obstacle"""
    for bg in [0]:
        for direction in ['down', 'up', 'left', 'right']:
            ok = True
            for inp, out in train_pairs:
                result = _apply_gravity(inp, bg, direction)
                if not grid_eq(result, out):
                    ok = False
                    break
            if ok:
                return {'bg': bg, 'direction': direction}
    return None


def _apply_gravity(g: Grid, bg: int, direction: str) -> Grid:
    """Apply gravity: non-bg cells fall in direction"""
    h, w = grid_shape(g)
    result = [[bg] * w for _ in range(h)]
    
    if direction == 'down':
        for c in range(w):
            col_vals = [g[r][c] for r in range(h) if g[r][c] != bg]
            # Place from bottom
            for i, v in enumerate(reversed(col_vals)):
                result[h - 1 - i][c] = v
    elif direction == 'up':
        for c in range(w):
            col_vals = [g[r][c] for r in range(h) if g[r][c] != bg]
            for i, v in enumerate(col_vals):
                result[i][c] = v
    elif direction == 'right':
        for r in range(h):
            row_vals = [g[r][c] for c in range(w) if g[r][c] != bg]
            for i, v in enumerate(reversed(row_vals)):
                result[r][w - 1 - i] = v
    elif direction == 'left':
        for r in range(h):
            row_vals = [g[r][c] for c in range(w) if g[r][c] != bg]
            for i, v in enumerate(row_vals):
                result[r][i] = v
    
    return result


def apply_gravity(inp: Grid, params: Dict) -> Optional[Grid]:
    return _apply_gravity(inp, params['bg'], params['direction'])


# ============================================================
# Gravity with obstacles (cells stop at non-bg cells)
# ============================================================

def learn_gravity_with_obstacles(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Gravity where cells stop when hitting other non-bg cells"""
    for bg in [0]:
        for direction in ['down', 'up', 'left', 'right']:
            # Try with different "moving" color vs "static" color
            # First try: all non-bg cells move
            ok = True
            for inp, out in train_pairs:
                result = _apply_gravity_obstacles(inp, bg, direction, None)
                if not grid_eq(result, out):
                    ok = False
                    break
            if ok:
                return {'bg': bg, 'direction': direction, 'moving_color': None}
            
            # Try: specific color moves, others are static
            colors = set()
            for inp, _ in train_pairs:
                for row in inp:
                    for v in row:
                        if v != bg:
                            colors.add(v)
            
            for mc in colors:
                ok = True
                for inp, out in train_pairs:
                    result = _apply_gravity_obstacles(inp, bg, direction, mc)
                    if not grid_eq(result, out):
                        ok = False
                        break
                if ok:
                    return {'bg': bg, 'direction': direction, 'moving_color': mc}
    
    return None


def _apply_gravity_obstacles(g: Grid, bg: int, direction: str, moving_color: Optional[int]) -> Grid:
    """Gravity with obstacles: moving cells stop at non-moving cells"""
    h, w = grid_shape(g)
    result = [row[:] for row in g]
    
    if direction == 'down':
        for c in range(w):
            # Collect moving cells
            moving = []
            for r in range(h):
                if moving_color is None:
                    if g[r][c] != bg:
                        moving.append((r, g[r][c]))
                        result[r][c] = bg
                else:
                    if g[r][c] == moving_color:
                        moving.append((r, g[r][c]))
                        result[r][c] = bg
            
            # Drop each moving cell down
            for orig_r, v in reversed(moving):
                # Find lowest empty position
                pos = h - 1
                while pos >= 0 and result[pos][c] != bg:
                    pos -= 1
                if pos >= 0:
                    result[pos][c] = v
    
    elif direction == 'up':
        for c in range(w):
            moving = []
            for r in range(h):
                if moving_color is None:
                    if g[r][c] != bg:
                        moving.append((r, g[r][c]))
                        result[r][c] = bg
                else:
                    if g[r][c] == moving_color:
                        moving.append((r, g[r][c]))
                        result[r][c] = bg
            
            for orig_r, v in moving:
                pos = 0
                while pos < h and result[pos][c] != bg:
                    pos += 1
                if pos < h:
                    result[pos][c] = v
    
    elif direction == 'right':
        for r in range(h):
            moving = []
            for c in range(w):
                if moving_color is None:
                    if g[r][c] != bg:
                        moving.append((c, g[r][c]))
                        result[r][c] = bg
                else:
                    if g[r][c] == moving_color:
                        moving.append((c, g[r][c]))
                        result[r][c] = bg
            
            for orig_c, v in reversed(moving):
                pos = w - 1
                while pos >= 0 and result[r][pos] != bg:
                    pos -= 1
                if pos >= 0:
                    result[r][pos] = v
    
    elif direction == 'left':
        for r in range(h):
            moving = []
            for c in range(w):
                if moving_color is None:
                    if g[r][c] != bg:
                        moving.append((c, g[r][c]))
                        result[r][c] = bg
                else:
                    if g[r][c] == moving_color:
                        moving.append((c, g[r][c]))
                        result[r][c] = bg
            
            for orig_c, v in moving:
                pos = 0
                while pos < w and result[r][pos] != bg:
                    pos += 1
                if pos < w:
                    result[r][pos] = v
    
    return result


def apply_gravity_with_obstacles(inp: Grid, params: Dict) -> Optional[Grid]:
    return _apply_gravity_obstacles(
        inp, params['bg'], params['direction'], params.get('moving_color'))


# ============================================================
# Symmetry completion
# ============================================================

def learn_symmetry_fill(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: complete a partially symmetric pattern"""
    # Try all possible background colors (most common first)
    bg_candidates = [0]
    # Add bg from first input's most common color
    from collections import Counter as _Counter
    flat0 = [c for row in train_pairs[0][0] for c in row]
    mc0 = _Counter(flat0).most_common()
    for color, _ in mc0:
        if color not in bg_candidates:
            bg_candidates.append(color)
    for bg in bg_candidates:
        for sym_type in ['mirror_hv', 'rot90', 'rot180', 'mirror_h', 'mirror_v']:
            ok = True
            for inp, out in train_pairs:
                h, w = grid_shape(inp)
                oh, ow = grid_shape(out)
                if (h, w) != (oh, ow):
                    ok = False; break
                
                result = _apply_symmetry_fill(inp, bg, sym_type)
                if not grid_eq(result, out):
                    ok = False; break
            if ok:
                return {'bg': bg, 'sym_type': sym_type}
    return None


def _apply_symmetry_fill(g: Grid, bg: int, sym_type: str) -> Grid:
    """Fill bg cells using symmetry of non-bg cells"""
    h, w = grid_shape(g)
    result = [row[:] for row in g]
    
    if sym_type == 'mirror_h':
        # Horizontal axis (top-bottom mirror)
        for r in range(h):
            for c in range(w):
                if result[r][c] == bg:
                    mr = h - 1 - r
                    if result[mr][c] != bg:
                        result[r][c] = result[mr][c]
    
    elif sym_type == 'mirror_v':
        # Vertical axis (left-right mirror)
        for r in range(h):
            for c in range(w):
                if result[r][c] == bg:
                    mc = w - 1 - c
                    if result[r][mc] != bg:
                        result[r][c] = result[r][mc]
    
    elif sym_type == 'mirror_hv':
        # Both axes
        for _ in range(2):
            for r in range(h):
                for c in range(w):
                    if result[r][c] == bg:
                        mr, mc = h - 1 - r, w - 1 - c
                        if result[mr][c] != bg:
                            result[r][c] = result[mr][c]
                        elif result[r][mc] != bg:
                            result[r][c] = result[r][mc]
                        elif result[mr][mc] != bg:
                            result[r][c] = result[mr][mc]
    
    elif sym_type == 'rot180':
        for r in range(h):
            for c in range(w):
                if result[r][c] == bg:
                    mr, mc = h - 1 - r, w - 1 - c
                    if result[mr][mc] != bg:
                        result[r][c] = result[mr][mc]
    
    elif sym_type == 'rot90':
        # Only for square grids
        if h != w:
            return result
        for r in range(h):
            for c in range(w):
                if result[r][c] == bg:
                    # Try all 90-degree rotations
                    positions = [(c, h-1-r), (h-1-r, w-1-c), (w-1-c, r)]
                    for pr, pc in positions:
                        if 0 <= pr < h and 0 <= pc < w and result[pr][pc] != bg:
                            result[r][c] = result[pr][pc]
                            break
    
    return result


def apply_symmetry_fill(inp: Grid, params: Dict) -> Optional[Grid]:
    return _apply_symmetry_fill(inp, params['bg'], params['sym_type'])


# ============================================================
# Pattern completion: fill between same-color objects
# ============================================================

def learn_flood_fill_enclosed(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: flood fill enclosed regions with a specific color"""
    for bg in [0]:
        # Check if output has more filled cells in enclosed areas
        ok = True
        fill_color = None
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            oh, ow = grid_shape(out)
            if (h, w) != (oh, ow):
                ok = False; break
            
            # Find cells that changed
            changed = []
            for r in range(h):
                for c in range(w):
                    if inp[r][c] != out[r][c]:
                        changed.append((r, c, out[r][c]))
            
            if not changed:
                ok = False; break
            
            fc = changed[0][2]
            if not all(v == fc for _, _, v in changed):
                ok = False; break
            
            if fill_color is None:
                fill_color = fc
            elif fill_color != fc:
                ok = False; break
        
        # Not a very specific rule, skip for now
        # This is already handled by fill_enclosed in cross_solver
    
    return None
