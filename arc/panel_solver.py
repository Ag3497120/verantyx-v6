"""
arc/panel_solver.py — Panel-based solver

Detects separator lines (rows/cols of uniform non-bg color), splits grid into panels,
learns per-panel transformation rules, and applies them.

Strategies:
1. panel_majority_fill: fill each panel with its majority non-bg, non-separator color
2. panel_unique_fill: fill each panel with its unique (non-shared) color
3. panel_count_color: color each panel based on count of non-bg cells
4. panel_position_color: learn color mapping by panel position pattern
5. panel_copy_template: one panel is the template, others get filled to match structure
6. panel_logic_op: AND/OR/XOR between panels
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
from arc.cross_engine import CrossPiece

Grid = List[List[int]]

def _g(g): return np.array(g, dtype=int)
def _l(a): return a.tolist()
def _bg(g): return int(Counter(g.flatten()).most_common(1)[0][0])


def _find_separators(grid, bg):
    """Find separator rows and cols (uniform single-color lines)"""
    H, W = grid.shape
    sep_rows = []
    sep_cols = []
    sep_color = None
    
    # Try non-bg first
    for r in range(H):
        row = grid[r, :]
        vals = set(int(v) for v in row)
        if len(vals) == 1 and int(row[0]) != bg:
            sep_rows.append(r)
            sep_color = int(row[0])
    
    for c in range(W):
        col = grid[:, c]
        vals = set(int(v) for v in col)
        if len(vals) == 1 and int(col[0]) != bg:
            sep_cols.append(c)
            if sep_color is None:
                sep_color = int(col[0])
    
    if sep_rows or sep_cols:
        return sep_rows, sep_cols, sep_color
    
    # Try any color that forms a grid pattern
    for color in range(10):
        c_rows = []
        c_cols = []
        for r in range(H):
            if np.all(grid[r, :] == color):
                c_rows.append(r)
        for c in range(W):
            if np.all(grid[:, c] == color):
                c_cols.append(c)
        
        if (c_rows or c_cols) and (len(c_rows) + len(c_cols)) >= 2:
            return c_rows, c_cols, color
    
    return [], [], None


def _find_separators_flexible(grid, bg):
    """Also detect bg-colored separators (full row/col of bg)"""
    H, W = grid.shape
    sep_rows, sep_cols, sep_color = _find_separators(grid, bg)
    
    if sep_rows or sep_cols:
        return sep_rows, sep_cols, sep_color
    
    # Try bg as separator
    for r in range(H):
        if np.all(grid[r, :] == bg):
            sep_rows.append(r)
    for c in range(W):
        if np.all(grid[:, c] == bg):
            sep_cols.append(c)
    
    if sep_rows or sep_cols:
        return sep_rows, sep_cols, bg
    
    return [], [], None


def _split_panels(grid, sep_rows, sep_cols):
    """Split grid into rectangular panels"""
    H, W = grid.shape
    
    # Get row boundaries
    row_bounds = []
    prev = 0
    for r in sorted(sep_rows):
        if r > prev:
            row_bounds.append((prev, r))
        prev = r + 1
    if prev < H:
        row_bounds.append((prev, H))
    
    # Get col boundaries
    col_bounds = []
    prev = 0
    for c in sorted(sep_cols):
        if c > prev:
            col_bounds.append((prev, c))
        prev = c + 1
    if prev < W:
        col_bounds.append((prev, W))
    
    if not row_bounds:
        row_bounds = [(0, H)]
    if not col_bounds:
        col_bounds = [(0, W)]
    
    panels = []
    for ri, (r1, r2) in enumerate(row_bounds):
        for ci, (c1, c2) in enumerate(col_bounds):
            panels.append({
                'row_idx': ri,
                'col_idx': ci,
                'r1': r1, 'r2': r2,
                'c1': c1, 'c2': c2,
                'data': grid[r1:r2, c1:c2].copy(),
            })
    
    return panels, row_bounds, col_bounds


def _panel_features(panel_data, bg, sep_color=None):
    """Extract features from a panel"""
    exclude = {bg}
    if sep_color is not None:
        exclude.add(sep_color)
    
    colors = Counter()
    for v in panel_data.flatten():
        v = int(v)
        if v not in exclude:
            colors[v] += 1
    
    nonbg_count = sum(1 for v in panel_data.flatten() if int(v) not in exclude)
    total = panel_data.size
    
    return {
        'colors': colors,
        'majority': colors.most_common(1)[0][0] if colors else bg,
        'n_colors': len(colors),
        'nonbg_count': nonbg_count,
        'nonbg_ratio': nonbg_count / total if total > 0 else 0,
        'unique_colors': set(colors.keys()),
    }


# === Strategy 1: Fill each panel with majority color ===

def learn_panel_majority_fill(train_pairs):
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if inp.shape != out.shape:
            return None
        
        bg = _bg(inp)
        sep_rows, sep_cols, sep_color = _find_separators_flexible(inp, bg)
        if not sep_rows and not sep_cols:
            return None
        
        panels_in, rb, cb = _split_panels(inp, sep_rows, sep_cols)
        panels_out, _, _ = _split_panels(out, sep_rows, sep_cols)
        
        if len(panels_in) < 2 or len(panels_in) != len(panels_out):
            return None
        
        for pi, po in zip(panels_in, panels_out):
            feat = _panel_features(pi['data'], bg, sep_color)
            maj = feat['majority']
            
            # Check if output panel is filled with majority color
            expected = np.full_like(po['data'], maj)
            if not np.array_equal(po['data'], expected):
                # Try: non-bg cells become majority, bg stays bg
                expected2 = np.where(pi['data'] != bg, maj, bg)
                if pi['data'].shape == po['data'].shape and not np.array_equal(po['data'], expected2):
                    return None
    
    return {'bg': bg, 'sep_color': sep_color, 'mode': 'majority_fill'}


# === Strategy 2: Panel logic operations (AND/OR/XOR) ===

def learn_panel_logic(train_pairs):
    """Two or more input panels combined with logic op → output panel"""
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if inp.shape != out.shape:
            # Output might be panel-sized
            pass
        
        bg = _bg(inp)
        sep_rows, sep_cols, sep_color = _find_separators_flexible(inp, bg)
        if not sep_rows and not sep_cols:
            return None
        
        panels_in, rb, cb = _split_panels(inp, sep_rows, sep_cols)
        
        if len(panels_in) < 2:
            return None
        
        # Check if all panels are same size
        shapes = set(p['data'].shape for p in panels_in)
        if len(shapes) != 1:
            return None
        
        panel_shape = list(shapes)[0]
        
        # Check if output is panel-sized
        if out.shape == panel_shape:
            # Output = result of combining panels
            pass
        elif inp.shape == out.shape:
            # Output has same structure — some panels change
            panels_out, _, _ = _split_panels(out, sep_rows, sep_cols)
            if len(panels_out) != len(panels_in):
                return None
            # Check which panels changed
            changed = []
            unchanged = []
            for i, (pi, po) in enumerate(zip(panels_in, panels_out)):
                if np.array_equal(pi['data'], po['data']):
                    unchanged.append(i)
                else:
                    changed.append(i)
            
            if not changed:
                return None
        else:
            return None
    
    # Try specific logic ops for panel-sized output
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        bg = _bg(inp)
        sep_rows, sep_cols, sep_color = _find_separators_flexible(inp, bg)
        panels_in, rb, cb = _split_panels(inp, sep_rows, sep_cols)
        
        panel_shape = panels_in[0]['data'].shape
        
        if out.shape == panel_shape:
            # Binary masks
            masks = [(p['data'] != bg).astype(int) for p in panels_in]
            out_mask = (out != bg).astype(int)
            
            for op_name, op_fn in [
                ('and', lambda a, b: a & b),
                ('or', lambda a, b: a | b),
                ('xor', lambda a, b: a ^ b),
            ]:
                result = masks[0]
                for m in masks[1:]:
                    result = op_fn(result, m)
                
                if np.array_equal(result, out_mask):
                    # Find the output color
                    out_colors = set(int(v) for v in out.flatten() if v != bg)
                    if len(out_colors) <= 1:
                        out_color = out_colors.pop() if out_colors else bg
                        return {'op': op_name, 'out_color': out_color, 'bg': bg,
                                'sep_color': sep_color, 'mode': 'logic_reduce'}
    
    return None


# === Strategy 3: Panel count → color mapping ===

def learn_panel_count_color(train_pairs):
    """Each panel's output color = f(count of non-bg cells in panel)"""
    count_to_color = {}
    
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if inp.shape != out.shape: return None
        
        bg = _bg(inp)
        sep_rows, sep_cols, sep_color = _find_separators_flexible(inp, bg)
        if not sep_rows and not sep_cols: return None
        
        panels_in, rb, cb = _split_panels(inp, sep_rows, sep_cols)
        panels_out, _, _ = _split_panels(out, sep_rows, sep_cols)
        
        if len(panels_in) < 2 or len(panels_in) != len(panels_out): return None
        
        for pi, po in zip(panels_in, panels_out):
            feat = _panel_features(pi['data'], bg, sep_color)
            count = feat['nonbg_count']
            
            # What color fills the output panel?
            out_colors = Counter(int(v) for v in po['data'].flatten() if int(v) != bg and (sep_color is None or int(v) != sep_color))
            if not out_colors:
                fill_color = bg
            else:
                fill_color = out_colors.most_common(1)[0][0]
            
            # Check panel is uniformly filled
            expected = np.full_like(po['data'], fill_color)
            if not np.array_equal(po['data'], expected):
                return None
            
            if count in count_to_color:
                if count_to_color[count] != fill_color:
                    return None
            else:
                count_to_color[count] = fill_color
    
    if not count_to_color or len(set(count_to_color.values())) <= 1:
        return None
    
    return {'count_to_color': count_to_color, 'bg': bg, 'sep_color': sep_color, 'mode': 'count_color'}


# === Strategy 4: Per-panel color based on position ===

def learn_panel_position_rule(train_pairs):
    """Learn mapping: (panel_row, panel_col) → transformation"""
    # First check we have consistent panel structure
    all_sep_info = []
    
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if inp.shape != out.shape: return None
        
        bg = _bg(inp)
        sep_rows, sep_cols, sep_color = _find_separators_flexible(inp, bg)
        if not sep_rows and not sep_cols: return None
        
        all_sep_info.append((sep_rows, sep_cols, sep_color, bg))
    
    # Check consistency: same number of panels
    bg = all_sep_info[0][3]
    
    # Learn: for each panel, what's the rule from input to output?
    # Try: each panel filled with its unique (only-in-this-panel) color
    
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        bg = _bg(inp)
        sep_rows, sep_cols, sep_color = _find_separators_flexible(inp, bg)
        panels_in, rb, cb = _split_panels(inp, sep_rows, sep_cols)
        panels_out, _, _ = _split_panels(out, sep_rows, sep_cols)
        
        if len(panels_in) != len(panels_out): return None
        
        # Get all panel color sets
        all_panel_colors = [_panel_features(p['data'], bg, sep_color)['unique_colors'] for p in panels_in]
        
        # Global color frequency across panels
        global_colors = Counter()
        for cs in all_panel_colors:
            for c in cs:
                global_colors[c] += 1
        
        for pi, po in zip(panels_in, panels_out):
            feat = _panel_features(pi['data'], bg, sep_color)
            
            # Find unique color (appears only in this panel)
            unique = [c for c in feat['unique_colors'] if global_colors[c] == 1]
            
            if len(unique) == 1:
                # Check if output panel is filled with this unique color
                expected = np.full_like(po['data'], unique[0])
                if not np.array_equal(po['data'], expected):
                    return None  # Doesn't match
            else:
                return None  # Can't determine unique color
    
    return {'bg': bg, 'sep_color': sep_color, 'mode': 'unique_fill'}


# === Strategy 5: Panel where non-bg pattern maps to specific output ===

def learn_panel_pattern_map(train_pairs):
    """Each panel's non-bg pattern (as binary mask) maps to a specific color output"""
    pattern_to_color = {}
    
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if inp.shape != out.shape: return None
        
        bg = _bg(inp)
        sep_rows, sep_cols, sep_color = _find_separators_flexible(inp, bg)
        if not sep_rows and not sep_cols: return None
        
        panels_in, rb, cb = _split_panels(inp, sep_rows, sep_cols)
        panels_out, _, _ = _split_panels(out, sep_rows, sep_cols)
        
        if len(panels_in) != len(panels_out): return None
        
        for pi, po in zip(panels_in, panels_out):
            # Binary mask of non-bg cells
            mask = (pi['data'] != bg).astype(int)
            mask_key = mask.tobytes()
            
            # Output panel should be uniform color
            out_vals = set(int(v) for v in po['data'].flatten())
            if len(out_vals) != 1: return None
            
            fill = int(po['data'][0, 0])
            
            if mask_key in pattern_to_color:
                if pattern_to_color[mask_key] != fill:
                    return None
            else:
                pattern_to_color[mask_key] = fill
    
    if not pattern_to_color or len(set(pattern_to_color.values())) <= 1:
        return None
    
    return {'pattern_to_color': pattern_to_color, 'bg': bg, 'sep_color': sep_color, 
            'mode': 'pattern_map', 'panel_shape': panels_in[0]['data'].shape}


# === Apply functions ===

def _apply_panel(inp_g, rule):
    inp = _g(inp_g)
    bg = rule['bg']
    sep_color = rule['sep_color']
    mode = rule['mode']
    
    sep_rows, sep_cols, _ = _find_separators_flexible(inp, bg)
    panels, rb, cb = _split_panels(inp, sep_rows, sep_cols)
    
    out = inp.copy()
    
    if mode == 'majority_fill':
        for p in panels:
            feat = _panel_features(p['data'], bg, sep_color)
            maj = feat['majority']
            out[p['r1']:p['r2'], p['c1']:p['c2']] = maj
    
    elif mode == 'unique_fill':
        all_panel_colors = [_panel_features(p['data'], bg, sep_color)['unique_colors'] for p in panels]
        global_colors = Counter()
        for cs in all_panel_colors:
            for c in cs:
                global_colors[c] += 1
        
        for p in panels:
            feat = _panel_features(p['data'], bg, sep_color)
            unique = [c for c in feat['unique_colors'] if global_colors[c] == 1]
            if unique:
                out[p['r1']:p['r2'], p['c1']:p['c2']] = unique[0]
            else:
                out[p['r1']:p['r2'], p['c1']:p['c2']] = bg
    
    elif mode == 'count_color':
        for p in panels:
            feat = _panel_features(p['data'], bg, sep_color)
            count = feat['nonbg_count']
            fill = rule['count_to_color'].get(count, bg)
            out[p['r1']:p['r2'], p['c1']:p['c2']] = fill
    
    elif mode == 'logic_reduce':
        masks = [(p['data'] != bg).astype(int) for p in panels]
        op_fns = {'and': lambda a, b: a & b, 'or': lambda a, b: a | b, 'xor': lambda a, b: a ^ b}
        result = masks[0]
        for m in masks[1:]:
            result = op_fns[rule['op']](result, m)
        
        panel_shape = panels[0]['data'].shape
        out_grid = np.where(result, rule['out_color'], bg)
        return _l(out_grid)
    
    elif mode == 'pattern_map':
        for p in panels:
            mask = (p['data'] != bg).astype(int)
            mask_key = mask.tobytes()
            fill = rule['pattern_to_color'].get(mask_key, bg)
            out[p['r1']:p['r2'], p['c1']:p['c2']] = fill
    
    # Restore separators
    if sep_color is not None and sep_color != bg:
        for r in sep_rows:
            out[r, :] = sep_color
        for c in sep_cols:
            out[:, c] = sep_color
    
    return _l(out)


def _verify(fn, train_pairs):
    for inp, out in train_pairs:
        pred = fn(inp)
        if pred is None or not np.array_equal(_g(pred), _g(out)):
            return False
    return True


def generate_panel_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    pieces = []
    if not train_pairs:
        return pieces
    
    inp0, out0 = _g(train_pairs[0][0]), _g(train_pairs[0][1])
    
    # Quick check: does input have separator structure?
    bg = _bg(inp0)
    sep_rows, sep_cols, sep_color = _find_separators_flexible(inp0, bg)
    if not sep_rows and not sep_cols:
        return pieces
    
    strategies = [
        ('panel_majority', learn_panel_majority_fill),
        ('panel_unique', learn_panel_position_rule),
        ('panel_count', learn_panel_count_color),
        ('panel_logic', learn_panel_logic),
        ('panel_pattern', learn_panel_pattern_map),
    ]
    
    for name, learn_fn in strategies:
        try:
            rule = learn_fn(train_pairs)
            if rule is None: continue
            fn = lambda inp, r=rule: _apply_panel(inp, r)
            if _verify(fn, train_pairs):
                pieces.append(CrossPiece(name=f"panel:{name}", apply_fn=fn))
                return pieces
        except Exception:
            continue
    
    return pieces
