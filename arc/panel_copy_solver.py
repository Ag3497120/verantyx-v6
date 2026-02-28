"""
arc/panel_copy_solver.py — Panel Copy/Mirror/Propagation Solver

Separator-divided grids where patterns propagate between panels.
Patterns:
- Copy non-bg pattern from one panel to another
- Mirror pattern across separator
- Fill panel based on another panel's pattern
- Propagate unique colors across panels
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
from scipy import ndimage

Grid = List[List[int]]


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.flatten())
    return counts.most_common(1)[0][0]


def _find_separators(grid: np.ndarray, bg: int):
    """Find separator rows and columns"""
    h, w = grid.shape
    sep_rows, sep_cols = [], []
    sep_color = None
    
    for r in range(h):
        vals = set(int(v) for v in grid[r])
        if len(vals) == 1 and list(vals)[0] != bg:
            sep_rows.append(r)
            sep_color = list(vals)[0]
    
    for c in range(w):
        vals = set(int(v) for v in grid[:, c])
        if len(vals) == 1 and list(vals)[0] != bg:
            sep_cols.append(c)
            if sep_color is None:
                sep_color = list(vals)[0]
    
    # Also check single-color rows/cols that ARE bg (separator could be bg color)
    # Skip for now
    
    return sep_rows, sep_cols, sep_color


def _extract_panels(grid: np.ndarray, sep_rows, sep_cols):
    """Extract panel regions"""
    h, w = grid.shape
    row_bounds = [0] + sorted(sep_rows) + [h]
    col_bounds = [0] + sorted(sep_cols) + [w]
    
    panels = []
    for i in range(len(row_bounds) - 1):
        for j in range(len(col_bounds) - 1):
            r0, r1 = row_bounds[i], row_bounds[i+1]
            c0, c1 = col_bounds[j], col_bounds[j+1]
            # Skip separator rows/cols
            if r0 in sep_rows: r0 += 1
            if c0 in sep_cols: c0 += 1
            if r1 > r0 and c1 > c0:
                panels.append((r0, c0, r1, c1))
    
    return panels


def _panel_content(grid, r0, c0, r1, c1, bg):
    """Extract panel content normalized (just the non-bg pattern)"""
    sub = grid[r0:r1, c0:c1].copy()
    return sub


def _panel_has_pattern(grid, r0, c0, r1, c1, bg):
    """Check if panel has non-bg content beyond the dominant"""
    sub = grid[r0:r1, c0:c1]
    colors = set(int(v) for v in sub.flatten()) - {bg}
    return len(colors) > 1  # More than just one fill color


def _count_nonbg_pattern(grid, r0, c0, r1, c1, bg):
    """Count cells that differ from the panel's dominant color"""
    sub = grid[r0:r1, c0:c1]
    dom = Counter(int(v) for v in sub.flatten()).most_common(1)[0][0]
    return int(np.sum(sub != dom))


def learn_panel_copy_rule(train_pairs: List[Tuple[Grid, Grid]], 
                          bg: int = None) -> Optional[Dict]:
    """Learn panel copy/propagation rule"""
    np_pairs = [(np.array(i), np.array(o)) for i, o in train_pairs]
    
    if bg is None:
        bg = _get_bg(np_pairs[0][0])
    
    # Check same-size
    for inp, out in np_pairs:
        if inp.shape != out.shape:
            return None
    
    # Find separators (must be consistent across pairs)
    sep_info = []
    for inp, out in np_pairs:
        sr, sc, scolor = _find_separators(inp, bg)
        sep_info.append((sr, sc, scolor))
    
    # Need at least one separator
    if not any(sr or sc for sr, sc, _ in sep_info):
        return None
    
    # Try each strategy
    strategies = [
        _try_panel_copy_to_all,
        _try_panel_mirror,
        _try_panel_unique_spread,
        _try_panel_template_fill,
    ]
    
    for strat_fn in strategies:
        try:
            rule = strat_fn(np_pairs, bg, sep_info)
            if rule is not None:
                return rule
        except Exception:
            continue
    
    return None


def _try_panel_copy_to_all(np_pairs, bg, sep_info):
    """Copy pattern from source panel to all other panels of same size"""
    for inp, out in np_pairs:
        sr, sc, scolor = _find_separators(inp, bg)
        panels = _extract_panels(inp, sr, sc)
        
        if len(panels) < 2:
            return None
        
        # Check if all panels are same size
        shapes = set((r1-r0, c1-c0) for r0, c0, r1, c1 in panels)
        if len(shapes) != 1:
            return None
        
        ph, pw = list(shapes)[0]
        
        # Find source panel (the one with the most pattern in output)
        panel_patterns = []
        for pi, (r0, c0, r1, c1) in enumerate(panels):
            out_sub = out[r0:r1, c0:c1]
            inp_sub = inp[r0:r1, c0:c1]
            # How much does this panel differ from bg-filled?
            pattern_count = _count_nonbg_pattern(out, r0, c0, r1, c1, bg)
            panel_patterns.append((pi, pattern_count, r0, c0, r1, c1))
        
        # Find panels that changed
        changed_panels = []
        source_panels = []
        for pi, (r0, c0, r1, c1) in enumerate(panels):
            inp_sub = inp[r0:r1, c0:c1]
            out_sub = out[r0:r1, c0:c1]
            if not np.array_equal(inp_sub, out_sub):
                changed_panels.append(pi)
            else:
                source_panels.append(pi)
        
        if not source_panels or not changed_panels:
            return None
        
        # Source = unchanged panel with most pattern
        source_pi = max(source_panels, 
                        key=lambda pi: _count_nonbg_pattern(inp, *panels[pi], bg))
        
        sr0, sc0, sr1, sc1 = panels[source_pi]
        source_pattern = inp[sr0:sr1, sc0:sc1].copy()
        
        # Check: does copying source pattern to changed panels produce output?
        predicted = inp.copy()
        for pi in changed_panels:
            r0, c0, r1, c1 = panels[pi]
            predicted[r0:r1, c0:c1] = source_pattern
        
        if not np.array_equal(predicted, out):
            return None
    
    return {
        'strategy': 'panel_copy_to_all',
        'bg': bg,
    }


def _try_panel_mirror(np_pairs, bg, sep_info):
    """Mirror pattern across separator (horizontal or vertical)"""
    for inp, out in np_pairs:
        sr, sc, scolor = _find_separators(inp, bg)
        
        # Try vertical separator mirror (left <-> right)
        if sc and len(sc) == 1:
            c_sep = sc[0]
            left = inp[:, :c_sep].copy()
            right = inp[:, c_sep+1:].copy()
            out_left = out[:, :c_sep]
            out_right = out[:, c_sep+1:]
            
            # Try: mirror right to left
            if left.shape == right.shape:
                predicted = inp.copy()
                right_flipped = np.fliplr(right)
                # Check which direction the copy goes
                if np.array_equal(out_right, inp[:, c_sep+1:]):
                    # Right unchanged, left gets right's pattern
                    # But mirrored? Or direct copy?
                    # Direct copy
                    predicted[:, :c_sep] = right
                    if np.array_equal(predicted, out):
                        return {'strategy': 'panel_mirror_r2l_direct', 'bg': bg}
                    # Mirrored copy
                    predicted[:, :c_sep] = right_flipped
                    if np.array_equal(predicted, out):
                        return {'strategy': 'panel_mirror_r2l_flip', 'bg': bg}
                elif np.array_equal(out_left, inp[:, :c_sep]):
                    # Left unchanged, right gets left's pattern
                    left_flipped = np.fliplr(left)
                    predicted = inp.copy()
                    predicted[:, c_sep+1:] = left
                    if np.array_equal(predicted, out):
                        return {'strategy': 'panel_mirror_l2r_direct', 'bg': bg}
                    predicted[:, c_sep+1:] = left_flipped
                    if np.array_equal(predicted, out):
                        return {'strategy': 'panel_mirror_l2r_flip', 'bg': bg}
                # Both changed: merge
                # Try: both sides get the OR of both patterns
                merged = np.where(left != bg, left, right_flipped)
                predicted = inp.copy()
                predicted[:, :c_sep] = merged
                merged_r = np.fliplr(merged)
                predicted[:, c_sep+1:] = merged_r
                if np.array_equal(predicted, out):
                    return {'strategy': 'panel_mirror_merge', 'bg': bg}
                
                # Try: right pattern replaces bg cells in left
                predicted = inp.copy()
                for r in range(left.shape[0]):
                    for c in range(left.shape[1]):
                        if left[r, c] == bg and right[r, c] != bg:
                            predicted[r, c] = right[r, c]
                        elif left[r, c] != bg and right[r, c] == bg:
                            rc = c_sep + 1 + c
                            if rc < inp.shape[1]:
                                predicted[r, rc] = left[r, c]
                if np.array_equal(predicted, out):
                    return {'strategy': 'panel_copy_symmetric', 'bg': bg}
        
        # Try horizontal separator mirror
        if sr and len(sr) == 1:
            r_sep = sr[0]
            top = inp[:r_sep, :].copy()
            bottom = inp[r_sep+1:, :].copy()
            
            if top.shape == bottom.shape:
                predicted = inp.copy()
                # Try: bottom's pattern copied to top
                predicted[:r_sep, :] = bottom
                if np.array_equal(predicted, out):
                    return {'strategy': 'panel_mirror_b2t_direct', 'bg': bg}
                predicted[:r_sep, :] = np.flipud(bottom)
                if np.array_equal(predicted, out):
                    return {'strategy': 'panel_mirror_b2t_flip', 'bg': bg}
                # Try: top's pattern copied to bottom
                predicted = inp.copy()
                predicted[r_sep+1:, :] = top
                if np.array_equal(predicted, out):
                    return {'strategy': 'panel_mirror_t2b_direct', 'bg': bg}
                predicted[r_sep+1:, :] = np.flipud(top)
                if np.array_equal(predicted, out):
                    return {'strategy': 'panel_mirror_t2b_flip', 'bg': bg}
    
    return None


def _try_panel_unique_spread(np_pairs, bg, sep_info):
    """Spread unique per-panel color to all panels in same row/col"""
    for inp, out in np_pairs:
        sr, sc, scolor = _find_separators(inp, bg)
        panels = _extract_panels(inp, sr, sc)
        
        if len(panels) < 2:
            return None
        
        # For each panel, find unique colors (not bg, not panel-fill)
        panel_dom = {}
        panel_unique = {}
        for pi, (r0, c0, r1, c1) in enumerate(panels):
            sub = inp[r0:r1, c0:c1]
            colors = Counter(int(v) for v in sub.flatten())
            dom = colors.most_common(1)[0][0]
            panel_dom[pi] = dom
            unique = [c for c, cnt in colors.items() if c != bg and c != dom]
            panel_unique[pi] = unique
        
        # Check: are changes in output = spreading unique colors?
        predicted = inp.copy()
        diff_mask = inp != out
        if not diff_mask.any():
            return None
        
        # Simple: if a panel has a unique dot, all other panels in same
        # row/column of panels get that dot at the same relative position
        # This is complex... skip for now
    
    return None


def _try_panel_template_fill(np_pairs, bg, sep_info):
    """One panel is a template; other panels get filled with template where they have markers"""
    for inp, out in np_pairs:
        sr, sc, scolor = _find_separators(inp, bg)
        panels = _extract_panels(inp, sr, sc)
        
        if len(panels) < 2:
            return None
        
        shapes = set((r1-r0, c1-c0) for r0, c0, r1, c1 in panels)
        if len(shapes) != 1:
            return None
        
        # Find template panel (most colored) and target panels
        best_pi = -1
        best_count = -1
        for pi, (r0, c0, r1, c1) in enumerate(panels):
            sub = inp[r0:r1, c0:c1]
            count = int(np.sum(sub != bg))
            if count > best_count:
                best_count = count
                best_pi = pi
        
        if best_pi < 0:
            return None
        
        tr0, tc0, tr1, tc1 = panels[best_pi]
        template = inp[tr0:tr1, tc0:tc1].copy()
        
        # Check: for each other panel, output = template OR output = merge
        predicted = inp.copy()
        for pi, (r0, c0, r1, c1) in enumerate(panels):
            if pi == best_pi:
                continue
            predicted[r0:r1, c0:c1] = template
        
        if np.array_equal(predicted, out):
            return {'strategy': 'panel_template_fill', 'bg': bg}
    
    return None


# ============================================================
# Apply functions
# ============================================================

def apply_panel_copy_rule(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply panel copy/propagation rule"""
    grid = np.array(inp)
    bg = rule.get('bg', 0)
    strategy = rule['strategy']
    h, w = grid.shape
    
    sr, sc, scolor = _find_separators(grid, bg)
    
    if strategy == 'panel_copy_to_all':
        panels = _extract_panels(grid, sr, sc)
        if len(panels) < 2:
            return None
        shapes = set((r1-r0, c1-c0) for r0, c0, r1, c1 in panels)
        if len(shapes) != 1:
            return None
        
        # Find source = panel with most pattern
        source_pi = max(range(len(panels)),
                        key=lambda pi: _count_nonbg_pattern(grid, *panels[pi], bg))
        sr0, sc0, sr1, sc1 = panels[source_pi]
        source = grid[sr0:sr1, sc0:sc1].copy()
        
        result = grid.copy()
        for pi, (r0, c0, r1, c1) in enumerate(panels):
            if pi != source_pi:
                result[r0:r1, c0:c1] = source
        return result.tolist()
    
    elif strategy == 'panel_template_fill':
        panels = _extract_panels(grid, sr, sc)
        if len(panels) < 2:
            return None
        
        best_pi = max(range(len(panels)),
                      key=lambda pi: int(np.sum(grid[panels[pi][0]:panels[pi][2], 
                                                      panels[pi][1]:panels[pi][3]] != bg)))
        tr0, tc0, tr1, tc1 = panels[best_pi]
        template = grid[tr0:tr1, tc0:tc1].copy()
        
        result = grid.copy()
        for pi, (r0, c0, r1, c1) in enumerate(panels):
            if pi != best_pi:
                result[r0:r1, c0:c1] = template
        return result.tolist()
    
    elif strategy.startswith('panel_mirror_'):
        if sc and len(sc) == 1:
            c_sep = sc[0]
            left = grid[:, :c_sep].copy()
            right = grid[:, c_sep+1:].copy()
            result = grid.copy()
            
            if strategy == 'panel_mirror_r2l_direct':
                result[:, :c_sep] = right
            elif strategy == 'panel_mirror_r2l_flip':
                result[:, :c_sep] = np.fliplr(right)
            elif strategy == 'panel_mirror_l2r_direct':
                result[:, c_sep+1:] = left
            elif strategy == 'panel_mirror_l2r_flip':
                result[:, c_sep+1:] = np.fliplr(left)
            elif strategy == 'panel_mirror_merge':
                right_f = np.fliplr(right)
                merged = np.where(left != bg, left, right_f)
                result[:, :c_sep] = merged
                result[:, c_sep+1:] = np.fliplr(merged)
            elif strategy == 'panel_copy_symmetric':
                for r in range(left.shape[0]):
                    for c in range(left.shape[1]):
                        if left[r, c] == bg and right[r, c] != bg:
                            result[r, c] = right[r, c]
                        elif left[r, c] != bg and right[r, c] == bg:
                            rc = c_sep + 1 + c
                            if rc < w:
                                result[r, rc] = left[r, c]
            return result.tolist()
        
        if sr and len(sr) == 1:
            r_sep = sr[0]
            top = grid[:r_sep, :].copy()
            bottom = grid[r_sep+1:, :].copy()
            result = grid.copy()
            
            if strategy == 'panel_mirror_b2t_direct':
                result[:r_sep, :] = bottom
            elif strategy == 'panel_mirror_b2t_flip':
                result[:r_sep, :] = np.flipud(bottom)
            elif strategy == 'panel_mirror_t2b_direct':
                result[r_sep+1:, :] = top
            elif strategy == 'panel_mirror_t2b_flip':
                result[r_sep+1:, :] = np.flipud(top)
            return result.tolist()
    
    return None


# ============================================================
# Integration
# ============================================================

def generate_panel_copy_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> list:
    from arc.cross_engine import CrossPiece
    from arc.grid import most_common_color
    
    pieces = []
    if not train_pairs:
        return pieces
    
    bg = most_common_color(train_pairs[0][0])
    rule = learn_panel_copy_rule(train_pairs, bg)
    
    if rule is not None:
        pieces.append(CrossPiece(
            f'panel_copy:{rule["strategy"]}',
            lambda inp, _r=rule: apply_panel_copy_rule(inp, _r)
        ))
    
    return pieces
