"""
arc/panel_extract_solver.py — Panel-based extraction and transformation solver

Handles tasks where grids are divided by lines into panels:
1. Extract specific panel (unique one, most colored, etc.)
2. Combine/overlay panels (AND, OR, XOR of panels)
3. Select panel by property (most non-bg, fewest colors, etc.)
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq
from arc.cross_engine import CrossPiece
import copy


def _find_dividers(grid: Grid, bg: int = 0):
    """Find horizontal and vertical divider lines.
    Returns (h_divs, v_divs, div_color) or None if no dividers."""
    h, w = grid_shape(grid)
    
    # Try each non-bg color as potential divider
    for div_color in range(1, 10):
        h_divs = []
        for r in range(h):
            if all(grid[r][c] == div_color for c in range(w)):
                h_divs.append(r)
        
        v_divs = []
        for c in range(w):
            if all(grid[r][c] == div_color for r in range(h)):
                v_divs.append(c)
        
        if h_divs or v_divs:
            return h_divs, v_divs, div_color
    
    return [], [], 0


def _split_panels(grid: Grid, h_divs: List[int], v_divs: List[int]) -> List[List[List[int]]]:
    """Split grid into panels based on divider positions.
    Returns list of panel grids."""
    h, w = grid_shape(grid)
    
    row_ranges = []
    prev = 0
    for r in h_divs:
        if r > prev:
            row_ranges.append((prev, r))
        prev = r + 1
    if prev < h:
        row_ranges.append((prev, h))
    
    col_ranges = []
    prev = 0
    for c in v_divs:
        if c > prev:
            col_ranges.append((prev, c))
        prev = c + 1
    if prev < w:
        col_ranges.append((prev, w))
    
    panels = []
    panel_positions = []  # (row_idx, col_idx, r1, c1)
    for ri, (r1, r2) in enumerate(row_ranges):
        for ci, (c1, c2) in enumerate(col_ranges):
            panel = [grid[r][c1:c2] for r in range(r1, r2)]
            panels.append(panel)
            panel_positions.append((ri, ci, r1, c1))
    
    return panels, panel_positions, row_ranges, col_ranges


def _panels_equal(p1, p2):
    """Check if two panels are identical."""
    if len(p1) != len(p2):
        return False
    for r1, r2 in zip(p1, p2):
        if r1 != r2:
            return False
    return True


def _panel_nonbg_count(panel, bg=0):
    """Count non-background cells in panel."""
    return sum(1 for row in panel for c in row if c != bg)


def _panel_colors(panel, bg=0):
    """Get set of non-bg colors in panel."""
    return set(c for row in panel for c in row if c != bg)


def _try_extract_unique_panel(train_pairs):
    """Extract the panel that is unique (all others are identical/similar)."""
    for inp, out in train_pairs:
        h_divs, v_divs, div_color = _find_dividers(inp)
        if not h_divs and not v_divs:
            return None
        panels, positions, _, _ = _split_panels(inp, h_divs, v_divs)
        if len(panels) < 2:
            return None
    
    def find_unique(panels):
        """Find the panel that differs from the majority."""
        n = len(panels)
        # Count how many other panels each panel matches
        match_counts = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if _panels_equal(panels[i], panels[j]):
                    match_counts[i] += 1
                    match_counts[j] += 1
        
        # The unique one has fewest matches
        min_matches = min(match_counts)
        unique_idx = match_counts.index(min_matches)
        return unique_idx
    
    # Check first: does extracting unique panel give the right output?
    def apply_fn(inp_grid):
        h_divs, v_divs, div_color = _find_dividers(inp_grid)
        if not h_divs and not v_divs:
            return None
        panels, positions, _, _ = _split_panels(inp_grid, h_divs, v_divs)
        if len(panels) < 2:
            return None
        idx = find_unique(panels)
        return panels[idx]
    
    for inp, out in train_pairs:
        result = apply_fn(inp)
        if result is None or not grid_eq(result, out):
            return None
    
    return CrossPiece('panel_extract_unique', apply_fn)


def _try_extract_panel_by_property(train_pairs):
    """Extract panel with most non-bg cells, most colors, etc."""
    
    strategies = [
        ('most_nonbg', lambda p, bg: _panel_nonbg_count(p, bg)),
        ('most_colors', lambda p, bg: len(_panel_colors(p, bg))),
        ('least_nonbg', lambda p, bg: -_panel_nonbg_count(p, bg)),
    ]
    
    for strat_name, scorer in strategies:
        def make_apply(score_fn):
            def apply_fn(inp_grid):
                h_divs, v_divs, div_color = _find_dividers(inp_grid)
                if not h_divs and not v_divs:
                    return None
                panels, positions, _, _ = _split_panels(inp_grid, h_divs, v_divs)
                if len(panels) < 2:
                    return None
                
                bg = 0
                scores = [(score_fn(p, bg), i) for i, p in enumerate(panels)]
                scores.sort(reverse=True)
                return panels[scores[0][1]]
            return apply_fn
        
        apply_fn = make_apply(scorer)
        ok = True
        for inp, out in train_pairs:
            result = apply_fn(inp)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        
        if ok:
            return CrossPiece(f'panel_extract_{strat_name}', apply_fn)
    
    return None


def _try_panel_overlay(train_pairs):
    """Combine panels using AND/OR/XOR operations."""
    
    for inp, out in train_pairs:
        h_divs, v_divs, div_color = _find_dividers(inp)
        if not h_divs and not v_divs:
            return None
        panels, positions, _, _ = _split_panels(inp, h_divs, v_divs)
        if len(panels) < 2:
            return None
        # All panels must be same size
        ph = len(panels[0])
        pw = len(panels[0][0]) if ph > 0 else 0
        if not all(len(p) == ph and len(p[0]) == pw for p in panels):
            return None
        # Output must be panel-sized
        oh, ow = grid_shape(out)
        if oh != ph or ow != pw:
            return None
    
    # Try different overlay operations
    operations = []
    
    # OR: non-bg from any panel
    def or_overlay(panels, bg=0):
        ph = len(panels[0])
        pw = len(panels[0][0])
        result = [[bg]*pw for _ in range(ph)]
        for p in panels:
            for r in range(ph):
                for c in range(pw):
                    if p[r][c] != bg and result[r][c] == bg:
                        result[r][c] = p[r][c]
        return result
    
    # AND: only cells that are non-bg in ALL panels
    def and_overlay(panels, bg=0):
        ph = len(panels[0])
        pw = len(panels[0][0])
        result = [[bg]*pw for _ in range(ph)]
        for r in range(ph):
            for c in range(pw):
                colors = [p[r][c] for p in panels if p[r][c] != bg]
                if len(colors) == len(panels):
                    # All panels have non-bg here - use first panel's color
                    result[r][c] = panels[0][r][c]
        return result
    
    # XOR: cells that are non-bg in exactly one panel
    def xor_overlay(panels, bg=0):
        ph = len(panels[0])
        pw = len(panels[0][0])
        result = [[bg]*pw for _ in range(ph)]
        for r in range(ph):
            for c in range(pw):
                non_bg = [(i, p[r][c]) for i, p in enumerate(panels) if p[r][c] != bg]
                if len(non_bg) == 1:
                    result[r][c] = non_bg[0][1]
        return result
    
    # Intersection: cells non-bg in all panels, keep color of first
    def intersection_overlay(panels, bg=0):
        ph = len(panels[0])
        pw = len(panels[0][0])
        result = [[bg]*pw for _ in range(ph)]
        for r in range(ph):
            for c in range(pw):
                if all(p[r][c] != bg for p in panels):
                    result[r][c] = panels[0][r][c]
        return result
    
    # Diff: cells non-bg in first panel but bg in second  
    # (only for 2 panels)
    
    for name, op_fn in [('or', or_overlay), ('and', and_overlay), 
                         ('xor', xor_overlay), ('intersection', intersection_overlay)]:
        def make_apply(op):
            def apply_fn(inp_grid):
                h_divs, v_divs, div_color = _find_dividers(inp_grid)
                if not h_divs and not v_divs:
                    return None
                panels, _, _, _ = _split_panels(inp_grid, h_divs, v_divs)
                if len(panels) < 2:
                    return None
                return op(panels)
            return apply_fn
        
        apply_fn = make_apply(op_fn)
        ok = True
        for inp, out in train_pairs:
            result = apply_fn(inp)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        
        if ok:
            return CrossPiece(f'panel_overlay_{name}', apply_fn)
    
    return None


def _try_panel_transform(train_pairs):
    """Learn per-panel transformations: each panel is modified independently.
    Common: fill panels based on some property (count, color, position)."""
    
    for inp, out in train_pairs:
        h_divs, v_divs, div_color = _find_dividers(inp)
        if not h_divs and not v_divs:
            return None
        
        oh, ow = grid_shape(out)
        ih, iw = grid_shape(inp)
        if oh != ih or ow != iw:
            return None
        
        # Check output has same divider structure
        h_divs2, v_divs2, div_color2 = _find_dividers(out)
        if h_divs != h_divs2 or v_divs != v_divs2:
            return None
    
    # Learn: for each panel, what changes from input to output?
    # Strategy: learn panel → single fill color mapping
    
    # Check if each output panel is a solid color
    def check_solid_panels(train_pairs):
        for inp, out in train_pairs:
            h_divs, v_divs, _ = _find_dividers(out)
            out_panels, _, _, _ = _split_panels(out, h_divs, v_divs)
            for p in out_panels:
                colors = set(c for row in p for c in row)
                if len(colors) > 1:
                    return False
        return True
    
    if check_solid_panels(train_pairs):
        # Each panel becomes a solid color. What determines the color?
        # Try: color = most common non-bg color in input panel
        def apply_fill_majority(inp_grid):
            h_divs, v_divs, div_color = _find_dividers(inp_grid)
            if not h_divs and not v_divs:
                return None
            panels, positions, row_ranges, col_ranges = _split_panels(inp_grid, h_divs, v_divs)
            
            result = [row[:] for row in inp_grid]
            
            pi = 0
            for ri, (r1, r2) in enumerate(row_ranges):
                for ci, (c1, c2) in enumerate(col_ranges):
                    panel = panels[pi]
                    bg = 0
                    colors = [c for row in panel for c in row if c != bg and c != div_color]
                    if colors:
                        fill = Counter(colors).most_common(1)[0][0]
                    else:
                        fill = bg
                    
                    for r in range(r1, r2):
                        for c in range(c1, c2):
                            result[r][c] = fill
                    pi += 1
            
            return result
        
        ok = True
        for inp, out in train_pairs:
            r = apply_fill_majority(inp)
            if r is None or not grid_eq(r, out):
                ok = False
                break
        if ok:
            return CrossPiece('panel_fill_majority', apply_fill_majority)
    
    return None


def generate_panel_extract_pieces(train_pairs) -> List[CrossPiece]:
    """Generate panel-based solver pieces."""
    pieces = []
    
    # Quick check: does the input have dividers?
    inp0, out0 = train_pairs[0]
    h_divs, v_divs, div_color = _find_dividers(inp0)
    if not h_divs and not v_divs:
        return pieces
    
    for solver in [
        _try_extract_unique_panel,
        _try_extract_panel_by_property,
        _try_panel_overlay,
        _try_panel_transform,
    ]:
        try:
            piece = solver(train_pairs)
            if piece is not None:
                pieces.append(piece)
        except Exception:
            pass
    
    return pieces
