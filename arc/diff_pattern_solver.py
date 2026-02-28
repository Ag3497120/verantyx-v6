"""
arc/diff_pattern_solver.py — Diff-based pattern learning solver

Learns transformation rules by analyzing input→output diffs:
1. For each changed cell, record local context (neighborhood, position relative to objects)
2. Learn a rule mapping context → output color
3. Apply rule to test input

This is more flexible than NB rules because it considers:
- Distance to nearest non-bg pixels
- Relative position within enclosed regions
- Row/column patterns
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import Counter, defaultdict


def _get_context(grid, r, c, bg=0):
    """Get rich context for a cell."""
    h, w = grid.shape
    val = int(grid[r, c])
    
    # 8-neighborhood colors
    nb8 = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                nb8.append(int(grid[nr, nc]))
            else:
                nb8.append(-1)
    
    # Count of each color in neighborhood
    nb_count_nonbg = sum(1 for v in nb8 if v > 0)
    nb_count_bg = sum(1 for v in nb8 if v == 0)
    
    # Distance to nearest non-bg in 4 directions
    dist_up = 0
    for rr in range(r - 1, -1, -1):
        if grid[rr, c] != bg:
            break
        dist_up += 1
    else:
        dist_up = r
    
    dist_down = 0
    for rr in range(r + 1, h):
        if grid[rr, c] != bg:
            break
        dist_down += 1
    else:
        dist_down = h - 1 - r
    
    dist_left = 0
    for cc in range(c - 1, -1, -1):
        if grid[r, cc] != bg:
            break
        dist_left += 1
    else:
        dist_left = c
    
    dist_right = 0
    for cc in range(c + 1, w):
        if grid[r, cc] != bg:
            break
        dist_right += 1
    else:
        dist_right = w - 1 - c
    
    # Nearest non-bg color in 4 directions
    color_up = -1
    for rr in range(r - 1, -1, -1):
        if grid[rr, c] != bg:
            color_up = int(grid[rr, c])
            break
    
    color_down = -1
    for rr in range(r + 1, h):
        if grid[rr, c] != bg:
            color_down = int(grid[rr, c])
            break
    
    color_left = -1
    for cc in range(c - 1, -1, -1):
        if grid[r, cc] != bg:
            color_left = int(grid[r, cc])
            break
    
    color_right = -1
    for cc in range(c + 1, w):
        if grid[r, cc] != bg:
            color_right = int(grid[r, cc])
            break
    
    return {
        'val': val,
        'nb8': tuple(nb8),
        'nb_nonbg': nb_count_nonbg,
        'dist': (dist_up, dist_down, dist_left, dist_right),
        'nearest_colors': (color_up, color_down, color_left, color_right),
    }


def _abstract_context(ctx, bg=0):
    """Create abstract version of context (color-independent)."""
    # Abstract nb8: replace specific colors with roles
    val = ctx['val']
    color_map = {}
    next_id = 1
    
    def abstract_color(c):
        nonlocal next_id
        if c == -1:
            return -1
        if c == bg:
            return 0
        if c == val:
            return 100  # self
        if c not in color_map:
            color_map[c] = next_id
            next_id += 1
        return color_map[c]
    
    abs_nb = tuple(abstract_color(v) for v in ctx['nb8'])
    abs_nearest = tuple(abstract_color(v) for v in ctx['nearest_colors'])
    
    return (abs_nb, ctx['nb_nonbg'], ctx['dist'], abs_nearest)


def generate_diff_pattern_pieces(train_pairs, bg=0):
    """Generate CrossPiece candidates using diff-based pattern learning."""
    from arc.cross_engine import CrossPiece
    
    pieces = []
    
    # Check: same size i/o for all pairs
    for inp, out in train_pairs:
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return pieces
    
    # Collect all diffs across train pairs
    all_contexts = []  # (abstract_context, output_color)
    fill_colors = set()
    change_srcs = set()
    
    for inp_grid, out_grid in train_pairs:
        inp = np.array(inp_grid)
        out = np.array(out_grid)
        
        if inp.shape != out.shape:
            return pieces
        
        diff = inp != out
        for r, c in zip(*np.where(diff)):
            ctx = _get_context(inp, r, c, bg)
            abs_ctx = _abstract_context(ctx, bg)
            out_color = int(out[r, c])
            all_contexts.append((abs_ctx, out_color))
            fill_colors.add(out_color)
            change_srcs.add(int(inp[r, c]))
    
    if not all_contexts:
        return pieces
    
    # Only handle single fill color for now
    if len(fill_colors) != 1:
        return pieces
    
    fill_color = fill_colors.pop()
    
    # Learn: which abstract contexts map to this fill_color?
    # Group by context → verify all map to same output
    context_map = defaultdict(set)
    for ctx, oc in all_contexts:
        context_map[ctx].add(oc)
    
    # All contexts should map to same fill_color
    consistent = all(len(v) == 1 for v in context_map.values())
    if not consistent:
        return pieces
    
    # Now: which cells in input should change?
    # We need to find which cells DON'T change too, to learn the negative pattern
    
    # For each unchanged cell, get its context
    unchanged_contexts = set()
    for inp_grid, out_grid in train_pairs:
        inp = np.array(inp_grid)
        out = np.array(out_grid)
        h, w = inp.shape
        for r in range(h):
            for c in range(w):
                if inp[r, c] == out[r, c]:
                    ctx = _get_context(inp, r, c, bg)
                    abs_ctx = _abstract_context(ctx, bg)
                    unchanged_contexts.add(abs_ctx)
    
    # Changed contexts = contexts that ONLY appear in changed cells
    changed_only = set(context_map.keys()) - unchanged_contexts
    
    if not changed_only:
        # Some contexts appear in both changed and unchanged — can't separate
        # Try: maybe we need more specific context (include exact position?)
        return pieces
    
    # If all changed contexts are unique to changed cells, we have a rule
    def apply_fn(inp_grid):
        inp = np.array(inp_grid)
        out = inp.copy()
        h, w = inp.shape
        
        for r in range(h):
            for c in range(w):
                ctx = _get_context(inp, r, c, bg)
                abs_ctx = _abstract_context(ctx, bg)
                if abs_ctx in changed_only:
                    out[r, c] = fill_color
        
        return out.tolist()
    
    # Verify on train
    ok = True
    for inp_grid, out_grid in train_pairs:
        if apply_fn(inp_grid) != out_grid:
            ok = False
            break
    
    if ok:
        pieces.append(CrossPiece(name=f"diff_pattern_fill_c{fill_color}", apply_fn=apply_fn, version=1))
    
    return pieces
