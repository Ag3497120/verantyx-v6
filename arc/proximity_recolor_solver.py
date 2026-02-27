"""
arc/proximity_recolor_solver.py — Recolor markers based on proximity to reference structures

Pattern: Small markers (dots) change color to match the nearest reference structure
(e.g., a horizontal line, a vertical line, or a colored border).

Examples:
- 2204b7a8: dots (color 3) → color of nearest horizontal divider line
- 50c07299: swap colors based on proximity to reference lines
"""

from __future__ import annotations
from typing import List, Tuple, Optional
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq
from arc.cross_engine import CrossPiece


def _find_reference_structures(grid: Grid, bg: int = 0):
    """Find horizontal/vertical lines and large regions as reference structures."""
    h, w = grid_shape(grid)
    
    refs = []
    
    # Full-width horizontal lines
    for r in range(h):
        vals = set(grid[r][c] for c in range(w))
        if len(vals) == 1 and list(vals)[0] != bg:
            refs.append({'type': 'h_line', 'row': r, 'color': list(vals)[0]})
    
    # Full-height vertical lines
    for c in range(w):
        vals = set(grid[r][c] for r in range(h))
        if len(vals) == 1 and list(vals)[0] != bg:
            refs.append({'type': 'v_line', 'col': c, 'color': list(vals)[0]})
    
    return refs


def _find_markers(grid: Grid, bg: int, ref_colors: set):
    """Find small markers (isolated non-bg, non-reference pixels)."""
    h, w = grid_shape(grid)
    markers = []
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color != bg and color not in ref_colors:
                markers.append((r, c, color))
    return markers


def _try_proximity_recolor(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossPiece]:
    """
    Markers change color to match the nearest reference structure's color.
    """
    
    def apply_fn(inp_grid: Grid) -> Optional[Grid]:
        h, w = grid_shape(inp_grid)
        bg = 0
        
        refs = _find_reference_structures(inp_grid, bg)
        if not refs:
            return None
        
        ref_colors = set(r['color'] for r in refs)
        markers = _find_markers(inp_grid, bg, ref_colors)
        
        if not markers:
            return None
        
        result = [row[:] for row in inp_grid]
        
        for mr, mc, mcolor in markers:
            # Find nearest reference structure
            min_dist = float('inf')
            nearest_color = mcolor
            
            for ref in refs:
                if ref['type'] == 'h_line':
                    dist = abs(mr - ref['row'])
                elif ref['type'] == 'v_line':
                    dist = abs(mc - ref['col'])
                else:
                    continue
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_color = ref['color']
            
            result[mr][mc] = nearest_color
        
        return result
    
    # Verify
    for inp, out in train_pairs:
        result = apply_fn(inp)
        if result is None or not grid_eq(result, out):
            return None
    
    return CrossPiece('proximity_recolor', apply_fn)


def _try_proximity_recolor_region(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossPiece]:
    """
    Variant: markers recolor based on which region they're in (divided by reference lines).
    Each region has a "region color" determined by the nearest reference.
    """
    
    def apply_fn(inp_grid: Grid) -> Optional[Grid]:
        h, w = grid_shape(inp_grid)
        bg = 0
        
        refs = _find_reference_structures(inp_grid, bg)
        if len(refs) < 2:
            return None
        
        ref_colors = set(r['color'] for r in refs)
        markers = _find_markers(inp_grid, bg, ref_colors)
        
        if not markers:
            return None
        
        result = [row[:] for row in inp_grid]
        
        # Determine marker color based on proximity
        h_refs = [r for r in refs if r['type'] == 'h_line']
        v_refs = [r for r in refs if r['type'] == 'v_line']
        
        for mr, mc, mcolor in markers:
            # Find nearest ref (any type)
            min_dist = float('inf')
            nearest_color = mcolor
            
            for ref in refs:
                if ref['type'] == 'h_line':
                    dist = abs(mr - ref['row'])
                elif ref['type'] == 'v_line':
                    dist = abs(mc - ref['col'])
                else:
                    continue
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_color = ref['color']
            
            result[mr][mc] = nearest_color
        
        return result
    
    for inp, out in train_pairs:
        result = apply_fn(inp)
        if result is None or not grid_eq(result, out):
            return None
    
    return CrossPiece('proximity_recolor_region', apply_fn)


def _try_swap_by_proximity(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossPiece]:
    """
    Variant: two reference colors swap in certain cells based on proximity.
    """
    # Learn from first pair: which color goes where
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    bg = 0
    
    # Find changed cells
    changes = {}
    for r in range(h):
        for c in range(w):
            if inp0[r][c] != out0[r][c]:
                fr, to = inp0[r][c], out0[r][c]
                changes.setdefault((fr, to), []).append((r, c))
    
    if not changes:
        return None
    
    # Check if it's a simple color swap pattern
    # For now, skip complex swaps
    return None


def generate_proximity_recolor_pieces(train_pairs) -> List[CrossPiece]:
    """Generate proximity-based recolor solver pieces."""
    pieces = []
    
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    oh, ow = grid_shape(out0)
    
    if h != oh or w != ow:
        return pieces
    
    for solver in [
        _try_proximity_recolor,
        _try_proximity_recolor_region,
    ]:
        try:
            piece = solver(train_pairs)
            if piece is not None:
                pieces.append(piece)
        except Exception:
            pass
    
    return pieces
