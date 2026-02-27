"""
arc/stripe_fill_solver.py — Row/Column Stripe Fill Solver for ARC-AGI-2

Handles tasks where:
1. Colored dots/pixels on a row → entire row filled with that color
2. Colored dots/pixels on a column → entire column filled with that color  
3. Can be periodic/repeating pattern
4. Supports both horizontal and vertical stripes

Example tasks:
- 0a938d79: dots define horizontal stripes that repeat periodically
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from arc.grid import Grid, grid_shape, grid_eq, most_common_color


def learn_stripe_fill_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn stripe fill rule from training examples.
    
    Returns rule dict with:
    - type: 'row_fill' or 'col_fill' or 'row_periodic' or 'col_periodic'
    - bg: background color
    - periodic: whether pattern repeats
    - period: period for repeating pattern
    """
    
    # Try horizontal row fill (simple)
    rule = _learn_row_fill(train_pairs, periodic=False)
    if rule:
        return rule
    
    # Try horizontal row fill with periodic repeat
    rule = _learn_row_fill(train_pairs, periodic=True)
    if rule:
        return rule
    
    # Try vertical column fill (simple)
    rule = _learn_col_fill(train_pairs, periodic=False)
    if rule:
        return rule
    
    # Try vertical column fill with periodic repeat
    rule = _learn_col_fill(train_pairs, periodic=True)
    if rule:
        return rule
    
    return None


def apply_stripe_fill_rule(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply learned stripe fill rule"""
    rule_type = rule['type']
    
    if rule_type in ['row_fill', 'row_periodic']:
        return _apply_row_fill(inp, rule)
    elif rule_type in ['col_fill', 'col_periodic']:
        return _apply_col_fill(inp, rule)
    
    return None


def _learn_row_fill(train_pairs: List[Tuple[Grid, Grid]], periodic: bool = False) -> Optional[Dict]:
    """Learn: colored dots on a row → fill entire row with that color.
    
    If periodic=True, pattern repeats every N rows.
    """
    for bg in [0]:
        ok = True
        period = None
        
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            if grid_shape(out) != (h, w):
                ok = False
                break
            
            # Find rows with non-bg pixels
            row_colors = {}  # row -> list of colors
            for r in range(h):
                colors = set()
                for c in range(w):
                    if inp[r][c] != bg:
                        colors.add(inp[r][c])
                if colors:
                    row_colors[r] = colors
            
            if not row_colors:
                ok = False
                break
            
            # Check if output fills these rows
            for r in range(h):
                if r in row_colors:
                    # This row should be filled
                    colors = row_colors[r]
                    if len(colors) != 1:
                        # Multiple colors on same row - pattern unclear
                        continue
                    
                    fill_color = list(colors)[0]
                    
                    # Check if row is filled in output
                    for c in range(w):
                        if out[r][c] != fill_color:
                            ok = False
                            break
                else:
                    # Row without colored dots - check if unchanged or follows pattern
                    if not periodic:
                        # Should be unchanged (bg)
                        for c in range(w):
                            if out[r][c] != inp[r][c]:
                                ok = False
                                break
            
            if not ok:
                break
            
            # If periodic, detect period
            if periodic and period is None:
                # Find smallest period
                filled_rows = sorted(row_colors.keys())
                if len(filled_rows) >= 2:
                    # Try to detect period from spacing
                    spacings = [filled_rows[i+1] - filled_rows[i] for i in range(len(filled_rows)-1)]
                    if spacings:
                        period = min(spacings)
                        # Check if this period works
                        for r in range(h):
                            if r in row_colors:
                                expected_color = list(row_colors[r])[0]
                                # Check if pattern repeats at r + k*period
                                for k in range(1, h // period + 1):
                                    repeat_r = r + k * period
                                    if repeat_r < h:
                                        # Check output at repeat position
                                        for c in range(w):
                                            if out[repeat_r][c] != expected_color:
                                                ok = False
                                                break
        
        if ok:
            return {
                'type': 'row_periodic' if periodic else 'row_fill',
                'bg': bg,
                'periodic': periodic,
                'period': period
            }
    
    return None


def _learn_col_fill(train_pairs: List[Tuple[Grid, Grid]], periodic: bool = False) -> Optional[Dict]:
    """Learn: colored dots on a column → fill entire column with that color.
    
    If periodic=True, pattern repeats every N columns.
    """
    for bg in [0]:
        ok = True
        period = None
        
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            if grid_shape(out) != (h, w):
                ok = False
                break
            
            # Find columns with non-bg pixels
            col_colors = {}  # col -> list of colors
            for c in range(w):
                colors = set()
                for r in range(h):
                    if inp[r][c] != bg:
                        colors.add(inp[r][c])
                if colors:
                    col_colors[c] = colors
            
            if not col_colors:
                ok = False
                break
            
            # Check if output fills these columns
            for c in range(w):
                if c in col_colors:
                    # This column should be filled
                    colors = col_colors[c]
                    if len(colors) != 1:
                        # Multiple colors on same column - pattern unclear
                        continue
                    
                    fill_color = list(colors)[0]
                    
                    # Check if column is filled in output
                    for r in range(h):
                        if out[r][c] != fill_color:
                            ok = False
                            break
                else:
                    # Column without colored dots - check if unchanged
                    if not periodic:
                        for r in range(h):
                            if out[r][c] != inp[r][c]:
                                ok = False
                                break
            
            if not ok:
                break
            
            # If periodic, detect period
            if periodic and period is None:
                filled_cols = sorted(col_colors.keys())
                if len(filled_cols) >= 2:
                    spacings = [filled_cols[i+1] - filled_cols[i] for i in range(len(filled_cols)-1)]
                    if spacings:
                        period = min(spacings)
        
        if ok:
            return {
                'type': 'col_periodic' if periodic else 'col_fill',
                'bg': bg,
                'periodic': periodic,
                'period': period
            }
    
    return None


def _apply_row_fill(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply row fill rule"""
    h, w = grid_shape(inp)
    out = [row[:] for row in inp]
    bg = rule['bg']
    periodic = rule.get('periodic', False)
    period = rule.get('period')
    
    # Find rows with non-bg pixels
    row_colors = {}
    for r in range(h):
        colors = set()
        for c in range(w):
            if inp[r][c] != bg:
                colors.add(inp[r][c])
        if colors and len(colors) == 1:
            row_colors[r] = list(colors)[0]
    
    # Fill rows
    for r in range(h):
        if r in row_colors:
            color = row_colors[r]
            for c in range(w):
                out[r][c] = color
            
            # If periodic, also fill repeated rows
            if periodic and period:
                for k in range(1, h // period + 1):
                    repeat_r = r + k * period
                    if repeat_r < h:
                        for c in range(w):
                            out[repeat_r][c] = color
    
    return out


def _apply_col_fill(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply column fill rule"""
    h, w = grid_shape(inp)
    out = [row[:] for row in inp]
    bg = rule['bg']
    periodic = rule.get('periodic', False)
    period = rule.get('period')
    
    # Find columns with non-bg pixels
    col_colors = {}
    for c in range(w):
        colors = set()
        for r in range(h):
            if inp[r][c] != bg:
                colors.add(inp[r][c])
        if colors and len(colors) == 1:
            col_colors[c] = list(colors)[0]
    
    # Fill columns
    for c in range(w):
        if c in col_colors:
            color = col_colors[c]
            for r in range(h):
                out[r][c] = color
            
            # If periodic, also fill repeated columns
            if periodic and period:
                for k in range(1, w // period + 1):
                    repeat_c = c + k * period
                    if repeat_c < w:
                        for r in range(h):
                            out[r][repeat_c] = color
    
    return out
