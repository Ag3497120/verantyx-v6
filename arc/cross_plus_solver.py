"""
arc/cross_plus_solver.py — Cross/Plus Pattern Solver for ARC-AGI-2

Handles tasks where:
1. Colored dots create cross/plus patterns around them
2. Pattern extends in 4 directions (up, down, left, right)
3. Each dot color may produce different cross patterns or colors
4. Can have varying cross sizes/shapes

Example tasks:
- 0ca9ddb6: each colored dot creates a cross pattern with new colors
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from arc.grid import Grid, grid_shape, grid_eq, most_common_color


def learn_cross_plus_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn cross/plus pattern rule from training examples.
    
    Returns rule dict with:
    - type: 'cross_pattern'
    - bg: background color
    - dot_patterns: dict mapping dot_color -> cross pattern info
    - cross_size: size of cross (radius from center)
    """
    
    # Try to learn cross pattern with varying sizes
    for cross_size in [1, 2, 3]:
        rule = _learn_cross_pattern(train_pairs, cross_size)
        if rule:
            return rule
    
    # Try to learn asymmetric cross patterns
    rule = _learn_asymmetric_cross(train_pairs)
    if rule:
        return rule
    
    return None


def apply_cross_plus_rule(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply learned cross/plus pattern rule"""
    rule_type = rule['type']
    
    if rule_type == 'cross_pattern':
        return _apply_cross_pattern(inp, rule)
    elif rule_type == 'asymmetric_cross':
        return _apply_asymmetric_cross(inp, rule)
    
    return None


def _learn_cross_pattern(train_pairs: List[Tuple[Grid, Grid]], cross_size: int) -> Optional[Dict]:
    """Learn: colored dots create symmetric cross patterns.
    
    For each colored dot at (r,c), creates a cross pattern:
    - Vertical arm: (r-cross_size..r+cross_size, c)
    - Horizontal arm: (r, c-cross_size..c+cross_size)
    """
    for bg in [0]:
        # Analyze first pair to learn pattern
        if not train_pairs:
            return None
        
        inp0, out0 = train_pairs[0]
        h, w = grid_shape(inp0)
        if grid_shape(out0) != (h, w):
            return None
        
        # Find all non-bg dots in input
        dots = []
        for r in range(h):
            for c in range(w):
                if inp0[r][c] != bg:
                    dots.append((r, c, inp0[r][c]))
        
        if not dots:
            return None
        
        # Learn what cross pattern each dot creates
        dot_patterns = {}
        
        for dot_r, dot_c, dot_color in dots:
            # Check what colors appear in cross around this dot
            cross_cells = []
            
            # Vertical arm
            for dr in range(-cross_size, cross_size + 1):
                r = dot_r + dr
                if 0 <= r < h:
                    cross_cells.append((r, dot_c))
            
            # Horizontal arm
            for dc in range(-cross_size, cross_size + 1):
                c = dot_c + dc
                if 0 <= c < w and (dot_r, c) not in cross_cells:
                    cross_cells.append((dot_r, c))
            
            # Record what colors appear in output at these positions
            pattern_colors = set()
            for r, c in cross_cells:
                if (r, c) != (dot_r, dot_c):  # Exclude center
                    color = out0[r][c]
                    if color != bg and color != dot_color:
                        pattern_colors.add(color)
            
            if pattern_colors:
                dot_patterns[dot_color] = {
                    'cross_colors': list(pattern_colors),
                    'keep_center': out0[dot_r][dot_c] == dot_color
                }
        
        if not dot_patterns:
            return None
        
        # Verify this rule on all training pairs
        ok = True
        for inp, out in train_pairs:
            result = _apply_cross_pattern_helper(inp, bg, cross_size, dot_patterns)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        
        if ok:
            return {
                'type': 'cross_pattern',
                'bg': bg,
                'cross_size': cross_size,
                'dot_patterns': dot_patterns
            }
    
    return None


def _learn_asymmetric_cross(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: colored dots create asymmetric cross patterns.
    
    Each dot may create crosses of different lengths in each direction.
    """
    for bg in [0]:
        if not train_pairs:
            return None
        
        inp0, out0 = train_pairs[0]
        h, w = grid_shape(inp0)
        if grid_shape(out0) != (h, w):
            return None
        
        # Find all non-bg dots
        dots = []
        for r in range(h):
            for c in range(w):
                if inp0[r][c] != bg:
                    dots.append((r, c, inp0[r][c]))
        
        if not dots:
            return None
        
        # Learn asymmetric cross for each dot
        dot_patterns = {}
        
        for dot_r, dot_c, dot_color in dots:
            # Detect cross arm lengths in each direction
            cross_info = {
                'up': 0, 'down': 0, 'left': 0, 'right': 0,
                'colors': set()
            }
            
            # Check up
            for dr in range(1, h):
                r = dot_r - dr
                if r < 0:
                    break
                if out0[r][dot_c] != bg and out0[r][dot_c] != inp0[r][dot_c]:
                    cross_info['up'] = dr
                    cross_info['colors'].add(out0[r][dot_c])
                else:
                    break
            
            # Check down
            for dr in range(1, h):
                r = dot_r + dr
                if r >= h:
                    break
                if out0[r][dot_c] != bg and out0[r][dot_c] != inp0[r][dot_c]:
                    cross_info['down'] = dr
                    cross_info['colors'].add(out0[r][dot_c])
                else:
                    break
            
            # Check left
            for dc in range(1, w):
                c = dot_c - dc
                if c < 0:
                    break
                if out0[dot_r][c] != bg and out0[dot_r][c] != inp0[dot_r][c]:
                    cross_info['left'] = dc
                    cross_info['colors'].add(out0[dot_r][c])
                else:
                    break
            
            # Check right
            for dc in range(1, w):
                c = dot_c + dc
                if c >= w:
                    break
                if out0[dot_r][c] != bg and out0[dot_r][c] != inp0[dot_r][c]:
                    cross_info['right'] = dc
                    cross_info['colors'].add(out0[dot_r][c])
                else:
                    break
            
            if any(cross_info[d] > 0 for d in ['up', 'down', 'left', 'right']):
                dot_patterns[dot_color] = cross_info
        
        if not dot_patterns:
            return None
        
        # Verify on all training pairs
        ok = True
        for inp, out in train_pairs:
            result = _apply_asymmetric_cross_helper(inp, bg, dot_patterns)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        
        if ok:
            return {
                'type': 'asymmetric_cross',
                'bg': bg,
                'dot_patterns': dot_patterns
            }
    
    return None


def _apply_cross_pattern(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply symmetric cross pattern rule"""
    bg = rule['bg']
    cross_size = rule['cross_size']
    dot_patterns = rule['dot_patterns']
    
    return _apply_cross_pattern_helper(inp, bg, cross_size, dot_patterns)


def _apply_cross_pattern_helper(inp: Grid, bg: int, cross_size: int, 
                                  dot_patterns: Dict) -> Optional[Grid]:
    """Helper to apply symmetric cross pattern"""
    h, w = grid_shape(inp)
    out = [row[:] for row in inp]
    
    # Find all dots in input
    dots = []
    for r in range(h):
        for c in range(w):
            if inp[r][c] != bg and inp[r][c] in dot_patterns:
                dots.append((r, c, inp[r][c]))
    
    # Draw cross for each dot
    for dot_r, dot_c, dot_color in dots:
        pattern = dot_patterns[dot_color]
        cross_colors = pattern['cross_colors']
        keep_center = pattern.get('keep_center', True)
        
        if not cross_colors:
            continue
        
        fill_color = cross_colors[0]  # Use first color
        
        # Vertical arm
        for dr in range(-cross_size, cross_size + 1):
            r = dot_r + dr
            if 0 <= r < h:
                if (r, dot_c) != (dot_r, dot_c) or not keep_center:
                    out[r][dot_c] = fill_color
        
        # Horizontal arm
        for dc in range(-cross_size, cross_size + 1):
            c = dot_c + dc
            if 0 <= c < w:
                if (dot_r, c) != (dot_r, dot_c) or not keep_center:
                    out[dot_r][c] = fill_color
    
    return out


def _apply_asymmetric_cross(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply asymmetric cross pattern rule"""
    bg = rule['bg']
    dot_patterns = rule['dot_patterns']
    
    return _apply_asymmetric_cross_helper(inp, bg, dot_patterns)


def _apply_asymmetric_cross_helper(inp: Grid, bg: int, dot_patterns: Dict) -> Optional[Grid]:
    """Helper to apply asymmetric cross pattern"""
    h, w = grid_shape(inp)
    out = [row[:] for row in inp]
    
    # Find all dots
    dots = []
    for r in range(h):
        for c in range(w):
            if inp[r][c] != bg and inp[r][c] in dot_patterns:
                dots.append((r, c, inp[r][c]))
    
    # Draw asymmetric cross for each dot
    for dot_r, dot_c, dot_color in dots:
        pattern = dot_patterns[dot_color]
        colors = list(pattern['colors'])
        
        if not colors:
            continue
        
        fill_color = colors[0]
        
        # Up
        for dr in range(1, pattern['up'] + 1):
            r = dot_r - dr
            if 0 <= r < h:
                out[r][dot_c] = fill_color
        
        # Down
        for dr in range(1, pattern['down'] + 1):
            r = dot_r + dr
            if 0 <= r < h:
                out[r][dot_c] = fill_color
        
        # Left
        for dc in range(1, pattern['left'] + 1):
            c = dot_c - dc
            if 0 <= c < w:
                out[dot_r][c] = fill_color
        
        # Right
        for dc in range(1, pattern['right'] + 1):
            c = dot_c + dc
            if 0 <= c < w:
                out[dot_r][c] = fill_color
    
    return out
