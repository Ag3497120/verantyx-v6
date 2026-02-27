"""
arc/flood_fill_solver.py — Flood Fill / Region Extension Solver

Patterns:
1. Line extension: colored segments extend to walls/boundaries
2. Rectangle fill: regions between markers get filled
3. Pattern repair: damaged tiling patterns get restored
4. Enclosed region fill: enclosed areas get filled with a color
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import ndimage
from collections import Counter

from arc.grid import Grid, grid_shape, grid_eq, most_common_color
from arc.cross_engine import CrossPiece


def _find_lines(grid: np.ndarray, color: int) -> List[Tuple[int, int, int, int, str]]:
    """Find horizontal and vertical line segments of a given color.
    Returns: [(r, c, length, color, 'h'|'v'), ...]
    """
    h, w = grid.shape
    lines = []
    # Horizontal
    for r in range(h):
        c_start = None
        for c in range(w + 1):
            if c < w and grid[r, c] == color:
                if c_start is None:
                    c_start = c
            else:
                if c_start is not None and (c - c_start) >= 2:
                    lines.append((r, c_start, c - c_start, color, 'h'))
                c_start = None
    # Vertical
    for c in range(w):
        r_start = None
        for r in range(h + 1):
            if r < h and grid[r, c] == color:
                if r_start is None:
                    r_start = r
            else:
                if r_start is not None and (r - r_start) >= 2:
                    lines.append((r_start, c, r - r_start, color, 'v'))
                r_start = None
    return lines


def try_line_extend_to_wall(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    Colored line segments extend to grid boundary or to another color.
    E.g., a vertical bar of color 7 at col 2, rows 7-10 → extends to fill col 2, rows 0-10.
    """
    pieces = []
    
    # Analyze: what changed between input and output?
    inp0, out0 = train_pairs[0]
    inp_np = np.array(inp0)
    out_np = np.array(out0)
    
    if inp_np.shape != out_np.shape:
        return pieces
    
    diff_mask = inp_np != out_np
    if not diff_mask.any():
        return pieces
    
    # Only bg cells should change
    changed_from = inp_np[diff_mask]
    if not all(v == bg for v in changed_from):
        return pieces
    
    changed_to = out_np[diff_mask]
    fill_colors = set(changed_to.tolist()) - {bg}
    
    # For each fill color, check if it extends existing segments
    for fill_color in fill_colors:
        # Find segments of this color in input
        in_positions = np.argwhere(inp_np == fill_color)
        out_positions = np.argwhere(out_np == fill_color)
        
        if len(in_positions) == 0:
            continue
        
        # Check: do existing segments extend horizontally to a wall?
        # Group by row
        in_rows = Counter(in_positions[:, 0].tolist())
        out_rows = Counter(out_positions[:, 0].tolist())
        
        # Check: do existing segments extend vertically to a wall?
        in_cols = Counter(in_positions[:, 1].tolist())
        out_cols = Counter(out_positions[:, 1].tolist())
    
    # Try: each non-bg segment extends in its major direction to fill the row/col
    # Strategy 1: extend to right wall
    for direction in ['right', 'left', 'up', 'down', 'both_h', 'both_v']:
        all_match = True
        for inp, out in train_pairs:
            inp_np = np.array(inp)
            out_np = np.array(out)
            h, w = inp_np.shape
            result = inp_np.copy()
            
            for color in range(1, 10):
                mask = (inp_np == color)
                if not mask.any():
                    continue
                
                labeled, n = ndimage.label(mask)
                for lbl in range(1, n + 1):
                    positions = np.argwhere(labeled == lbl)
                    r_min, c_min = positions.min(axis=0)
                    r_max, c_max = positions.max(axis=0)
                    
                    if direction == 'right':
                        for r in range(r_min, r_max + 1):
                            for c in range(c_max + 1, w):
                                if result[r, c] == bg:
                                    result[r, c] = color
                                else:
                                    break
                    elif direction == 'left':
                        for r in range(r_min, r_max + 1):
                            for c in range(c_min - 1, -1, -1):
                                if result[r, c] == bg:
                                    result[r, c] = color
                                else:
                                    break
                    elif direction == 'down':
                        for c in range(c_min, c_max + 1):
                            for r in range(r_max + 1, h):
                                if result[r, c] == bg:
                                    result[r, c] = color
                                else:
                                    break
                    elif direction == 'up':
                        for c in range(c_min, c_max + 1):
                            for r in range(r_min - 1, -1, -1):
                                if result[r, c] == bg:
                                    result[r, c] = color
                                else:
                                    break
                    elif direction == 'both_h':
                        for r in range(r_min, r_max + 1):
                            for c in range(c_max + 1, w):
                                if result[r, c] == bg:
                                    result[r, c] = color
                                else:
                                    break
                            for c in range(c_min - 1, -1, -1):
                                if result[r, c] == bg:
                                    result[r, c] = color
                                else:
                                    break
                    elif direction == 'both_v':
                        for c in range(c_min, c_max + 1):
                            for r in range(r_max + 1, h):
                                if result[r, c] == bg:
                                    result[r, c] = color
                                else:
                                    break
                            for r in range(r_min - 1, -1, -1):
                                if result[r, c] == bg:
                                    result[r, c] = color
                                else:
                                    break
            
            if not np.array_equal(result, out_np):
                all_match = False
                break
        
        if all_match:
            def _apply(inp, _bg=bg, _dir=direction):
                inp_np = np.array(inp)
                h, w = inp_np.shape
                result = inp_np.copy()
                for color in range(1, 10):
                    mask = (inp_np == color)
                    if not mask.any():
                        continue
                    labeled, n = ndimage.label(mask)
                    for lbl in range(1, n + 1):
                        positions = np.argwhere(labeled == lbl)
                        r_min, c_min = positions.min(axis=0)
                        r_max, c_max = positions.max(axis=0)
                        if _dir in ('right', 'both_h'):
                            for r in range(r_min, r_max + 1):
                                for c in range(c_max + 1, w):
                                    if result[r, c] == _bg: result[r, c] = color
                                    else: break
                        if _dir in ('left', 'both_h'):
                            for r in range(r_min, r_max + 1):
                                for c in range(c_min - 1, -1, -1):
                                    if result[r, c] == _bg: result[r, c] = color
                                    else: break
                        if _dir in ('down', 'both_v'):
                            for c in range(c_min, c_max + 1):
                                for r in range(r_max + 1, h):
                                    if result[r, c] == _bg: result[r, c] = color
                                    else: break
                        if _dir in ('up', 'both_v'):
                            for c in range(c_min, c_max + 1):
                                for r in range(r_min - 1, -1, -1):
                                    if result[r, c] == _bg: result[r, c] = color
                                    else: break
                return result.tolist()
            
            pieces.append(CrossPiece(f'flood:line_extend_{direction}', _apply))
    
    return pieces


def try_enclosed_fill(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    Fill enclosed regions: areas surrounded by non-bg cells get filled.
    """
    pieces = []
    
    # Check if output fills interior bg regions with a specific color
    all_match = True
    fill_color = None
    
    for inp, out in train_pairs:
        inp_np = np.array(inp)
        out_np = np.array(out)
        
        if inp_np.shape != out_np.shape:
            return pieces
        
        # Find interior bg regions (not connected to border)
        bg_mask = (inp_np == bg)
        labeled, n = ndimage.label(bg_mask)
        
        # Find border-connected labels
        h, w = inp_np.shape
        border_labels = set()
        border_labels.update(labeled[0, :].tolist())
        border_labels.update(labeled[-1, :].tolist())
        border_labels.update(labeled[:, 0].tolist())
        border_labels.update(labeled[:, -1].tolist())
        border_labels.discard(0)
        
        interior_mask = np.zeros_like(bg_mask)
        for lbl in range(1, n + 1):
            if lbl not in border_labels:
                interior_mask |= (labeled == lbl)
        
        if not interior_mask.any():
            all_match = False
            break
        
        # What color fills the interior in the output?
        interior_colors = out_np[interior_mask]
        c = Counter(interior_colors.tolist())
        dominant = c.most_common(1)[0][0]
        
        if fill_color is None:
            fill_color = dominant
        elif fill_color != dominant:
            all_match = False
            break
        
        # Verify: interior cells change from bg to fill_color, nothing else changes
        expected = inp_np.copy()
        expected[interior_mask] = fill_color
        if not np.array_equal(expected, out_np):
            all_match = False
            break
    
    if all_match and fill_color is not None:
        def _apply(inp, _bg=bg, _fc=fill_color):
            inp_np = np.array(inp)
            bg_mask = (inp_np == _bg)
            labeled, n = ndimage.label(bg_mask)
            h, w = inp_np.shape
            border_labels = set()
            border_labels.update(labeled[0, :].tolist())
            border_labels.update(labeled[-1, :].tolist())
            border_labels.update(labeled[:, 0].tolist())
            border_labels.update(labeled[:, -1].tolist())
            border_labels.discard(0)
            result = inp_np.copy()
            for lbl in range(1, n + 1):
                if lbl not in border_labels:
                    result[labeled == lbl] = _fc
            return result.tolist()
        
        pieces.append(CrossPiece(f'flood:enclosed_fill_c{fill_color}', _apply))
    
    # Also try: fill each enclosed region with the color of its surrounding border
    all_match2 = True
    for inp, out in train_pairs:
        inp_np = np.array(inp)
        out_np = np.array(out)
        if inp_np.shape != out_np.shape:
            all_match2 = False
            break
        
        bg_mask = (inp_np == bg)
        labeled, n = ndimage.label(bg_mask)
        h, w = inp_np.shape
        border_labels = set()
        border_labels.update(labeled[0, :].tolist())
        border_labels.update(labeled[-1, :].tolist())
        border_labels.update(labeled[:, 0].tolist())
        border_labels.update(labeled[:, -1].tolist())
        border_labels.discard(0)
        
        result = inp_np.copy()
        for lbl in range(1, n + 1):
            if lbl in border_labels:
                continue
            region = (labeled == lbl)
            # Find surrounding color (dilate region and check border)
            dilated = ndimage.binary_dilation(region, iterations=1)
            border = dilated & ~region
            border_colors = inp_np[border]
            border_colors = border_colors[border_colors != bg]
            if len(border_colors) == 0:
                continue
            surround_color = Counter(border_colors.tolist()).most_common(1)[0][0]
            result[region] = surround_color
        
        if not np.array_equal(result, out_np):
            all_match2 = False
            break
    
    if all_match2:
        def _apply2(inp, _bg=bg):
            inp_np = np.array(inp)
            bg_mask = (inp_np == _bg)
            labeled, n = ndimage.label(bg_mask)
            h, w = inp_np.shape
            border_labels = set()
            border_labels.update(labeled[0, :].tolist())
            border_labels.update(labeled[-1, :].tolist())
            border_labels.update(labeled[:, 0].tolist())
            border_labels.update(labeled[:, -1].tolist())
            border_labels.discard(0)
            result = inp_np.copy()
            for lbl in range(1, n + 1):
                if lbl in border_labels:
                    continue
                region = (labeled == lbl)
                dilated = ndimage.binary_dilation(region, iterations=1)
                border = dilated & ~region
                border_colors = inp_np[border]
                border_colors = border_colors[border_colors != _bg]
                if len(border_colors) == 0:
                    continue
                surround_color = Counter(border_colors.tolist()).most_common(1)[0][0]
                result[region] = surround_color
            return result.tolist()
        
        pieces.append(CrossPiece('flood:enclosed_fill_by_border', _apply2))
    
    return pieces


def try_pattern_repair(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    Repair damaged tiling patterns: find the tile period and fill holes.
    """
    pieces = []
    
    inp0, out0 = train_pairs[0]
    out_np = np.array(out0)
    inp_np = np.array(inp0)
    h, w = inp_np.shape
    
    if inp_np.shape != out_np.shape:
        return pieces
    
    # Find tile period from output (which is the complete pattern)
    # Try periods from 1 to h//2
    for ph in range(1, h // 2 + 1):
        for pw in range(1, w // 2 + 1):
            tile = out_np[:ph, :pw]
            # Check if output is a perfect tiling of this tile
            is_tiled = True
            for r in range(h):
                for c in range(w):
                    if out_np[r, c] != tile[r % ph, c % pw]:
                        is_tiled = False
                        break
                if not is_tiled:
                    break
            
            if not is_tiled:
                continue
            
            # Verify: input is the same tile but with some cells replaced by bg
            damaged = (inp_np == bg) & (out_np != bg)
            if not damaged.any():
                continue
            
            # Undamaged cells should match the tile
            undamaged = ~damaged
            if not np.array_equal(inp_np[undamaged], out_np[undamaged]):
                continue
            
            # Verify on all training pairs
            all_ok = True
            for inp2, out2 in train_pairs:
                inp2_np = np.array(inp2)
                out2_np = np.array(out2)
                h2, w2 = inp2_np.shape
                
                # Reconstruct from undamaged cells
                # Find tile from non-bg cells
                tile2 = np.zeros((ph, pw), dtype=inp2_np.dtype)
                tile_filled = np.zeros((ph, pw), dtype=bool)
                
                for r in range(h2):
                    for c in range(w2):
                        tr, tc = r % ph, c % pw
                        if inp2_np[r, c] != bg:
                            tile2[tr, tc] = inp2_np[r, c]
                            tile_filled[tr, tc] = True
                
                if not tile_filled.all():
                    # Try to fill from output
                    all_ok = False
                    break
                
                # Apply tile
                result = np.zeros_like(inp2_np)
                for r in range(h2):
                    for c in range(w2):
                        result[r, c] = tile2[r % ph, c % pw]
                
                if not np.array_equal(result, out2_np):
                    all_ok = False
                    break
            
            if all_ok:
                def _apply(inp, _bg=bg, _ph=ph, _pw=pw):
                    inp_np = np.array(inp)
                    h, w = inp_np.shape
                    tile = np.zeros((_ph, _pw), dtype=inp_np.dtype)
                    tile_filled = np.zeros((_ph, _pw), dtype=bool)
                    for r in range(h):
                        for c in range(w):
                            tr, tc = r % _ph, c % _pw
                            if inp_np[r, c] != _bg and not tile_filled[tr, tc]:
                                tile[tr, tc] = inp_np[r, c]
                                tile_filled[tr, tc] = True
                    result = np.zeros_like(inp_np)
                    for r in range(h):
                        for c in range(w):
                            result[r, c] = tile[r % _ph, c % _pw]
                    return result.tolist()
                
                pieces.append(CrossPiece(f'flood:tile_repair_{ph}x{pw}', _apply))
                return pieces  # found one, done
    
    return pieces


def generate_flood_fill_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """Generate flood fill transform pieces."""
    pieces = []
    
    inp0, out0 = train_pairs[0]
    bg = most_common_color(inp0)
    
    pieces.extend(try_line_extend_to_wall(train_pairs, bg))
    pieces.extend(try_enclosed_fill(train_pairs, bg))
    pieces.extend(try_pattern_repair(train_pairs, bg))
    
    return pieces
