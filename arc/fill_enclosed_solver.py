"""
arc/fill_enclosed_solver.py — Fill enclosed rectangular regions

Handles tasks where rectangular frames (made of a single color) 
have their interior filled with a new color based on region properties
(e.g., smallest interior gets filled, or all interiors get filled).

Also handles: connect-the-dots between same-colored objects,
and row/col pattern projection.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from collections import Counter


def _find_rectangular_frames(grid: np.ndarray, bg: int = 0) -> List[Dict]:
    """Find rectangular frames in the grid.
    A frame is a rectangle border made of a single non-bg color with bg interior."""
    h, w = grid.shape
    frames = []
    
    for color in range(1, 10):
        positions = list(zip(*np.where(grid == color)))
        if len(positions) < 4:
            continue
        
        rs = [p[0] for p in positions]
        cs = [p[1] for p in positions]
        r_min, r_max = min(rs), max(rs)
        c_min, c_max = min(cs), max(cs)
        
        if r_max - r_min < 2 or c_max - c_min < 2:
            continue
        
        # Check if this forms a rectangular frame
        expected_frame = set()
        for r in range(r_min, r_max + 1):
            expected_frame.add((r, c_min))
            expected_frame.add((r, c_max))
        for c in range(c_min, c_max + 1):
            expected_frame.add((r_min, c))
            expected_frame.add((r_max, c))
        
        actual = set(positions)
        # Allow some tolerance (frame might have gaps)
        if len(actual & expected_frame) >= len(expected_frame) * 0.8:
            # Check interior is bg
            interior = []
            for r in range(r_min + 1, r_max):
                for c in range(c_min + 1, c_max):
                    interior.append(grid[r, c])
            
            if interior and all(v == bg for v in interior):
                frames.append({
                    'color': color,
                    'r_min': r_min, 'r_max': r_max,
                    'c_min': c_min, 'c_max': c_max,
                    'interior_size': len(interior),
                    'is_exact': actual == expected_frame,
                })
    
    return frames


def _flood_fill_regions(grid: np.ndarray, bg: int = 0) -> List[Dict]:
    """Find connected bg regions enclosed by non-bg cells."""
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    regions = []
    
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg or visited[r, c]:
                continue
            
            # BFS to find connected bg region
            queue = [(r, c)]
            visited[r, c] = True
            cells = []
            touches_border = False
            border_colors = set()
            
            while queue:
                cr, cc = queue.pop(0)
                cells.append((cr, cc))
                
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if grid[nr, nc] == bg and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                        elif grid[nr, nc] != bg:
                            border_colors.add(int(grid[nr, nc]))
            
            if not touches_border and len(cells) > 0:
                rs = [p[0] for p in cells]
                cs = [p[1] for p in cells]
                regions.append({
                    'cells': cells,
                    'size': len(cells),
                    'r_min': min(rs), 'r_max': max(rs),
                    'c_min': min(cs), 'c_max': max(cs),
                    'border_colors': border_colors,
                    'is_rect': len(cells) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1),
                })
    
    return regions


def generate_fill_enclosed_pieces(train_pairs, bg=0):
    """Generate CrossPiece candidates for enclosed-region filling tasks."""
    from arc.cross_engine import CrossPiece
    
    pieces = []
    
    # Strategy 1: Fill enclosed bg regions with a learned color
    try:
        piece = _try_fill_enclosed(train_pairs, bg)
        if piece:
            pieces.append(piece)
    except Exception:
        pass
    
    # Strategy 2: Connect same-colored objects with lines
    try:
        piece = _try_connect_objects(train_pairs, bg)
        if piece:
            pieces.append(piece)
    except Exception:
        pass
    
    # Strategy 3: Row/col pattern projection
    try:
        piece = _try_pattern_projection(train_pairs, bg)
        if piece:
            pieces.append(piece)
    except Exception:
        pass
    
    return pieces


def _try_fill_enclosed(train_pairs, bg=0):
    """Try: find enclosed bg regions, fill them with appropriate color."""
    from arc.cross_engine import CrossPiece
    
    # Analyze what filling rule is used across train pairs
    fill_rules = []
    
    for inp_grid, out_grid in train_pairs:
        inp = np.array(inp_grid)
        out = np.array(out_grid)
        if inp.shape != out.shape:
            return None
        
        diff = inp != out
        if not diff.any():
            continue
        
        # Only bg cells should change
        changed_from = set(inp[diff].tolist())
        if changed_from != {bg}:
            return None
        
        fill_color = set(out[diff].tolist())
        if len(fill_color) != 1:
            return None
        fill_color = fill_color.pop()
        
        # Find enclosed regions
        regions = _flood_fill_regions(inp, bg)
        if not regions:
            return None
        
        # Which regions were filled?
        filled_regions = []
        unfilled_regions = []
        for reg in regions:
            sample_r, sample_c = reg['cells'][0]
            if out[sample_r, sample_c] != bg:
                filled_regions.append(reg)
            else:
                unfilled_regions.append(reg)
        
        if not filled_regions:
            return None
        
        fill_rules.append({
            'fill_color': fill_color,
            'n_filled': len(filled_regions),
            'n_unfilled': len(unfilled_regions),
            'filled_sizes': sorted([r['size'] for r in filled_regions]),
            'unfilled_sizes': sorted([r['size'] for r in unfilled_regions]),
            'regions': regions,
            'filled': filled_regions,
        })
    
    if not fill_rules:
        return None
    
    # Determine the fill rule
    # Common patterns:
    # 1. Fill ALL enclosed regions
    # 2. Fill smallest enclosed region
    # 3. Fill by some size criterion
    
    all_same_color = len(set(r['fill_color'] for r in fill_rules)) == 1
    if not all_same_color:
        return None
    
    fill_color = fill_rules[0]['fill_color']
    
    # Check if all regions are filled (simplest rule)
    all_filled = all(r['n_unfilled'] == 0 for r in fill_rules)
    
    # Check if only smallest is filled
    smallest_filled = all(
        r['n_filled'] == 1 and 
        r['filled'][0]['size'] == min(reg['size'] for reg in r['regions'])
        for r in fill_rules
    )
    
    if all_filled:
        rule_type = 'all'
    elif smallest_filled:
        rule_type = 'smallest'
    else:
        # Try: fill regions with specific size range
        rule_type = 'all'  # fallback
    
    def apply_fn(inp_grid):
        inp = np.array(inp_grid)
        out = inp.copy()
        regions = _flood_fill_regions(inp, bg)
        
        if rule_type == 'smallest' and regions:
            regions = [min(regions, key=lambda r: r['size'])]
        
        for reg in regions:
            for r, c in reg['cells']:
                out[r, c] = fill_color
        
        return out.tolist()
    
    # Verify on train
    for inp_grid, out_grid in train_pairs:
        result = apply_fn(inp_grid)
        if result != out_grid:
            return None
    
    name = f"fill_enclosed_{rule_type}_c{fill_color}"
    return CrossPiece(name=name, apply_fn=apply_fn, version=1)


def _try_connect_objects(train_pairs, bg=0):
    """Try: connect same-colored objects with lines/fills."""
    from arc.cross_engine import CrossPiece
    
    # Check if output fills gaps between same-colored pixels in same row/col
    fill_configs = []
    
    for inp_grid, out_grid in train_pairs:
        inp = np.array(inp_grid)
        out = np.array(out_grid)
        if inp.shape != out.shape:
            return None
        
        diff = inp != out
        if not diff.any():
            continue
        
        changed_from = set(inp[diff].tolist())
        changed_to = set(out[diff].tolist())
        
        if len(changed_to) != 1:
            return None
        
        fill_color = changed_to.pop()
        
        # Check: are all filled cells between same-colored objects in same row?
        positions = list(zip(*np.where(diff)))
        
        # Test: fill between same-color pixels on same row
        valid = True
        for r, c in positions:
            # Check if there's a same-color-as-fill_color pixel on same row to left and right
            row = inp[r]
            left_found = any(row[cc] == fill_color for cc in range(c))
            right_found = any(row[cc] == fill_color for cc in range(c + 1, inp.shape[1]))
            
            col = inp[:, c]
            top_found = any(col[rr] == fill_color for rr in range(r))
            bottom_found = any(col[rr] == fill_color for rr in range(r + 1, inp.shape[0]))
            
            if not ((left_found and right_found) or (top_found and bottom_found)):
                valid = False
                break
        
        if not valid:
            return None
        
        fill_configs.append({'fill_color': fill_color})
    
    if not fill_configs:
        return None
    
    fill_color = fill_configs[0]['fill_color']
    if not all(fc['fill_color'] == fill_color for fc in fill_configs):
        return None
    
    # Try each mode: row-only, col-only, both
    best_mode = None
    for mode in ['row', 'col', 'both']:
        def _make_fn(m):
            def fn(inp_grid):
                return _apply_connect(inp_grid, fill_color, bg, m)
            return fn
        
        test_fn = _make_fn(mode)
        ok = all(test_fn(ig) == og for ig, og in train_pairs)
        if ok:
            best_mode = mode
            break
    
    if best_mode is None:
        return None
    
    apply_fn = _make_fn(best_mode)
    
    # Verify
    for inp_grid, out_grid in train_pairs:
        if apply_fn(inp_grid) != out_grid:
            return None
    
    return CrossPiece(name=f"connect_objects_{best_mode}_c{fill_color}", apply_fn=apply_fn, version=1)


def _apply_connect(inp_grid, fill_color, bg, mode):
    inp = np.array(inp_grid)
    out = inp.copy()
    h, w = inp.shape
    
    # Fill between same-colored pixels on same row
    if mode in ('row', 'both'):
        for r in range(h):
                positions = [c for c in range(w) if inp[r, c] == fill_color]
                if len(positions) >= 2:
                    for c in range(min(positions), max(positions) + 1):
                        if out[r, c] == bg:
                            out[r, c] = fill_color
        
        # Fill between same-colored pixels on same col
        if mode in ('col', 'both'):
            for c in range(w):
                positions = [r for r in range(h) if inp[r, c] == fill_color]
                if len(positions) >= 2:
                    for r in range(min(positions), max(positions) + 1):
                        if out[r, c] == bg:
                            out[r, c] = fill_color
        
        return out.tolist()


def _try_pattern_projection(train_pairs, bg=0):
    """Try: project row pattern onto column markers or vice versa.
    
    Pattern: row0 has pattern [0,5,0,5,5,0,0,5,0,0]
    Certain rows have a marker (single pixel) at col 9
    → Those rows get the row0 pattern stamped in a different color
    """
    from arc.cross_engine import CrossPiece
    
    configs = []
    
    for inp_grid, out_grid in train_pairs:
        inp = np.array(inp_grid)
        out = np.array(out_grid)
        if inp.shape != out.shape:
            return None
        
        diff = inp != out
        if not diff.any():
            continue
        
        changed_from = set(inp[diff].tolist())
        changed_to = set(out[diff].tolist())
        
        if changed_from != {bg} or len(changed_to) != 1:
            return None
        
        stamp_color = changed_to.pop()
        h, w = inp.shape
        
        # Find which rows were modified
        modified_rows = set()
        modified_cols = set()
        for r, c in zip(*np.where(diff)):
            modified_rows.add(int(r))
            modified_cols.add(int(c))
        
        # Check: is there a "template row" with a pattern, and "marker" rows/cols
        # that indicate where to stamp?
        
        # Find rows with non-bg content in input
        content_rows = {}  # row -> list of (col, color)
        for r in range(h):
            non_bg = [(c, int(inp[r, c])) for c in range(w) if inp[r, c] != bg]
            if non_bg:
                content_rows[r] = non_bg
        
        # Find cols with non-bg content  
        content_cols = {}
        for c in range(w):
            non_bg = [(r, int(inp[r, c])) for r in range(h) if inp[r, c] != bg]
            if non_bg:
                content_cols[c] = non_bg
        
        # Pattern: template in first/last row, markers in a specific column
        # Check if modified rows correlate with marker positions
        
        # Find a column that has markers exactly at modified rows
        found = False
        for mc in range(w):
            marker_rows = set(r for r in range(h) if inp[r, mc] != bg and r not in content_rows or (r in content_rows and len(content_rows[r]) == 1 and content_rows[r][0][0] == mc))
            
            # Actually simpler: check if there's a column with single pixels
            # at exactly the modified rows
            marker_rows_simple = set()
            marker_color = None
            for r in range(h):
                if inp[r, mc] != bg:
                    row_content = [(c, int(inp[r, c])) for c in range(w) if inp[r, c] != bg]
                    if len(row_content) == 1 and row_content[0][0] == mc:
                        marker_rows_simple.add(r)
                        marker_color = row_content[0][1]
            
            if marker_rows_simple == modified_rows and marker_color is not None:
                # Find the template row (a row with the pattern)
                # Template = a row that has the same column positions as the stamped pattern
                stamped_cols = set()
                for r in modified_rows:
                    for c in range(w):
                        if diff[r, c]:
                            stamped_cols.add(c)
                
                # Find template row
                for tr in range(h):
                    if tr in modified_rows:
                        continue
                    template_cols = set(c for c in range(w) if inp[tr, c] != bg)
                    if template_cols == stamped_cols:
                        configs.append({
                            'marker_col': mc,
                            'marker_color': marker_color,
                            'template_row': tr,
                            'stamp_color': stamp_color,
                        })
                        found = True
                        break
                
                if found:
                    break
        
        # Try column markers (template in first/last col, markers in a row)
        if not found:
            for mr in range(h):
                marker_cols_simple = set()
                marker_color = None
                for c in range(w):
                    if inp[mr, c] != bg:
                        col_content = [(r, int(inp[r, c])) for r in range(h) if inp[r, c] != bg]
                        if len(col_content) == 1 and col_content[0][0] == mr:
                            marker_cols_simple.add(c)
                            marker_color = col_content[0][1]
                
                if marker_cols_simple == modified_cols and marker_color is not None:
                    stamped_rows = set()
                    for r in range(h):
                        for c in modified_cols:
                            if diff[r, c]:
                                stamped_rows.add(r)
                    
                    for tc in range(w):
                        if tc in modified_cols:
                            continue
                        template_rows = set(r for r in range(h) if inp[r, tc] != bg)
                        if template_rows == stamped_rows:
                            configs.append({
                                'marker_row': mr,
                                'marker_color': marker_color,
                                'template_col': tc,
                                'stamp_color': stamp_color,
                            })
                            found = True
                            break
                    
                    if found:
                        break
        
        if not found:
            return None
    
    if not configs:
        return None
    
    # Check consistency
    if 'marker_col' in configs[0]:
        # Row-based stamping
        def apply_fn(inp_grid):
            inp = np.array(inp_grid)
            out = inp.copy()
            h, w = inp.shape
            cfg = configs[0]
            mc = cfg['marker_col']
            tr = cfg['template_row']
            sc = cfg['stamp_color']
            
            template = [int(inp[tr, c]) for c in range(w)]
            
            for r in range(h):
                row_content = [(c, int(inp[r, c])) for c in range(w) if inp[r, c] != bg]
                if len(row_content) == 1 and row_content[0][0] == mc:
                    for c in range(w):
                        if template[c] != bg and out[r, c] == bg:
                            out[r, c] = sc
            
            return out.tolist()
    else:
        # Col-based stamping
        def apply_fn(inp_grid):
            inp = np.array(inp_grid)
            out = inp.copy()
            h, w = inp.shape
            cfg = configs[0]
            mr = cfg['marker_row']
            tc = cfg['template_col']
            sc = cfg['stamp_color']
            
            template = [int(inp[r, tc]) for r in range(h)]
            
            for c in range(w):
                col_content = [(r, int(inp[r, c])) for r in range(h) if inp[r, c] != bg]
                if len(col_content) == 1 and col_content[0][0] == mr:
                    for r in range(h):
                        if template[r] != bg and out[r, c] == bg:
                            out[r, c] = sc
            
            return out.tolist()
    
    # Verify
    for inp_grid, out_grid in train_pairs:
        if apply_fn(inp_grid) != out_grid:
            return None
    
    return CrossPiece(name=f"pattern_projection", apply_fn=apply_fn, version=1)
