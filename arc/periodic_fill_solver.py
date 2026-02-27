"""
arc/periodic_fill_solver.py — Periodic dot expansion solver

Pattern: A few colored dots on bg grid → expand as full rows/cols,
repeating periodically downward/rightward.

Examples:
- 0a938d79: 2 dots → full row fills, repeating with period = row gap
"""

from __future__ import annotations
from typing import List, Tuple, Optional
from arc.grid import Grid, grid_shape, grid_eq
from arc.cross_engine import CrossPiece
import copy


def _find_dots(grid: Grid, bg: int = 0) -> List[Tuple[int, int, int]]:
    """Find non-bg pixels: returns [(row, col, color), ...]"""
    h, w = grid_shape(grid)
    dots = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                dots.append((r, c, grid[r][c]))
    return dots


def _try_periodic_axis_fill(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossPiece]:
    """
    Pattern: N dots define rows or columns to fill, repeating periodically.
    Auto-detects axis (row vs col) per example based on which works.
    Some tasks switch axis based on grid orientation.
    """
    
    def _fill_axis(inp_grid, axis):
        """axis=0: row fill, axis=1: col fill"""
        h, w = grid_shape(inp_grid)
        dots = _find_dots(inp_grid)
        if len(dots) < 2:
            return None
        
        if axis == 0:
            # Sort by row
            dots.sort(key=lambda x: x[0])
            positions = [r for r, c, color in dots]
            colors = [color for r, c, color in dots]
            grid_len = h
        else:
            # Sort by col
            dots.sort(key=lambda x: x[1])
            positions = [c for r, c, color in dots]
            colors = [color for r, c, color in dots]
            grid_len = w
        
        if len(positions) < 2:
            return None
        
        gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        if len(set(gaps)) == 1:
            # Equal gaps: full cycle = n_dots * gap
            period = len(dots) * gaps[0]
        else:
            # Unequal gaps: full cycle = sum of all gaps + repeat gap
            period = positions[-1] - positions[0] + gaps[0]  # assume repeats from last
        
        if period <= 0:
            return None
        
        result = [row[:] for row in inp_grid]
        base_offsets = [p - positions[0] for p in positions]
        
        # Fill in both directions
        # Fill from first dot position downward/rightward only
        for start in range(positions[0], grid_len + period, period):
            for i, offset in enumerate(base_offsets):
                pos = start + offset
                if 0 <= pos < grid_len:
                    if axis == 0:
                        for cc in range(w):
                            result[pos][cc] = colors[i]
                    else:
                        for rr in range(h):
                            result[rr][pos] = colors[i]
        
        return result
    
    # Determine strategy: does this task use row fill, col fill, or auto?
    # Check if consistent axis across all train pairs
    for axis_mode in ['row', 'col', 'auto_shape']:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            if axis_mode == 'row':
                result = _fill_axis(inp, 0)
            elif axis_mode == 'col':
                result = _fill_axis(inp, 1)
            else:  # auto_shape: tall→row, wide→col
                axis = 0 if h >= w else 1
                result = _fill_axis(inp, axis)
            
            if result is None or not grid_eq(result, out):
                ok = False
                break
        
        if ok:
            def make_apply(mode):
                def apply_fn(inp_grid):
                    h, w = grid_shape(inp_grid)
                    if mode == 'row':
                        return _fill_axis(inp_grid, 0)
                    elif mode == 'col':
                        return _fill_axis(inp_grid, 1)
                    else:  # auto_shape
                        axis = 0 if h >= w else 1
                        return _fill_axis(inp_grid, axis)
                return apply_fn
            
            return CrossPiece(f'periodic_{axis_mode}_fill', make_apply(axis_mode))
    
    return None


def _try_periodic_col_fill(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossPiece]:
    """Same as row fill but for columns."""
    for pair_idx, (inp, out) in enumerate(train_pairs):
        h, w = grid_shape(inp)
        dots = _find_dots(inp)
        if len(dots) < 2 or len(dots) > 6:
            return None
        
        # Check: in output, are dot columns fully filled?
        for r, c, color in dots:
            for rr in range(h):
                if out[rr][c] != color:
                    return None
    
    inp0, out0 = train_pairs[0]
    h0, w0 = grid_shape(inp0)
    dots0 = _find_dots(inp0)
    dots0.sort(key=lambda x: x[1])
    dot_cols = [c for r, c, color in dots0]
    dot_colors = [color for r, c, color in dots0]
    
    if len(dot_cols) >= 2:
        gaps = [dot_cols[i+1] - dot_cols[i] for i in range(len(dot_cols)-1)]
        if len(set(gaps)) == 1:
            period = gaps[0]
        else:
            period = sum(gaps)
    else:
        return None
    
    def apply_fn(inp_grid: Grid) -> Optional[Grid]:
        h, w = grid_shape(inp_grid)
        dots = _find_dots(inp_grid)
        if len(dots) < 2:
            return None
        
        dots.sort(key=lambda x: x[1])
        d_cols = [c for r, c, color in dots]
        d_colors = [color for r, c, color in dots]
        
        if len(d_cols) >= 2:
            d_gaps = [d_cols[i+1] - d_cols[i] for i in range(len(d_cols)-1)]
            if len(set(d_gaps)) == 1:
                d_period = d_gaps[0]
            else:
                d_period = sum(d_gaps)
        else:
            return None
        
        result = [row[:] for row in inp_grid]
        base_offsets = [c - d_cols[0] for c in d_cols]
        
        for cycle_start in range(d_cols[0], w, d_period):
            for i, offset in enumerate(base_offsets):
                c = cycle_start + offset
                if 0 <= c < w:
                    for rr in range(h):
                        result[rr][c] = d_colors[i]
        
        for cycle_start in range(d_cols[0] - d_period, -d_period * 10, -d_period):
            any_valid = False
            for i, offset in enumerate(base_offsets):
                c = cycle_start + offset
                if 0 <= c < w:
                    for rr in range(h):
                        result[rr][c] = d_colors[i]
                    any_valid = True
            if not any_valid:
                break
        
        return result
    
    for inp, out in train_pairs:
        result = apply_fn(inp)
        if result is None or not grid_eq(result, out):
            return None
    
    return CrossPiece('periodic_col_fill', apply_fn)


def _try_periodic_row_col_fill(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossPiece]:
    """
    Pattern: dots define both row and column lines.
    Each dot extends its row AND column, creating a grid pattern.
    Rows and columns repeat periodically.
    """
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        dots = _find_dots(inp)
        if len(dots) < 2 or len(dots) > 6:
            return None
    
    def apply_fn(inp_grid: Grid) -> Optional[Grid]:
        h, w = grid_shape(inp_grid)
        dots = _find_dots(inp_grid)
        if len(dots) < 2:
            return None
        
        result = [row[:] for row in inp_grid]
        
        # Try: each dot fills its row AND column
        for r, c, color in dots:
            for cc in range(w):
                if result[r][cc] == 0:
                    result[r][cc] = color
            for rr in range(h):
                if result[rr][c] == 0:
                    result[rr][c] = color
        
        return result
    
    for inp, out in train_pairs:
        result = apply_fn(inp)
        if result is None or not grid_eq(result, out):
            return None
    
    return CrossPiece('dot_row_col_fill', apply_fn)


def _try_ray_from_dots(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossPiece]:
    """
    Pattern: colored dots shoot rays in specific directions until hitting boundary or another color.
    Try all 4 directions and combinations.
    """
    directions = [
        ('down', 1, 0), ('up', -1, 0), ('right', 0, 1), ('left', 0, -1),
    ]
    
    # Try each combination of directions
    for dir_combo_bits in range(1, 16):
        dir_combo = [d for i, d in enumerate(directions) if dir_combo_bits & (1 << i)]
        
        def make_apply(dirs):
            def apply_fn(inp_grid: Grid) -> Optional[Grid]:
                h, w = grid_shape(inp_grid)
                dots = _find_dots(inp_grid)
                if not dots:
                    return None
                
                result = [row[:] for row in inp_grid]
                
                for r, c, color in dots:
                    for name, dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < h and 0 <= nc < w:
                            if result[nr][nc] != 0:
                                break
                            result[nr][nc] = color
                            nr += dr
                            nc += dc
                
                return result
            return apply_fn
        
        apply_fn = make_apply(dir_combo)
        
        ok = True
        for inp, out in train_pairs:
            result = apply_fn(inp)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        
        if ok:
            dir_names = '+'.join(d[0] for d in dir_combo)
            return CrossPiece(f'ray_{dir_names}', apply_fn)
    
    return None


def _try_flood_enclosed_regions(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossPiece]:
    """
    Pattern: colored lines form boundaries. Enclosed bg regions get filled
    based on the nearest/surrounding colored pixel.
    """
    from collections import deque
    
    def find_enclosed_regions(grid, bg=0):
        """Find bg regions and whether they touch the border."""
        h, w = grid_shape(grid)
        visited = [[False]*w for _ in range(h)]
        regions = []
        
        for sr in range(h):
            for sc in range(w):
                if visited[sr][sc] or grid[sr][sc] != bg:
                    continue
                # BFS
                q = deque([(sr, sc)])
                visited[sr][sc] = True
                cells = []
                touches_border = False
                neighbor_colors = set()
                
                while q:
                    r, c = q.popleft()
                    cells.append((r, c))
                    if r == 0 or r == h-1 or c == 0 or c == w-1:
                        touches_border = True
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if not visited[nr][nc]:
                                if grid[nr][nc] == bg:
                                    visited[nr][nc] = True
                                    q.append((nr, nc))
                                else:
                                    neighbor_colors.add(grid[nr][nc])
                        
                regions.append({
                    'cells': cells,
                    'touches_border': touches_border,
                    'neighbor_colors': neighbor_colors
                })
        return regions
    
    # Strategy 1: Fill enclosed (non-border-touching) regions with their neighbor color
    def apply_enclosed_fill(inp_grid: Grid) -> Optional[Grid]:
        h, w = grid_shape(inp_grid)
        regions = find_enclosed_regions(inp_grid)
        result = [row[:] for row in inp_grid]
        
        for reg in regions:
            if reg['touches_border']:
                continue
            nc = reg['neighbor_colors']
            if len(nc) == 1:
                fill_color = list(nc)[0]
                for r, c in reg['cells']:
                    result[r][c] = fill_color
        
        return result
    
    ok = True
    for inp, out in train_pairs:
        result = apply_enclosed_fill(inp)
        if result is None or not grid_eq(result, out):
            ok = False
            break
    if ok:
        return CrossPiece('flood_enclosed_single_color', apply_enclosed_fill)
    
    # Strategy 2: Fill enclosed regions - use most common neighbor color
    def apply_enclosed_fill_v2(inp_grid: Grid) -> Optional[Grid]:
        h, w = grid_shape(inp_grid)
        regions = find_enclosed_regions(inp_grid)
        result = [row[:] for row in inp_grid]
        
        for reg in regions:
            if reg['touches_border']:
                continue
            if reg['neighbor_colors']:
                # Count neighbor occurrences
                from collections import Counter
                neighbor_counts = Counter()
                for r, c in reg['cells']:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and inp_grid[nr][nc] != 0:
                            neighbor_counts[inp_grid[nr][nc]] += 1
                if neighbor_counts:
                    fill_color = neighbor_counts.most_common(1)[0][0]
                    for r, c in reg['cells']:
                        result[r][c] = fill_color
        
        return result
    
    ok = True
    for inp, out in train_pairs:
        result = apply_enclosed_fill_v2(inp)
        if result is None or not grid_eq(result, out):
            ok = False
            break
    if ok:
        return CrossPiece('flood_enclosed_majority', apply_enclosed_fill_v2)
    
    return None


def _try_row_col_repeat_pattern(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossPiece]:
    """
    Pattern: A small seed pattern (few non-bg rows/cols) is repeated 
    to fill the entire grid. The seed's spacing defines the period.
    """
    # Try row-based repeat
    def apply_row_repeat(inp_grid: Grid) -> Optional[Grid]:
        h, w = grid_shape(inp_grid)
        # Find non-bg rows
        non_bg_rows = []
        for r in range(h):
            if any(inp_grid[r][c] != 0 for c in range(w)):
                non_bg_rows.append(r)
        
        if len(non_bg_rows) < 1 or len(non_bg_rows) > h // 2:
            return None
        
        result = [row[:] for row in inp_grid]
        
        if len(non_bg_rows) == 1:
            # Single row: fill all rows with same pattern
            src_r = non_bg_rows[0]
            for r in range(h):
                result[r] = list(inp_grid[src_r])
            return result
        
        # Multiple non-bg rows: find period
        period = non_bg_rows[1] - non_bg_rows[0]
        if period <= 0:
            return None
        
        # Build seed: the rows from first non-bg row, length = period
        seed_start = non_bg_rows[0]
        seed = []
        for i in range(period):
            r = seed_start + i
            if r < h:
                seed.append(list(inp_grid[r]))
            else:
                seed.append([0] * w)
        
        # Tile seed across entire grid
        for r in range(h):
            phase = (r - seed_start) % period
            if 0 <= phase < len(seed):
                result[r] = list(seed[phase])
        
        return result
    
    ok = True
    for inp, out in train_pairs:
        result = apply_row_repeat(inp)
        if result is None or not grid_eq(result, out):
            ok = False
            break
    if ok:
        return CrossPiece('row_repeat_pattern', apply_row_repeat)
    
    return None


def generate_periodic_fill_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """Generate periodic fill solver pieces."""
    pieces = []
    
    # Only try on tasks with few non-bg pixels or simple structure
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    oh, ow = grid_shape(out0)
    
    # Same size only
    if h != oh or w != ow:
        return pieces
    
    for solver in [
        _try_periodic_axis_fill,
        _try_periodic_col_fill,
        _try_periodic_row_col_fill,
        _try_ray_from_dots,
        _try_flood_enclosed_regions,
        _try_row_col_repeat_pattern,
    ]:
        try:
            piece = solver(train_pairs)
            if piece is not None:
                pieces.append(piece)
        except Exception:
            pass
    
    return pieces
