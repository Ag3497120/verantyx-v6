"""
arc/cross_probe_fill.py — Cross Probe Fill Engine

Uses cross-structure probing to solve:
1. Line fill between same-column/row dots (fill gaps with color propagation)
2. Rectangular region expansion from corner/edge dots
3. Cross-directional expansion from center objects
4. Object merge via probe intersection

Core idea: inject measurement probes (cross structures) into the grid,
then determine fill rules from how probes connect, intersect, or get blocked.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Set, Dict
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color


def _try_column_line_fill(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[callable]:
    """
    Pattern: dots in same column → fill the entire column between extremes.
    Each dot's color fills upward/downward until the next dot's color takes over.
    """
    for fill_mode in ['top_to_bottom', 'bottom_to_top', 'nearest']:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            bg = most_common_color(inp)
            result = _apply_column_fill(inp, h, w, bg, fill_mode)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        if ok:
            def fn(inp, _mode=fill_mode):
                h, w = grid_shape(inp)
                bg = most_common_color(inp)
                return _apply_column_fill(inp, h, w, bg, _mode)
            return fn
    
    # Try row-based fill
    for fill_mode in ['left_to_right', 'right_to_left', 'nearest']:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            bg = most_common_color(inp)
            result = _apply_row_fill(inp, h, w, bg, fill_mode)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        if ok:
            def fn(inp, _mode=fill_mode):
                h, w = grid_shape(inp)
                bg = most_common_color(inp)
                return _apply_row_fill(inp, h, w, bg, _mode)
            return fn
    
    return None


def _apply_column_fill(inp: Grid, h: int, w: int, bg: int, mode: str) -> Optional[Grid]:
    """Fill columns: dots propagate color vertically."""
    result = [list(row) for row in inp]
    
    for c in range(w):
        # Find non-bg cells in this column
        dots = [(r, inp[r][c]) for r in range(h) if inp[r][c] != bg]
        if len(dots) < 2:
            continue
        
        if mode == 'top_to_bottom':
            # Fill from top: first dot's color fills down to next dot
            for r in range(h):
                if result[r][c] == bg:
                    # Find nearest dot above and below
                    above = [(dr, dc) for dr, dc in dots if dr < r]
                    below = [(dr, dc) for dr, dc in dots if dr > r]
                    if above:
                        result[r][c] = above[-1][1]  # nearest above
                    elif below:
                        result[r][c] = below[0][1]   # nearest below
        
        elif mode == 'bottom_to_top':
            for r in range(h - 1, -1, -1):
                if result[r][c] == bg:
                    below = [(dr, dc) for dr, dc in dots if dr > r]
                    above = [(dr, dc) for dr, dc in dots if dr < r]
                    if below:
                        result[r][c] = below[0][1]
                    elif above:
                        result[r][c] = above[-1][1]
        
        elif mode == 'nearest':
            for r in range(h):
                if result[r][c] == bg:
                    # Find nearest dot
                    nearest = min(dots, key=lambda d: abs(d[0] - r))
                    result[r][c] = nearest[1]
    
    return result


def _apply_row_fill(inp: Grid, h: int, w: int, bg: int, mode: str) -> Optional[Grid]:
    """Fill rows: dots propagate color horizontally."""
    result = [list(row) for row in inp]
    
    for r in range(h):
        dots = [(c, inp[r][c]) for c in range(w) if inp[r][c] != bg]
        if len(dots) < 2:
            continue
        
        if mode == 'left_to_right':
            for c in range(w):
                if result[r][c] == bg:
                    left = [(dc, dv) for dc, dv in dots if dc < c]
                    right = [(dc, dv) for dc, dv in dots if dc > c]
                    if left:
                        result[r][c] = left[-1][1]
                    elif right:
                        result[r][c] = right[0][1]
        
        elif mode == 'right_to_left':
            for c in range(w - 1, -1, -1):
                if result[r][c] == bg:
                    right = [(dc, dv) for dc, dv in dots if dc > c]
                    left = [(dc, dv) for dc, dv in dots if dc < c]
                    if right:
                        result[r][c] = right[0][1]
                    elif left:
                        result[r][c] = left[-1][1]
        
        elif mode == 'nearest':
            for c in range(w):
                if result[r][c] == bg:
                    nearest = min(dots, key=lambda d: abs(d[0] - c))
                    result[r][c] = nearest[1]
    
    return result


def _try_rect_expansion(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[callable]:
    """
    Pattern: dots at edges/corners define rectangular regions.
    Each dot expands into a rectangle toward a wall or another dot.
    """
    ok = True
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        bg = most_common_color(inp)
        result = _apply_rect_expansion(inp, h, w, bg)
        if result is None or not grid_eq(result, out):
            ok = False
            break
    
    if ok:
        def fn(inp):
            h, w = grid_shape(inp)
            bg = most_common_color(inp)
            return _apply_rect_expansion(inp, h, w, bg)
        return fn
    return None


def _apply_rect_expansion(inp: Grid, h: int, w: int, bg: int) -> Optional[Grid]:
    """Expand dots into rectangles based on their position relative to grid edges."""
    result = [[bg] * w for _ in range(h)]
    
    # Find all non-bg cells and their colors
    dots = []
    for r in range(h):
        for c in range(w):
            if inp[r][c] != bg:
                dots.append((r, c, inp[r][c]))
    
    if not dots:
        return None
    
    # Group dots by color
    color_dots: Dict[int, List[Tuple[int, int]]] = {}
    for r, c, color in dots:
        color_dots.setdefault(color, []).append((r, c))
    
    # For each color, find bounding region and fill
    for color, positions in color_dots.items():
        if len(positions) == 1:
            r, c = positions[0]
            # Single dot: expand to nearest edge
            # Try: fill from dot to nearest corner
            result[r][c] = color
        else:
            # Multiple dots: fill rectangle between them
            rs = [r for r, c in positions]
            cs = [c for r, c in positions]
            for r in range(min(rs), max(rs) + 1):
                for c in range(min(cs), max(cs) + 1):
                    result[r][c] = color
    
    return result


def _try_cross_expansion(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[callable]:
    """
    Pattern: center object expands cross-shaped probes in 4 directions.
    Inner color becomes outer, outer becomes inner (color swap).
    """
    ok = True
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        bg = most_common_color(inp)
        result = _apply_cross_expansion(inp, h, w, bg)
        if result is None or not grid_eq(result, out):
            ok = False
            break
    
    if ok:
        def fn(inp):
            h, w = grid_shape(inp)
            bg = most_common_color(inp)
            return _apply_cross_expansion(inp, h, w, bg)
        return fn
    return None


def _apply_cross_expansion(inp: Grid, h: int, w: int, bg: int) -> Optional[Grid]:
    """Center object shoots cross probes in 4 directions, colors swap."""
    from collections import deque
    
    # Find connected components
    visited = set()
    objects = []
    for r in range(h):
        for c in range(w):
            if inp[r][c] != bg and (r, c) not in visited:
                comp = set()
                q = deque([(r, c)])
                while q:
                    cr, cc = q.popleft()
                    if (cr, cc) in comp or cr < 0 or cr >= h or cc < 0 or cc >= w:
                        continue
                    if inp[cr][cc] == bg:
                        continue
                    comp.add((cr, cc))
                    visited.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        q.append((cr + dr, cc + dc))
                colors = set(inp[r][c] for r, c in comp)
                objects.append({'cells': comp, 'colors': colors})
    
    if len(objects) != 1:
        return None
    
    obj = objects[0]
    if len(obj['colors']) != 2:
        return None
    
    # Find outer and inner colors
    color_counts = Counter(inp[r][c] for r, c in obj['cells'])
    outer_color = color_counts.most_common()[0][0]
    inner_color = color_counts.most_common()[1][0]
    
    # Get bounding box
    rs = [r for r, c in obj['cells']]
    cs = [c for r, c in obj['cells']]
    r1, r2 = min(rs), max(rs)
    c1, c2 = min(cs), max(cs)
    obj_h = r2 - r1 + 1
    obj_w = c2 - c1 + 1
    
    result = [list(row) for row in inp]
    
    # Swap colors in original object
    for r, c in obj['cells']:
        if inp[r][c] == outer_color:
            result[r][c] = inner_color
        else:
            result[r][c] = outer_color
    
    # Extend outer_color in 4 directions from the object bbox
    # Up
    for r in range(r1 - 1, -1, -1):
        for c in range(c1, c2 + 1):
            result[r][c] = outer_color
    # Down
    for r in range(r2 + 1, h):
        for c in range(c1, c2 + 1):
            result[r][c] = outer_color
    # Left
    for c in range(c1 - 1, -1, -1):
        for r in range(r1, r2 + 1):
            result[r][c] = outer_color
    # Right
    for c in range(c2 + 1, w):
        for r in range(r1, r2 + 1):
            result[r][c] = outer_color
    
    return result


def _try_dot_line_connect(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[callable]:
    """
    Pattern: pairs of same-color dots → draw line between them.
    The line follows row then column (or vice versa).
    """
    for line_mode in ['h_then_v', 'v_then_h', 'direct_h', 'direct_v']:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            bg = most_common_color(inp)
            result = _apply_dot_line_connect(inp, h, w, bg, line_mode)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        if ok:
            def fn(inp, _mode=line_mode):
                h, w = grid_shape(inp)
                bg = most_common_color(inp)
                return _apply_dot_line_connect(inp, h, w, bg, _mode)
            return fn
    return None


def _apply_dot_line_connect(inp: Grid, h: int, w: int, bg: int, mode: str) -> Optional[Grid]:
    """Connect same-color dot pairs with lines."""
    result = [list(row) for row in inp]
    
    # Find dots by color
    color_dots: Dict[int, List[Tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if inp[r][c] != bg:
                color_dots.setdefault(inp[r][c], []).append((r, c))
    
    for color, dots in color_dots.items():
        if len(dots) < 2:
            continue
        
        # Connect pairs (for 2 dots: connect them; for more: connect sequentially)
        for i in range(len(dots)):
            for j in range(i + 1, len(dots)):
                r1, c1 = dots[i]
                r2, c2 = dots[j]
                
                if mode == 'direct_h' and r1 == r2:
                    for c in range(min(c1, c2), max(c1, c2) + 1):
                        result[r1][c] = color
                elif mode == 'direct_v' and c1 == c2:
                    for r in range(min(r1, r2), max(r1, r2) + 1):
                        result[r][c1] = color
                elif mode == 'h_then_v':
                    # Horizontal from (r1,c1) to (r1,c2), then vertical to (r2,c2)
                    for c in range(min(c1, c2), max(c1, c2) + 1):
                        result[r1][c] = color
                    for r in range(min(r1, r2), max(r1, r2) + 1):
                        result[r][c2] = color
                elif mode == 'v_then_h':
                    for r in range(min(r1, r2), max(r1, r2) + 1):
                        result[r][c1] = color
                    for c in range(min(c1, c2), max(c1, c2) + 1):
                        result[r2][c] = color
    
    return result


def _try_probe_merge(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[callable]:
    """
    Cross probe merge: objects of same color merge by filling the space
    between them (rectangular hull or axis-aligned connection).
    """
    for merge_mode in ['rect_hull', 'axis_fill']:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            bg = most_common_color(inp)
            result = _apply_probe_merge(inp, h, w, bg, merge_mode)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        if ok:
            def fn(inp, _mode=merge_mode):
                h, w = grid_shape(inp)
                bg = most_common_color(inp)
                return _apply_probe_merge(inp, h, w, bg, _mode)
            return fn
    return None


def _apply_probe_merge(inp: Grid, h: int, w: int, bg: int, mode: str) -> Optional[Grid]:
    """Merge same-color objects by filling between them."""
    result = [list(row) for row in inp]
    
    color_cells: Dict[int, List[Tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if inp[r][c] != bg:
                color_cells.setdefault(inp[r][c], []).append((r, c))
    
    for color, cells in color_cells.items():
        if len(cells) < 2:
            continue
        
        if mode == 'rect_hull':
            # Fill bounding rectangle of all cells of this color
            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            for r in range(min(rs), max(rs) + 1):
                for c in range(min(cs), max(cs) + 1):
                    result[r][c] = color
        
        elif mode == 'axis_fill':
            # Fill along rows and columns that contain cells
            rows_with = set(r for r, c in cells)
            cols_with = set(c for r, c in cells)
            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            rmin, rmax = min(rs), max(rs)
            cmin, cmax = min(cs), max(cs)
            for r in rows_with:
                for c in range(cmin, cmax + 1):
                    result[r][c] = color
            for c in cols_with:
                for r in range(rmin, rmax + 1):
                    result[r][c] = color
    
    return result


def _try_dot_to_line(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[callable]:
    """
    Pattern: each non-bg cell extends into a full row or full column.
    Direction (row vs col) may be per-dot based on position, or uniform.
    """
    # Try uniform row fill
    for mode in ['row', 'col', 'edge_detect']:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            bg = most_common_color(inp)
            result = _apply_dot_to_line(inp, h, w, bg, mode)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        if ok:
            def fn(inp, _mode=mode):
                h, w = grid_shape(inp)
                bg = most_common_color(inp)
                return _apply_dot_to_line(inp, h, w, bg, _mode)
            return fn
    return None


def _apply_dot_to_line(inp: Grid, h: int, w: int, bg: int, mode: str) -> Optional[Grid]:
    """Extend dots into full rows or columns."""
    result = [[bg] * w for _ in range(h)]
    
    dots = [(r, c, inp[r][c]) for r in range(h) for c in range(w) if inp[r][c] != bg]
    if not dots:
        return None
    
    for r, c, color in dots:
        if mode == 'row':
            for cc in range(w):
                result[r][cc] = color
        elif mode == 'col':
            for rr in range(h):
                result[rr][c] = color
        elif mode == 'edge_detect':
            # If dot is on left/right edge → row fill; top/bottom edge → col fill
            on_left = (c == 0)
            on_right = (c == w - 1)
            on_top = (r == 0)
            on_bottom = (r == h - 1)
            
            if on_left or on_right:
                for cc in range(w):
                    result[r][cc] = color
            elif on_top or on_bottom:
                for rr in range(h):
                    result[rr][c] = color
            else:
                # Interior dot — try row by default
                for cc in range(w):
                    result[r][cc] = color
    
    return result


def _try_dot_to_cross_rect(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[callable]:
    """
    Pattern: dots define rectangles in a grid.
    Each dot's color owns a region. Within that region:
    - Dot row = full horizontal line
    - Region start/end rows = full horizontal line  
    - Other rows = left/right border only
    Regions are split at midpoints between consecutive dots.
    """
    ok = True
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        bg = most_common_color(inp)
        result = _apply_dot_to_cross_rect(inp, h, w, bg)
        if result is None or not grid_eq(result, out):
            ok = False
            break
    
    if ok:
        def fn(inp):
            h, w = grid_shape(inp)
            bg = most_common_color(inp)
            return _apply_dot_to_cross_rect(inp, h, w, bg)
        return fn
    return None


def _apply_dot_to_cross_rect(inp: Grid, h: int, w: int, bg: int) -> Optional[Grid]:
    """Dots → rectangular regions with cross borders."""
    result = [[bg] * w for _ in range(h)]
    
    dots = sorted(
        [(r, c, inp[r][c]) for r in range(h) for c in range(w) if inp[r][c] != bg],
        key=lambda d: d[0]
    )
    if not dots:
        return None
    
    # Compute regions: split at midpoints between consecutive dots
    regions = []
    for i, (r, c, color) in enumerate(dots):
        if i == 0:
            r_start = 0
        else:
            r_start = (dots[i - 1][0] + r) // 2 + 1
        if i == len(dots) - 1:
            r_end = h - 1
        else:
            r_end = (r + dots[i + 1][0]) // 2
        regions.append((color, r_start, r_end, r))
    
    for color, r_start, r_end, dot_row in regions:
        # Find which rows should be full lines:
        # - The dot row itself
        # - The region boundary that touches another color or grid edge
        # Determine full rows: dot_row + the farthest row from dot
        # The other full row is at grid edge (row 0 or row h-1)
        # Only if this region touches the grid edge
        other_full = None
        if r_start == 0:
            other_full = 0
        elif r_end == h - 1:
            other_full = h - 1
        
        for r in range(r_start, r_end + 1):
            if r == dot_row or (other_full is not None and r == other_full):
                for c in range(w):
                    result[r][c] = color
            else:
                result[r][0] = color
                result[r][w - 1] = color
    
    return result


# ==================== Main Entry ====================

def generate_cross_probe_pieces(train_pairs: List[Tuple[Grid, Grid]]):
    """Generate CrossPiece candidates from cross probe fill strategies."""
    from arc.cross_engine import CrossPiece
    pieces = []
    
    strategies = [
        ('cross_probe:column_fill', _try_column_line_fill),
        ('cross_probe:rect_expansion', _try_rect_expansion),
        ('cross_probe:cross_expansion', _try_cross_expansion),
        ('cross_probe:dot_line_connect', _try_dot_line_connect),
        ('cross_probe:probe_merge', _try_probe_merge),
        ('cross_probe:dot_to_line', _try_dot_to_line),
        ('cross_probe:dot_to_cross_rect', _try_dot_to_cross_rect),
    ]
    
    for name, try_fn in strategies:
        try:
            fn = try_fn(train_pairs)
            if fn is not None:
                pieces.append(CrossPiece(name, fn))
        except Exception:
            pass
    
    return pieces
