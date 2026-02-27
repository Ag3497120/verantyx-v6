"""
arc/concentric_fill_solver.py — Concentric/spiral rectangle fill solver

Pattern: A rectangular border in input → output fills interior with
concentric rectangles of alternating colors.

Examples:
- b7fb29bc: 3-bordered rect → spiral fill with colors 3,4,2 alternating
- 516b51b7: 1-bordered rect → concentric rings filling inward 1,2,3,...
"""

from __future__ import annotations
from typing import List, Tuple, Optional
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq
from arc.cross_engine import CrossPiece


def _find_rectangles(grid: Grid, bg: int = 0):
    """Find rectangular borders (single-color outlines) in grid."""
    h, w = grid_shape(grid)
    rects = []
    visited = set()
    
    for sr in range(h):
        for sc in range(w):
            if grid[sr][sc] == bg or (sr, sc) in visited:
                continue
            color = grid[sr][sc]
            
            # Try to find a rectangle starting at (sr, sc)
            # Look for top-left corner of a rect border
            # Scan right for top edge
            tr = sr
            tc_end = sc
            while tc_end + 1 < w and grid[tr][tc_end + 1] == color:
                tc_end += 1
            
            if tc_end == sc:
                continue  # single pixel, not a rect
            
            # Scan down for right edge
            br = tr
            while br + 1 < h and grid[br + 1][tc_end] == color:
                br += 1
            
            if br == tr:
                continue
            
            # Verify it's a complete rectangle border
            rect_w = tc_end - sc + 1
            rect_h = br - tr + 1
            
            if rect_w < 3 or rect_h < 3:
                continue
            
            # Check top edge
            top_ok = all(grid[tr][c] == color for c in range(sc, tc_end + 1))
            # Check bottom edge
            bot_ok = all(grid[br][c] == color for c in range(sc, tc_end + 1))
            # Check left edge
            left_ok = all(grid[r][sc] == color for r in range(tr, br + 1))
            # Check right edge
            right_ok = all(grid[r][tc_end] == color for r in range(tr, br + 1))
            
            if top_ok and bot_ok and left_ok and right_ok:
                # Check interior is mostly bg
                interior_bg = 0
                interior_total = 0
                for r in range(tr + 1, br):
                    for c in range(sc + 1, tc_end):
                        interior_total += 1
                        if grid[r][c] == bg:
                            interior_bg += 1
                
                if interior_total > 0 and interior_bg >= interior_total * 0.5:
                    rects.append({
                        'r1': tr, 'c1': sc, 'r2': br, 'c2': tc_end,
                        'color': color,
                        'h': rect_h, 'w': rect_w
                    })
                    # Mark visited
                    for r in range(tr, br + 1):
                        for c in range(sc, tc_end + 1):
                            visited.add((r, c))
    
    return rects


def _try_concentric_fill(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossPiece]:
    """
    Pattern: rect border in input → concentric rings filling inward.
    Learn the color sequence from the output.
    """
    
    # Analyze first train pair to learn the pattern
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    bg = 0
    
    rects = _find_rectangles(inp0, bg)
    if not rects:
        return None
    
    # For each rect, check if output has concentric fill
    for rect in rects:
        r1, c1, r2, c2 = rect['r1'], rect['c1'], rect['r2'], rect['c2']
        border_color = rect['color']
        
        # Read the concentric ring colors from output
        ring_colors = [border_color]
        r, c = r1, c1
        depth = 1
        while r1 + depth < r2 - depth + 1 and c1 + depth < c2 - depth + 1:
            # Sample a cell in the next ring
            ring_r = r1 + depth
            ring_c = c1 + depth
            if ring_r <= r2 - depth and ring_c <= c2 - depth:
                color = out0[ring_r][ring_c]
                ring_colors.append(color)
            depth += 1
        
        if len(ring_colors) < 2:
            continue
        
        # Verify: apply concentric fill and check
        def make_fill(ring_cols):
            def fill_rect(grid, r1, c1, r2, c2, bg=0):
                result = [row[:] for row in grid]
                depth = 0
                while r1 + depth <= r2 - depth and c1 + depth <= c2 - depth:
                    color = ring_cols[depth % len(ring_cols)] if depth < len(ring_cols) else ring_cols[-1]
                    # Fill this ring
                    for c in range(c1 + depth, c2 - depth + 1):
                        result[r1 + depth][c] = color
                        result[r2 - depth][c] = color
                    for r in range(r1 + depth, r2 - depth + 1):
                        result[r][c1 + depth] = color
                        result[r][c2 - depth] = color
                    depth += 1
                return result
            return fill_rect
        
        fill_fn = make_fill(ring_colors)
        test_result = fill_fn(inp0, r1, c1, r2, c2)
        
        if not grid_eq(test_result, out0):
            continue
        
        # Learn: apply to all rects in any grid
        def make_apply(ring_cols):
            def apply_fn(inp_grid):
                h, w = grid_shape(inp_grid)
                bg = 0
                rects = _find_rectangles(inp_grid, bg)
                if not rects:
                    return None
                
                result = [row[:] for row in inp_grid]
                for rect in rects:
                    r1, c1, r2, c2 = rect['r1'], rect['c1'], rect['r2'], rect['c2']
                    depth = 0
                    while r1 + depth <= r2 - depth and c1 + depth <= c2 - depth:
                        color = ring_cols[depth % len(ring_cols)] if depth < len(ring_cols) else ring_cols[-1]
                        for c in range(c1 + depth, c2 - depth + 1):
                            result[r1 + depth][c] = color
                            result[r2 - depth][c] = color
                        for r in range(r1 + depth, r2 - depth + 1):
                            result[r][c1 + depth] = color
                            result[r][c2 - depth] = color
                        depth += 1
                return result
            return apply_fn
        
        apply_fn = make_apply(ring_colors)
        
        # Verify on all train pairs
        ok = True
        for inp, out in train_pairs:
            result = apply_fn(inp)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        
        if ok:
            return CrossPiece('concentric_fill', apply_fn)
    
    return None


def _try_distance_fill(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossPiece]:
    """
    Pattern: Fill cells based on their distance from the nearest non-bg cell.
    Each distance maps to a specific color.
    """
    from collections import deque
    
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    bg = 0
    
    # BFS from all non-bg cells to compute distance
    def compute_distances(grid, bg=0):
        h, w = grid_shape(grid)
        dist = [[-1]*w for _ in range(h)]
        q = deque()
        
        for r in range(h):
            for c in range(w):
                if grid[r][c] != bg:
                    dist[r][c] = 0
                    q.append((r, c))
        
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and dist[nr][nc] == -1:
                    dist[nr][nc] = dist[r][c] + 1
                    q.append((nr, nc))
        
        return dist
    
    # Learn distance → color mapping from first pair
    distances = compute_distances(inp0, bg)
    dist_color_map = {}
    consistent = True
    
    for r in range(h):
        for c in range(w):
            d = distances[r][c]
            oc = out0[r][c]
            if d in dist_color_map:
                if dist_color_map[d] != oc:
                    consistent = False
                    break
            else:
                dist_color_map[d] = oc
        if not consistent:
            break
    
    if not consistent:
        return None
    
    def apply_fn(inp_grid):
        h, w = grid_shape(inp_grid)
        distances = compute_distances(inp_grid, bg)
        result = [row[:] for row in inp_grid]
        
        for r in range(h):
            for c in range(w):
                d = distances[r][c]
                if d in dist_color_map:
                    result[r][c] = dist_color_map[d]
                elif d >= 0:
                    # Use max known distance color
                    max_d = max(dist_color_map.keys())
                    result[r][c] = dist_color_map[max_d]
        
        return result
    
    ok = True
    for inp, out in train_pairs:
        result = apply_fn(inp)
        if result is None or not grid_eq(result, out):
            ok = False
            break
    
    if ok:
        return CrossPiece('distance_fill', apply_fn)
    
    return None


def _try_border_distance_fill(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossPiece]:
    """
    Pattern: Fill based on distance from the grid border or from specific objects.
    516b51b7: distance from the outer boundary of the 1-colored rectangle.
    """
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    
    # For each cell, compute min distance to border of grid
    def border_dist(r, c, h, w):
        return min(r, c, h-1-r, w-1-c)
    
    # Check if output color = f(border_distance) where non-bg cells define border
    # Actually: check if output color = f(distance to nearest non-bg cell)
    # This is handled by _try_distance_fill above
    
    # Try: distance from edge of the grid itself
    dist_color_map = {}
    consistent = True
    for r in range(h):
        for c in range(w):
            d = border_dist(r, c, h, w)
            oc = out0[r][c]
            if d in dist_color_map:
                if dist_color_map[d] != oc:
                    consistent = False
                    break
            else:
                dist_color_map[d] = oc
        if not consistent:
            break
    
    if not consistent:
        return None
    
    def apply_fn(inp_grid):
        h, w = grid_shape(inp_grid)
        result = [row[:] for row in inp_grid]
        for r in range(h):
            for c in range(w):
                d = border_dist(r, c, h, w)
                if d in dist_color_map:
                    result[r][c] = dist_color_map[d]
        return result
    
    ok = True
    for inp, out in train_pairs:
        result = apply_fn(inp)
        if result is None or not grid_eq(result, out):
            ok = False
            break
    
    if ok:
        return CrossPiece('border_distance_fill', apply_fn)
    
    return None


def generate_concentric_fill_pieces(train_pairs) -> List[CrossPiece]:
    """Generate concentric/distance fill solver pieces."""
    pieces = []
    
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    oh, ow = grid_shape(out0)
    
    if h != oh or w != ow:
        return pieces
    
    for solver in [
        _try_concentric_fill,
        _try_distance_fill,
        _try_border_distance_fill,
    ]:
        try:
            piece = solver(train_pairs)
            if piece is not None:
                pieces.append(piece)
        except Exception:
            pass
    
    return pieces
