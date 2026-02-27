"""
arc/extend_to_divider_solver.py — Extend objects to divider lines

Pattern: Objects near a divider line extend toward it, filling
the space between the object and the divider with the object's color(s).

Examples:
- 13713586: colored shapes extend rightward to vertical divider
- 05a7bcf2: colored shapes extend toward a horizontal/vertical divider
"""

from __future__ import annotations
from typing import List, Tuple, Optional
from arc.grid import Grid, grid_shape, grid_eq
from arc.cross_engine import CrossPiece
from collections import deque


def _find_full_dividers(grid: Grid):
    """Find full-span horizontal and vertical divider lines."""
    h, w = grid_shape(grid)
    
    h_divs = []
    for r in range(h):
        vals = set(grid[r][c] for c in range(w))
        if len(vals) == 1 and list(vals)[0] != 0:
            h_divs.append((r, list(vals)[0]))
    
    v_divs = []
    for c in range(w):
        vals = set(grid[r][c] for r in range(h))
        if len(vals) == 1 and list(vals)[0] != 0:
            v_divs.append((c, list(vals)[0]))
    
    return h_divs, v_divs


def _find_objects_bfs(grid: Grid, bg: int = 0, exclude_colors=set()):
    """Find connected components excluding bg and specified colors."""
    h, w = grid_shape(grid)
    visited = [[False]*w for _ in range(h)]
    objects = []
    
    for sr in range(h):
        for sc in range(w):
            if visited[sr][sc] or grid[sr][sc] == bg or grid[sr][sc] in exclude_colors:
                continue
            q = deque([(sr, sc)])
            visited[sr][sc] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c, grid[r][c]))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc]:
                        if grid[nr][nc] != bg and grid[nr][nc] not in exclude_colors:
                            visited[nr][nc] = True
                            q.append((nr, nc))
            
            rows = [r for r, c, _ in cells]
            cols = [c for r, c, _ in cells]
            objects.append({
                'cells': cells,
                'bbox': (min(rows), min(cols), max(rows), max(cols)),
                'colors': set(v for _, _, v in cells),
            })
    
    return objects


def _try_extend_to_divider(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossPiece]:
    """
    Try extending each object toward the nearest divider in each of 4 directions.
    """
    directions = ['right', 'left', 'down', 'up']
    
    for direction in directions:
        def make_apply(d):
            def apply_fn(inp_grid):
                h, w = grid_shape(inp_grid)
                h_divs, v_divs = _find_full_dividers(inp_grid)
                
                if not h_divs and not v_divs:
                    return None
                
                div_colors = set()
                for _, color in h_divs:
                    div_colors.add(color)
                for _, color in v_divs:
                    div_colors.add(color)
                
                objects = _find_objects_bfs(inp_grid, 0, div_colors)
                if not objects:
                    return None
                
                result = [row[:] for row in inp_grid]
                
                for obj in objects:
                    r1, c1, r2, c2 = obj['bbox']
                    
                    if d == 'right' and v_divs:
                        # Find nearest v_div to the right
                        targets = [vc for vc, _ in v_divs if vc > c2]
                        if not targets:
                            targets = [w]  # extend to grid edge
                        target_c = min(targets)
                        
                        # Extend each row of the object rightward
                        for r, c, color in obj['cells']:
                            for nc in range(c + 1, target_c):
                                if result[r][nc] == 0:
                                    result[r][nc] = color
                    
                    elif d == 'left' and v_divs:
                        targets = [vc for vc, _ in v_divs if vc < c1]
                        if not targets:
                            targets = [-1]
                        target_c = max(targets)
                        
                        for r, c, color in obj['cells']:
                            for nc in range(target_c + 1, c):
                                if result[r][nc] == 0:
                                    result[r][nc] = color
                    
                    elif d == 'down' and h_divs:
                        targets = [hr for hr, _ in h_divs if hr > r2]
                        if not targets:
                            targets = [h]
                        target_r = min(targets)
                        
                        for r, c, color in obj['cells']:
                            for nr in range(r + 1, target_r):
                                if result[nr][c] == 0:
                                    result[nr][c] = color
                    
                    elif d == 'up' and h_divs:
                        targets = [hr for hr, _ in h_divs if hr < r1]
                        if not targets:
                            targets = [-1]
                        target_r = max(targets)
                        
                        for r, c, color in obj['cells']:
                            for nr in range(target_r + 1, r):
                                if result[nr][c] == 0:
                                    result[nr][c] = color
                
                return result
            return apply_fn
        
        apply_fn = make_apply(direction)
        ok = True
        for inp, out in train_pairs:
            result = apply_fn(inp)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        
        if ok:
            return CrossPiece(f'extend_to_div_{direction}', apply_fn)
    
    # Try: extend in ALL directions simultaneously (toward nearest divider)
    def apply_all_dirs(inp_grid):
        h, w = grid_shape(inp_grid)
        h_divs, v_divs = _find_full_dividers(inp_grid)
        
        if not h_divs and not v_divs:
            return None
        
        div_colors = set()
        h_div_rows = set()
        v_div_cols = set()
        for r, color in h_divs:
            div_colors.add(color)
            h_div_rows.add(r)
        for c, color in v_divs:
            div_colors.add(color)
            v_div_cols.add(c)
        
        objects = _find_objects_bfs(inp_grid, 0, div_colors)
        if not objects:
            return None
        
        result = [row[:] for row in inp_grid]
        
        for obj in objects:
            r1, c1, r2, c2 = obj['bbox']
            
            # Extend right to nearest v_div
            targets_r = [vc for vc in v_div_cols if vc > c2]
            target_r = min(targets_r) if targets_r else w
            
            # Extend left to nearest v_div
            targets_l = [vc for vc in v_div_cols if vc < c1]
            target_l = max(targets_l) if targets_l else -1
            
            # Extend down to nearest h_div
            targets_d = [hr for hr in h_div_rows if hr > r2]
            target_d = min(targets_d) if targets_d else h
            
            # Extend up to nearest h_div
            targets_u = [hr for hr in h_div_rows if hr < r1]
            target_u = max(targets_u) if targets_u else -1
            
            for r, c, color in obj['cells']:
                # Right
                for nc in range(c + 1, target_r):
                    if result[r][nc] == 0:
                        result[r][nc] = color
                # Left
                for nc in range(target_l + 1, c):
                    if result[r][nc] == 0:
                        result[r][nc] = color
                # Down
                for nr in range(r + 1, target_d):
                    if result[nr][c] == 0:
                        result[nr][c] = color
                # Up
                for nr in range(target_u + 1, r):
                    if result[nr][c] == 0:
                        result[nr][c] = color
        
        return result
    
    ok = True
    for inp, out in train_pairs:
        result = apply_all_dirs(inp)
        if result is None or not grid_eq(result, out):
            ok = False
            break
    if ok:
        return CrossPiece('extend_to_div_all', apply_all_dirs)
    
    # Try: extend bbox (not individual cells) to divider
    for direction in directions:
        def make_bbox_apply(d):
            def apply_fn(inp_grid):
                h, w = grid_shape(inp_grid)
                h_divs, v_divs = _find_full_dividers(inp_grid)
                
                if not h_divs and not v_divs:
                    return None
                
                div_colors = set()
                for _, color in h_divs:
                    div_colors.add(color)
                for _, color in v_divs:
                    div_colors.add(color)
                
                objects = _find_objects_bfs(inp_grid, 0, div_colors)
                if not objects:
                    return None
                
                result = [row[:] for row in inp_grid]
                
                for obj in objects:
                    r1, c1, r2, c2 = obj['bbox']
                    # Get the dominant color
                    from collections import Counter
                    color_counts = Counter(v for _, _, v in obj['cells'])
                    main_color = color_counts.most_common(1)[0][0]
                    
                    if d == 'right' and v_divs:
                        targets = [vc for vc, _ in v_divs if vc > c2]
                        target_c = min(targets) if targets else w
                        for r in range(r1, r2 + 1):
                            for c in range(c2 + 1, target_c):
                                if result[r][c] == 0:
                                    result[r][c] = main_color
                    elif d == 'left' and v_divs:
                        targets = [vc for vc, _ in v_divs if vc < c1]
                        target_c = max(targets) + 1 if targets else 0
                        for r in range(r1, r2 + 1):
                            for c in range(target_c, c1):
                                if result[r][c] == 0:
                                    result[r][c] = main_color
                    elif d == 'down' and h_divs:
                        targets = [hr for hr, _ in h_divs if hr > r2]
                        target_r = min(targets) if targets else h
                        for r in range(r2 + 1, target_r):
                            for c in range(c1, c2 + 1):
                                if result[r][c] == 0:
                                    result[r][c] = main_color
                    elif d == 'up' and h_divs:
                        targets = [hr for hr, _ in h_divs if hr < r1]
                        target_r = max(targets) + 1 if targets else 0
                        for r in range(target_r, r1):
                            for c in range(c1, c2 + 1):
                                if result[r][c] == 0:
                                    result[r][c] = main_color
                
                return result
            return apply_fn
        
        apply_fn = make_bbox_apply(direction)
        ok = True
        for inp, out in train_pairs:
            result = apply_fn(inp)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        if ok:
            return CrossPiece(f'extend_bbox_to_div_{direction}', apply_fn)
    
    return None


def generate_extend_to_divider_pieces(train_pairs) -> List[CrossPiece]:
    """Generate extend-to-divider solver pieces."""
    pieces = []
    
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    oh, ow = grid_shape(out0)
    
    if h != oh or w != ow:
        return pieces
    
    # Quick check: does input have dividers?
    h_divs, v_divs = _find_full_dividers(inp0)
    if not h_divs and not v_divs:
        return pieces
    
    try:
        piece = _try_extend_to_divider(train_pairs)
        if piece is not None:
            pieces.append(piece)
    except Exception:
        pass
    
    return pieces
