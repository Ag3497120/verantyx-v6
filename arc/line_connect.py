"""
arc/line_connect.py — Line drawing and connection transforms for ARC-AGI-2

Handles:
1. L-shape connection between two points
2. Cross projection from dot positions (horizontal + vertical lines)
3. Line extension to border
4. Connect same-color objects with lines
"""

from typing import List, Tuple, Optional, Dict, Set
from arc.grid import Grid, grid_shape, grid_eq, most_common_color, grid_colors
from arc.objects import detect_objects


def _find_non_bg_cells(grid: Grid, bg: int) -> List[Tuple[int, int, int]]:
    """Find all non-background cells: (row, col, color)."""
    h, w = grid_shape(grid)
    cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                cells.append((r, c, grid[r][c]))
    return cells


def _diff_cells(inp: Grid, out: Grid) -> List[Tuple[int, int, int, int]]:
    """Find cells that changed: (row, col, old_val, new_val)."""
    h, w = grid_shape(inp)
    changes = []
    for r in range(h):
        for c in range(w):
            if inp[r][c] != out[r][c]:
                changes.append((r, c, inp[r][c], out[r][c]))
    return changes


# === L-shape Connection ===

def learn_l_connect(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn L-shaped line connection between two non-bg cells.
    
    Pattern: Two isolated dots connected by an L-shaped path
    using a specific color (often a new color or the dots' color).
    """
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    
    for pair_idx, (inp, out) in enumerate(train_pairs):
        changes = _diff_cells(inp, out)
        if not changes:
            return None
        
        # All changes should be bg -> some color
        if not all(old == bg for _, _, old, _ in changes):
            return None
    
    # Analyze first pair to learn the rule
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    changes0 = _diff_cells(inp0, out0)
    
    # Find the line color
    line_colors = set(new for _, _, _, new in changes0)
    if len(line_colors) != 1:
        return None
    line_color = line_colors.pop()
    
    # The dots in input
    dots = _find_non_bg_cells(inp0, bg)
    
    # Need exactly 2 dots of some color(s) that get connected
    # Try finding dot pairs that form L-shapes matching the changes
    added_cells = set((r, c) for r, c, _, _ in changes0)
    
    # Try all pairs of dots (any colors)
    if len(dots) != 2:
        return None
    
    for _ in [0]:  # just once
        (r1, c1, _), (r2, c2, _) = dots
        
        # Try L-shape: go vertical from p1, then horizontal to p2
        l_path_a = set()
        for r in range(min(r1, r2), max(r1, r2) + 1):
            if (r, c1) != (r1, c1) and (r, c1) != (r2, c2):
                l_path_a.add((r, c1))
        for c in range(min(c1, c2), max(c1, c2) + 1):
            if (r2, c) != (r1, c1) and (r2, c) != (r2, c2):
                l_path_a.add((r2, c))
        
        # Try L-shape: go horizontal from p1, then vertical to p2
        l_path_b = set()
        for c in range(min(c1, c2), max(c1, c2) + 1):
            if (r1, c) != (r1, c1) and (r1, c) != (r2, c2):
                l_path_b.add((r1, c))
        for r in range(min(r1, r2), max(r1, r2) + 1):
            if (r, c2) != (r1, c1) and (r, c2) != (r2, c2):
                l_path_b.add((r, c2))
        
        order_a = 'vert_first' if l_path_a == added_cells else None
        order_b = 'horiz_first' if l_path_b == added_cells else None
        matched_order = order_a or order_b
        
        if matched_order:
            # Determine what property of the input selects vert vs horiz
            # Try: top dot color determines order
            top_color = dots[0][2] if r1 <= r2 else dots[1][2]
            
            # Try fixed order first
            rule = {'type': 'l_connect', 'line_color': line_color,
                    'order': matched_order, 'bg': bg}
            if _verify_l_connect(rule, train_pairs):
                return rule
            
            # Try color-based order
            rule = {'type': 'l_connect', 'line_color': line_color,
                    'order': 'color_based',
                    'vert_first_color': top_color if matched_order == 'vert_first' else None,
                    'horiz_first_color': top_color if matched_order == 'horiz_first' else None,
                    'bg': bg}
            if _verify_l_connect(rule, train_pairs):
                return rule
            
            # Auto fallback
            rule = {'type': 'l_connect', 'line_color': line_color,
                    'order': 'auto', 'bg': bg}
            if _verify_l_connect(rule, train_pairs):
                return rule
    
    return None


def _verify_l_connect(rule: Dict, train_pairs: List[Tuple[Grid, Grid]]) -> bool:
    for inp, out in train_pairs:
        result = apply_l_connect(rule, inp)
        if result is None or not grid_eq(result, out):
            return False
    return True


def apply_l_connect(rule: Dict, inp: Grid) -> Optional[Grid]:
    """Apply L-connect rule."""
    h, w = grid_shape(inp)
    bg = rule['bg']
    line_color = rule['line_color']
    order = rule['order']
    
    dots = [(r, c) for r in range(h) for c in range(w) if inp[r][c] != bg]
    if len(dots) != 2:
        return None
    
    (r1, c1), (r2, c2) = dots
    result = [row[:] for row in inp]
    
    if order == 'color_based':
        top_r, top_c = (r1, c1) if r1 <= r2 else (r2, c2)
        top_color = inp[top_r][top_c]
        vert_color = rule.get('vert_first_color')
        order = 'vert_first' if top_color == vert_color else 'horiz_first'
    
    if order == 'auto':
        # Pick the L-path whose corner cell is closer to bg structure.
        # Heuristic: choose the path that avoids non-bg cells along the route.
        # Count non-bg obstacles along each path.
        def path_obstacles(order_try):
            obs = 0
            if order_try == 'vert_first':
                for r in range(min(r1,r2), max(r1,r2)+1):
                    if inp[r][c1] != bg and (r,c1) not in [(r1,c1),(r2,c2)]:
                        obs += 1
                for c in range(min(c1,c2), max(c1,c2)+1):
                    if inp[r2][c] != bg and (r2,c) not in [(r1,c1),(r2,c2)]:
                        obs += 1
            else:
                for c in range(min(c1,c2), max(c1,c2)+1):
                    if inp[r1][c] != bg and (r1,c) not in [(r1,c1),(r2,c2)]:
                        obs += 1
                for r in range(min(r1,r2), max(r1,r2)+1):
                    if inp[r][c2] != bg and (r,c2) not in [(r1,c1),(r2,c2)]:
                        obs += 1
            return obs
        obs_vert = path_obstacles('vert_first')
        obs_horiz = path_obstacles('horiz_first')
        order = 'horiz_first' if obs_horiz < obs_vert else 'vert_first'
    
    if order == 'vert_first':
        for r in range(min(r1, r2), max(r1, r2) + 1):
            if result[r][c1] == bg:
                result[r][c1] = line_color
        for c in range(min(c1, c2), max(c1, c2) + 1):
            if result[r2][c] == bg:
                result[r2][c] = line_color
    else:  # horiz_first
        for c in range(min(c1, c2), max(c1, c2) + 1):
            if result[r1][c] == bg:
                result[r1][c] = line_color
        for r in range(min(r1, r2), max(r1, r2) + 1):
            if result[r][c2] == bg:
                result[r][c2] = line_color
    
    return result


# === Cross Projection from Dots ===

def learn_cross_projection(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn cross/line projection from non-bg cells.
    
    Pattern: Each non-bg cell projects lines in 4 directions.
    Lines may stop at grid boundary or at other non-bg cells.
    The projected lines get a specific color.
    """
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    
    changes = _diff_cells(inp0, out0)
    if not changes:
        return None
    
    # Find what color the projected lines use
    new_colors = set(new for _, _, old, new in changes if old == bg)
    
    # Try different projection strategies
    for strategy in ['project_same_color', 'project_fixed_color', 'project_from_each']:
        rule = _try_cross_projection_strategy(strategy, train_pairs, bg)
        if rule is not None:
            return rule
    
    return None


def _try_cross_projection_strategy(strategy: str, train_pairs, bg) -> Optional[Dict]:
    """Try a specific cross projection strategy."""
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    
    dots = _find_non_bg_cells(inp0, bg)
    
    if strategy == 'project_same_color':
        # Each dot projects lines in its own color
        result = [row[:] for row in inp0]
        for r, c, color in dots:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                while 0 <= nr < h and 0 <= nc < w:
                    if result[nr][nc] == bg:
                        result[nr][nc] = color
                    elif result[nr][nc] != color:
                        break
                    nr, nc = nr+dr, nc+dc
        
        if grid_eq(result, out0):
            rule = {'type': 'cross_project', 'strategy': 'same_color', 'bg': bg}
            if all(grid_eq(apply_cross_projection(rule, inp), out) for inp, out in train_pairs):
                return rule
    
    elif strategy == 'project_fixed_color':
        # All dots project in a single fixed color
        changes = _diff_cells(inp0, out0)
        line_colors = set(new for _, _, old, new in changes if old == bg)
        
        for lc in line_colors:
            result = [row[:] for row in inp0]
            for r, c, color in dots:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    while 0 <= nr < h and 0 <= nc < w:
                        if result[nr][nc] == bg:
                            result[nr][nc] = lc
                        else:
                            break
                        nr, nc = nr+dr, nc+dc
            
            if grid_eq(result, out0):
                rule = {'type': 'cross_project', 'strategy': 'fixed_color',
                        'line_color': lc, 'bg': bg}
                if all(grid_eq(apply_cross_projection(rule, inp), out) for inp, out in train_pairs):
                    return rule
    
    return None


def apply_cross_projection(rule: Dict, inp: Grid) -> Optional[Grid]:
    """Apply cross projection rule."""
    h, w = grid_shape(inp)
    bg = rule['bg']
    strategy = rule['strategy']
    
    result = [row[:] for row in inp]
    dots = _find_non_bg_cells(inp, bg)
    
    if strategy == 'same_color':
        for r, c, color in dots:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                while 0 <= nr < h and 0 <= nc < w:
                    if result[nr][nc] == bg:
                        result[nr][nc] = color
                    elif result[nr][nc] != color:
                        break
                    nr, nc = nr+dr, nc+dc
    
    elif strategy == 'fixed_color':
        lc = rule['line_color']
        for r, c, color in dots:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                while 0 <= nr < h and 0 <= nc < w:
                    if result[nr][nc] == bg:
                        result[nr][nc] = lc
                    else:
                        break
                    nr, nc = nr+dr, nc+dc
    
    return result


# === Line Extension to Border ===

def learn_line_extension(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn line extension: short line segments get extended to grid border.
    
    Pattern: Non-bg cells that form short horizontal/vertical segments
    get extended in their direction until they hit the border.
    """
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    
    changes = _diff_cells(inp0, out0)
    if not changes:
        return None
    
    # All added cells should be bg -> some color
    if not all(old == bg for _, _, old, _ in changes):
        return None
    
    # Try: each non-bg cell extends in all 4 directions to border, in its own color
    result = [row[:] for row in inp0]
    for r in range(h):
        for c in range(w):
            if inp0[r][c] != bg:
                color = inp0[r][c]
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    while 0 <= nr < h and 0 <= nc < w:
                        if result[nr][nc] == bg:
                            result[nr][nc] = color
                        nr, nc = nr+dr, nc+dc
    
    if grid_eq(result, out0):
        rule = {'type': 'line_extend_all', 'bg': bg}
        if all(grid_eq(apply_line_extension(rule, inp), out) for inp, out in train_pairs):
            return rule
    
    # Try: extend only in the direction the segment is pointing
    rule = _try_directional_extension(train_pairs, bg)
    if rule is not None:
        return rule
    
    # Try single-direction fills: right, left, down, up
    for dir_name, dr, dc in [('right', 0, 1), ('left', 0, -1), ('down', 1, 0), ('up', -1, 0)]:
        result = [row[:] for row in inp0]
        for r in range(h):
            for c in range(w):
                if inp0[r][c] != bg:
                    color = inp0[r][c]
                    nr, nc = r+dr, c+dc
                    while 0 <= nr < h and 0 <= nc < w and inp0[nr][nc] == bg:
                        result[nr][nc] = color
                        nr, nc = nr+dr, nc+dc
        
        if grid_eq(result, out0):
            rule = {'type': f'fill_{dir_name}_until_obstacle', 'bg': bg}
            if all(grid_eq(apply_line_extension(rule, inp), out) for inp, out in train_pairs):
                return rule
    
    # Try: segment-direction-aware fill (perpendicular to segment orientation)
    # v segments → fill right or left; h segments → fill down or up
    for v_dir, h_dir in [('right', 'down'), ('left', 'down'), ('right', 'up'), ('left', 'up')]:
        v_dr, v_dc = {'right': (0,1), 'left': (0,-1)}[v_dir]
        h_dr, h_dc = {'down': (1,0), 'up': (-1,0)}[h_dir]
        
        ok_all = True
        for inp, out in train_pairs:
            h_g, w_g = grid_shape(inp)
            b = most_common_color(inp)
            result = [row[:] for row in inp]
            segs = _find_segments(inp, b)
            for color, cells, direction in segs:
                for r, c in cells:
                    if direction == 'v':
                        nr, nc = r+v_dr, c+v_dc
                        while 0 <= nr < h_g and 0 <= nc < w_g and inp[nr][nc] == b:
                            result[nr][nc] = color
                            nr, nc = nr+v_dr, nc+v_dc
                    elif direction == 'h':
                        nr, nc = r+h_dr, c+h_dc
                        while 0 <= nr < h_g and 0 <= nc < w_g and inp[nr][nc] == b:
                            result[nr][nc] = color
                            nr, nc = nr+h_dr, nc+h_dc
                    elif direction == 'point':
                        for d_r, d_c in [(v_dr,v_dc),(h_dr,h_dc)]:
                            nr, nc = r+d_r, c+d_c
                            while 0 <= nr < h_g and 0 <= nc < w_g and inp[nr][nc] == b:
                                result[nr][nc] = color
                                nr, nc = nr+d_r, nc+d_c
            if not grid_eq(result, out):
                ok_all = False
                break
        
        if ok_all:
            return {'type': 'fill_segment_perp', 'v_dir': v_dir, 'h_dir': h_dir, 'bg': bg}
    
    # Try: fill in all 4 directions from each non-bg cell until obstacle
    for directions_name, directions in [
        ('fill_all_until_obstacle', [(-1,0),(1,0),(0,-1),(0,1)]),
    ]:
        result = [row[:] for row in inp0]
        for r in range(h):
            for c in range(w):
                if inp0[r][c] != bg:
                    color = inp0[r][c]
                    for d_r, d_c in directions:
                        nr, nc = r+d_r, c+d_c
                        while 0 <= nr < h and 0 <= nc < w and inp0[nr][nc] == bg:
                            result[nr][nc] = color
                            nr, nc = nr+d_r, nc+d_c
        
        if grid_eq(result, out0):
            rule = {'type': directions_name, 'bg': bg}
            if all(grid_eq(apply_line_extension(rule, inp), out) for inp, out in train_pairs):
                return rule
    
    return None


def _try_directional_extension(train_pairs, bg) -> Optional[Dict]:
    """Try extending each colored segment in its own direction only."""
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    
    # Find segments: groups of same-color cells forming h or v lines
    segments = _find_segments(inp0, bg)
    
    # For each segment, extend in its direction
    result = [row[:] for row in inp0]
    for color, cells, direction in segments:
        if direction == 'h':
            # Extend horizontally
            r = cells[0][0]
            c_min = min(c for _, c in cells)
            c_max = max(c for _, c in cells)
            for c in range(0, w):
                if result[r][c] == bg:
                    result[r][c] = color
        elif direction == 'v':
            # Extend vertically
            c = cells[0][1]
            r_min = min(r for r, _ in cells)
            r_max = max(r for r, _ in cells)
            for r in range(0, h):
                if result[r][c] == bg:
                    result[r][c] = color
        elif direction == 'point':
            # Single cell: extend in all 4 directions
            r, c = cells[0]
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                while 0 <= nr < h and 0 <= nc < w:
                    if result[nr][nc] == bg:
                        result[nr][nc] = color
                    nr, nc = nr+dr, nc+dc
    
    if grid_eq(result, out0):
        rule = {'type': 'line_extend_directional', 'bg': bg}
        if all(grid_eq(apply_line_extension(rule, inp), out) for inp, out in train_pairs):
            return rule
    
    return None


def _find_segments(grid: Grid, bg: int) -> List[Tuple[int, List[Tuple[int,int]], str]]:
    """Find colored segments (horizontal, vertical, or single point)."""
    h, w = grid_shape(grid)
    visited = set()
    segments = []
    
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and (r, c) not in visited:
                color = grid[r][c]
                # BFS to find connected same-color cells
                cells = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and grid[nr][nc] == color:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                
                # Determine direction
                rows = set(r for r, c in cells)
                cols = set(c for r, c in cells)
                if len(cells) == 1:
                    direction = 'point'
                elif len(rows) == 1:
                    direction = 'h'
                elif len(cols) == 1:
                    direction = 'v'
                else:
                    direction = 'mixed'
                
                segments.append((color, cells, direction))
    
    return segments


def apply_line_extension(rule: Dict, inp: Grid) -> Optional[Grid]:
    """Apply line extension rule."""
    h, w = grid_shape(inp)
    bg = rule['bg']
    
    if rule['type'] == 'line_extend_all':
        result = [row[:] for row in inp]
        for r in range(h):
            for c in range(w):
                if inp[r][c] != bg:
                    color = inp[r][c]
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        while 0 <= nr < h and 0 <= nc < w:
                            if result[nr][nc] == bg:
                                result[nr][nc] = color
                            nr, nc = nr+dr, nc+dc
        return result
    
    elif rule['type'] == 'line_extend_directional':
        result = [row[:] for row in inp]
        segments = _find_segments(inp, bg)
        
        for color, cells, direction in segments:
            if direction == 'h':
                r = cells[0][0]
                for c in range(w):
                    if result[r][c] == bg:
                        result[r][c] = color
            elif direction == 'v':
                c = cells[0][1]
                for r_i in range(h):
                    if result[r_i][c] == bg:
                        result[r_i][c] = color
            elif direction == 'point':
                r, c = cells[0]
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    while 0 <= nr < h and 0 <= nc < w:
                        if result[nr][nc] == bg:
                            result[nr][nc] = color
                        nr, nc = nr+dr, nc+dc
        return result
    
    elif rule['type'] == 'fill_right_until_obstacle':
        result = [row[:] for row in inp]
        for r in range(h):
            for c in range(w):
                if inp[r][c] != bg:
                    color = inp[r][c]
                    nc = c + 1
                    while nc < w and inp[r][nc] == bg:
                        result[r][nc] = color
                        nc += 1
        return result
    
    elif rule['type'] == 'fill_down_until_obstacle':
        result = [row[:] for row in inp]
        for r in range(h):
            for c in range(w):
                if inp[r][c] != bg:
                    color = inp[r][c]
                    nr = r + 1
                    while nr < h and inp[nr][c] == bg:
                        result[nr][c] = color
                        nr += 1
        return result
    
    elif rule['type'] == 'fill_all_until_obstacle':
        result = [row[:] for row in inp]
        for r in range(h):
            for c in range(w):
                if inp[r][c] != bg:
                    color = inp[r][c]
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        while 0 <= nr < h and 0 <= nc < w and inp[nr][nc] == bg:
                            result[nr][nc] = color
                            nr, nc = nr+dr, nc+dc
        return result
    
    elif rule['type'] == 'fill_segment_perp':
        v_dir = rule['v_dir']
        h_dir = rule['h_dir']
        v_dr, v_dc = {'right': (0,1), 'left': (0,-1)}[v_dir]
        h_dr, h_dc = {'down': (1,0), 'up': (-1,0)}[h_dir]
        
        result = [row[:] for row in inp]
        segs = _find_segments(inp, bg)
        for color, cells, direction in segs:
            for r, c in cells:
                if direction == 'v':
                    nr, nc = r+v_dr, c+v_dc
                    while 0 <= nr < h and 0 <= nc < w and inp[nr][nc] == bg:
                        result[nr][nc] = color
                        nr, nc = nr+v_dr, nc+v_dc
                elif direction == 'h':
                    nr, nc = r+h_dr, c+h_dc
                    while 0 <= nr < h and 0 <= nc < w and inp[nr][nc] == bg:
                        result[nr][nc] = color
                        nr, nc = nr+h_dr, nc+h_dc
                elif direction == 'point':
                    for d_r, d_c in [(v_dr,v_dc),(h_dr,h_dc)]:
                        nr, nc = r+d_r, c+d_c
                        while 0 <= nr < h and 0 <= nc < w and inp[nr][nc] == bg:
                            result[nr][nc] = color
                            nr, nc = nr+d_r, nc+d_c
        return result
    
    elif rule['type'].startswith('fill_') and rule['type'].endswith('_until_obstacle'):
        # Generic single-direction fill
        dir_map = {'right': (0,1), 'left': (0,-1), 'down': (1,0), 'up': (-1,0)}
        dir_name = rule['type'].replace('fill_', '').replace('_until_obstacle', '')
        if dir_name in dir_map:
            dr, dc = dir_map[dir_name]
            result = [row[:] for row in inp]
            for r in range(h):
                for c in range(w):
                    if inp[r][c] != bg:
                        color = inp[r][c]
                        nr, nc = r+dr, c+dc
                        while 0 <= nr < h and 0 <= nc < w and inp[nr][nc] == bg:
                            result[nr][nc] = color
                            nr, nc = nr+dr, nc+dc
            return result
    
    return None


# === Connect Same-Color Objects ===

def learn_connect_objects(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn connecting same-color objects with lines.
    
    Pattern: Objects of the same color get connected by horizontal
    or vertical lines between them.
    """
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    inp0, out0 = train_pairs[0]
    
    changes = _diff_cells(inp0, out0)
    if not changes:
        return None
    
    # All changes should be bg -> some_color
    if not all(old == bg for _, _, old, _ in changes):
        return None
    
    # Check if added cells have colors matching nearby objects
    # and form lines between object pairs
    
    # Try: for each pair of same-color objects, draw a line between centers
    for connect_type in ['center_h', 'center_v', 'nearest_edge']:
        rule = {'type': 'connect_same_color', 'connect': connect_type, 'bg': bg}
        result = apply_connect_objects(rule, inp0)
        if result is not None and grid_eq(result, out0):
            if all(grid_eq(apply_connect_objects(rule, inp), out) for inp, out in train_pairs):
                return rule
    
    return None


def apply_connect_objects(rule: Dict, inp: Grid) -> Optional[Grid]:
    """Apply object connection rule."""
    h, w = grid_shape(inp)
    bg = rule['bg']
    connect = rule['connect']
    
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]
    
    # Group objects by color
    by_color = {}
    for obj in objs:
        by_color.setdefault(obj.color, []).append(obj)
    
    for color, group in by_color.items():
        if len(group) < 2:
            continue
        
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                o1, o2 = group[i], group[j]
                cr1, cc1 = int(o1.center[0]), int(o1.center[1])
                cr2, cc2 = int(o2.center[0]), int(o2.center[1])
                
                if connect == 'center_h' or connect == 'nearest_edge':
                    # Draw horizontal line on average row
                    if abs(cr1 - cr2) <= 1:
                        row = (cr1 + cr2) // 2
                        c_min = min(cc1, cc2)
                        c_max = max(cc1, cc2)
                        for c in range(c_min, c_max + 1):
                            if result[row][c] == bg:
                                result[row][c] = color
                
                if connect == 'center_v' or connect == 'nearest_edge':
                    if abs(cc1 - cc2) <= 1:
                        col = (cc1 + cc2) // 2
                        r_min = min(cr1, cr2)
                        r_max = max(cr1, cr2)
                        for r in range(r_min, r_max + 1):
                            if result[r][col] == bg:
                                result[r][col] = color
    
    return result
