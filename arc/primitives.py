"""
New DSL Primitives for Beam Search (Module 20+)

Each primitive is an atomic transformation that can be chained.
Design principle: simple, composable, useful as intermediate steps.
"""

from typing import List, Tuple, Optional, Dict
from collections import Counter, deque
from arc.grid import Grid, grid_eq, grid_shape, most_common_color


# ============================================================
# Color operations
# ============================================================

def remove_color(grid: Grid, color: int) -> Grid:
    """Replace all cells of `color` with bg (most common color)."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    return [[bg if grid[r][c] == color else grid[r][c] for c in range(w)] for r in range(h)]


def swap_colors(grid: Grid, a: int, b: int) -> Grid:
    """Swap two colors."""
    h, w = grid_shape(grid)
    return [[b if grid[r][c] == a else (a if grid[r][c] == b else grid[r][c]) for c in range(w)] for r in range(h)]


def recolor_all_nonbg(grid: Grid, new_color: int) -> Grid:
    """Recolor all non-bg cells to new_color."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    return [[new_color if grid[r][c] != bg else bg for c in range(w)] for r in range(h)]


def keep_only_color(grid: Grid, color: int) -> Grid:
    """Keep only cells of `color`, replace rest with bg."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    return [[color if grid[r][c] == color else bg for c in range(w)] for r in range(h)]


# ============================================================
# Geometric operations  
# ============================================================

def flip_horizontal(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]

def flip_vertical(grid: Grid) -> Grid:
    return grid[::-1]

def rotate_90(grid: Grid) -> Grid:
    h, w = grid_shape(grid)
    return [[grid[h-1-c][r] for c in range(h)] for r in range(w)]

def rotate_180(grid: Grid) -> Grid:
    h, w = grid_shape(grid)
    return [[grid[h-1-r][w-1-c] for c in range(w)] for r in range(h)]

def rotate_270(grid: Grid) -> Grid:
    h, w = grid_shape(grid)
    return [[grid[c][w-1-r] for c in range(h)] for r in range(w)]

def transpose(grid: Grid) -> Grid:
    h, w = grid_shape(grid)
    return [[grid[r][c] for r in range(h)] for c in range(w)]


# ============================================================
# Region operations
# ============================================================

def fill_enclosed_regions(grid: Grid) -> Grid:
    """Fill enclosed bg regions with the surrounding color."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    
    # BFS from borders to find exterior bg cells
    visited = set()
    q = deque()
    for r in range(h):
        for c in [0, w-1]:
            if grid[r][c] == bg and (r,c) not in visited:
                visited.add((r,c)); q.append((r,c))
    for c in range(w):
        for r in [0, h-1]:
            if grid[r][c] == bg and (r,c) not in visited:
                visited.add((r,c)); q.append((r,c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == bg and (nr,nc) not in visited:
                visited.add((nr,nc)); q.append((nr,nc))
    
    # Fill interior bg cells with adjacent non-bg color
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg and (r,c) not in visited:
                adj = [grid[r+dr][c+dc] for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                       if 0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr][c+dc] != bg]
                if adj:
                    result[r][c] = Counter(adj).most_common(1)[0][0]
    return result


def remove_isolated_cells(grid: Grid) -> Grid:
    """Remove cells with no same-color neighbors (noise removal)."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                has_neighbor = False
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == grid[r][c]:
                        has_neighbor = True; break
                if not has_neighbor:
                    result[r][c] = bg
    return result


def remove_small_components(grid: Grid, min_size: int = 3) -> Grid:
    """Remove connected components smaller than min_size."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    visited = [[False]*w for _ in range(h)]
    result = [row[:] for row in grid]
    
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and not visited[r][c]:
                # BFS to find component
                comp = []
                q = deque([(r,c)])
                visited[r][c] = True
                color = grid[r][c]
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True; q.append((nr,nc))
                if len(comp) < min_size:
                    for cr, cc in comp:
                        result[cr][cc] = bg
    return result


def keep_largest_component(grid: Grid) -> Grid:
    """Keep only the largest connected component."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    visited = [[False]*w for _ in range(h)]
    components = []
    
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and not visited[r][c]:
                comp = []
                q = deque([(r,c)])
                visited[r][c] = True
                color = grid[r][c]
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True; q.append((nr,nc))
                components.append((color, comp))
    
    if not components:
        return [row[:] for row in grid]
    
    largest = max(components, key=lambda x: len(x[1]))
    keep = set(largest[1])
    result = [[bg]*w for _ in range(h)]
    for r, c in keep:
        result[r][c] = grid[r][c]
    return result


# ============================================================
# Drawing operations
# ============================================================

def draw_diagonals_from_dots(grid: Grid) -> Grid:
    """From each non-bg cell, draw diagonal lines (X pattern)."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    dots = [(r, c) for r in range(h) for c in range(w) if grid[r][c] != bg]
    result = [row[:] for row in grid]
    for dr, dc in dots:
        color = grid[dr][dc]
        for ddr, ddc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = dr+ddr, dc+ddc
            while 0 <= nr < h and 0 <= nc < w:
                if result[nr][nc] == bg:
                    result[nr][nc] = color
                nr += ddr; nc += ddc
    return result


def draw_cross_from_dots(grid: Grid) -> Grid:
    """From each non-bg cell, draw horizontal+vertical lines."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    dots = [(r, c) for r in range(h) for c in range(w) if grid[r][c] != bg]
    result = [row[:] for row in grid]
    for dr, dc in dots:
        color = grid[dr][dc]
        for r in range(h):
            if result[r][dc] == bg:
                result[r][dc] = color
        for c in range(w):
            if result[dr][c] == bg:
                result[dr][c] = color
    return result


def fill_between_h(grid: Grid) -> Grid:
    """Fill gaps between same-color cells horizontally."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    result = [row[:] for row in grid]
    for r in range(h):
        for color in range(10):
            if color == bg: continue
            positions = [c for c in range(w) if grid[r][c] == color]
            if len(positions) >= 2:
                for i in range(len(positions)-1):
                    for c in range(positions[i]+1, positions[i+1]):
                        if result[r][c] == bg:
                            result[r][c] = color
    return result


def fill_between_v(grid: Grid) -> Grid:
    """Fill gaps between same-color cells vertically."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    result = [row[:] for row in grid]
    for c in range(w):
        for color in range(10):
            if color == bg: continue
            positions = [r for r in range(h) if grid[r][c] == color]
            if len(positions) >= 2:
                for i in range(len(positions)-1):
                    for r in range(positions[i]+1, positions[i+1]):
                        if result[r][c] == bg:
                            result[r][c] = color
    return result


def outline_objects(grid: Grid) -> Grid:
    """Draw border around each non-bg object (expand by 1 pixel)."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and result[nr][nc] == bg:
                        result[nr][nc] = grid[r][c]
    return result


def erode_objects(grid: Grid) -> Grid:
    """Remove border pixels of objects (shrink by 1 pixel)."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if not (0 <= nr < h and 0 <= nc < w) or grid[nr][nc] == bg:
                        result[r][c] = bg
                        break
    return result


# ============================================================
# Symmetry operations
# ============================================================

def complete_symmetry_h(grid: Grid) -> Grid:
    """Complete horizontal symmetry (mirror top↔bottom)."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    result = [row[:] for row in grid]
    for r in range(h):
        mr = h - 1 - r
        for c in range(w):
            if result[r][c] == bg and result[mr][c] != bg:
                result[r][c] = result[mr][c]
            elif result[mr][c] == bg and result[r][c] != bg:
                result[mr][c] = result[r][c]
    return result


def complete_symmetry_v(grid: Grid) -> Grid:
    """Complete vertical symmetry (mirror left↔right)."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            mc = w - 1 - c
            if result[r][c] == bg and result[r][mc] != bg:
                result[r][c] = result[r][mc]
            elif result[r][mc] == bg and result[r][c] != bg:
                result[r][mc] = result[r][c]
    return result


def complete_symmetry_diag(grid: Grid) -> Grid:
    """Complete diagonal symmetry (mirror across main diagonal)."""
    h, w = grid_shape(grid)
    if h != w:
        return [row[:] for row in grid]
    bg = most_common_color(grid)
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if result[r][c] == bg and result[c][r] != bg:
                result[r][c] = result[c][r]
            elif result[c][r] == bg and result[r][c] != bg:
                result[c][r] = result[r][c]
    return result


# ============================================================
# Extraction / Cropping
# ============================================================

def crop_to_content(grid: Grid) -> Grid:
    """Crop to bounding box of all non-bg content."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    rows = [r for r in range(h) if any(grid[r][c] != bg for c in range(w))]
    cols = [c for c in range(w) if any(grid[r][c] != bg for r in range(h))]
    if not rows or not cols:
        return [row[:] for row in grid]
    return [[grid[r][c] for c in range(min(cols), max(cols)+1)] for r in range(min(rows), max(rows)+1)]


def extract_largest_object(grid: Grid) -> Grid:
    """Extract and crop the largest connected component."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    visited = [[False]*w for _ in range(h)]
    best = None
    
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and not visited[r][c]:
                comp = []
                q = deque([(r,c)])
                visited[r][c] = True
                color = grid[r][c]
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True; q.append((nr,nc))
                if best is None or len(comp) > len(best[1]):
                    best = (color, comp)
    
    if best is None:
        return [row[:] for row in grid]
    
    color, cells = best
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    cell_set = set(cells)
    r1, r2, c1, c2 = min(rs), max(rs), min(cs), max(cs)
    return [[color if (r,c) in cell_set else bg for c in range(c1, c2+1)] for r in range(r1, r2+1)]


# ============================================================
# Tiling / Scaling
# ============================================================

def scale_up_2x(grid: Grid) -> Grid:
    """Scale grid up by 2x."""
    h, w = grid_shape(grid)
    return [[grid[r//2][c//2] for c in range(w*2)] for r in range(h*2)]


def scale_up_3x(grid: Grid) -> Grid:
    """Scale grid up by 3x."""
    h, w = grid_shape(grid)
    return [[grid[r//3][c//3] for c in range(w*3)] for r in range(h*3)]


def tile_2x2(grid: Grid) -> Grid:
    """Tile grid in 2x2 pattern."""
    h, w = grid_shape(grid)
    return [[grid[r % h][c % w] for c in range(w*2)] for r in range(h*2)]


def tile_3x3(grid: Grid) -> Grid:
    """Tile grid in 3x3 pattern."""
    h, w = grid_shape(grid)
    return [[grid[r % h][c % w] for c in range(w*3)] for r in range(h*3)]


def kronecker_self(grid: Grid) -> Grid:
    """Kronecker product: use grid as template to tile itself.
    Non-bg cells → place copy of grid; bg cells → bg block."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    oh, ow = h*h, w*w
    result = [[bg]*ow for _ in range(oh)]
    for i in range(h):
        for j in range(w):
            if grid[i][j] != bg:
                for di in range(h):
                    for dj in range(w):
                        result[i*h+di][j*w+dj] = grid[di][dj]
    return result


# ============================================================
# Key pixel / marker operations
# ============================================================

def key_pixel_recolor(grid: Grid) -> Optional[Grid]:
    """Find singleton color pixel, recolor all non-bg cells to that color, remove key."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    cc = Counter(grid[r][c] for r in range(h) for c in range(w))
    singletons = [(c, cnt) for c, cnt in cc.items() if cnt == 1 and c != bg]
    if len(singletons) != 1:
        return None
    key_color = singletons[0][0]
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if result[r][c] == key_color:
                result[r][c] = bg
            elif result[r][c] != bg:
                result[r][c] = key_color
    return result


# ============================================================
# Gravity operations
# ============================================================

def gravity_down(grid: Grid) -> Grid:
    """Drop all non-bg cells downward."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    result = [[bg]*w for _ in range(h)]
    for c in range(w):
        non_bg = [grid[r][c] for r in range(h) if grid[r][c] != bg]
        for i, v in enumerate(non_bg):
            result[h - len(non_bg) + i][c] = v
    return result


def gravity_up(grid: Grid) -> Grid:
    """Push all non-bg cells upward."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    result = [[bg]*w for _ in range(h)]
    for c in range(w):
        non_bg = [grid[r][c] for r in range(h) if grid[r][c] != bg]
        for i, v in enumerate(non_bg):
            result[i][c] = v
    return result


def gravity_left(grid: Grid) -> Grid:
    """Push all non-bg cells leftward."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    result = [[bg]*w for _ in range(h)]
    for r in range(h):
        non_bg = [grid[r][c] for c in range(w) if grid[r][c] != bg]
        for i, v in enumerate(non_bg):
            result[r][i] = v
    return result


def gravity_right(grid: Grid) -> Grid:
    """Push all non-bg cells rightward."""
    h, w = grid_shape(grid)
    bg = most_common_color(grid)
    result = [[bg]*w for _ in range(h)]
    for r in range(h):
        non_bg = [grid[r][c] for c in range(w) if grid[r][c] != bg]
        for i, v in enumerate(non_bg):
            result[r][w - len(non_bg) + i] = v
    return result


# ============================================================
# Master list of all primitives
# ============================================================

# Each entry: (name, fn, needs_params)
# If needs_params=True, fn takes (grid, **params) and must be learned
# If needs_params=False, fn takes (grid) and is parameter-free

PARAMETERLESS_PRIMITIVES = [
    ('flip_h', flip_horizontal),
    ('flip_v', flip_vertical),
    ('rot90', rotate_90),
    ('rot180', rotate_180),
    ('rot270', rotate_270),
    ('transpose', transpose),
    ('fill_enclosed', fill_enclosed_regions),
    ('remove_isolated', remove_isolated_cells),
    ('remove_small_3', lambda g: remove_small_components(g, 3)),
    ('remove_small_5', lambda g: remove_small_components(g, 5)),
    ('keep_largest_comp', keep_largest_component),
    ('draw_diag', draw_diagonals_from_dots),
    ('draw_cross', draw_cross_from_dots),
    ('fill_between_h', fill_between_h),
    ('fill_between_v', fill_between_v),
    ('outline', outline_objects),
    ('erode', erode_objects),
    ('sym_h', complete_symmetry_h),
    ('sym_v', complete_symmetry_v),
    ('sym_diag', complete_symmetry_diag),
    ('crop', crop_to_content),
    ('extract_largest', extract_largest_object),
    ('scale_2x', scale_up_2x),
    ('scale_3x', scale_up_3x),
    ('tile_2x2', tile_2x2),
    ('tile_3x3', tile_3x3),
    ('kronecker', kronecker_self),
    ('key_recolor', key_pixel_recolor),
    ('gravity_down', gravity_down),
    ('gravity_up', gravity_up),
    ('gravity_left', gravity_left),
    ('gravity_right', gravity_right),
]

# Color-parameterized (generated dynamically based on input colors)
def get_color_primitives(grid: Grid) -> list:
    """Generate color-specific primitives based on colors in the grid."""
    h, w = grid_shape(grid)
    colors = set(grid[r][c] for r in range(h) for c in range(w))
    bg = most_common_color(grid)
    prims = []
    for c in colors:
        if c != bg:
            prims.append((f'remove_c{c}', lambda g, _c=c: remove_color(g, _c)))
            prims.append((f'keep_c{c}', lambda g, _c=c: keep_only_color(g, _c)))
            prims.append((f'recolor_to_{c}', lambda g, _c=c: recolor_all_nonbg(g, _c)))
    # Color swaps (limited to avoid explosion)
    non_bg = sorted(colors - {bg})
    for i, a in enumerate(non_bg):
        for b in non_bg[i+1:]:
            prims.append((f'swap_{a}_{b}', lambda g, _a=a, _b=b: swap_colors(g, _a, _b)))
    return prims
