"""
arc/program_search.py — Test-time program synthesis

Inspired by DeepMind's approach: instead of hand-coding each solver,
define atomic primitives and search for compositions that work.

Atomic primitives operate on numpy grids. Each takes (grid, bg) and returns grid.
Search: depth 1 (single op), depth 2 (compose two), depth 3 if needed.
Verify: must pass ALL train examples.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from collections import Counter
from itertools import product
from scipy.ndimage import label as connected_components
from arc.cross_engine import CrossPiece

Grid = List[List[int]]

def _g(g): return np.array(g, dtype=int)
def _l(a): return a.tolist()
def _bg(g): return int(Counter(g.flatten()).most_common(1)[0][0])


# ============================================================
# ATOMIC PRIMITIVES
# Each: (grid: ndarray, bg: int) -> ndarray
# ============================================================

def rot90(g, bg): return np.rot90(g, 1)
def rot180(g, bg): return np.rot90(g, 2)
def rot270(g, bg): return np.rot90(g, 3)
def flip_h(g, bg): return g[:, ::-1]
def flip_v(g, bg): return g[::-1, :]
def transpose(g, bg): return g.T

def gravity_down(g, bg):
    H, W = g.shape
    out = np.full_like(g, bg)
    for c in range(W):
        vals = [int(g[r, c]) for r in range(H) if g[r, c] != bg]
        for i, v in enumerate(vals):
            out[H - len(vals) + i, c] = v
    return out

def gravity_up(g, bg):
    H, W = g.shape
    out = np.full_like(g, bg)
    for c in range(W):
        vals = [int(g[r, c]) for r in range(H) if g[r, c] != bg]
        for i, v in enumerate(vals):
            out[i, c] = v
    return out

def gravity_right(g, bg):
    H, W = g.shape
    out = np.full_like(g, bg)
    for r in range(H):
        vals = [int(g[r, c]) for c in range(W) if g[r, c] != bg]
        for i, v in enumerate(vals):
            out[r, W - len(vals) + i] = v
    return out

def gravity_left(g, bg):
    H, W = g.shape
    out = np.full_like(g, bg)
    for r in range(H):
        vals = [int(g[r, c]) for c in range(W) if g[r, c] != bg]
        for i, v in enumerate(vals):
            out[r, i] = v
    return out

def sort_rows_by_nonbg(g, bg):
    H, W = g.shape
    counts = [sum(1 for c in range(W) if g[r, c] != bg) for r in range(H)]
    order = sorted(range(H), key=lambda r: counts[r])
    return g[order, :]

def sort_rows_by_nonbg_desc(g, bg):
    H, W = g.shape
    counts = [sum(1 for c in range(W) if g[r, c] != bg) for r in range(H)]
    order = sorted(range(H), key=lambda r: -counts[r])
    return g[order, :]

def sort_cols_by_nonbg(g, bg):
    H, W = g.shape
    counts = [sum(1 for r in range(H) if g[r, c] != bg) for c in range(W)]
    order = sorted(range(W), key=lambda c: counts[c])
    return g[:, order]

def fill_enclosed(g, bg):
    """Fill enclosed bg regions with surrounding color"""
    H, W = g.shape
    out = g.copy()
    bg_mask = (g == bg).astype(int)
    labeled, n = connected_components(bg_mask)
    
    for i in range(1, n + 1):
        cells = np.where(labeled == i)
        rows, cols = cells
        
        touches_border = (rows.min() == 0 or rows.max() == H-1 or
                         cols.min() == 0 or cols.max() == W-1)
        
        if not touches_border:
            surround = Counter()
            for r, c in zip(rows, cols):
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and g[nr, nc] != bg:
                        surround[int(g[nr, nc])] += 1
            if surround:
                fill = surround.most_common(1)[0][0]
                for r, c in zip(rows, cols):
                    out[r, c] = fill
    return out

def remove_isolated(g, bg):
    """Remove cells with no same-color neighbor"""
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if g[r, c] == bg:
                continue
            has_neighbor = False
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == g[r, c]:
                    has_neighbor = True
                    break
            if not has_neighbor:
                out[r, c] = bg
    return out

def keep_largest_object(g, bg):
    """Keep only the largest connected component"""
    mask = (g != bg).astype(int)
    labeled, n = connected_components(mask)
    if n == 0:
        return g.copy()
    
    sizes = {}
    for i in range(1, n + 1):
        sizes[i] = np.sum(labeled == i)
    
    largest = max(sizes, key=sizes.get)
    out = np.full_like(g, bg)
    out[labeled == largest] = g[labeled == largest]
    return out

def keep_smallest_object(g, bg):
    """Keep only the smallest connected component"""
    mask = (g != bg).astype(int)
    labeled, n = connected_components(mask)
    if n == 0:
        return g.copy()
    
    sizes = {}
    for i in range(1, n + 1):
        sizes[i] = np.sum(labeled == i)
    
    smallest = min(sizes, key=sizes.get)
    out = np.full_like(g, bg)
    out[labeled == smallest] = g[labeled == smallest]
    return out

def mirror_h_complete(g, bg):
    """Mirror left half to right"""
    H, W = g.shape
    out = g.copy()
    mid = W // 2
    for r in range(H):
        for c in range(mid):
            if g[r, c] != bg and g[r, W-1-c] == bg:
                out[r, W-1-c] = g[r, c]
            elif g[r, W-1-c] != bg and g[r, c] == bg:
                out[r, c] = g[r, W-1-c]
    return out

def mirror_v_complete(g, bg):
    """Mirror top half to bottom"""
    H, W = g.shape
    out = g.copy()
    mid = H // 2
    for c in range(W):
        for r in range(mid):
            if g[r, c] != bg and g[H-1-r, c] == bg:
                out[H-1-r, c] = g[r, c]
            elif g[H-1-r, c] != bg and g[r, c] == bg:
                out[r, c] = g[H-1-r, c]
    return out

def extend_lines_h(g, bg):
    """Extend horizontal lines of non-bg to fill the row"""
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        colors = [int(g[r, c]) for c in range(W) if g[r, c] != bg]
        if len(colors) >= 2:
            most_common = Counter(colors).most_common(1)[0][0]
            for c in range(W):
                if out[r, c] == bg:
                    out[r, c] = most_common
    return out

def extend_lines_v(g, bg):
    """Extend vertical lines of non-bg to fill the column"""
    H, W = g.shape
    out = g.copy()
    for c in range(W):
        colors = [int(g[r, c]) for r in range(H) if g[r, c] != bg]
        if len(colors) >= 2:
            most_common = Counter(colors).most_common(1)[0][0]
            for r in range(H):
                if out[r, c] == bg:
                    out[r, c] = most_common
    return out

def invert_colors(g, bg):
    """Swap bg and non-bg: bg cells get most common non-bg, non-bg becomes bg"""
    nonbg_colors = Counter(int(v) for v in g.flatten() if int(v) != bg)
    if not nonbg_colors:
        return g.copy()
    most_common = nonbg_colors.most_common(1)[0][0]
    out = g.copy()
    out[g == bg] = most_common
    out[g != bg] = bg
    return out

def crop_nonbg(g, bg):
    """Crop to bounding box of non-bg"""
    rows, cols = np.where(g != bg)
    if len(rows) == 0:
        return g.copy()
    return g[rows.min():rows.max()+1, cols.min():cols.max()+1].copy()

def dedup_rows(g, bg):
    """Remove duplicate consecutive rows"""
    rows = []
    prev = None
    for r in range(g.shape[0]):
        key = g[r, :].tobytes()
        if key != prev:
            rows.append(r)
            prev = key
    return g[rows, :]

def dedup_cols(g, bg):
    """Remove duplicate consecutive columns"""
    cols = []
    prev = None
    for c in range(g.shape[1]):
        key = g[:, c].tobytes()
        if key != prev:
            cols.append(c)
            prev = key
    return g[:, cols]


# ============================================================
# PRIMITIVE REGISTRY
# ============================================================

# Shape-preserving primitives (can be freely composed)
SAME_SHAPE_OPS = [
    ('rot90', rot90),
    ('rot180', rot180),
    ('rot270', rot270),
    ('flip_h', flip_h),
    ('flip_v', flip_v),
    ('grav_d', gravity_down),
    ('grav_u', gravity_up),
    ('grav_r', gravity_right),
    ('grav_l', gravity_left),
    ('fill_enc', fill_enclosed),
    ('rm_iso', remove_isolated),
    ('mir_h', mirror_h_complete),
    ('mir_v', mirror_v_complete),
    ('ext_h', extend_lines_h),
    ('ext_v', extend_lines_v),
    ('inv', invert_colors),
]

# Shape-changing primitives (only at the end of a pipeline)
SHAPE_CHANGE_OPS = [
    ('transpose', transpose),
    ('crop', crop_nonbg),
    ('dedup_r', dedup_rows),
    ('dedup_c', dedup_cols),
    ('keep_big', keep_largest_object),
    ('keep_small', keep_smallest_object),
    ('sort_r_asc', sort_rows_by_nonbg),
    ('sort_r_desc', sort_rows_by_nonbg_desc),
    ('sort_c_asc', sort_cols_by_nonbg),
]

def extract_minority_color_bbox(g, bg):
    """Extract bbox of the least common non-bg color."""
    colors = Counter(g.flatten())
    non_bg = {c: n for c, n in colors.items() if c != bg and n > 0}
    if not non_bg:
        return g
    minority = min(non_bg, key=non_bg.get)
    rows, cols = np.where(g == minority)
    if len(rows) == 0:
        return g
    return g[rows.min():rows.max()+1, cols.min():cols.max()+1]


def extract_unique_3x3(g, bg):
    """Find the unique 3x3 subgrid (different from all others)."""
    H, W = g.shape
    if H < 3 or W < 3:
        return g
    patches = []
    for r in range(H-2):
        for c in range(W-2):
            patches.append(((r,c), g[r:r+3, c:c+3].tobytes()))
    
    from collections import Counter
    patch_counts = Counter(p[1] for p in patches)
    # Find the rarest patch
    for (r,c), key in patches:
        if patch_counts[key] == 1:
            return g[r:r+3, c:c+3]
    return g


def remove_bg_rows_cols(g, bg):
    """Remove rows and columns that are entirely bg."""
    mask_r = np.any(g != bg, axis=1)
    mask_c = np.any(g != bg, axis=0)
    if not mask_r.any() or not mask_c.any():
        return g
    return g[np.ix_(mask_r, mask_c)]


def extract_top_left_quarter(g, bg):
    H, W = g.shape
    return g[:H//2, :W//2]


def extract_top_right_quarter(g, bg):
    H, W = g.shape
    return g[:H//2, W//2:]


def extract_bottom_left_quarter(g, bg):
    H, W = g.shape
    return g[H//2:, :W//2]


def extract_bottom_right_quarter(g, bg):
    H, W = g.shape
    return g[H//2:, W//2:]


def xor_halves_h(g, bg):
    """XOR top and bottom halves (non-bg in one but not both)."""
    H, W = g.shape
    if H % 2 != 0:
        return g
    top = g[:H//2]
    bot = g[H//2:]
    result = np.full_like(top, bg)
    for r in range(H//2):
        for c in range(W):
            t, b = top[r,c], bot[r,c]
            if t != bg and b == bg:
                result[r,c] = t
            elif b != bg and t == bg:
                result[r,c] = b
    return result


def xor_halves_v(g, bg):
    """XOR left and right halves."""
    H, W = g.shape
    if W % 2 != 0:
        return g
    left = g[:, :W//2]
    right = g[:, W//2:]
    result = np.full_like(left, bg)
    for r in range(H):
        for c in range(W//2):
            l, ri = left[r,c], right[r,c]
            if l != bg and ri == bg:
                result[r,c] = l
            elif ri != bg and l == bg:
                result[r,c] = ri
    return result


def or_halves_h(g, bg):
    """OR top and bottom halves."""
    H, W = g.shape
    if H % 2 != 0:
        return g
    top = g[:H//2]
    bot = g[H//2:]
    result = np.full_like(top, bg)
    for r in range(H//2):
        for c in range(W):
            if top[r,c] != bg:
                result[r,c] = top[r,c]
            elif bot[r,c] != bg:
                result[r,c] = bot[r,c]
    return result


def or_halves_v(g, bg):
    """OR left and right halves."""
    H, W = g.shape
    if W % 2 != 0:
        return g
    left = g[:, :W//2]
    right = g[:, W//2:]
    result = np.full_like(left, bg)
    for r in range(H):
        for c in range(W//2):
            if left[r,c] != bg:
                result[r,c] = left[r,c]
            elif right[r,c] != bg:
                result[r,c] = right[r,c]
    return result


def and_halves_h(g, bg):
    """AND top and bottom halves (keep cells non-bg in both)."""
    H, W = g.shape
    if H % 2 != 0:
        return g
    top = g[:H//2]
    bot = g[H//2:]
    result = np.full_like(top, bg)
    for r in range(H//2):
        for c in range(W):
            if top[r,c] != bg and bot[r,c] != bg:
                result[r,c] = top[r,c]
    return result


def and_halves_v(g, bg):
    """AND left and right halves."""
    H, W = g.shape
    if W % 2 != 0:
        return g
    left = g[:, :W//2]
    right = g[:, W//2:]
    result = np.full_like(left, bg)
    for r in range(H):
        for c in range(W//2):
            if left[r,c] != bg and right[r,c] != bg:
                result[r,c] = left[r,c]
    return result


def fill_obj_interior(g, bg):
    """Fill interior of each object with its dominant color"""
    mask = (g != bg).astype(int)
    labeled, n = connected_components(mask)
    out = g.copy()
    H, W = g.shape
    for oid in range(1, n+1):
        obj_mask = (labeled == oid)
        rows, cols = np.where(obj_mask)
        if len(rows) == 0: continue
        color = Counter(int(g[r,c]) for r,c in zip(rows,cols)).most_common(1)[0][0]
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        for r in range(r0, r1+1):
            row_in = [c for c in range(c0, c1+1) if obj_mask[r, c]]
            if len(row_in) >= 2:
                for c in range(min(row_in), max(row_in)+1):
                    if g[r, c] == bg:
                        out[r, c] = color
    return out

def recolor_by_size(g, bg):
    """Recolor each object by its size rank (smallest=1, largest=max_color)"""
    mask = (g != bg).astype(int)
    labeled, n = connected_components(mask)
    if n == 0: return g.copy()
    sizes = [(np.sum(labeled == i), i) for i in range(1, n+1)]
    sizes.sort()
    out = np.full_like(g, bg)
    for rank, (sz, oid) in enumerate(sizes, 1):
        out[labeled == oid] = rank
    return out

def outline_objects(g, bg):
    """Keep only border cells of each object"""
    mask = (g != bg).astype(int)
    labeled, n = connected_components(mask)
    H, W = g.shape
    out = np.full_like(g, bg)
    for r in range(H):
        for c in range(W):
            if g[r, c] == bg: continue
            oid = labeled[r, c]
            is_border = any(
                not (0 <= r+dr < H and 0 <= c+dc < W) or labeled[r+dr, c+dc] != oid
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
            )
            if is_border:
                out[r, c] = g[r, c]
    return out

def flood_from_corners(g, bg):
    """Flood fill from corners - mark all bg-connected-to-corner cells"""
    from collections import deque
    H, W = g.shape
    out = g.copy()
    visited = np.zeros((H, W), dtype=bool)
    q = deque()
    for r, c in [(0,0),(0,W-1),(H-1,0),(H-1,W-1)]:
        if g[r, c] == bg:
            q.append((r, c))
            visited[r, c] = True
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and g[nr, nc] == bg:
                visited[nr, nc] = True
                q.append((nr, nc))
    # Fill non-visited bg cells with most common non-bg
    nonbg = Counter(int(v) for v in g.flatten() if int(v) != bg)
    if nonbg:
        fill = nonbg.most_common(1)[0][0]
        for r in range(H):
            for c in range(W):
                if g[r, c] == bg and not visited[r, c]:
                    out[r, c] = fill
    return out

def swap_two_colors(g, bg):
    """Swap the two most common non-bg colors"""
    nonbg = Counter(int(v) for v in g.flatten() if int(v) != bg)
    if len(nonbg) < 2: return g.copy()
    c1, c2 = [c for c, _ in nonbg.most_common(2)]
    out = g.copy()
    out[g == c1] = c2
    out[g == c2] = c1
    return out

def connect_same_color_h(g, bg):
    """Connect same-color cells horizontally (fill bg between them)"""
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for color in range(1, 10):
            cols = [c for c in range(W) if g[r, c] == color]
            if len(cols) >= 2:
                for c in range(min(cols), max(cols)+1):
                    if out[r, c] == bg:
                        out[r, c] = color
    return out

def connect_same_color_v(g, bg):
    """Connect same-color cells vertically"""
    H, W = g.shape
    out = g.copy()
    for c in range(W):
        for color in range(1, 10):
            rows = [r for r in range(H) if g[r, c] == color]
            if len(rows) >= 2:
                for r in range(min(rows), max(rows)+1):
                    if out[r, c] == bg:
                        out[r, c] = color
    return out

def dilate(g, bg):
    """Dilate non-bg cells by 1 pixel"""
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if g[r, c] != bg:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and out[nr, nc] == bg:
                        out[nr, nc] = g[r, c]
    return out

def erode(g, bg):
    """Erode non-bg cells by 1 pixel"""
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if g[r, c] != bg:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if not (0 <= nr < H and 0 <= nc < W) or g[nr, nc] == bg:
                        out[r, c] = bg
                        break
    return out

def copy_obj_to_center(g, bg):
    """Copy the single non-bg object to the center of grid"""
    rows, cols = np.where(g != bg)
    if len(rows) == 0: return g.copy()
    obj = g[rows.min():rows.max()+1, cols.min():cols.max()+1]
    H, W = g.shape
    oh, ow = obj.shape
    r0 = (H - oh) // 2
    c0 = (W - ow) // 2
    out = np.full_like(g, bg)
    out[r0:r0+oh, c0:c0+ow] = obj
    return out


def replace_color(g, bg):
    """Replace all non-bg minority color with majority non-bg color"""
    nonbg = Counter(int(v) for v in g.flatten() if int(v) != bg)
    if len(nonbg) < 2: return g.copy()
    colors = nonbg.most_common()
    majority = colors[0][0]
    minority = colors[-1][0]
    out = g.copy()
    out[g == minority] = majority
    return out

def replace_bg_neighbor(g, bg):
    """Replace bg cells that have exactly 1 non-bg neighbor with that neighbor's color"""
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if g[r, c] != bg: continue
            neighbors = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and g[nr, nc] != bg:
                    neighbors.append(g[nr, nc])
            if len(neighbors) == 1:
                out[r, c] = neighbors[0]
    return out

def fill_between_objects(g, bg):
    """Fill bg cells between same-color non-bg cells in each row"""
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for color in range(1, 10):
            cols = [c for c in range(W) if g[r, c] == color]
            if len(cols) >= 2:
                for c in range(min(cols), max(cols) + 1):
                    if out[r, c] == bg:
                        out[r, c] = color
    for c in range(W):
        for color in range(1, 10):
            rows = [r for r in range(H) if g[r, c] == color]
            if len(rows) >= 2:
                for r in range(min(rows), max(rows) + 1):
                    if out[r, c] == bg:
                        out[r, c] = color
    return out

def shrink_2x(g, bg):
    """Shrink grid by 2x (take every other cell)"""
    return g[::2, ::2].copy()

def upscale_2x(g, bg):
    """Upscale grid by 2x"""
    return np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)

def l_ray_nearest2(g, bg):
    """Each non-bg cell extends lines toward its 2 nearest grid edges"""
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if g[r, c] != bg:
                color = int(g[r, c])
                dists = [('top', r), ('bot', H-1-r), ('left', c), ('right', W-1-c)]
                dists.sort(key=lambda x: x[1])
                for edge, _ in dists[:2]:
                    if edge == 'top':
                        out[:r, c] = color
                    elif edge == 'bot':
                        out[r+1:, c] = color
                    elif edge == 'left':
                        out[r, :c] = color
                    elif edge == 'right':
                        out[r, c+1:] = color
    return out

def l_ray_nearest1(g, bg):
    """Each non-bg cell extends a line toward its nearest grid edge"""
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if g[r, c] != bg:
                color = int(g[r, c])
                dists = [('top', r), ('bot', H-1-r), ('left', c), ('right', W-1-c)]
                dists.sort(key=lambda x: x[1])
                edge = dists[0][0]
                if edge == 'top':
                    out[:r, c] = color
                elif edge == 'bot':
                    out[r+1:, c] = color
                elif edge == 'left':
                    out[r, :c] = color
                elif edge == 'right':
                    out[r, c+1:] = color
    return out

def cross_ray_full(g, bg):
    """Each non-bg cell extends lines in all 4 cardinal directions to edges"""
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if g[r, c] != bg:
                color = int(g[r, c])
                out[r, :] = np.where(out[r, :] == bg, color, out[r, :])
                out[:, c] = np.where(out[:, c] == bg, color, out[:, c])
    return out

def per_color_bbox_fill(g, bg):
    """Fill the bounding box of each non-bg color"""
    out = g.copy()
    for c in range(10):
        if c == bg:
            continue
        mask = g == c
        if not np.any(mask):
            continue
        rows, cols = np.where(mask)
        out[min(rows):max(rows)+1, min(cols):max(cols)+1] = c
    return out

def connect_same_color_hv(g, bg):
    """Connect same-color cells with lines (horizontal + vertical)"""
    out = g.copy()
    H, W = g.shape
    for color in range(10):
        if color == bg:
            continue
        pts = list(zip(*np.where(g == color)))
        if len(pts) < 2:
            continue
        row_pts = {}
        for r, c in pts:
            row_pts.setdefault(r, []).append(c)
        for r, cols in row_pts.items():
            cols.sort()
            for i in range(len(cols) - 1):
                out[r, cols[i]:cols[i+1]+1] = color
        col_pts = {}
        for r, c in pts:
            col_pts.setdefault(c, []).append(r)
        for c, rows in col_pts.items():
            rows.sort()
            for i in range(len(rows) - 1):
                out[rows[i]:rows[i+1]+1, c] = color
    return out

def remove_border_ring(g, bg):
    """Set border cells to bg"""
    H, W = g.shape
    if H < 3 or W < 3:
        return g
    out = g.copy()
    out[0, :] = bg
    out[-1, :] = bg
    out[:, 0] = bg
    out[:, -1] = bg
    return out

def keep_border_ring(g, bg):
    """Keep only border cells, set interior to bg"""
    H, W = g.shape
    out = np.full_like(g, bg)
    out[0, :] = g[0, :]
    out[-1, :] = g[-1, :]
    out[:, 0] = g[:, 0]
    out[:, -1] = g[:, -1]
    return out

def color_count_to_grid(g, bg):
    """Replace each non-bg object with a single cell of its color (compress)"""
    mask = (g != bg).astype(int)
    labeled, n = connected_components(mask)
    if n == 0: return g.copy()
    colors = []
    for oid in range(1, n + 1):
        cells = g[labeled == oid]
        colors.append(int(Counter(cells.flatten()).most_common(1)[0][0]))
    # Create 1xN grid
    out = np.array([colors], dtype=int)
    return out

def overlay_halves_h(g, bg):
    """Overlay left half onto right half (non-bg wins)"""
    H, W = g.shape
    hw = W // 2
    left = g[:, :hw]
    right = g[:, hw:hw+hw] if W % 2 == 0 else g[:, hw+1:hw+1+hw]
    if left.shape != right.shape: return g.copy()
    out = right.copy()
    mask = left != bg
    out[mask] = left[mask]
    return out

def overlay_halves_v(g, bg):
    """Overlay top half onto bottom half (non-bg wins)"""
    H, W = g.shape
    hh = H // 2
    top = g[:hh, :]
    bot = g[hh:hh+hh, :] if H % 2 == 0 else g[hh+1:hh+1+hh, :]
    if top.shape != bot.shape: return g.copy()
    out = bot.copy()
    mask = top != bg
    out[mask] = top[mask]
    return out


SAME_SHAPE_OPS_NEW = [
    ('fill_int', fill_obj_interior),
    ('outline', outline_objects),
    ('flood_corner', flood_from_corners),
    ('swap_colors', swap_two_colors),
    ('conn_h', connect_same_color_h),
    ('conn_v', connect_same_color_v),
    ('dilate', dilate),
    ('erode', erode),
    ('copy_center', copy_obj_to_center),
    ('repl_color', replace_color),
    ('repl_bg_nb', replace_bg_neighbor),
    ('fill_between', fill_between_objects),
]


SHAPE_CHANGE_OPS_NEW2 = [
    ('shrink2x', shrink_2x),
    ('upscale2x', upscale_2x),
    ('overlay_h', overlay_halves_h),
    ('overlay_v', overlay_halves_v),
]

SAME_SHAPE_OPS_V77 = [
    ('l_ray2', l_ray_nearest2),
    ('l_ray1', l_ray_nearest1),
    ('cross_ray', cross_ray_full),
    ('bbox_fill', per_color_bbox_fill),
    ('conn_hv', connect_same_color_hv),
    ('rm_border', remove_border_ring),
    ('keep_border', keep_border_ring),
]

SHAPE_CHANGE_OPS_EXTRA = [
    ('ext_minority_bbox', extract_minority_color_bbox),
    ('ext_unique_3x3', extract_unique_3x3),
    ('rm_bg_rc', remove_bg_rows_cols),
    ('ext_tl', extract_top_left_quarter),
    ('ext_tr', extract_top_right_quarter),
    ('ext_bl', extract_bottom_left_quarter),
    ('ext_br', extract_bottom_right_quarter),
    ('xor_h', xor_halves_h),
    ('xor_v', xor_halves_v),
    ('or_h', or_halves_h),
    ('or_v', or_halves_v),
    ('and_h', and_halves_h),
    ('and_v', and_halves_v),
]

ALL_OPS = SAME_SHAPE_OPS + SAME_SHAPE_OPS_NEW + SAME_SHAPE_OPS_V77 + SHAPE_CHANGE_OPS + SHAPE_CHANGE_OPS_NEW2 + SHAPE_CHANGE_OPS_EXTRA


def _compose(ops):
    """Create a composed function from a list of (name, fn) pairs"""
    def composed(grid_list, bg):
        g = grid_list
        for _, fn in ops:
            g = fn(g, bg)
        return g
    name = '+'.join(n for n, _ in ops)
    return name, composed


# ============================================================
# SEARCH ENGINE
# ============================================================

def _verify_program(fn, train_pairs, bg=None):
    """Check if fn(input) == output for all train pairs"""
    for inp_g, out_g in train_pairs:
        inp = _g(inp_g)
        out = _g(out_g)
        if bg is None:
            b = _bg(inp)
        else:
            b = bg
        
        try:
            pred = fn(inp, b)
        except Exception:
            return False
        
        if pred is None or not np.array_equal(pred, out):
            return False
    
    return True


def search_programs(train_pairs, max_depth=2, time_limit_ops=6000):
    """Search for programs that solve all train examples.
    
    Returns list of (name, fn) that work.
    Stops after finding the first valid program.
    """
    bg_vals = set()
    for inp_g, _ in train_pairs:
        bg_vals.add(_bg(_g(inp_g)))
    
    # Use most common bg
    bg = max(bg_vals, key=lambda b: sum(1 for inp_g, _ in train_pairs if _bg(_g(inp_g)) == b))
    
    # Check output shape to filter ops
    inp0, out0 = _g(train_pairs[0][0]), _g(train_pairs[0][1])
    same_shape = (inp0.shape == out0.shape)
    
    ops_tried = 0
    
    # Depth 1: single op
    ops_to_try = ALL_OPS if not same_shape else SAME_SHAPE_OPS + SHAPE_CHANGE_OPS
    
    for name, fn in ops_to_try:
        ops_tried += 1
        if _verify_program(fn, train_pairs, bg):
            return [(name, fn, bg)]
    
    if max_depth < 2:
        return []
    
    # Depth 2: compose two ops
    first_ops = (SAME_SHAPE_OPS + SAME_SHAPE_OPS_NEW) if same_shape else ALL_OPS
    second_ops = ALL_OPS
    
    for n1, f1 in first_ops:
        for n2, f2 in second_ops:
            ops_tried += 1
            if ops_tried > time_limit_ops:
                return []
            
            def composed(g, b, _f1=f1, _f2=f2):
                return _f2(_f1(g, b), b)
            
            if _verify_program(composed, train_pairs, bg):
                return [(f"{n1}+{n2}", composed, bg)]
    
    return []


def generate_search_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """Generate pieces via program search"""
    pieces = []
    if not train_pairs or len(train_pairs) < 2:
        return pieces
    
    results = search_programs(train_pairs, max_depth=2)
    
    for name, fn, bg in results:
        def apply_fn(inp_g, _fn=fn, _bg=bg):
            return _l(_fn(_g(inp_g), _bg))
        
        pieces.append(CrossPiece(name=f"search:{name}", apply_fn=apply_fn))
        return pieces  # Return first valid
    
    return pieces
