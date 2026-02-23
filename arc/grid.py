"""
arc/grid.py — Grid representation and basic operations for ARC-AGI-2

Grid = 2D list of ints (0-9, where 0 = black/background)
Colors: 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=grey, 6=magenta, 7=orange, 8=azure, 9=maroon
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
import copy


Grid = List[List[int]]


@dataclass
class GridInfo:
    """Static analysis of a grid — Verantyx IR for ARC"""
    height: int
    width: int
    colors: Set[int]  # unique colors present
    bg_color: int  # most common color (background)
    color_counts: dict  # color → count
    regions: List[dict] = field(default_factory=list)  # connected components
    symmetry: dict = field(default_factory=dict)  # h_sym, v_sym, rot90, rot180
    adjacency: List[Tuple] = field(default_factory=list)  # region adjacency pairs
    periodicity: dict = field(default_factory=dict)  # row/col periodicity
    cell_neighbors: dict = field(default_factory=dict)  # per-cell neighbor color info


def parse_grid(raw: List[List[int]]) -> Grid:
    """Ensure grid is well-formed"""
    return [list(row) for row in raw]


def grid_shape(g: Grid) -> Tuple[int, int]:
    """(height, width)"""
    if not g:
        return (0, 0)
    return (len(g), len(g[0]))


def grid_colors(g: Grid) -> Set[int]:
    return {c for row in g for c in row}


def grid_eq(a: Grid, b: Grid) -> bool:
    if grid_shape(a) != grid_shape(b):
        return False
    return all(a[r][c] == b[r][c] for r in range(len(a)) for c in range(len(a[0])))


# ── Geometric transforms ──

def rotate_90(g: Grid) -> Grid:
    """Rotate 90° clockwise"""
    h, w = grid_shape(g)
    return [[g[h - 1 - r][c] for r in range(h)] for c in range(w)]


def rotate_180(g: Grid) -> Grid:
    return rotate_90(rotate_90(g))


def rotate_270(g: Grid) -> Grid:
    return rotate_90(rotate_90(rotate_90(g)))


def flip_h(g: Grid) -> Grid:
    """Flip horizontally (left-right)"""
    return [row[::-1] for row in g]


def flip_v(g: Grid) -> Grid:
    """Flip vertically (top-bottom)"""
    return g[::-1]


def transpose(g: Grid) -> Grid:
    h, w = grid_shape(g)
    return [[g[r][c] for r in range(h)] for c in range(w)]


# ── Tiling ──

def tile(g: Grid, repeat_h: int, repeat_w: int) -> Grid:
    """Tile grid repeat_h × repeat_w"""
    h, w = grid_shape(g)
    result = []
    for rr in range(repeat_h):
        for r in range(h):
            row = []
            for cr in range(repeat_w):
                row.extend(g[r])
            result.append(row)
    return result


def tile_with_flip(g: Grid, repeat_h: int, repeat_w: int) -> Grid:
    """Tile with alternating row flips"""
    h, w = grid_shape(g)
    g_fh = flip_h(g)
    
    result = []
    for rr in range(repeat_h):
        src = g if rr % 2 == 0 else g_fh
        for r in range(h):
            row = list(src[r]) * repeat_w
            result.append(row)
    return result


def tile_checkerboard(g: Grid, repeat_h: int, repeat_w: int) -> Grid:
    """Tile with checkerboard flip pattern"""
    h, w = grid_shape(g)
    g_fh = flip_h(g)
    g_fv = flip_v(g)
    g_fhv = flip_h(flip_v(g))
    variants = [[g, g_fh], [g_fv, g_fhv]]
    
    result = []
    for rr in range(repeat_h):
        for r in range(h):
            row = []
            for cr in range(repeat_w):
                src = variants[rr % 2][cr % 2]
                row.extend(src[r])
            result.append(row)
    return result


# ── Color operations ──

def recolor(g: Grid, color_map: dict) -> Grid:
    """Apply color mapping"""
    return [[color_map.get(c, c) for c in row] for row in g]


def mask_color(g: Grid, color: int) -> Grid:
    """Keep only specified color, rest → 0"""
    return [[c if c == color else 0 for c in row] for row in g]


# ── Region extraction ──

def flood_fill_regions(g: Grid, ignore_bg: bool = True) -> List[dict]:
    """Find connected regions (4-connected) in grid"""
    h, w = grid_shape(g)
    if h == 0:
        return []
    
    bg = most_common_color(g) if ignore_bg else -1
    visited = [[False] * w for _ in range(h)]
    regions = []
    
    for r in range(h):
        for c in range(w):
            if visited[r][c] or (ignore_bg and g[r][c] == bg):
                continue
            # BFS
            color = g[r][c]
            cells = []
            queue = [(r, c)]
            visited[r][c] = True
            while queue:
                cr, cc = queue.pop(0)
                cells.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] == color:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
            
            min_r = min(r for r, c in cells)
            max_r = max(r for r, c in cells)
            min_c = min(c for r, c in cells)
            max_c = max(c for r, c in cells)
            
            regions.append({
                'color': color,
                'cells': cells,
                'size': len(cells),
                'bbox': (min_r, min_c, max_r, max_c),
                'bbox_h': max_r - min_r + 1,
                'bbox_w': max_c - min_c + 1,
            })
    
    return sorted(regions, key=lambda r: r['size'], reverse=True)


def most_common_color(g: Grid) -> int:
    """Most frequent color in grid"""
    counts = {}
    for row in g:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    return max(counts, key=counts.get) if counts else 0


def extract_subgrid(g: Grid, r1: int, c1: int, r2: int, c2: int) -> Grid:
    """Extract subgrid [r1:r2+1, c1:c2+1]"""
    return [row[c1:c2 + 1] for row in g[r1:r2 + 1]]


def place_subgrid(g: Grid, sub: Grid, r: int, c: int) -> Grid:
    """Place subgrid at position (r, c), returns new grid"""
    result = copy.deepcopy(g)
    for sr in range(len(sub)):
        for sc in range(len(sub[0])):
            tr, tc = r + sr, c + sc
            if 0 <= tr < len(result) and 0 <= tc < len(result[0]):
                result[tr][tc] = sub[sr][sc]
    return result


# ── Symmetry detection ──

def check_symmetry(g: Grid) -> dict:
    """Check various symmetries"""
    return {
        'h_sym': grid_eq(g, flip_h(g)),
        'v_sym': grid_eq(g, flip_v(g)),
        'rot90': grid_eq(g, rotate_90(g)),
        'rot180': grid_eq(g, rotate_180(g)),
        'transpose': grid_eq(g, transpose(g)),
    }


# ── Analyze grid ──

def detect_periodicity(g: Grid) -> dict:
    """Detect row/column periodicity"""
    h, w = grid_shape(g)
    result = {'row_period': 0, 'col_period': 0}
    
    # Row periodicity: smallest p where row[i] == row[i+p] for all valid i
    for p in range(1, h):
        match = all(g[r] == g[r + p] for r in range(h - p))
        if match:
            result['row_period'] = p
            break
    
    # Column periodicity
    for p in range(1, w):
        match = all(g[r][c] == g[r][c + p] for r in range(h) for c in range(w - p))
        if match:
            result['col_period'] = p
            break
    
    return result


def region_adjacency(regions: List[dict], h: int, w: int) -> List[Tuple]:
    """Find which regions are adjacent (share a border)"""
    # Build cell→region_idx map
    cell_to_reg = {}
    for idx, reg in enumerate(regions):
        for r, c in reg['cells']:
            cell_to_reg[(r, c)] = idx
    
    adj = set()
    for (r, c), idx in cell_to_reg.items():
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nidx = cell_to_reg.get((nr, nc))
            if nidx is not None and nidx != idx:
                adj.add((min(idx, nidx), max(idx, nidx)))
    
    return sorted(adj)


def analyze(g: Grid) -> GridInfo:
    """Full static analysis of grid — Verantyx IR"""
    h, w = grid_shape(g)
    colors = grid_colors(g)
    counts = {}
    for row in g:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    bg = max(counts, key=counts.get) if counts else 0
    regions = flood_fill_regions(g)
    
    return GridInfo(
        height=h,
        width=w,
        colors=colors,
        bg_color=bg,
        color_counts=counts,
        regions=regions,
        symmetry=check_symmetry(g),
        adjacency=region_adjacency(regions, h, w),
        periodicity=detect_periodicity(g),
    )


def grid_to_str(g: Grid) -> str:
    """Pretty-print grid"""
    color_chars = '⬛🔵🔴🟢🟡⬜🟣🟠🔷🟤'
    lines = []
    for row in g:
        lines.append(''.join(color_chars[c] if c < 10 else str(c) for c in row))
    return '\n'.join(lines)
