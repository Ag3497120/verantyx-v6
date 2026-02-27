"""
arc/cross_classifier.py — Cross-structure based task classification

Instead of LLM-based surface classification, classify tasks by their
cross-structural properties:
1. Cross worlds: how many independent color regions exist
2. Cross probe fill: what happens when measurement probes are injected
3. Cross connectivity: how objects connect/disconnect between input→output
4. Cross symmetry: what symmetries the cross structure exhibits
5. Cross size relation: input→output size mapping
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter, deque
from arc.grid import Grid, grid_shape, grid_eq, most_common_color


def _detect_objects(grid: Grid, bg: int) -> List[Dict]:
    """Detect connected components (4-connected, same color)."""
    h, w = grid_shape(grid)
    visited: Set[Tuple[int, int]] = set()
    objects = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and (r, c) not in visited:
                color = grid[r][c]
                comp: Set[Tuple[int, int]] = set()
                q = deque([(r, c)])
                while q:
                    cr, cc = q.popleft()
                    if (cr, cc) in comp or cr < 0 or cr >= h or cc < 0 or cc >= w:
                        continue
                    if grid[cr][cc] != color:
                        continue
                    comp.add((cr, cc))
                    visited.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        q.append((cr + dr, cc + dc))
                rs = [r for r, c in comp]
                cs = [c for r, c in comp]
                objects.append({
                    'color': color,
                    'cells': comp,
                    'size': len(comp),
                    'bbox': (min(rs), min(cs), max(rs), max(cs)),
                    'center': ((min(rs) + max(rs)) / 2, (min(cs) + max(cs)) / 2),
                })
    return objects


def _detect_mc_objects(grid: Grid, bg: int) -> List[Dict]:
    """Detect multicolor connected components."""
    h, w = grid_shape(grid)
    visited: Set[Tuple[int, int]] = set()
    objects = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and (r, c) not in visited:
                comp: Set[Tuple[int, int]] = set()
                q = deque([(r, c)])
                while q:
                    cr, cc = q.popleft()
                    if (cr, cc) in comp or cr < 0 or cr >= h or cc < 0 or cc >= w:
                        continue
                    if grid[cr][cc] == bg:
                        continue
                    comp.add((cr, cc))
                    visited.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        q.append((cr + dr, cc + dc))
                colors = Counter(grid[r][c] for r, c in comp)
                rs = [r for r, c in comp]
                cs = [c for r, c in comp]
                objects.append({
                    'colors': dict(colors),
                    'cells': comp,
                    'size': len(comp),
                    'bbox': (min(rs), min(cs), max(rs), max(cs)),
                    'n_colors': len(colors),
                })
    return objects


def _cross_probe_fill(grid: Grid, bg: int) -> Dict:
    """
    Inject cross probes and measure structural properties.
    Returns cross-structural features of the grid.
    """
    h, w = grid_shape(grid)
    
    # Feature: row/column completeness
    full_rows = sum(1 for r in range(h) if all(grid[r][c] != bg for c in range(w)))
    full_cols = sum(1 for c in range(w) if all(grid[r][c] != bg for r in range(h)))
    
    # Feature: separator lines (single-color full rows/cols)
    sep_rows = []
    for r in range(h):
        row_colors = set(grid[r])
        if len(row_colors) == 1 and grid[r][0] != bg:
            sep_rows.append((r, grid[r][0]))
    
    sep_cols = []
    for c in range(w):
        col_colors = set(grid[r][c] for r in range(h))
        if len(col_colors) == 1 and grid[0][c] != bg:
            sep_cols.append((c, grid[0][c]))
    
    # Feature: enclosed regions (bg cells not reachable from edge)
    reachable = set()
    edge_q = deque()
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and grid[r][c] == bg:
                if (r, c) not in reachable:
                    reachable.add((r, c))
                    edge_q.append((r, c))
    while edge_q:
        cr, cc = edge_q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reachable and grid[nr][nc] == bg:
                reachable.add((nr, nc))
                edge_q.append((nr, nc))
    
    total_bg = sum(1 for r in range(h) for c in range(w) if grid[r][c] == bg)
    enclosed_bg = total_bg - len(reachable)
    
    # Feature: symmetry
    h_sym = all(grid[r][c] == grid[r][w - 1 - c] for r in range(h) for c in range(w // 2) if grid[r][c] != bg)
    v_sym = all(grid[r][c] == grid[h - 1 - r][c] for r in range(h // 2) for c in range(w) if grid[r][c] != bg)
    
    return {
        'full_rows': full_rows,
        'full_cols': full_cols,
        'sep_rows': sep_rows,
        'sep_cols': sep_cols,
        'enclosed_bg': enclosed_bg,
        'h_sym': h_sym,
        'v_sym': v_sym,
    }


def classify_task_cross(task_data: dict) -> Dict:
    """
    Classify a task by its cross-structural properties.
    Returns a feature dict for clustering.
    """
    train = task_data['train']
    inp0 = train[0]['input']
    out0 = train[0]['output']
    
    ih, iw = grid_shape(inp0)
    oh, ow = grid_shape(out0)
    bg_in = most_common_color(inp0)
    bg_out = most_common_color(out0)
    
    # === Size relation ===
    if (ih, iw) == (oh, ow):
        size_rel = 'same'
    elif oh < ih and ow < iw:
        size_rel = 'shrink'
    elif oh > ih and ow > iw:
        size_rel = 'grow'
    elif oh * ow < ih * iw:
        size_rel = 'shrink_mixed'
    else:
        size_rel = 'grow_mixed'
    
    # === Cross worlds ===
    in_objs = _detect_objects(inp0, bg_in)
    out_objs = _detect_objects(out0, bg_out)
    in_mc = _detect_mc_objects(inp0, bg_in)
    out_mc = _detect_mc_objects(out0, bg_out)
    
    in_colors = set(c for row in inp0 for c in row if c != bg_in)
    out_colors = set(c for row in out0 for c in row if c != bg_out)
    
    # Cross world count = number of distinct color groups
    n_in_worlds = len(in_colors)
    n_out_worlds = len(out_colors)
    new_colors = out_colors - in_colors
    lost_colors = in_colors - out_colors
    
    # === Cross probe features ===
    in_probe = _cross_probe_fill(inp0, bg_in)
    out_probe = _cross_probe_fill(out0, bg_out)
    
    # === Connectivity change ===
    n_in_objs = len(in_objs)
    n_out_objs = len(out_objs)
    n_in_mc = len(in_mc)
    n_out_mc = len(out_mc)
    
    if n_out_mc < n_in_mc:
        connectivity = 'merge'  # objects merge
    elif n_out_mc > n_in_mc:
        connectivity = 'split'  # objects split/create
    else:
        connectivity = 'same'
    
    # === Object movement detection ===
    static_count = sum(1 for io in in_objs if any(
        io['cells'] == oo['cells'] and io['color'] == oo['color'] for oo in out_objs
    ))
    moved_count = n_in_objs - static_count
    
    # === Nonbg cell change ===
    in_nonbg = sum(1 for r in range(ih) for c in range(iw) if inp0[r][c] != bg_in)
    out_nonbg = sum(1 for r in range(oh) for c in range(ow) if out0[r][c] != bg_out)
    
    if out_nonbg > in_nonbg * 1.5:
        fill_change = 'expand'
    elif out_nonbg < in_nonbg * 0.67:
        fill_change = 'contract'
    else:
        fill_change = 'stable'
    
    # === Wall/separator structure ===
    has_walls = len(in_probe['sep_rows']) >= 2 or len(in_probe['sep_cols']) >= 2
    has_separator = (1 <= len(in_probe['sep_rows']) <= 3) or (1 <= len(in_probe['sep_cols']) <= 3)
    has_enclosure = in_probe['enclosed_bg'] > 0
    
    # === Cross classification ===
    cross_type = _determine_cross_type(
        size_rel, connectivity, fill_change, 
        n_in_worlds, new_colors, has_walls, has_separator, has_enclosure,
        moved_count, static_count, n_in_objs,
        in_probe, out_probe
    )
    
    return {
        'cross_type': cross_type,
        'size_rel': size_rel,
        'connectivity': connectivity,
        'fill_change': fill_change,
        'n_in_worlds': n_in_worlds,
        'n_out_worlds': n_out_worlds,
        'new_colors': len(new_colors),
        'lost_colors': len(lost_colors),
        'n_in_objs': n_in_objs,
        'n_out_objs': n_out_objs,
        'n_in_mc': n_in_mc,
        'n_out_mc': n_out_mc,
        'static_objs': static_count,
        'moved_objs': moved_count,
        'has_walls': has_walls,
        'has_separator': has_separator,
        'has_enclosure': has_enclosure,
        'in_sym_h': in_probe['h_sym'],
        'in_sym_v': in_probe['v_sym'],
        'out_sym_h': out_probe['h_sym'],
        'out_sym_v': out_probe['v_sym'],
        'in_nonbg': in_nonbg,
        'out_nonbg': out_nonbg,
    }


def _determine_cross_type(size_rel, connectivity, fill_change,
                           n_worlds, new_colors, has_walls, has_separator, has_enclosure,
                           moved_count, static_count, n_objs,
                           in_probe, out_probe) -> str:
    """Determine the cross-structural type of the task."""
    
    # A: Wall absorption — 2+ color worlds with wall boundaries
    if has_walls and n_worlds >= 2 and size_rel == 'same':
        return 'A_wall_absorb'
    
    # B: Separator panel — grid divided by separators into panels
    if has_separator and size_rel in ('same', 'shrink', 'shrink_mixed'):
        return 'B_separator_panel'
    
    # C: Object extraction — output is smaller, extracting a subgrid
    if size_rel in ('shrink', 'shrink_mixed') and not has_separator:
        return 'C_object_extract'
    
    # D: Cross expansion — objects expand, nonbg grows significantly
    if fill_change == 'expand' and size_rel == 'same':
        if has_enclosure:
            return 'D1_fill_enclosed'
        return 'D2_cross_expand'
    
    # E: Object movement — objects move but count stays similar
    if size_rel == 'same' and moved_count > 0 and fill_change == 'stable':
        if static_count > 0:
            return 'E1_anchor_slide'
        return 'E2_free_movement'
    
    # F: Merge/split — object count changes significantly
    if connectivity == 'merge':
        return 'F1_cross_merge'
    if connectivity == 'split':
        return 'F2_cross_split'
    
    # G: Recolor — same structure, different colors
    if size_rel == 'same' and fill_change == 'stable' and moved_count == 0:
        if new_colors:
            return 'G1_cross_recolor_new'
        return 'G2_cross_recolor'
    
    # H: Scale/tile — output is larger
    if size_rel in ('grow', 'grow_mixed'):
        return 'H_cross_scale'
    
    # I: Complex — doesn't fit simple categories
    return 'I_complex'


def classify_all_unsolved(task_dir: str, solved_ids: Set[str]) -> Dict[str, List[str]]:
    """Classify all unsolved tasks by cross type."""
    import os, json
    
    categories: Dict[str, List[str]] = {}
    
    for fname in sorted(os.listdir(task_dir)):
        if not fname.endswith('.json'):
            continue
        tid = fname.replace('.json', '')
        if tid in solved_ids:
            continue
        
        with open(os.path.join(task_dir, fname)) as f:
            task = json.load(f)
        
        try:
            features = classify_task_cross(task)
            ct = features['cross_type']
            categories.setdefault(ct, []).append(tid)
        except Exception:
            categories.setdefault('X_error', []).append(tid)
    
    return categories
