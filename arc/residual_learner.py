"""
arc/residual_learner.py — Learn transformations from input-output pixel diffs

Instead of trying to understand the "meaning" of a transform,
directly learns the diff pattern and tries to apply it.

Strategies:
1. Color substitution (color A → color B everywhere)
2. Cell-by-cell rule (neighborhood-based for changed cells)
3. Connected-component based fill (fill bboxes, fill between, etc.)
4. Row/column operations
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color


def learn_color_substitution(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: simple color replacement map (not 1:1 necessarily)"""
    for bg_try in [None]:  # Don't filter by bg
        cmap = {}
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            oh, ow = grid_shape(out)
            if (h, w) != (oh, ow):
                ok = False; break
            for r in range(h):
                for c in range(w):
                    vi, vo = inp[r][c], out[r][c]
                    if vi in cmap:
                        if cmap[vi] != vo:
                            ok = False; break
                    else:
                        cmap[vi] = vo
                if not ok: break
            if not ok: break
        
        if ok and any(k != v for k, v in cmap.items()):
            return {'type': 'color_sub', 'map': cmap}
    
    return None


def apply_color_substitution(inp: Grid, params: Dict) -> Optional[Grid]:
    cmap = params['map']
    h, w = grid_shape(inp)
    return [[cmap.get(inp[r][c], inp[r][c]) for c in range(w)] for r in range(h)]


def learn_fill_between_same_color(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: fill gaps between same-color cells along rows, cols, or both.
    
    Only fills with the SAME color as the endpoints.
    Tries multiple bg candidates + auto-bg.
    """
    bgs = _get_candidate_bgs(train_pairs) + ['auto']
    
    for bg in bgs:
        for mode in ['h', 'v', 'both']:
            ok = True
            for inp, out in train_pairs:
                pair_bg = most_common_color(inp) if bg == 'auto' else bg
                result = _apply_fill_between(inp, pair_bg, mode)
                if not grid_eq(result, out):
                    ok = False; break
            if ok:
                return {'type': 'fill_between', 'bg': bg, 'mode': mode}
    
    return None


def _apply_fill_between(inp: Grid, bg: int, mode: str) -> Grid:
    h, w = grid_shape(inp)
    result = [row[:] for row in inp]
    
    if mode in ('both', 'h'):
        for r in range(h):
            for color in range(10):
                if color == bg: continue
                positions = [c for c in range(w) if inp[r][c] == color]
                if len(positions) >= 2:
                    for i in range(len(positions) - 1):
                        for c in range(positions[i]+1, positions[i+1]):
                            if result[r][c] == bg:
                                result[r][c] = color
    
    if mode in ('both', 'v'):
        for c in range(w):
            for color in range(10):
                if color == bg: continue
                positions = [r for r in range(h) if inp[r][c] == color]
                if len(positions) >= 2:
                    for i in range(len(positions) - 1):
                        for r in range(positions[i]+1, positions[i+1]):
                            if result[r][c] == bg:
                                result[r][c] = color
    
    return result


def apply_fill_between(inp: Grid, params: Dict) -> Optional[Grid]:
    bg = most_common_color(inp) if params.get('bg') == 'auto' else params['bg']
    return _apply_fill_between(inp, bg, params['mode'])


def learn_project_rows_or_cols(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: project non-bg cells across entire row or column."""
    bgs = _get_candidate_bgs(train_pairs)
    
    for bg in bgs:
        for mode in ['row', 'col', 'both']:
            ok = True
            for inp, out in train_pairs:
                result = _apply_project(inp, bg, mode)
                if not grid_eq(result, out):
                    ok = False; break
            if ok:
                return {'type': 'project', 'bg': bg, 'mode': mode}
    
    return None


def _apply_project(inp: Grid, bg: int, mode: str) -> Grid:
    h, w = grid_shape(inp)
    result = [row[:] for row in inp]
    
    if mode in ('both', 'row'):
        for r in range(h):
            colors = [inp[r][c] for c in range(w) if inp[r][c] != bg]
            if len(set(colors)) == 1:
                color = colors[0]
                for c in range(w):
                    if result[r][c] == bg:
                        result[r][c] = color
    
    if mode in ('both', 'col'):
        for c in range(w):
            colors = [inp[r][c] for r in range(h) if inp[r][c] != bg]
            if len(set(colors)) == 1:
                color = colors[0]
                for r in range(h):
                    if result[r][c] == bg:
                        result[r][c] = color
    
    return result


def apply_project(inp: Grid, params: Dict) -> Optional[Grid]:
    return _apply_project(inp, params['bg'], params['mode'])


def learn_cross_from_dots(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: draw cross lines (full row + col) from each non-bg cell."""
    bgs = _get_candidate_bgs(train_pairs)
    
    for bg in bgs:
        for fill_mode in ['same_color', 'single_color']:
            ok = True
            fill_color = None
            
            for inp, out in train_pairs:
                result = _apply_cross_dots(inp, bg, fill_mode, fill_color)
                if fill_mode == 'single_color' and fill_color is None:
                    # Infer fill color from first changed cell
                    h, w = grid_shape(inp)
                    for r in range(h):
                        for c in range(w):
                            if inp[r][c] == bg and out[r][c] != bg:
                                fill_color = out[r][c]
                                break
                        if fill_color is not None:
                            break
                    result = _apply_cross_dots(inp, bg, fill_mode, fill_color)
                
                if not grid_eq(result, out):
                    ok = False; break
            
            if ok:
                return {'type': 'cross_dots', 'bg': bg, 'fill_mode': fill_mode, 'fill_color': fill_color}
    
    return None


def _apply_cross_dots(inp: Grid, bg: int, fill_mode: str, fill_color: Optional[int]) -> Grid:
    h, w = grid_shape(inp)
    dots = [(r, c, inp[r][c]) for r in range(h) for c in range(w) if inp[r][c] != bg]
    
    result = [row[:] for row in inp]
    for dr, dc, dot_color in dots:
        color = dot_color if fill_mode == 'same_color' else (fill_color or dot_color)
        for r in range(h):
            if result[r][dc] == bg:
                result[r][dc] = color
        for c in range(w):
            if result[dr][c] == bg:
                result[dr][c] = color
    
    return result


def apply_cross_dots(inp: Grid, params: Dict) -> Optional[Grid]:
    return _apply_cross_dots(inp, params['bg'], params['fill_mode'], params.get('fill_color'))


def learn_expand_objects(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: expand each non-bg object to fill its bounding box."""
    bgs = _get_candidate_bgs(train_pairs)
    
    for bg in bgs:
        ok = True
        for inp, out in train_pairs:
            result = _apply_expand_objects(inp, bg)
            if not grid_eq(result, out):
                ok = False; break
        if ok:
            return {'type': 'expand_objects', 'bg': bg}
    
    return None


def _apply_expand_objects(inp: Grid, bg: int) -> Grid:
    """Fill bounding box of each connected component with its color."""
    from arc.objects import detect_objects
    h, w = grid_shape(inp)
    result = [row[:] for row in inp]
    objs = detect_objects(inp, bg)
    for obj in objs:
        r1, c1, r2, c2 = obj.bbox
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if result[r][c] == bg:
                    result[r][c] = obj.color
    return result


def apply_expand_objects(inp: Grid, params: Dict) -> Optional[Grid]:
    return _apply_expand_objects(inp, params['bg'])


def learn_mirror_across_axis(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: mirror non-bg content across a separator line."""
    bgs = _get_candidate_bgs(train_pairs)
    
    for bg in bgs:
        h0, w0 = grid_shape(train_pairs[0][0])
        
        # Find separator rows (single non-bg color full row)
        for sep_r in range(h0):
            row = train_pairs[0][0][sep_r]
            if len(set(row)) == 1 and row[0] != bg:
                # Try mirror across this row
                ok = True
                for inp, out in train_pairs:
                    h, w = grid_shape(inp)
                    result = [r[:] for r in inp]
                    for r in range(h):
                        mr = 2 * sep_r - r
                        if 0 <= mr < h and r != sep_r:
                            for c in range(w):
                                if result[r][c] == bg and inp[mr][c] != bg and inp[mr][c] != inp[sep_r][0]:
                                    result[r][c] = inp[mr][c]
                                if result[mr][c] == bg and inp[r][c] != bg and inp[r][c] != inp[sep_r][0]:
                                    result[mr][c] = inp[r][c]
                    if not grid_eq(result, out):
                        ok = False; break
                if ok:
                    return {'type': 'mirror_row', 'bg': bg, 'sep_r': sep_r}
        
        # Find separator cols
        for sep_c in range(w0):
            col = [train_pairs[0][0][r][sep_c] for r in range(h0)]
            if len(set(col)) == 1 and col[0] != bg:
                ok = True
                for inp, out in train_pairs:
                    h, w = grid_shape(inp)
                    result = [r[:] for r in inp]
                    for c in range(w):
                        mc = 2 * sep_c - c
                        if 0 <= mc < w and c != sep_c:
                            for r in range(h):
                                if result[r][c] == bg and inp[r][mc] != bg and inp[r][mc] != inp[0][sep_c]:
                                    result[r][c] = inp[r][mc]
                                if result[r][mc] == bg and inp[r][c] != bg and inp[r][c] != inp[0][sep_c]:
                                    result[r][mc] = inp[r][c]
                    if not grid_eq(result, out):
                        ok = False; break
                if ok:
                    return {'type': 'mirror_col', 'bg': bg, 'sep_c': sep_c}
    
    return None


def apply_mirror_across_axis(inp: Grid, params: Dict) -> Optional[Grid]:
    bg = params['bg']
    h, w = grid_shape(inp)
    result = [r[:] for r in inp]
    
    if params['type'] == 'mirror_row':
        sep_r = params['sep_r']
        sep_color = inp[sep_r][0]
        for r in range(h):
            mr = 2 * sep_r - r
            if 0 <= mr < h and r != sep_r:
                for c in range(w):
                    if result[r][c] == bg and inp[mr][c] != bg and inp[mr][c] != sep_color:
                        result[r][c] = inp[mr][c]
                    if result[mr][c] == bg and inp[r][c] != bg and inp[r][c] != sep_color:
                        result[mr][c] = inp[r][c]
    
    elif params['type'] == 'mirror_col':
        sep_c = params['sep_c']
        sep_color = inp[0][sep_c]
        for c in range(w):
            mc = 2 * sep_c - c
            if 0 <= mc < w and c != sep_c:
                for r in range(h):
                    if result[r][c] == bg and inp[r][mc] != bg and inp[r][mc] != sep_color:
                        result[r][c] = inp[r][mc]
                    if result[r][mc] == bg and inp[r][c] != bg and inp[r][c] != sep_color:
                        result[r][mc] = inp[r][c]
    
    return result


def _get_candidate_bgs(train_pairs: List[Tuple[Grid, Grid]]) -> List[int]:
    """Get candidate bg colors from most common in inputs."""
    counts = Counter()
    for inp, _ in train_pairs:
        for row in inp:
            counts.update(row)
    bgs = [c for c, _ in counts.most_common(3)]
    if 0 not in bgs:
        bgs.append(0)
    return bgs


# ============================================================
# Master learner: try all strategies
# ============================================================

ALL_LEARNERS = [
    ('color_sub', learn_color_substitution, apply_color_substitution),
    ('fill_between', learn_fill_between_same_color, apply_fill_between),
    ('project', learn_project_rows_or_cols, apply_project),
    ('cross_dots', learn_cross_from_dots, apply_cross_dots),
    ('expand_obj', learn_expand_objects, apply_expand_objects),
    ('mirror_axis', learn_mirror_across_axis, apply_mirror_across_axis),
]


def learn_residual(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Tuple[str, Dict, callable]]:
    """Try all residual learners, return first that works."""
    for name, learn_fn, apply_fn in ALL_LEARNERS:
        try:
            rule = learn_fn(train_pairs)
            if rule is not None:
                # Double-verify
                ok = True
                for inp, out in train_pairs:
                    result = apply_fn(inp, rule)
                    if result is None or not grid_eq(result, out):
                        ok = False; break
                if ok:
                    return (name, rule, apply_fn)
        except Exception:
            pass
    return None


def learn_cross_with_marker(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: draw cross from dots, replace dot with marker color, add diag color.
    Uses auto-bg (most common color per pair) to handle varying backgrounds."""
    ok = True
    marker_color = None
    diag_color = None
    dot_color = None
    
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        if grid_shape(out) != (h, w):
            return None
        
        bg = most_common_color(inp)
        dots = [(r, c) for r in range(h) for c in range(w) if inp[r][c] != bg]
        if not dots:
            return None
        
        dc = inp[dots[0][0]][dots[0][1]]
        if dot_color is None:
            dot_color = dc
        elif dot_color != dc:
            return None
        
        # Build expected result: draw crosses
        result = [row[:] for row in inp]
        for dr, dcc in dots:
            for r in range(h):
                if result[r][dcc] == bg:
                    result[r][dcc] = dot_color
            for c in range(w):
                if result[dr][c] == bg:
                    result[dr][c] = dot_color
        
        # Detect marker and diag from output
        for dr, dcc in dots:
            mc = out[dr][dcc]
            if mc != dot_color:
                if marker_color is None:
                    marker_color = mc
                elif marker_color != mc:
                    return None
            
            for ddr, ddcc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = dr+ddr, dcc+ddcc
                if 0 <= nr < h and 0 <= nc < w:
                    ov = out[nr][nc]
                    if ov != dot_color and ov != bg and ov != (marker_color or -1):
                        if diag_color is None:
                            diag_color = ov
                        elif diag_color != ov:
                            return None
        
        # Apply marker + diag
        for dr, dcc in dots:
            if marker_color:
                result[dr][dcc] = marker_color
            if diag_color:
                for ddr, ddcc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = dr+ddr, dcc+ddcc
                    if 0 <= nr < h and 0 <= nc < w and result[nr][nc] == bg:
                        result[nr][nc] = diag_color
        
        if not grid_eq(result, out):
            return None
    
    if dot_color is None:
        return None
    return {'type': 'cross_marker', 'bg': 'auto', 'dot_color': dot_color,
            'marker_color': marker_color, 'diag_color': diag_color}


def apply_cross_with_marker(inp: Grid, params: Dict) -> Optional[Grid]:
    dot_color = params['dot_color']
    marker_color = params.get('marker_color')
    diag_color = params.get('diag_color')
    bg = most_common_color(inp) if params.get('bg') == 'auto' else params.get('bg', 0)
    
    h, w = grid_shape(inp)
    dots = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == dot_color]
    
    result = [row[:] for row in inp]
    for dr, dc in dots:
        for r in range(h):
            if result[r][dc] == bg:
                result[r][dc] = dot_color
        for c in range(w):
            if result[dr][c] == bg:
                result[dr][c] = dot_color
    
    for dr, dc in dots:
        if marker_color is not None:
            result[dr][dc] = marker_color
        if diag_color is not None:
            for ddr, ddc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = dr+ddr, dc+ddc
                if 0 <= nr < h and 0 <= nc < w and result[nr][nc] == bg:
                    result[nr][nc] = diag_color
    
    return result


# Add to ALL_LEARNERS
ALL_LEARNERS.append(('cross_marker', learn_cross_with_marker, apply_cross_with_marker))
