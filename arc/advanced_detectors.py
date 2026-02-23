"""
arc/advanced_detectors.py — Advanced ARC-AGI-2 transform detectors

1. Conditional fill — color/position-based cell changes
2. Region-based transforms — per-region operations
3. Pattern completion — symmetry-based fill
4. Count-based — color count / region size based transforms
5. Composite rules — multi-atom compositions

Each detector returns List[TransformAtom] and has a corresponding apply function.
All detectors verify against ALL training pairs (Verantyx: verify before answer).
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Set
from arc.grid import (
    Grid, grid_shape, grid_eq, grid_colors,
    most_common_color, flood_fill_regions, extract_subgrid, place_subgrid,
    analyze, flip_h, flip_v, rotate_90, rotate_180, rotate_270, transpose,
    check_symmetry, GridInfo,
)
from arc.pattern_atoms import TransformAtom
import copy


# ═══════════════════════════════════════════════════════════
# 1. CONDITIONAL FILL
# ═══════════════════════════════════════════════════════════

def detect_conditional_fill(inp: Grid, out: Grid) -> List[TransformAtom]:
    """Detect rules like: if cell==X, change to Y; if neighbor==Z, change to W"""
    atoms = []
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    
    if (ih, iw) != (oh, ow):
        return atoms
    
    # Collect all cell changes
    changes = []
    for r in range(ih):
        for c in range(iw):
            if inp[r][c] != out[r][c]:
                neighbors = _get_neighbors(inp, r, c)
                changes.append({
                    'r': r, 'c': c,
                    'old': inp[r][c], 'new': out[r][c],
                    'neighbors': neighbors,
                })
    
    if not changes:
        return atoms
    
    # --- Rule 1: Simple color replacement (old→new, conditional on neighbors) ---
    # Group by (old_color, new_color)
    by_transition = {}
    for ch in changes:
        key = (ch['old'], ch['new'])
        by_transition.setdefault(key, []).append(ch)
    
    for (old_c, new_c), group in by_transition.items():
        if len(group) == len(changes):
            # All changes are same transition
            atoms.append(TransformAtom(
                'conditional', 'color_replace',
                {'old': old_c, 'new': new_c},
                0.85, f'Replace color {old_c} → {new_c}'
            ))
    
    # --- Rule 2: Fill based on neighbor presence ---
    # Check if changed cells all have a specific neighbor color
    for (old_c, new_c), group in by_transition.items():
        for check_color in range(10):
            if all(check_color in ch['neighbors'].values() for ch in group):
                # Verify: unchanged cells with same old_color do NOT have this neighbor
                unchanged_with_old = []
                for r in range(ih):
                    for c in range(iw):
                        if inp[r][c] == old_c and out[r][c] == old_c:
                            nb = _get_neighbors(inp, r, c)
                            unchanged_with_old.append(check_color in nb.values())
                
                if unchanged_with_old and not any(unchanged_with_old):
                    atoms.append(TransformAtom(
                        'conditional', 'neighbor_fill',
                        {'old': old_c, 'new': new_c, 'neighbor_color': check_color},
                        0.9, f'If cell={old_c} and has neighbor={check_color}, fill with {new_c}'
                    ))
    
    # --- Rule 3: Fill background between colored regions ---
    bg = most_common_color(inp)
    if all(ch['old'] == bg for ch in changes):
        atoms.append(TransformAtom(
            'conditional', 'fill_between',
            {'bg': bg, 'fill_positions': [(ch['r'], ch['c'], ch['new']) for ch in changes]},
            0.6, f'Fill background cells with pattern'
        ))
    
    # --- Rule 4: Gravity / flood fill direction ---
    bg = most_common_color(inp)
    non_bg_colors = grid_colors(inp) - {bg}
    for color in non_bg_colors:
        # Check if color "falls down" (gravity)
        in_positions = [(r, c) for r in range(ih) for c in range(iw) if inp[r][c] == color]
        out_positions = [(r, c) for r in range(oh) for c in range(ow) if out[r][c] == color]
        
        if in_positions and out_positions:
            # Same columns, rows shifted down?
            in_cols = set(c for _, c in in_positions)
            out_cols = set(c for _, c in out_positions)
            if in_cols == out_cols:
                # Check gravity: for each column, are output positions at bottom?
                gravity = True
                for col in in_cols:
                    in_rows = sorted(r for r, c in in_positions if c == col)
                    out_rows = sorted(r for r, c in out_positions if c == col)
                    if len(in_rows) != len(out_rows):
                        gravity = False
                        break
                    # Output should be at bottom
                    expected = list(range(ih - len(out_rows), ih))
                    if out_rows != expected:
                        gravity = False
                        break
                
                if gravity and len(in_positions) > 1:
                    atoms.append(TransformAtom(
                        'conditional', 'gravity_down',
                        {'color': color, 'bg': bg},
                        0.9, f'Gravity: color {color} falls to bottom'
                    ))
    
    return atoms


def apply_conditional_fill(atom: TransformAtom, inp: Grid) -> Optional[Grid]:
    """Apply a conditional fill transform"""
    op = atom.operation
    params = atom.params
    h, w = grid_shape(inp)
    result = copy.deepcopy(inp)
    
    if op == 'color_replace':
        old, new = params['old'], params['new']
        for r in range(h):
            for c in range(w):
                if result[r][c] == old:
                    result[r][c] = new
        return result
    
    elif op == 'neighbor_fill':
        old, new, nc = params['old'], params['new'], params['neighbor_color']
        for r in range(h):
            for c in range(w):
                if inp[r][c] == old:
                    neighbors = _get_neighbors(inp, r, c)
                    if nc in neighbors.values():
                        result[r][c] = new
        return result
    
    elif op == 'gravity_down':
        color, bg = params['color'], params['bg']
        for c in range(w):
            count = sum(1 for r in range(h) if inp[r][c] == color)
            for r in range(h):
                if inp[r][c] == color:
                    result[r][c] = bg
            for i in range(count):
                result[h - 1 - i][c] = color
        return result
    
    return None


# ═══════════════════════════════════════════════════════════
# 2. REGION-BASED TRANSFORMS
# ═══════════════════════════════════════════════════════════

def detect_region_transforms(inp: Grid, out: Grid) -> List[TransformAtom]:
    """Detect per-region operations: move, copy, transform individual regions"""
    atoms = []
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    
    bg = most_common_color(inp)
    in_regions = flood_fill_regions(inp)
    out_regions = flood_fill_regions(out)
    
    if not in_regions:
        return atoms
    
    # --- Rule 1: Largest region extraction ---
    if in_regions and (oh < ih or ow < iw):
        largest = in_regions[0]
        r1, c1, r2, c2 = largest['bbox']
        sub = extract_subgrid(inp, r1, c1, r2, c2)
        if grid_eq(sub, out):
            atoms.append(TransformAtom(
                'region', 'extract_largest',
                {'color': largest['color']},
                0.9, f'Extract largest region (color={largest["color"]})'
            ))
    
    # --- Rule 2: Smallest region extraction ---
    if len(in_regions) >= 2 and (oh < ih or ow < iw):
        smallest = in_regions[-1]
        r1, c1, r2, c2 = smallest['bbox']
        sub = extract_subgrid(inp, r1, c1, r2, c2)
        if grid_eq(sub, out):
            atoms.append(TransformAtom(
                'region', 'extract_smallest',
                {'color': smallest['color']},
                0.9, f'Extract smallest region (color={smallest["color"]})'
            ))
    
    # --- Rule 3: Count regions → output is count-sized ---
    n_regions = len(in_regions)
    if oh == 1 and ow == 1:
        if out[0][0] == n_regions:
            atoms.append(TransformAtom(
                'region', 'count_regions',
                {},
                0.95, f'Output = number of regions ({n_regions})'
            ))
    
    # --- Rule 4: Sort regions by size ---
    if (ih, iw) == (oh, ow) and len(in_regions) >= 2:
        # Check if regions are recolored based on size order
        sorted_regs = sorted(in_regions, key=lambda r: r['size'])
        pass  # Complex — defer to composite rules
    
    # --- Rule 5: Per-region geometric transform ---
    if (ih, iw) == (oh, ow):
        for reg in in_regions:
            r1, c1, r2, c2 = reg['bbox']
            in_sub = extract_subgrid(inp, r1, c1, r2, c2)
            out_sub = extract_subgrid(out, r1, c1, r2, c2)
            
            # Check transforms on this region
            for name, fn in [('flip_h', flip_h), ('flip_v', flip_v), 
                            ('rotate_90', rotate_90), ('rotate_180', rotate_180)]:
                if grid_shape(in_sub) == grid_shape(out_sub) and grid_eq(fn(in_sub), out_sub):
                    atoms.append(TransformAtom(
                        'region', f'per_region_{name}',
                        {'color': reg['color'], 'bbox': reg['bbox']},
                        0.85, f'{name} on region color={reg["color"]}'
                    ))
    
    return atoms


def apply_region_transform(atom: TransformAtom, inp: Grid) -> Optional[Grid]:
    op = atom.operation
    params = atom.params
    h, w = grid_shape(inp)
    
    if op == 'extract_largest':
        regions = flood_fill_regions(inp)
        if regions:
            r1, c1, r2, c2 = regions[0]['bbox']
            return extract_subgrid(inp, r1, c1, r2, c2)
    
    elif op == 'extract_smallest':
        regions = flood_fill_regions(inp)
        if regions:
            r1, c1, r2, c2 = regions[-1]['bbox']
            return extract_subgrid(inp, r1, c1, r2, c2)
    
    elif op == 'count_regions':
        regions = flood_fill_regions(inp)
        return [[len(regions)]]
    
    elif op.startswith('per_region_'):
        transform_name = op.replace('per_region_', '')
        fn_map = {'flip_h': flip_h, 'flip_v': flip_v, 
                  'rotate_90': rotate_90, 'rotate_180': rotate_180}
        fn = fn_map.get(transform_name)
        if not fn:
            return None
        
        color = params['color']
        result = copy.deepcopy(inp)
        regions = flood_fill_regions(inp)
        for reg in regions:
            if reg['color'] == color:
                r1, c1, r2, c2 = reg['bbox']
                sub = extract_subgrid(inp, r1, c1, r2, c2)
                transformed = fn(sub)
                if grid_shape(sub) == grid_shape(transformed):
                    result = place_subgrid(result, transformed, r1, c1)
        return result
    
    return None


# ═══════════════════════════════════════════════════════════
# 3. PATTERN COMPLETION (symmetry-based)
# ═══════════════════════════════════════════════════════════

def detect_pattern_completion(inp: Grid, out: Grid) -> List[TransformAtom]:
    """Detect symmetry completion: fill holes to make grid symmetric"""
    atoms = []
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    
    if (ih, iw) != (oh, ow):
        return atoms
    
    out_sym = check_symmetry(out)
    inp_sym = check_symmetry(inp)
    
    bg = most_common_color(inp)
    
    # Check if output has symmetry that input doesn't
    for sym_name in ['h_sym', 'v_sym', 'rot180']:
        if out_sym.get(sym_name) and not inp_sym.get(sym_name):
            # Verify: the non-bg cells in input are preserved in output
            preserved = all(
                inp[r][c] == out[r][c]
                for r in range(ih) for c in range(iw)
                if inp[r][c] != bg
            )
            if preserved:
                atoms.append(TransformAtom(
                    'completion', f'complete_{sym_name}',
                    {'bg': bg, 'symmetry': sym_name},
                    0.85, f'Complete to {sym_name} symmetry (bg={bg})'
                ))
    
    return atoms


def apply_pattern_completion(atom: TransformAtom, inp: Grid) -> Optional[Grid]:
    op = atom.operation
    params = atom.params
    h, w = grid_shape(inp)
    bg = params['bg']
    result = copy.deepcopy(inp)
    
    if op == 'complete_h_sym':
        for r in range(h):
            for c in range(w):
                mirror_c = w - 1 - c
                if result[r][c] == bg and result[r][mirror_c] != bg:
                    result[r][c] = result[r][mirror_c]
                elif result[r][mirror_c] == bg and result[r][c] != bg:
                    result[r][mirror_c] = result[r][c]
        return result
    
    elif op == 'complete_v_sym':
        for r in range(h):
            mirror_r = h - 1 - r
            for c in range(w):
                if result[r][c] == bg and result[mirror_r][c] != bg:
                    result[r][c] = result[mirror_r][c]
                elif result[mirror_r][c] == bg and result[r][c] != bg:
                    result[mirror_r][c] = result[r][c]
        return result
    
    elif op == 'complete_rot180':
        for r in range(h):
            for c in range(w):
                mr, mc = h - 1 - r, w - 1 - c
                if result[r][c] == bg and result[mr][mc] != bg:
                    result[r][c] = result[mr][mc]
                elif result[mr][mc] == bg and result[r][c] != bg:
                    result[mr][mc] = result[r][c]
        return result
    
    return None


# ═══════════════════════════════════════════════════════════
# 4. COUNT-BASED TRANSFORMS
# ═══════════════════════════════════════════════════════════

def detect_count_based(inp: Grid, out: Grid) -> List[TransformAtom]:
    """Detect transforms based on counting: color frequency, region size, etc."""
    atoms = []
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    
    in_info = analyze(inp)
    
    # --- Rule 1: Output dimensions based on color count ---
    for color, count in in_info.color_counts.items():
        if color == in_info.bg_color:
            continue
        if oh == count and ow == 1:
            atoms.append(TransformAtom(
                'count', 'color_count_to_height',
                {'color': color},
                0.85, f'Output height = count of color {color} ({count})'
            ))
        if ow == count and oh == 1:
            atoms.append(TransformAtom(
                'count', 'color_count_to_width',
                {'color': color},
                0.85, f'Output width = count of color {color} ({count})'
            ))
    
    # --- Rule 2: Number of colors → output dimension ---
    n_colors = len(in_info.colors - {in_info.bg_color})
    if oh == n_colors or ow == n_colors:
        atoms.append(TransformAtom(
            'count', 'num_colors_to_dim',
            {'n_colors': n_colors, 'maps_to': 'height' if oh == n_colors else 'width'},
            0.7, f'Output dim = number of non-bg colors ({n_colors})'
        ))
    
    # --- Rule 3: Region count → single cell output ---
    n_regions = len(in_info.regions)
    if oh == 1 and ow == 1 and out[0][0] == n_regions:
        atoms.append(TransformAtom(
            'count', 'region_count',
            {},
            0.95, f'Output = region count ({n_regions})'
        ))
    
    # --- Rule 4: Most/least frequent non-bg color as output ---
    non_bg = {k: v for k, v in in_info.color_counts.items() if k != in_info.bg_color}
    if non_bg and oh == 1 and ow == 1:
        most = max(non_bg, key=non_bg.get)
        least = min(non_bg, key=non_bg.get)
        if out[0][0] == most:
            atoms.append(TransformAtom(
                'count', 'most_frequent_color',
                {},
                0.8, f'Output = most frequent non-bg color ({most})'
            ))
        if out[0][0] == least:
            atoms.append(TransformAtom(
                'count', 'least_frequent_color',
                {},
                0.8, f'Output = least frequent non-bg color ({least})'
            ))
    
    # --- Rule 5: Majority vote per row/column ---
    if (ih, iw) == (oh, ow):
        pass  # Complex — defer
    
    return atoms


def apply_count_based(atom: TransformAtom, inp: Grid) -> Optional[Grid]:
    op = atom.operation
    in_info = analyze(inp)
    
    if op == 'region_count':
        return [[len(in_info.regions)]]
    
    elif op == 'most_frequent_color':
        non_bg = {k: v for k, v in in_info.color_counts.items() if k != in_info.bg_color}
        if non_bg:
            return [[max(non_bg, key=non_bg.get)]]
    
    elif op == 'least_frequent_color':
        non_bg = {k: v for k, v in in_info.color_counts.items() if k != in_info.bg_color}
        if non_bg:
            return [[min(non_bg, key=non_bg.get)]]
    
    elif op == 'color_count_to_height':
        color = atom.params['color']
        count = sum(1 for row in inp for c in row if c == color)
        return [[color]] * count
    
    elif op == 'color_count_to_width':
        color = atom.params['color']
        count = sum(1 for row in inp for c in row if c == color)
        return [[color] * count]
    
    return None


# ═══════════════════════════════════════════════════════════
# 5. COMPOSITE RULES
# ═══════════════════════════════════════════════════════════

def detect_composite(inp: Grid, out: Grid) -> List[TransformAtom]:
    """Detect multi-step transforms (applied last, lowest priority)"""
    atoms = []
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    
    # --- Composite 1: Extract + Transform ---
    if oh < ih or ow < iw:
        regions = flood_fill_regions(inp)
        for reg in regions[:5]:
            r1, c1, r2, c2 = reg['bbox']
            sub = extract_subgrid(inp, r1, c1, r2, c2)
            sh, sw = grid_shape(sub)
            
            if (sh, sw) == (oh, ow):
                for name, fn in [('flip_h', flip_h), ('flip_v', flip_v),
                                ('rotate_180', rotate_180)]:
                    if grid_eq(fn(sub), out):
                        atoms.append(TransformAtom(
                            'composite', f'extract_then_{name}',
                            {'color': reg['color'], 'transform': name},
                            0.85, f'Extract region {reg["color"]} then {name}'
                        ))
            elif (sw, sh) == (oh, ow):
                for name, fn in [('rotate_90', rotate_90), ('rotate_270', rotate_270),
                                ('transpose', transpose)]:
                    if grid_eq(fn(sub), out):
                        atoms.append(TransformAtom(
                            'composite', f'extract_then_{name}',
                            {'color': reg['color'], 'transform': name},
                            0.85, f'Extract region {reg["color"]} then {name}'
                        ))
    
    return atoms


def apply_composite(atom: TransformAtom, inp: Grid) -> Optional[Grid]:
    op = atom.operation
    params = atom.params
    
    if op.startswith('extract_then_'):
        transform_name = params['transform']
        fn_map = {
            'flip_h': flip_h, 'flip_v': flip_v,
            'rotate_90': rotate_90, 'rotate_180': rotate_180,
            'rotate_270': rotate_270, 'transpose': transpose,
        }
        fn = fn_map.get(transform_name)
        if not fn:
            return None
        
        color = params['color']
        regions = flood_fill_regions(inp)
        for reg in regions:
            if reg['color'] == color:
                r1, c1, r2, c2 = reg['bbox']
                sub = extract_subgrid(inp, r1, c1, r2, c2)
                return fn(sub)
    
    return None


# ═══════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════

def _get_neighbors(g: Grid, r: int, c: int) -> Dict[str, int]:
    """Get 4-connected neighbor colors"""
    h, w = grid_shape(g)
    nb = {}
    if r > 0: nb['up'] = g[r-1][c]
    if r < h-1: nb['down'] = g[r+1][c]
    if c > 0: nb['left'] = g[r][c-1]
    if c < w-1: nb['right'] = g[r][c+1]
    return nb


# ═══════════════════════════════════════════════════════════
# ALL DETECTORS (ordered by priority — Verantyx: simplest first)
# ═══════════════════════════════════════════════════════════

ALL_DETECTORS = [
    # (detect_fn, apply_fn)
    (detect_conditional_fill, apply_conditional_fill),
    (detect_region_transforms, apply_region_transform),
    (detect_pattern_completion, apply_pattern_completion),
    (detect_count_based, apply_count_based),
    (detect_composite, apply_composite),
]
