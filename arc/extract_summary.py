"""
arc/extract_summary.py — Fixed-Output-Size Summary/Extraction for ARC-AGI-2

Handles tasks where output is always a fixed size (typically 3x3, 5x5, etc.)
regardless of input dimensions. These are "summary" tasks where the output
encodes properties of the input.

Common patterns:
- Grid partition → majority color per block
- Object count encoding in fixed grid
- Extract unique/dominant pattern from input
- Region property mapping to fixed-size output
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color
from arc.objects import detect_objects


def learn_fixed_output_summary(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """
    Learn fixed-output summary transformation.

    Detects if output is always fixed size, then tries various strategies:
    1. Grid partition → majority color per block
    2. Largest solid block color → fill output
    3. Object count encoding
    4. Unique pattern extraction

    Returns rule dict or None.
    """
    if not train_pairs:
        return None

    # Check if output size is fixed across all training examples
    output_sizes = [grid_shape(out) for _, out in train_pairs]
    if len(set(output_sizes)) != 1:
        return None

    out_h, out_w = output_sizes[0]

    # Only handle small fixed outputs (2x2 to 5x5 typically)
    if out_h > 10 or out_w > 10 or out_h < 1 or out_w < 1:
        return None

    # Try different strategies in priority order
    strategies = [
        _try_largest_solid_block_fill,
        _try_grid_partition_majority,
        _try_unique_block_extraction,
        _try_object_count_grid,
        _try_color_histogram_blocks,
    ]

    for strategy_fn in strategies:
        rule = strategy_fn(train_pairs, out_h, out_w)
        if rule is not None:
            # Verify rule on all training pairs
            ok = True
            for inp, expected_out in train_pairs:
                result = apply_fixed_output_summary(inp, rule)
                if result is None or not grid_eq(result, expected_out):
                    ok = False
                    break

            if ok:
                return rule

    return None


def apply_fixed_output_summary(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply learned fixed-output summary transformation."""
    rule_type = params.get('type')

    if rule_type == 'largest_solid_block_fill':
        return _apply_largest_solid_block_fill(inp, params)
    elif rule_type == 'grid_partition_majority':
        return _apply_grid_partition_majority(inp, params)
    elif rule_type == 'unique_block_extraction':
        return _apply_unique_block_extraction(inp, params)
    elif rule_type == 'object_count_grid':
        return _apply_object_count_grid(inp, params)
    elif rule_type == 'color_histogram_blocks':
        return _apply_color_histogram_blocks(inp, params)

    return None


# ============================================================
# Strategy 1: Largest solid block → fill output
# ============================================================

def _find_largest_solid_blocks(inp: Grid, bg: int = 0) -> List[Dict]:
    """Find all solid rectangular blocks of non-bg color in input."""
    arr = np.array(inp)
    h, w = arr.shape
    blocks = []

    # For each color, find solid blocks
    colors = set(arr.flatten()) - {bg}

    for color in colors:
        # Find all positions of this color
        mask = (arr == color)

        # Try to find rectangular solid blocks
        # Use a simple approach: scan for maximal rectangles
        visited = np.zeros_like(mask, dtype=bool)

        for r in range(h):
            for c in range(w):
                if mask[r, c] and not visited[r, c]:
                    # Try to expand into largest rectangle from (r, c)
                    # Find max width in this row
                    max_c = c
                    while max_c + 1 < w and mask[r, max_c + 1]:
                        max_c += 1

                    # Try to extend downward
                    max_r = r
                    while max_r + 1 < h:
                        # Check if row max_r+1 has same span
                        if all(mask[max_r + 1, c2] for c2 in range(c, max_c + 1)):
                            max_r += 1
                        else:
                            break

                    # Check if this is actually a solid block
                    block_h = max_r - r + 1
                    block_w = max_c - c + 1
                    block_size = block_h * block_w

                    if block_size >= 4:  # At least 2x2
                        blocks.append({
                            'color': int(color),
                            'bbox': (r, c, max_r, max_c),
                            'size': block_size,
                            'height': block_h,
                            'width': block_w,
                        })
                        # Mark as visited
                        visited[r:max_r+1, c:max_c+1] = True

    return sorted(blocks, key=lambda b: b['size'], reverse=True)


def _try_largest_solid_block_fill(train_pairs: List[Tuple[Grid, Grid]],
                                   out_h: int, out_w: int) -> Optional[Dict]:
    """
    Strategy: Find largest solid rectangular block in input, fill output with that color.
    Example: task 3194b014 where a large 3x3 or 4x5 solid block appears, output = all that color.

    Tries multiple selection criteria:
    1. Absolute largest block
    2. Among large blocks (>=threshold), sum cells by color, select color with most total cells
    3. Among large blocks (>=threshold), select first by position (top-left)
    """
    for bg in [0, most_common_color(train_pairs[0][0])]:
        # Try strategy 1: absolute largest
        ok = True
        for inp, out in train_pairs:
            arr_out = np.array(out)
            blocks = _find_largest_solid_blocks(inp, bg)
            if not blocks:
                ok = False
                break
            if not np.all(arr_out == blocks[0]['color']):
                ok = False
                break
        if ok:
            return {'type': 'largest_solid_block_fill', 'bg': bg, 'out_h': out_h, 'out_w': out_w,
                    'selection': 'largest'}

        # Try strategy 2: among large blocks, select color with most total cells
        for min_size in [20, 16, 12, 9, 6, 4]:
            ok = True
            for inp, out in train_pairs:
                arr_out = np.array(out)
                blocks = _find_largest_solid_blocks(inp, bg)
                if not blocks:
                    ok = False
                    break

                # Sum cells by color for large blocks
                large_blocks = [b for b in blocks if b['size'] >= min_size]
                if not large_blocks:
                    ok = False
                    break

                color_totals = Counter()
                for b in large_blocks:
                    color_totals[b['color']] += b['size']

                expected_color = color_totals.most_common(1)[0][0]

                if not np.all(arr_out == expected_color):
                    ok = False
                    break

            if ok:
                return {'type': 'largest_solid_block_fill', 'bg': bg, 'out_h': out_h, 'out_w': out_w,
                        'selection': 'most_total_cells', 'min_size': min_size}

        # Try strategy 3: among large blocks, select first by position
        for min_size in [20, 16, 12, 9, 4]:
            ok = True
            for inp, out in train_pairs:
                arr_out = np.array(out)
                blocks = _find_largest_solid_blocks(inp, bg)
                if not blocks:
                    ok = False
                    break

                # Filter to large blocks
                large_blocks = [b for b in blocks if b['size'] >= min_size]
                if not large_blocks:
                    ok = False
                    break

                # Sort by position (row, col)
                large_blocks_by_pos = sorted(large_blocks, key=lambda b: (b['bbox'][0], b['bbox'][1]))
                expected_color = large_blocks_by_pos[0]['color']

                if not np.all(arr_out == expected_color):
                    ok = False
                    break

            if ok:
                return {'type': 'largest_solid_block_fill', 'bg': bg, 'out_h': out_h, 'out_w': out_w,
                        'selection': 'large_first_pos', 'min_size': min_size}

    return None


def _apply_largest_solid_block_fill(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply largest solid block fill strategy."""
    bg = params['bg']
    out_h = params['out_h']
    out_w = params['out_w']
    selection = params.get('selection', 'largest')

    blocks = _find_largest_solid_blocks(inp, bg)

    if not blocks:
        return None

    if selection == 'largest':
        fill_color = blocks[0]['color']
    elif selection == 'most_total_cells':
        min_size = params.get('min_size', 12)
        large_blocks = [b for b in blocks if b['size'] >= min_size]
        if not large_blocks:
            return None
        # Sum cells by color
        color_totals = Counter()
        for b in large_blocks:
            color_totals[b['color']] += b['size']
        fill_color = color_totals.most_common(1)[0][0]
    elif selection == 'large_first_pos':
        min_size = params.get('min_size', 20)
        large_blocks = [b for b in blocks if b['size'] >= min_size]
        if not large_blocks:
            return None
        # Sort by position
        large_blocks_by_pos = sorted(large_blocks, key=lambda b: (b['bbox'][0], b['bbox'][1]))
        fill_color = large_blocks_by_pos[0]['color']
    else:
        return None

    return [[fill_color] * out_w for _ in range(out_h)]


# ============================================================
# Strategy 2: Grid partition → majority color per block
# ============================================================

def _try_grid_partition_majority(train_pairs: List[Tuple[Grid, Grid]],
                                 out_h: int, out_w: int) -> Optional[Dict]:
    """
    Strategy: Divide input into out_h × out_w blocks, output[r][c] = majority color in block.
    """
    for bg in [0]:
        for count_bg in [False, True]:  # Whether to include bg in majority
            ok = True

            for inp, out in train_pairs:
                arr_inp = np.array(inp)
                arr_out = np.array(out)
                in_h, in_w = arr_inp.shape

                # Partition input into out_h × out_w regions
                result = np.zeros((out_h, out_w), dtype=int)

                for br in range(out_h):
                    for bc in range(out_w):
                        # Define block boundaries
                        r1 = br * in_h // out_h
                        r2 = (br + 1) * in_h // out_h
                        c1 = bc * in_w // out_w
                        c2 = (bc + 1) * in_w // out_w

                        block = arr_inp[r1:r2, c1:c2].flatten()

                        if count_bg:
                            # Include bg in majority vote
                            if len(block) > 0:
                                result[br, bc] = np.bincount(block).argmax()
                        else:
                            # Only count non-bg colors
                            non_bg = block[block != bg]
                            if len(non_bg) > 0:
                                result[br, bc] = np.bincount(non_bg).argmax()
                            else:
                                result[br, bc] = bg

                if not np.array_equal(result, arr_out):
                    ok = False
                    break

            if ok:
                return {
                    'type': 'grid_partition_majority',
                    'bg': bg,
                    'count_bg': count_bg,
                    'out_h': out_h,
                    'out_w': out_w,
                }

    return None


def _apply_grid_partition_majority(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply grid partition majority strategy."""
    bg = params['bg']
    count_bg = params['count_bg']
    out_h = params['out_h']
    out_w = params['out_w']

    arr_inp = np.array(inp)
    in_h, in_w = arr_inp.shape
    result = np.zeros((out_h, out_w), dtype=int)

    for br in range(out_h):
        for bc in range(out_w):
            r1 = br * in_h // out_h
            r2 = (br + 1) * in_h // out_h
            c1 = bc * in_w // out_w
            c2 = (bc + 1) * in_w // out_w

            block = arr_inp[r1:r2, c1:c2].flatten()

            if count_bg:
                if len(block) > 0:
                    result[br, bc] = np.bincount(block).argmax()
            else:
                non_bg = block[block != bg]
                if len(non_bg) > 0:
                    result[br, bc] = np.bincount(non_bg).argmax()
                else:
                    result[br, bc] = bg

    return result.tolist()


# ============================================================
# Strategy 3: Unique block extraction
# ============================================================

def _try_unique_block_extraction(train_pairs: List[Tuple[Grid, Grid]],
                                 out_h: int, out_w: int) -> Optional[Dict]:
    """
    Strategy: Extract a unique out_h × out_w block that appears in input.
    Look for patterns that appear once or are distinct.
    """
    # Try finding a consistent block position across all training pairs
    # This is complex - simplified version: extract centered block

    for bg in [0]:
        ok = True

        for inp, out in train_pairs:
            arr_inp = np.array(inp)
            arr_out = np.array(out)
            in_h, in_w = arr_inp.shape

            # Try: extract center block
            cr = (in_h - out_h) // 2
            cc = (in_w - out_w) // 2

            if cr < 0 or cc < 0:
                ok = False
                break

            extracted = arr_inp[cr:cr+out_h, cc:cc+out_w]

            if not np.array_equal(extracted, arr_out):
                ok = False
                break

        if ok:
            return {
                'type': 'unique_block_extraction',
                'mode': 'center',
                'out_h': out_h,
                'out_w': out_w,
            }

    return None


def _apply_unique_block_extraction(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply unique block extraction strategy."""
    mode = params['mode']
    out_h = params['out_h']
    out_w = params['out_w']

    arr_inp = np.array(inp)
    in_h, in_w = arr_inp.shape

    if mode == 'center':
        cr = (in_h - out_h) // 2
        cc = (in_w - out_w) // 2

        if cr < 0 or cc < 0:
            return None

        return arr_inp[cr:cr+out_h, cc:cc+out_w].tolist()

    return None


# ============================================================
# Strategy 4: Object count encoding
# ============================================================

def _try_object_count_grid(train_pairs: List[Tuple[Grid, Grid]],
                           out_h: int, out_w: int) -> Optional[Dict]:
    """
    Strategy: Count objects by color and encode in output grid.
    Output[r][c] might represent count of objects with certain property.
    """
    # This is complex - skip for now, could be added later
    return None


def _apply_object_count_grid(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply object count encoding strategy."""
    return None


# ============================================================
# Strategy 5: Color histogram blocks
# ============================================================

def _try_color_histogram_blocks(train_pairs: List[Tuple[Grid, Grid]],
                                out_h: int, out_w: int) -> Optional[Dict]:
    """
    Strategy: Each output cell encodes color frequency from input region.
    Similar to partition but might use different encoding (e.g., presence vs count).
    """
    # Could implement variations of partition strategy
    return None


def _apply_color_histogram_blocks(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply color histogram blocks strategy."""
    return None
