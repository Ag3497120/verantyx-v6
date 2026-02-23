"""
arc/grid_summarize.py — Grid-to-Summary transforms for ARC-AGI-2

Handles large-grid → small-grid reductions:
1. Block-based classification (grid divided by separator lines → property per block)
2. Column/row reduction (fold columns/rows via some operation)
3. Count-based summary (count objects/regions → output as Kx1)
"""

from typing import List, Tuple, Optional, Dict, Set
from arc.grid import Grid, grid_shape, grid_eq, most_common_color, grid_colors


def find_separator_lines(grid: Grid, bg: int) -> Tuple[List[int], List[int]]:
    """Find rows and columns that act as separators (all same non-bg color)."""
    h, w = grid_shape(grid)
    
    sep_rows = []
    for r in range(h):
        vals = set(grid[r])
        if len(vals) == 1 and vals.pop() != bg:
            sep_rows.append(r)
    
    sep_cols = []
    for c in range(w):
        vals = set(grid[r][c] for r in range(h))
        if len(vals) == 1 and vals.pop() != bg:
            sep_cols.append(c)
    
    return sep_rows, sep_cols


def extract_blocks(grid: Grid, sep_rows: List[int], sep_cols: List[int]) -> List[List[Grid]]:
    """Extract sub-grids between separator lines."""
    h, w = grid_shape(grid)
    
    row_ranges = []
    prev = 0
    for sr in sep_rows:
        if sr > prev:
            row_ranges.append((prev, sr))
        prev = sr + 1
    if prev < h:
        row_ranges.append((prev, h))
    
    col_ranges = []
    prev = 0
    for sc in sep_cols:
        if sc > prev:
            col_ranges.append((prev, sc))
        prev = sc + 1
    if prev < w:
        col_ranges.append((prev, w))
    
    blocks = []
    for r_start, r_end in row_ranges:
        row_blocks = []
        for c_start, c_end in col_ranges:
            block = [grid[r][c_start:c_end] for r in range(r_start, r_end)]
            row_blocks.append(block)
        blocks.append(row_blocks)
    
    return blocks


def _block_property_count(block: Grid, bg: int, sep_color: int) -> int:
    """Count non-bg, non-separator cells in a block."""
    count = 0
    for row in block:
        for v in row:
            if v != bg and v != sep_color:
                count += 1
    return count


def _block_property_has_color(block: Grid, bg: int, sep_color: int, target: int) -> bool:
    for row in block:
        for v in row:
            if v == target:
                return True
    return False


def _block_unique_non_bg_colors(block: Grid, bg: int, sep_color: int) -> Set[int]:
    colors = set()
    for row in block:
        for v in row:
            if v != bg and v != sep_color:
                colors.add(v)
    return colors


def learn_block_classifier(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn a block-based grid-to-summary rule.
    
    Detects separator lines, extracts blocks, learns what property
    of each block maps to the output value.
    """
    inp0, out0 = train_pairs[0]
    bg = most_common_color(inp0)
    oh, ow = grid_shape(out0)
    
    sep_rows, sep_cols = find_separator_lines(inp0, bg)
    if not sep_rows and not sep_cols:
        return None
    
    # Determine separator color
    h, w = grid_shape(inp0)
    sep_color = inp0[sep_rows[0]][0] if sep_rows else inp0[0][sep_cols[0]]
    
    blocks = extract_blocks(inp0, sep_rows, sep_cols)
    
    n_block_rows = len(blocks)
    n_block_cols = len(blocks[0]) if blocks else 0
    
    # Output dimensions must match block grid
    if n_block_rows != oh or n_block_cols != ow:
        return None
    
    # Try different classifiers
    for classifier_name, classifier_fn in _get_block_classifiers(bg, sep_color):
        # Test on all training pairs
        ok = True
        for inp, out in train_pairs:
            sr, sc = find_separator_lines(inp, bg)
            blks = extract_blocks(inp, sr, sc)
            o_h, o_w = grid_shape(out)
            if len(blks) != o_h or (blks and len(blks[0]) != o_w):
                ok = False; break
            
            for br in range(o_h):
                for bc in range(o_w):
                    predicted = classifier_fn(blks[br][bc])
                    if predicted != out[br][bc]:
                        ok = False; break
                if not ok: break
            if not ok: break
        
        if ok:
            return {
                'type': 'block_classifier',
                'classifier': classifier_name,
                'bg': bg,
                'sep_color': sep_color,
            }
    
    return None


def _get_block_classifiers(bg: int, sep_color: int):
    """Generate candidate classifier functions.
    
    Yields all reasonable classifier types.
    """
    # Binary classifiers for all possible (lo, hi) pairs
    for lo in range(10):
        for hi in range(lo + 1, 10):
            # Count-based thresholds
            for threshold in range(1, 10):
                name = f'count_ge_{threshold}_map_{lo}_{hi}'
                def make_fn(t=threshold, l=lo, h=hi):
                    def fn(block):
                        c = _block_property_count(block, bg, sep_color)
                        return h if c >= t else l
                    return fn
                yield name, make_fn()
            
            # Has specific color
            for color in range(10):
                if color == bg or color == sep_color:
                    continue
                name = f'has_color_{color}_map_{lo}_{hi}'
                def make_fn(col=color, l=lo, h=hi):
                    def fn(block):
                        return h if _block_property_has_color(block, bg, sep_color, col) else l
                    return fn
                yield name, make_fn()
            
            # Number of unique colors >= threshold
            for threshold in range(1, 5):
                name = f'unique_colors_ge_{threshold}_map_{lo}_{hi}'
                def make_fn(t=threshold, l=lo, h=hi):
                    def fn(block):
                        return h if len(_block_unique_non_bg_colors(block, bg, sep_color)) >= t else l
                    return fn
                yield name, make_fn()
    
    # Multi-value: count directly
    name = 'count_direct'
    def count_fn(block):
        return _block_property_count(block, bg, sep_color)
    yield name, count_fn
    
    # Multi-value: most common non-bg color
    name = 'most_common_color'
    def mc_fn(block):
        from collections import Counter
        counts = Counter()
        for row in block:
            for v in row:
                if v != bg and v != sep_color:
                    counts[v] += 1
        return counts.most_common(1)[0][0] if counts else bg
    yield name, mc_fn
    
    # Multi-value: minority color (least common non-bg)
    name = 'minority_color'
    def min_fn(block):
        from collections import Counter
        counts = Counter()
        for row in block:
            for v in row:
                if v != bg and v != sep_color:
                    counts[v] += 1
        return counts.most_common()[-1][0] if counts else bg
    yield name, min_fn


def apply_block_classifier(rule: Dict, inp: Grid) -> Optional[Grid]:
    """Apply a learned block classifier rule."""
    bg = rule['bg']
    sep_color = rule['sep_color']
    
    sep_rows, sep_cols = find_separator_lines(inp, bg)
    blocks = extract_blocks(inp, sep_rows, sep_cols)
    
    if not blocks:
        return None
    
    n_rows = len(blocks)
    n_cols = len(blocks[0])
    
    classifier_name = rule['classifier']
    
    # Reconstruct the classifier function
    for name, fn in _get_block_classifiers(bg, sep_color):
        if name == classifier_name:
            result = []
            for br in range(n_rows):
                row = []
                for bc in range(n_cols):
                    row.append(fn(blocks[br][bc]))
                result.append(row)
            return result
    
    return None


# === Column/Row Folding ===

def learn_fold_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn grid folding/reduction rules.
    
    Handles cases where output = some operation on groups of columns/rows.
    E.g., 8x9 -> 8x2 by folding 3 groups of 3 columns.
    """
    inp0, out0 = train_pairs[0]
    ih, iw = grid_shape(inp0)
    oh, ow = grid_shape(out0)
    bg = most_common_color(inp0)
    
    # Check for separator columns (all-0 or all-bg columns in input)
    sep_cols = []
    for c in range(iw):
        if all(inp0[r][c] == bg for r in range(ih)):
            sep_cols.append(c)
    
    # Check for separator rows
    sep_rows = []
    for r in range(ih):
        if all(inp0[r][c] == bg for c in range(iw)):
            sep_rows.append(r)
    
    # Try column-group folding (same height, reduced width)
    if ih == oh and ow < iw:
        # Split columns into groups by separators
        col_groups = _split_by_seps(iw, sep_cols)
        
        if len(col_groups) >= 2:
            # Each group should produce ow columns
            # Try: output = XOR/OR/AND of groups, or specific group selection
            for op_name, op_fn in [
                ('xor', lambda a, b: a if b == bg else (b if a == bg else (bg if a == b else a))),
                ('or', lambda a, b: a if b == bg else b),
                ('diff', lambda a, b: a if a != b and a != bg else (b if b != bg and a != b else bg)),
            ]:
                result = _try_fold_cols(inp0, col_groups, ow, oh, op_fn, bg)
                if result is not None and grid_eq(result, out0):
                    # Verify on all pairs
                    ok = True
                    for inp, out in train_pairs[1:]:
                        h2, w2 = grid_shape(inp)
                        sc2 = [c for c in range(w2) if all(inp[r][c] == bg for r in range(h2))]
                        cg2 = _split_by_seps(w2, sc2)
                        r2 = _try_fold_cols(inp, cg2, grid_shape(out)[1], grid_shape(out)[0], op_fn, bg)
                        if r2 is None or not grid_eq(r2, out):
                            ok = False; break
                    if ok:
                        return {'type': 'fold_cols', 'op': op_name}
    
    # Try row-group folding (same width, reduced height)
    if iw == ow and oh < ih:
        row_groups = _split_by_seps(ih, sep_rows)
        
        if len(row_groups) >= 2:
            for op_name, op_fn in [
                ('xor', lambda a, b: a if b == bg else (b if a == bg else (bg if a == b else a))),
                ('or', lambda a, b: a if b == bg else b),
                ('overlay', lambda a, b: b if b != bg else a),
            ]:
                result = _try_fold_rows(inp0, row_groups, oh, ow, op_fn, bg)
                if result is not None and grid_eq(result, out0):
                    ok = True
                    for inp, out in train_pairs[1:]:
                        h2, w2 = grid_shape(inp)
                        sr2 = [r for r in range(h2) if all(inp[r][c] == bg for c in range(w2))]
                        rg2 = _split_by_seps(h2, sr2)
                        r2 = _try_fold_rows(inp, rg2, grid_shape(out)[0], grid_shape(out)[1], op_fn, bg)
                        if r2 is None or not grid_eq(r2, out):
                            ok = False; break
                    if ok:
                        return {'type': 'fold_rows', 'op': op_name}
    
    return None


def _split_by_seps(total: int, seps: List[int]) -> List[Tuple[int, int]]:
    """Split range [0, total) by separator positions into groups."""
    groups = []
    prev = 0
    for s in sorted(seps):
        if s > prev:
            groups.append((prev, s))
        prev = s + 1
    if prev < total:
        groups.append((prev, total))
    return groups


def _try_fold_cols(inp: Grid, col_groups, ow, oh, op_fn, bg) -> Optional[Grid]:
    """Try folding column groups into output width."""
    # Each group has some width. We need to reduce to ow columns.
    # If all groups have the same width == ow, fold them pairwise.
    group_widths = [end - start for start, end in col_groups]
    
    if all(w == ow for w in group_widths):
        # Fold all groups
        result = [[bg] * ow for _ in range(oh)]
        for g_start, g_end in col_groups:
            for r in range(oh):
                for c in range(ow):
                    result[r][c] = op_fn(result[r][c], inp[r][g_start + c])
        return result
    
    return None


def _try_fold_rows(inp: Grid, row_groups, oh, ow, op_fn, bg) -> Optional[Grid]:
    """Try folding row groups into output height."""
    group_heights = [end - start for start, end in row_groups]
    
    if all(h == oh for h in group_heights):
        result = [[bg] * ow for _ in range(oh)]
        for g_start, g_end in row_groups:
            for r in range(oh):
                for c in range(ow):
                    result[r][c] = op_fn(result[r][c], inp[g_start + r][c])
        return result
    
    return None


def apply_fold_rule(rule: Dict, inp: Grid) -> Optional[Grid]:
    """Apply a learned fold rule."""
    h, w = grid_shape(inp)
    bg = most_common_color(inp)
    
    if rule['type'] == 'fold_cols':
        sep_cols = [c for c in range(w) if all(inp[r][c] == bg for r in range(h))]
        col_groups = _split_by_seps(w, sep_cols)
        if not col_groups:
            return None
        ow = col_groups[0][1] - col_groups[0][0]
        op_fn = _get_op_fn(rule['op'], bg)
        return _try_fold_cols(inp, col_groups, ow, h, op_fn, bg)
    
    elif rule['type'] == 'fold_rows':
        sep_rows = [r for r in range(h) if all(inp[r][c] == bg for c in range(w))]
        row_groups = _split_by_seps(h, sep_rows)
        if not row_groups:
            return None
        oh = row_groups[0][1] - row_groups[0][0]
        op_fn = _get_op_fn(rule['op'], bg)
        return _try_fold_rows(inp, row_groups, oh, w, op_fn, bg)
    
    return None


def _get_op_fn(op_name: str, bg: int):
    if op_name == 'xor':
        return lambda a, b: a if b == bg else (b if a == bg else (bg if a == b else a))
    elif op_name == 'or':
        return lambda a, b: a if b == bg else b
    elif op_name == 'overlay':
        return lambda a, b: b if b != bg else a
    elif op_name == 'diff':
        return lambda a, b: a if a != b and a != bg else (b if b != bg and a != b else bg)
    return lambda a, b: b


# === Count-based Summary ===

def learn_count_summary(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn count-based grid → Kx1 summary.
    
    E.g., count connected components of non-bg color, or count
    objects per row, etc.
    """
    from arc.objects import detect_objects
    
    inp0, out0 = train_pairs[0]
    oh, ow = grid_shape(out0)
    bg = most_common_color(inp0)
    
    if ow != 1:
        return None  # Only handle Kx1 outputs
    
    # Try: output height = number of distinct non-bg colors
    n_colors = len(grid_colors(inp0) - {bg})
    if oh == n_colors:
        # Check all pairs
        ok = True
        for inp, out in train_pairs:
            b = most_common_color(inp)
            nc = len(grid_colors(inp) - {b})
            o_h, o_w = grid_shape(out)
            if o_w != 1 or o_h != nc:
                ok = False; break
        if ok:
            # What value? Check if all zeros
            all_zero = all(out[r][0] == 0 for _, out in train_pairs for r in range(grid_shape(out)[0]))
            if all_zero:
                return {'type': 'count_colors_zeros', 'bg': bg}
    
    # Try: output height = number of connected components of non-bg
    objs0 = detect_objects(inp0, bg)
    if oh == len(objs0):
        ok = True
        for inp, out in train_pairs:
            b = most_common_color(inp)
            objs = detect_objects(inp, b)
            o_h, o_w = grid_shape(out)
            if o_w != 1 or o_h != len(objs):
                ok = False; break
        if ok:
            all_zero = all(out[r][0] == 0 for _, out in train_pairs for r in range(grid_shape(out)[0]))
            if all_zero:
                return {'type': 'count_objects_zeros', 'bg': bg}
    
    return None


def apply_count_summary(rule: Dict, inp: Grid) -> Optional[Grid]:
    """Apply a count-based summary rule."""
    bg = rule['bg']
    
    if rule['type'] == 'count_colors_zeros':
        n = len(grid_colors(inp) - {bg})
        return [[0] for _ in range(n)]
    
    elif rule['type'] == 'count_objects_zeros':
        from arc.objects import detect_objects
        objs = detect_objects(inp, bg)
        return [[0] for _ in range(len(objs))]
    
    return None
