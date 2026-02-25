"""Block-level IR: detect and solve block-grid tasks.

Many ARC tasks have grids composed of repeating blocks with separators.
This module detects such structure and allows solving at the block level.
"""
from typing import Optional, Tuple, List, Dict
import numpy as np
from arc.grid import Grid, grid_eq, most_common_color


def detect_block_grid(grid: Grid, bg: int = None) -> Optional[Dict]:
    """Detect if grid is a block grid with separators.
    
    Returns dict with:
        block_h, block_w: size of each block
        sep_w: separator width (0 or 1)
        sep_color: separator color
        n_rows, n_cols: number of blocks
        blocks: list of list of block contents (as grids)
        block_colors: np.array of dominant color per block (for single-color blocks)
    """
    arr = np.array(grid)
    h, w = arr.shape
    if bg is None:
        bg = most_common_color(grid)
    
    results = []
    
    # Try different block sizes and separator widths
    for bh in range(1, min(h // 2 + 1, 8)):
        for bw in range(1, min(w // 2 + 1, 8)):
            for sep_w in [0, 1]:
                # Calculate expected grid size
                # With sep: n_rows * bh + (n_rows - 1) * sep_w = h
                # n_rows = (h + sep_w) / (bh + sep_w)
                if (h + sep_w) % (bh + sep_w) != 0:
                    continue
                if (w + sep_w) % (bw + sep_w) != 0:
                    continue
                
                n_rows = (h + sep_w) // (bh + sep_w)
                n_cols = (w + sep_w) // (bw + sep_w)
                
                if n_rows < 2 or n_cols < 2:
                    continue
                if n_rows > 20 or n_cols > 20:
                    continue
                
                # Verify separators
                valid = True
                sep_color = -1
                
                if sep_w > 0:
                    # Check horizontal separators
                    for i in range(1, n_rows):
                        row_idx = i * (bh + sep_w) - sep_w
                        for s in range(sep_w):
                            if row_idx + s >= h:
                                valid = False
                                break
                            row = arr[row_idx + s]
                            unique = set(int(v) for v in row)
                            if len(unique) != 1:
                                valid = False
                                break
                            if sep_color == -1:
                                sep_color = int(row[0])
                            elif int(row[0]) != sep_color:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Check vertical separators
                        for j in range(1, n_cols):
                            col_idx = j * (bw + sep_w) - sep_w
                            for s in range(sep_w):
                                if col_idx + s >= w:
                                    valid = False
                                    break
                                col = arr[:, col_idx + s]
                                unique = set(int(v) for v in col)
                                if len(unique) != 1 or int(col[0]) != sep_color:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if not valid:
                    continue
                
                if sep_w == 0:
                    sep_color = bg
                
                # Extract blocks
                blocks = []
                block_colors = np.zeros((n_rows, n_cols), dtype=int)
                
                for br in range(n_rows):
                    row_blocks = []
                    for bc in range(n_cols):
                        r0 = br * (bh + sep_w)
                        c0 = bc * (bw + sep_w)
                        block = arr[r0:r0 + bh, c0:c0 + bw]
                        row_blocks.append(block.tolist())
                        
                        # Dominant color
                        vals = block.flatten()
                        unique = set(int(v) for v in vals)
                        if len(unique) == 1:
                            block_colors[br, bc] = int(vals[0])
                        else:
                            # Most common non-separator color
                            from collections import Counter
                            counts = Counter(int(v) for v in vals)
                            if sep_color in counts:
                                del counts[sep_color]
                            block_colors[br, bc] = counts.most_common(1)[0][0] if counts else int(vals[0])
                    
                    blocks.append(row_blocks)
                
                # Score: prefer configs where blocks are uniform color
                uniform_count = sum(
                    1 for br in range(n_rows) for bc in range(n_cols)
                    if len(set(int(v) for v in np.array(blocks[br][bc]).flatten())) == 1
                )
                uniformity = uniform_count / max(n_rows * n_cols, 1)
                
                # Must have >50% uniform blocks to be a valid block grid
                if uniformity < 0.5:
                    continue
                
                # Block size 1x1 with no separator is trivial (just the grid itself)
                if bh == 1 and bw == 1 and sep_w == 0:
                    continue
                
                # Prefer: separator present, larger blocks, reasonable grid size
                score = uniformity * 100
                if sep_w > 0:
                    score *= 10  # Strongly prefer grids with separators
                if bh >= 2 and bw >= 2:
                    score *= 3  # Prefer non-trivial blocks
                
                results.append({
                    'block_h': bh, 'block_w': bw,
                    'sep_w': sep_w, 'sep_color': sep_color,
                    'n_rows': n_rows, 'n_cols': n_cols,
                    'blocks': blocks,
                    'block_colors': block_colors,
                    'score': score,
                })
    
    if not results:
        return None
    
    # Return best (highest score)
    return max(results, key=lambda x: x['score'])


def block_colors_to_grid(block_colors: np.ndarray, block_h: int, block_w: int,
                         sep_w: int, sep_color: int) -> Grid:
    """Convert block color grid back to pixel grid."""
    n_rows, n_cols = block_colors.shape
    h = n_rows * block_h + max(0, n_rows - 1) * sep_w
    w = n_cols * block_w + max(0, n_cols - 1) * sep_w
    
    result = np.full((h, w), sep_color, dtype=int)
    
    for br in range(n_rows):
        for bc in range(n_cols):
            r0 = br * (block_h + sep_w)
            c0 = bc * (block_w + sep_w)
            result[r0:r0 + block_h, c0:c0 + block_w] = block_colors[br, bc]
    
    return result.tolist()


def blocks_to_grid(blocks: list, block_h: int, block_w: int,
                   sep_w: int, sep_color: int) -> Grid:
    """Convert block content grid back to pixel grid."""
    n_rows = len(blocks)
    n_cols = len(blocks[0])
    h = n_rows * block_h + max(0, n_rows - 1) * sep_w
    w = n_cols * block_w + max(0, n_cols - 1) * sep_w
    
    result = np.full((h, w), sep_color, dtype=int)
    
    for br in range(n_rows):
        for bc in range(n_cols):
            r0 = br * (block_h + sep_w)
            c0 = bc * (block_w + sep_w)
            block = np.array(blocks[br][bc])
            result[r0:r0 + block_h, c0:c0 + block_w] = block
    
    return result.tolist()


def _between_fill_same_color(inp, bg=0):
    """Fill between same-color cells in rows and cols."""
    arr = np.array(inp)
    h, w = arr.shape
    result = arr.copy()
    
    # Row-wise
    for r in range(h):
        for color in range(10):
            if color == bg:
                continue
            cols = [c for c in range(w) if arr[r, c] == color]
            if len(cols) >= 2:
                for c in range(min(cols), max(cols) + 1):
                    if result[r, c] == bg:
                        result[r, c] = color
    
    # Col-wise
    for c in range(w):
        for color in range(10):
            if color == bg:
                continue
            rows = [r for r in range(h) if arr[r, c] == color]
            if len(rows) >= 2:
                for r in range(min(rows), max(rows) + 1):
                    if result[r, c] == bg:
                        result[r, c] = color
    
    return result.tolist()


def solve_at_block_level(train_pairs, test_inputs):
    """Try to solve by detecting block grid and solving at block color level."""
    from arc.cross_engine import _generate_cross_pieces, CrossSimulator, CrossPiece
    
    # Detect block grid for first training pair
    bg0 = most_common_color(train_pairs[0][0])
    ir0 = detect_block_grid(train_pairs[0][0], bg0)
    if ir0 is None:
        return None, []
    
    bh = ir0['block_h']
    bw = ir0['block_w']
    sep_w = ir0['sep_w']
    sep_color = ir0['sep_color']
    
    # Verify all pairs have same block size (grid size may differ)
    block_train = []
    for inp, out in train_pairs:
        ir_in = detect_block_grid(inp, bg0)
        ir_out = detect_block_grid(out, bg0)
        
        if ir_in is None or ir_out is None:
            return None, []
        if ir_in['block_h'] != bh or ir_in['block_w'] != bw:
            return None, []
        if ir_out['block_h'] != bh or ir_out['block_w'] != bw:
            return None, []
        if ir_in['sep_w'] != sep_w or ir_out['sep_w'] != sep_w:
            return None, []
        
        block_in = ir_in['block_colors'].tolist()
        block_out = ir_out['block_colors'].tolist()
        block_train.append((block_in, block_out))
    
    # Detect block grid for test inputs
    block_test_irs = []
    for test_inp in test_inputs:
        ir_t = detect_block_grid(test_inp, bg0)
        if ir_t is None or ir_t['block_h'] != bh or ir_t['block_w'] != bw:
            return None, []
        block_test_irs.append(ir_t)
    
    block_test = [ir_t['block_colors'].tolist() for ir_t in block_test_irs]
    
    # Try solving at block level
    sim = CrossSimulator()
    verified = []
    
    # 0. Try between_fill_same_color first (high-confidence block-level op)
    block_bg = most_common_color(block_train[0][0])
    _bf_ok = True
    for b_in, b_out in block_train:
        _bf_pred = _between_fill_same_color(b_in, block_bg)
        if not grid_eq(_bf_pred, b_out):
            _bf_ok = False
            break
    if _bf_ok:
        verified.append(('block_ir',
            type('Piece', (), {
                'name': 'block_between_fill_same_color',
                'apply': lambda inp, _bg=block_bg: _between_fill_same_color(inp, _bg)
            })()))
    
    # 1. Try cross_engine pieces on block-level grids
    pieces = _generate_cross_pieces(block_train)
    for piece in pieces:
        if sim.verify(piece, block_train):
            verified.append(('block_ir', piece))
            if len(verified) >= 2:
                break
    
    if not verified:
        return None, []
    
    # Apply to test and convert back
    predictions = []
    for i, test_block in enumerate(block_test):
        attempts = []
        _tir = block_test_irs[i]
        for kind, piece in verified:
            try:
                block_result = piece.apply(test_block)
            except Exception:
                block_result = None
            if block_result is not None:
                pixel_result = block_colors_to_grid(
                    np.array(block_result),
                    _tir['block_h'], _tir['block_w'],
                    _tir['sep_w'], _tir['sep_color'])
                attempts.append(pixel_result)
        predictions.append(attempts)
    
    return predictions, verified
