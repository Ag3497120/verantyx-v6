"""
arc/cross6_scale.py — 多スケールCross + サイズ変更対応

1. 多スケール: 2x2, 3x3ブロック単位でCross記述子を計算
2. サイズ変更: 入出力サイズが異なるタスク対応
   - crop (切り出し)
   - tile (タイリング)
   - scale (拡大/縮小)
   - extract (オブジェクト抽出)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
from scipy.ndimage import label as scipy_label
from arc.grid import grid_eq, grid_shape, most_common_color


# ─── サイズ変更ソルバー ───

def _detect_crop(train_pairs) -> Optional[Dict]:
    """出力が入力の部分切り出しか検出"""
    for strategy in ['find_subgrid', 'find_object']:
        results = []
        for inp, out in train_pairs:
            gi = np.array(inp)
            go = np.array(out)
            hi, wi = gi.shape
            ho, wo = go.shape
            
            if ho > hi or wo > wi:
                return None
            
            # Search for exact subgrid match
            found = False
            for r in range(hi - ho + 1):
                for c in range(wi - wo + 1):
                    if np.array_equal(gi[r:r+ho, c:c+wo], go):
                        results.append((r, c, ho, wo))
                        found = True
                        break
                if found:
                    break
            
            if not found:
                return None
        
        if len(results) == len(train_pairs):
            # Check if position rule is consistent
            # Option A: fixed position
            if len(set(results)) == 1:
                r, c, h, w = results[0]
                return {'type': 'crop_fixed', 'r': r, 'c': c, 'h': h, 'w': w}
            
            # Option B: relative to some feature (non-bg bounding box)
            # Try: crop around non-bg content
            offsets = []
            for (inp, out), (r, c, h, w) in zip(train_pairs, results):
                gi = np.array(inp)
                bg = most_common_color(inp)
                fg_mask = gi != bg
                if not fg_mask.any():
                    break
                fg_rows, fg_cols = np.where(fg_mask)
                fg_r0 = fg_rows.min()
                fg_c0 = fg_cols.min()
                offsets.append((r - fg_r0, c - fg_c0))
            
            if len(offsets) == len(train_pairs) and len(set(offsets)) == 1:
                return {'type': 'crop_relative_fg', 'offset': offsets[0], 
                        'h': results[0][2], 'w': results[0][3]}
    
    return None


def _detect_scale(train_pairs) -> Optional[Dict]:
    """出力が入力のスケーリングか検出"""
    ratios = set()
    for inp, out in train_pairs:
        hi, wi = len(inp), len(inp[0])
        ho, wo = len(out), len(out[0])
        
        if ho % hi != 0 or wo % wi != 0:
            return None
        
        rh = ho // hi
        rw = wo // wi
        if rh != rw:
            return None
        ratios.add(rh)
    
    if len(ratios) != 1:
        return None
    
    scale = ratios.pop()
    if scale <= 1:
        return None
    
    # Verify: each cell becomes scale x scale block
    for inp, out in train_pairs:
        gi = np.array(inp)
        go = np.array(out)
        hi, wi = gi.shape
        
        for r in range(hi):
            for c in range(wi):
                block = go[r*scale:(r+1)*scale, c*scale:(c+1)*scale]
                if not np.all(block == gi[r, c]):
                    return None
    
    return {'type': 'upscale', 'scale': scale}


def _detect_tile(train_pairs) -> Optional[Dict]:
    """出力が入力のタイリング"""
    for inp, out in train_pairs:
        hi, wi = len(inp), len(inp[0])
        ho, wo = len(out), len(out[0])
        
        if ho % hi != 0 or wo % wi != 0:
            return None
        
        rh = ho // hi
        rw = wo // wi
        
        gi = np.array(inp)
        go = np.array(out)
        
        for tr in range(rh):
            for tc in range(rw):
                block = go[tr*hi:(tr+1)*hi, tc*wi:(tc+1)*wi]
                if not np.array_equal(block, gi):
                    return None
    
    rh = len(train_pairs[0][1]) // len(train_pairs[0][0])
    rw = len(train_pairs[0][1][0]) // len(train_pairs[0][0][0])
    return {'type': 'tile', 'rh': rh, 'rw': rw}


def _detect_extract_object(train_pairs) -> Optional[Dict]:
    """特定のオブジェクトを抽出"""
    rules = []
    
    for inp, out in train_pairs:
        gi = np.array(inp)
        go = np.array(out)
        bg = most_common_color(inp)
        ho, wo = go.shape
        
        # Find all connected components
        fg_mask = gi != bg
        labeled, n = scipy_label(fg_mask)
        
        found = False
        for i in range(1, n + 1):
            cells = np.where(labeled == i)
            r0, c0 = cells[0].min(), cells[1].min()
            r1, c1 = cells[0].max(), cells[1].max()
            h, w = r1 - r0 + 1, c1 - c0 + 1
            
            if h == ho and w == wo:
                subgrid = gi[r0:r0+h, c0:c0+w].copy()
                # Fill non-object cells with bg
                for r in range(h):
                    for c in range(w):
                        if labeled[r0+r, c0+c] != i:
                            subgrid[r, c] = bg
                
                if np.array_equal(subgrid, go):
                    obj_size = int((labeled == i).sum())
                    rules.append(('exact', obj_size, i))
                    found = True
                    break
                
                # Also try without bg fill
                if np.array_equal(gi[r0:r0+h, c0:c0+w], go):
                    obj_size = int((labeled == i).sum())
                    rules.append(('bbox', obj_size, i))
                    found = True
                    break
        
        if not found:
            return None
    
    if not rules:
        return None
    
    # Determine selection criterion
    # Check if it's always the smallest / largest / specific color
    # For now: check if same size-rank works
    for rank_fn in ['largest', 'smallest', 'second_largest']:
        consistent = True
        for inp, out in train_pairs:
            gi = np.array(inp)
            go = np.array(out)
            bg = most_common_color(inp)
            ho, wo = go.shape
            
            fg_mask = gi != bg
            labeled, n = scipy_label(fg_mask)
            
            sizes = []
            for i in range(1, n + 1):
                sizes.append((int((labeled == i).sum()), i))
            sizes.sort(key=lambda x: -x[0])
            
            if rank_fn == 'largest':
                target_idx = 0
            elif rank_fn == 'smallest':
                target_idx = -1
            elif rank_fn == 'second_largest':
                target_idx = 1 if len(sizes) > 1 else 0
            
            if target_idx >= len(sizes):
                consistent = False
                break
            
            _, label_id = sizes[target_idx]
            cells = np.where(labeled == label_id)
            r0, c0 = cells[0].min(), cells[1].min()
            r1, c1 = cells[0].max(), cells[1].max()
            h, w = r1 - r0 + 1, c1 - c0 + 1
            
            if h != ho or w != wo:
                consistent = False
                break
            
            extracted = gi[r0:r0+h, c0:c0+w].copy()
            if not np.array_equal(extracted, go):
                # Try with bg fill
                for r in range(h):
                    for c in range(w):
                        if labeled[r0+r, c0+c] != label_id:
                            extracted[r, c] = bg
                if not np.array_equal(extracted, go):
                    consistent = False
                    break
        
        if consistent:
            return {'type': 'extract_object', 'rank': rank_fn}
    
    return None


def _detect_downscale(train_pairs) -> Optional[Dict]:
    """入力を縮小"""
    ratios = set()
    for inp, out in train_pairs:
        hi, wi = len(inp), len(inp[0])
        ho, wo = len(out), len(out[0])
        
        if hi % ho != 0 or wi % wo != 0:
            return None
        rh = hi // ho
        rw = wi // wo
        if rh != rw:
            return None
        ratios.add(rh)
    
    if len(ratios) != 1:
        return None
    
    scale = ratios.pop()
    if scale <= 1:
        return None
    
    # What reduction? majority vote per block
    for mode in ['majority', 'minority', 'any_fg']:
        ok = True
        for inp, out in train_pairs:
            gi = np.array(inp)
            go = np.array(out)
            bg = most_common_color(inp)
            ho, wo = go.shape
            
            for r in range(ho):
                for c in range(wo):
                    block = gi[r*scale:(r+1)*scale, c*scale:(c+1)*scale]
                    
                    if mode == 'majority':
                        pred = Counter(block.flatten()).most_common(1)[0][0]
                    elif mode == 'minority':
                        counts = Counter(block.flatten())
                        pred = counts.most_common()[-1][0]
                    elif mode == 'any_fg':
                        fg = [int(x) for x in block.flatten() if x != bg]
                        pred = fg[0] if fg else bg
                    
                    if pred != go[r, c]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return {'type': 'downscale', 'scale': scale, 'mode': mode}
    
    return None


def size_change_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """サイズ変更タスクソルバー"""
    
    for detector in [_detect_upscale_apply, _detect_tile_apply, 
                     _detect_crop_apply, _detect_extract_apply,
                     _detect_downscale_apply]:
        result = detector(train_pairs, test_input)
        if result is not None:
            return result
    
    return None


def _detect_upscale_apply(train_pairs, test_input):
    info = _detect_scale(train_pairs)
    if info is None:
        return None
    s = info['scale']
    gi = np.array(test_input)
    h, w = gi.shape
    result = np.zeros((h*s, w*s), dtype=gi.dtype)
    for r in range(h):
        for c in range(w):
            result[r*s:(r+1)*s, c*s:(c+1)*s] = gi[r, c]
    return result.tolist()


def _detect_tile_apply(train_pairs, test_input):
    info = _detect_tile(train_pairs)
    if info is None:
        return None
    gi = np.array(test_input)
    h, w = gi.shape
    rh, rw = info['rh'], info['rw']
    result = np.tile(gi, (rh, rw))
    return result.tolist()


def _detect_crop_apply(train_pairs, test_input):
    info = _detect_crop(train_pairs)
    if info is None:
        return None
    gi = np.array(test_input)
    
    if info['type'] == 'crop_fixed':
        r, c, h, w = info['r'], info['c'], info['h'], info['w']
        return gi[r:r+h, c:c+w].tolist()
    elif info['type'] == 'crop_relative_fg':
        bg = most_common_color(test_input)
        fg_mask = gi != bg
        if not fg_mask.any():
            return None
        fg_rows, fg_cols = np.where(fg_mask)
        fg_r0, fg_c0 = fg_rows.min(), fg_cols.min()
        dr, dc = info['offset']
        r, c = fg_r0 + dr, fg_c0 + dc
        h, w = info['h'], info['w']
        if r < 0 or c < 0 or r + h > gi.shape[0] or c + w > gi.shape[1]:
            return None
        return gi[r:r+h, c:c+w].tolist()
    
    return None


def _detect_extract_apply(train_pairs, test_input):
    info = _detect_extract_object(train_pairs)
    if info is None:
        return None
    
    gi = np.array(test_input)
    bg = most_common_color(test_input)
    fg_mask = gi != bg
    labeled, n = scipy_label(fg_mask)
    
    sizes = []
    for i in range(1, n + 1):
        sizes.append((int((labeled == i).sum()), i))
    sizes.sort(key=lambda x: -x[0])
    
    rank = info['rank']
    if rank == 'largest':
        idx = 0
    elif rank == 'smallest':
        idx = -1
    elif rank == 'second_largest':
        idx = 1 if len(sizes) > 1 else 0
    
    if idx >= len(sizes):
        return None
    
    _, label_id = sizes[idx]
    cells = np.where(labeled == label_id)
    r0, c0 = cells[0].min(), cells[1].min()
    r1, c1 = cells[0].max(), cells[1].max()
    
    extracted = gi[r0:r1+1, c0:c1+1].copy()
    # Fill non-object cells with bg
    for r in range(extracted.shape[0]):
        for c in range(extracted.shape[1]):
            if labeled[r0+r, c0+c] != label_id:
                extracted[r, c] = bg
    
    # Verify on train first
    for inp, out in train_pairs:
        pred = _detect_extract_apply([], inp)  # won't work recursively
    
    return extracted.tolist()


def _detect_downscale_apply(train_pairs, test_input):
    info = _detect_downscale(train_pairs)
    if info is None:
        return None
    
    s = info['scale']
    mode = info['mode']
    gi = np.array(test_input)
    bg = most_common_color(test_input)
    h, w = gi.shape
    ho, wo = h // s, w // s
    
    if ho == 0 or wo == 0:
        return None
    
    result = np.zeros((ho, wo), dtype=gi.dtype)
    for r in range(ho):
        for c in range(wo):
            block = gi[r*s:(r+1)*s, c*s:(c+1)*s]
            if mode == 'majority':
                result[r, c] = Counter(block.flatten()).most_common(1)[0][0]
            elif mode == 'minority':
                result[r, c] = Counter(block.flatten()).most_common()[-1][0]
            elif mode == 'any_fg':
                fg = [int(x) for x in block.flatten() if x != bg]
                result[r, c] = fg[0] if fg else bg
    
    return result.tolist()


# ─── 多スケールCrossソルバー ───

def _downscale_grid(grid, block_size):
    g = np.array(grid)
    h, w = g.shape
    nh, nw = h // block_size, w // block_size
    if nh < 2 or nw < 2:
        return None
    result = []
    for r in range(nh):
        row = []
        for c in range(nw):
            block = g[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size]
            row.append(int(Counter(block.flatten()).most_common(1)[0][0]))
        result.append(row)
    return result


def multiscale_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """多スケールで6軸Crossソルバーを適用"""
    from arc.cross6axis import cross6_solve
    from arc.cross6_fill_v2 import cross6_fill_v2_solve
    
    for bs in [2, 3]:
        # Downscale all grids
        ds_pairs = []
        valid = True
        for inp, out in train_pairs:
            ds_in = _downscale_grid(inp, bs)
            ds_out = _downscale_grid(out, bs)
            if ds_in is None or ds_out is None:
                valid = False
                break
            ds_pairs.append((ds_in, ds_out))
        
        if not valid:
            continue
        
        ds_test = _downscale_grid(test_input, bs)
        if ds_test is None:
            continue
        
        # Try solving in downscaled space
        for solver in [cross6_solve, cross6_fill_v2_solve]:
            r = solver(ds_pairs, ds_test)
            if r is not None:
                # Upscale result
                result = np.zeros_like(np.array(test_input))
                for row_i, row in enumerate(r):
                    for col_i, val in enumerate(row):
                        result[row_i*bs:(row_i+1)*bs, col_i*bs:(col_i+1)*bs] = val
                return result.tolist()
    
    return None
