"""
arc/cross6_brute.py — ブルートフォース操作ソルバー

学習（ルール検出）を介さず、全操作を直接train全例に適用して
一致するものを見つける。最も信頼性が高い。
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import Counter
from scipy.ndimage import label as scipy_label
from arc.grid import grid_eq, grid_shape, most_common_color


# ──── 操作関数群 ────

def _gravity(grid, direction):
    g = np.array(grid, dtype=np.int8)
    h, w = g.shape
    bg = Counter(g.flatten()).most_common(1)[0][0]
    if direction == 'down':
        for c in range(w):
            col = [g[r, c] for r in range(h) if g[r, c] != bg]
            for r in range(h): g[r, c] = bg
            for i, v in enumerate(reversed(col)): g[h-1-i, c] = v
    elif direction == 'up':
        for c in range(w):
            col = [g[r, c] for r in range(h) if g[r, c] != bg]
            for r in range(h): g[r, c] = bg
            for i, v in enumerate(col): g[i, c] = v
    elif direction == 'right':
        for r in range(h):
            row = [g[r, c] for c in range(w) if g[r, c] != bg]
            for c in range(w): g[r, c] = bg
            for i, v in enumerate(reversed(row)): g[r, w-1-i] = v
    elif direction == 'left':
        for r in range(h):
            row = [g[r, c] for c in range(w) if g[r, c] != bg]
            for c in range(w): g[r, c] = bg
            for i, v in enumerate(row): g[r, i] = v
    return g.tolist()


def _gravity_with_walls(grid, direction):
    """壁（非bg非対象色）に当たると止まる重力"""
    g = np.array(grid, dtype=np.int8)
    h, w = g.shape
    bg = Counter(g.flatten()).most_common(1)[0][0]
    result = g.copy()

    if direction in ('down', 'up'):
        for c in range(w):
            segments = []
            current_start = 0
            for r in range(h):
                # 壁はそのまま
                pass
            # セル単位で落とす
            occupied = np.zeros(h, dtype=bool)
            # まず壁を固定
            walls = set()
            for r in range(h):
                # 壁の判定: 他のfgセルが同列で一番多い色
                pass
            # Simple approach: just move non-bg cells
            if direction == 'down':
                col = []
                for r in range(h):
                    if g[r, c] != bg:
                        col.append(int(g[r, c]))
                    result[r, c] = bg
                for i, v in enumerate(reversed(col)):
                    result[h-1-i, c] = v
            else:
                col = []
                for r in range(h):
                    if g[r, c] != bg:
                        col.append(int(g[r, c]))
                    result[r, c] = bg
                for i, v in enumerate(col):
                    result[i, c] = v
    else:
        for r in range(h):
            if direction == 'right':
                row = [int(g[r, c]) for c in range(w) if g[r, c] != bg]
                for c in range(w): result[r, c] = bg
                for i, v in enumerate(reversed(row)): result[r, w-1-i] = v
            else:
                row = [int(g[r, c]) for c in range(w) if g[r, c] != bg]
                for c in range(w): result[r, c] = bg
                for i, v in enumerate(row): result[r, i] = v

    return result.tolist()


def _flood_fill_enclosed(grid, fill_color):
    g = np.array(grid, dtype=np.int8)
    h, w = g.shape
    bg = Counter(g.flatten()).most_common(1)[0][0]
    bg_mask = (g == bg)
    labeled, n = scipy_label(bg_mask)
    border_labels = set()
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h-1 or c == 0 or c == w-1) and labeled[r, c] > 0:
                border_labels.add(labeled[r, c])
    result = g.copy()
    for i in range(1, n + 1):
        if i not in border_labels:
            result[labeled == i] = fill_color
    return result.tolist()


def _flood_fill_each_color(grid):
    """各閉じた背景領域を、囲んでいるfg色で塗る"""
    g = np.array(grid, dtype=np.int8)
    h, w = g.shape
    bg = Counter(g.flatten()).most_common(1)[0][0]
    bg_mask = (g == bg)
    labeled, n = scipy_label(bg_mask)
    border_labels = set()
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h-1 or c == 0 or c == w-1) and labeled[r, c] > 0:
                border_labels.add(labeled[r, c])
    result = g.copy()
    for i in range(1, n + 1):
        if i in border_labels:
            continue
        # 囲んでいる色を特定
        adj_colors = Counter()
        for r, c in zip(*np.where(labeled == i)):
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != bg and labeled[nr, nc] != i:
                    adj_colors[int(g[nr, nc])] += 1
        if adj_colors:
            fill_c = adj_colors.most_common(1)[0][0]
            result[labeled == i] = fill_c
    return result.tolist()


def _symmetrize(grid, axis):
    g = np.array(grid, dtype=np.int8)
    h, w = g.shape
    bg = Counter(g.flatten()).most_common(1)[0][0]
    result = g.copy()
    if axis == 'h':
        for r in range(h):
            for c in range(w):
                mc = w - 1 - c
                if result[r, c] == bg and result[r, mc] != bg:
                    result[r, c] = result[r, mc]
                elif result[r, c] != bg and result[r, mc] == bg:
                    result[r, mc] = result[r, c]
    elif axis == 'v':
        for r in range(h):
            for c in range(w):
                mr = h - 1 - r
                if result[r, c] == bg and result[mr, c] != bg:
                    result[r, c] = result[mr, c]
                elif result[r, c] != bg and result[mr, c] == bg:
                    result[mr, c] = result[r, c]
    elif axis == 'both':
        for r in range(h):
            for c in range(w):
                candidates = [
                    result[r, c], result[r, w-1-c],
                    result[h-1-r, c], result[h-1-r, w-1-c]
                ]
                non_bg = [v for v in candidates if v != bg]
                if non_bg:
                    fill = Counter(non_bg).most_common(1)[0][0]
                    for (pr, pc) in [(r,c),(r,w-1-c),(h-1-r,c),(h-1-r,w-1-c)]:
                        if result[pr, pc] == bg:
                            result[pr, pc] = fill
    elif axis == 'rot4':
        for r in range(h):
            for c in range(w):
                positions = [(r,c),(c,h-1-r),(h-1-r,w-1-c),(w-1-c,r)]
                vals = [int(result[pr,pc]) for pr,pc in positions
                        if 0<=pr<h and 0<=pc<w and result[pr,pc] != bg]
                if vals:
                    fill = Counter(vals).most_common(1)[0][0]
                    for pr, pc in positions:
                        if 0<=pr<h and 0<=pc<w and result[pr,pc] == bg:
                            result[pr, pc] = fill
    return result.tolist()


def _color_swap(grid, c1, c2):
    g = np.array(grid, dtype=np.int8)
    result = g.copy()
    result[g == c1] = c2
    result[g == c2] = c1
    return result.tolist()


def _color_map(grid, cmap):
    g = np.array(grid, dtype=np.int8)
    result = g.copy()
    for k, v in cmap.items():
        result[g == k] = v
    return result.tolist()


def _global_transform(grid, name):
    g = np.array(grid)
    if name == 'flip_h': return g[:,::-1].tolist()
    if name == 'flip_v': return g[::-1,:].tolist()
    if name == 'rot90': return np.rot90(g, -1).tolist()
    if name == 'rot180': return np.rot90(g, 2).tolist()
    if name == 'rot270': return np.rot90(g, 1).tolist()
    if name == 'transpose': return g.T.tolist()
    return None


def _upscale(grid, scale):
    g = np.array(grid)
    h, w = g.shape
    result = np.zeros((h*scale, w*scale), dtype=g.dtype)
    for r in range(h):
        for c in range(w):
            result[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = g[r, c]
    return result.tolist()


def _tile(grid, rh, rw):
    g = np.array(grid)
    return np.tile(g, (rh, rw)).tolist()


def _extract_largest(grid):
    g = np.array(grid)
    bg = Counter(g.flatten()).most_common(1)[0][0]
    fg = g != bg
    labeled, n = scipy_label(fg)
    if n == 0: return None
    sizes = [(int((labeled==i).sum()), i) for i in range(1, n+1)]
    sizes.sort(key=lambda x: -x[0])
    _, lid = sizes[0]
    cells = np.where(labeled == lid)
    r0, c0 = cells[0].min(), cells[1].min()
    r1, c1 = cells[0].max(), cells[1].max()
    sub = g[r0:r1+1, c0:c1+1].copy()
    mask = labeled[r0:r1+1, c0:c1+1] == lid
    sub[~mask] = bg
    return sub.tolist()


def _extract_smallest(grid):
    g = np.array(grid)
    bg = Counter(g.flatten()).most_common(1)[0][0]
    fg = g != bg
    labeled, n = scipy_label(fg)
    if n == 0: return None
    sizes = [(int((labeled==i).sum()), i) for i in range(1, n+1)]
    sizes.sort(key=lambda x: x[0])
    _, lid = sizes[0]
    cells = np.where(labeled == lid)
    r0, c0 = cells[0].min(), cells[1].min()
    r1, c1 = cells[0].max(), cells[1].max()
    sub = g[r0:r1+1, c0:c1+1].copy()
    mask = labeled[r0:r1+1, c0:c1+1] == lid
    sub[~mask] = bg
    return sub.tolist()


def _crop_to_content(grid):
    g = np.array(grid)
    bg = Counter(g.flatten()).most_common(1)[0][0]
    fg = g != bg
    if not fg.any(): return None
    rows, cols = np.where(fg)
    return g[rows.min():rows.max()+1, cols.min():cols.max()+1].tolist()


def _downscale(grid, scale, mode='majority'):
    g = np.array(grid)
    h, w = g.shape
    nh, nw = h // scale, w // scale
    if nh == 0 or nw == 0: return None
    bg = Counter(g.flatten()).most_common(1)[0][0]
    result = np.zeros((nh, nw), dtype=g.dtype)
    for r in range(nh):
        for c in range(nw):
            block = g[r*scale:(r+1)*scale, c*scale:(c+1)*scale]
            if mode == 'majority':
                result[r, c] = Counter(block.flatten()).most_common(1)[0][0]
            elif mode == 'any_fg':
                fg = [int(x) for x in block.flatten() if x != bg]
                result[r, c] = fg[0] if fg else bg
    return result.tolist()


def _connect_same_color(grid, connect_color=None):
    """同色オブジェクトを水平/垂直の線で接続"""
    g = np.array(grid, dtype=np.int8)
    h, w = g.shape
    bg = Counter(g.flatten()).most_common(1)[0][0]
    result = g.copy()

    fg_mask = g != bg
    labeled, n = scipy_label(fg_mask)

    # 各オブジェクトの色と中心
    objs = []
    for i in range(1, n + 1):
        cells = np.where(labeled == i)
        color = int(Counter(g[cells].flatten()).most_common(1)[0][0])
        cr = cells[0].mean()
        cc = cells[1].mean()
        r0, c0 = cells[0].min(), cells[1].min()
        r1, c1 = cells[0].max(), cells[1].max()
        objs.append({'label': i, 'color': color, 'cr': cr, 'cc': cc,
                      'r0': r0, 'c0': c0, 'r1': r1, 'c1': c1})

    # 同色ペアを接続
    used_color = connect_color if connect_color is not None else None
    for i in range(len(objs)):
        for j in range(i+1, len(objs)):
            a, b = objs[i], objs[j]
            if a['color'] != b['color']:
                continue
            lc = used_color if used_color is not None else a['color']

            # 同じ行範囲 → 水平接続
            if a['r0'] <= b['r1'] and b['r0'] <= a['r1']:
                r = int(round((a['cr'] + b['cr']) / 2))
                c_start = min(a['c1'], b['c1']) + 1
                c_end = max(a['c0'], b['c0'])
                for c in range(c_start, c_end):
                    if 0 <= c < w and result[r, c] == bg:
                        result[r, c] = lc

            # 同じ列範囲 → 垂直接続
            if a['c0'] <= b['c1'] and b['c0'] <= a['c1']:
                c = int(round((a['cc'] + b['cc']) / 2))
                r_start = min(a['r1'], b['r1']) + 1
                r_end = max(a['r0'], b['r0'])
                for r in range(r_start, r_end):
                    if 0 <= r < h and result[r, c] == bg:
                        result[r, c] = lc

    return result.tolist()


# ──── ブルートフォース試行 ────

def brute_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """全操作をbrute-forceで試行"""
    same_size = all(grid_shape(i) == grid_shape(o) for i, o in train_pairs)
    all_colors = set()
    for inp, out in train_pairs:
        all_colors.update(int(x) for x in np.array(inp).flatten())
        all_colors.update(int(x) for x in np.array(out).flatten())

    # === Same-size operations ===
    if same_size:
        # Gravity
        for d in ['down', 'up', 'left', 'right']:
            if _check(train_pairs, lambda g: _gravity(g, d)):
                return _gravity(test_input, d)

        # Symmetrize
        for axis in ['h', 'v', 'both', 'rot4']:
            if _check(train_pairs, lambda g, a=axis: _symmetrize(g, a)):
                return _symmetrize(test_input, axis)

        # Flood fill (enclosed regions)
        for fc in all_colors:
            if _check(train_pairs, lambda g, c=fc: _flood_fill_enclosed(g, c)):
                return _flood_fill_enclosed(test_input, fc)

        # Flood fill (each color from neighbors)
        if _check(train_pairs, _flood_fill_each_color):
            return _flood_fill_each_color(test_input)

        # Color swap
        colors_list = sorted(all_colors)
        for i in range(len(colors_list)):
            for j in range(i+1, len(colors_list)):
                c1, c2 = colors_list[i], colors_list[j]
                if _check(train_pairs, lambda g, a=c1, b=c2: _color_swap(g, a, b)):
                    return _color_swap(test_input, c1, c2)

        # Global transforms
        for t in ['flip_h', 'flip_v', 'rot90', 'rot180', 'rot270', 'transpose']:
            if _check(train_pairs, lambda g, n=t: _global_transform(g, n)):
                return _global_transform(test_input, t)

        # Connect same color
        if _check(train_pairs, _connect_same_color):
            return _connect_same_color(test_input)
        for cc in all_colors:
            if _check(train_pairs, lambda g, c=cc: _connect_same_color(g, c)):
                return _connect_same_color(test_input, cc)

        # Color map (learn from first pair, verify on rest)
        cmap = _learn_colormap(train_pairs)
        if cmap is not None:
            if _check(train_pairs, lambda g, m=cmap: _color_map(g, m)):
                return _color_map(test_input, cmap)

    # === Size-change operations ===
    # Upscale
    for scale in [2, 3, 4, 5]:
        if _check(train_pairs, lambda g, s=scale: _upscale(g, s)):
            return _upscale(test_input, scale)

    # Tile
    if train_pairs:
        hi, wi = len(train_pairs[0][0]), len(train_pairs[0][0][0])
        ho, wo = len(train_pairs[0][1]), len(train_pairs[0][1][0])
        if ho > hi and wo > wi and ho % hi == 0 and wo % wi == 0:
            rh, rw = ho // hi, wo // wi
            if _check(train_pairs, lambda g, a=rh, b=rw: _tile(g, a, b)):
                return _tile(test_input, rh, rw)

    # Downscale
    for scale in [2, 3, 4, 5]:
        for mode in ['majority', 'any_fg']:
            if _check(train_pairs, lambda g, s=scale, m=mode: _downscale(g, s, m)):
                r = _downscale(test_input, scale, mode)
                if r is not None:
                    return r

    # Extract
    for fn in [_extract_largest, _extract_smallest, _crop_to_content]:
        if _check(train_pairs, fn):
            return fn(test_input)

    return None


def _check(train_pairs, fn):
    """全train例でfnが正解を出すか"""
    for inp, out in train_pairs:
        try:
            pred = fn(inp)
            if pred is None or not grid_eq(pred, out):
                return False
        except:
            return False
    return True


def _learn_colormap(train_pairs):
    """色マップを学習"""
    cmap = {}
    for inp, out in train_pairs:
        gi = np.array(inp)
        go = np.array(out)
        if gi.shape != go.shape:
            return None
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                ic, oc = int(gi[r,c]), int(go[r,c])
                if ic in cmap:
                    if cmap[ic] != oc:
                        return None
                else:
                    cmap[ic] = oc
    if all(k == v for k, v in cmap.items()):
        return None
    return cmap
