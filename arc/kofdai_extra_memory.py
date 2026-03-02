"""
arc/kofdai_extra_memory.py — 追加のフル記憶ソルバー

kofdaiが言ったが未実装だったパターン + ARC頻出パターン

=== 追加パターン ===
A. 重力（物が落ちる）— 上下左右4方向
B. 線を引く（2点間をconnect）
C. クロップ/抽出（一番目立つ部分を切り出す）
D. 拡大/縮小（スケール変換）
E. separator → panel操作（XOR/AND/OR）
F. 増減パターン（オブジェクトサイズの等差数列）
G. テンプレートマッチング（柄の類似）
H. flood_fill（色の広がり）
I. 辺の等しさ→変換の選択
J. 交互塗り（行/列ごとに交互）
K. 最大/最小オブジェクト抽出
L. オブジェクト数え上げ→出力
M. dedup（重複行/列の削除）
N. rotate/flip全探索
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import Counter
from scipy.ndimage import label as scipy_label
from arc.grid import grid_eq, most_common_color


def _objects(g, bg):
    mask = (g != bg).astype(int)
    labeled, n = scipy_label(mask)
    objs = []
    for i in range(1, n + 1):
        cells = list(zip(*np.where(labeled == i)))
        colors = [int(g[r, c]) for r, c in cells]
        r_min = min(r for r,c in cells)
        c_min = min(c for r,c in cells)
        r_max = max(r for r,c in cells)
        c_max = max(c for r,c in cells)
        objs.append({
            'cells': cells, 'size': len(cells),
            'color': Counter(colors).most_common(1)[0][0],
            'colors': set(colors),
            'bbox': (r_min, c_min, r_max, c_max),
            'shape': frozenset((r-r_min, c-c_min) for r,c in cells),
            'grid': g[r_min:r_max+1, c_min:c_max+1].copy(),
        })
    return objs


# ══════════════════════════════════════════════════════════════
# A. 重力
# ══════════════════════════════════════════════════════════════

def gravity_solve(train_pairs, test_input):
    """物が落ちる: 4方向の重力"""
    for direction in ['down', 'up', 'left', 'right']:
        result = _apply_gravity(test_input, direction)
        if result is not None:
            # train検証
            ok = all(grid_eq(_apply_gravity(inp, direction), out) 
                     for inp, out in train_pairs
                     if _apply_gravity(inp, direction) is not None)
            if ok:
                return result
    return None

def _apply_gravity(grid, direction):
    g = np.array(grid)
    h, w = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    result = np.full_like(g, bg)
    
    if direction in ('down', 'up'):
        for c in range(w):
            colors = [int(g[r, c]) for r in range(h) if g[r, c] != bg]
            if not colors: continue
            if direction == 'down':
                for i, color in enumerate(reversed(colors)):
                    result[h-1-i, c] = color
            else:
                for i, color in enumerate(colors):
                    result[i, c] = color
    else:  # left, right
        for r in range(h):
            colors = [int(g[r, c]) for c in range(w) if g[r, c] != bg]
            if not colors: continue
            if direction == 'right':
                for i, color in enumerate(reversed(colors)):
                    result[r, w-1-i] = color
            else:
                for i, color in enumerate(colors):
                    result[r, i] = color
    
    return result.tolist()


# ══════════════════════════════════════════════════════════════
# B. 2点間connect（線を引く）
# ══════════════════════════════════════════════════════════════

def connect_solve(train_pairs, test_input):
    """同色2点間に線を引く"""
    gi = np.array(test_input)
    h, w = gi.shape
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    # trainから「どの色の点を結ぶか」学習
    connect_colors = set()
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        bg_t = int(Counter(ga.flatten()).most_common(1)[0][0])
        
        added = list(zip(*np.where((ga == bg_t) & (go != bg_t))))
        for r, c in added:
            connect_colors.add(int(go[r, c]))
    
    if not connect_colors:
        return None
    
    result = [row[:] for row in test_input]
    changed = False
    
    for color in connect_colors:
        points = [(r, c) for r in range(h) for c in range(w) if gi[r, c] == color]
        if len(points) < 2:
            continue
        
        # 全ペアで水平/垂直の線を引く
        for i, (r1, c1) in enumerate(points):
            for r2, c2 in points[i+1:]:
                if r1 == r2:  # 水平
                    for c in range(min(c1,c2)+1, max(c1,c2)):
                        if result[r1][c] == bg:
                            result[r1][c] = color
                            changed = True
                elif c1 == c2:  # 垂直
                    for r in range(min(r1,r2)+1, max(r1,r2)):
                        if result[r][c1] == bg:
                            result[r][c1] = color
                            changed = True
    
    return result if changed else None


# ══════════════════════════════════════════════════════════════
# C. クロップ/抽出
# ══════════════════════════════════════════════════════════════

def crop_solve(train_pairs, test_input):
    """オブジェクトの切り出し"""
    # trainの出力サイズパターンから切り出しルールを学習
    gi = np.array(test_input)
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    objs = _objects(gi, bg)
    
    if not objs:
        return None
    
    for inp, out in train_pairs:
        ga = np.array(inp)
        go = np.array(out)
        oh, ow = go.shape
        bg_t = int(Counter(ga.flatten()).most_common(1)[0][0])
        train_objs = _objects(ga, bg_t)
        
        # 出力 = 最大オブジェクトのbbox?
        for strategy in ['largest', 'smallest', 'specific_color']:
            if strategy == 'largest':
                sorted_objs = sorted(train_objs, key=lambda o: -o['size'])
            elif strategy == 'smallest':
                sorted_objs = sorted(train_objs, key=lambda o: o['size'])
            else:
                continue
            
            for obj in sorted_objs:
                r1, c1, r2, c2 = obj['bbox']
                if r2-r1+1 == oh and c2-c1+1 == ow:
                    crop = ga[r1:r2+1, c1:c2+1]
                    if np.array_equal(crop, go):
                        # test に同じ戦略を適用
                        test_sorted = sorted(objs, key=lambda o: -o['size'] if strategy == 'largest' else o['size'])
                        for tobj in test_sorted:
                            tr1, tc1, tr2, tc2 = tobj['bbox']
                            return gi[tr1:tr2+1, tc1:tc2+1].tolist()
        
        # 出力 = 非bgのbbox全体?
        fg = np.where(ga != bg_t)
        if len(fg[0]) > 0:
            rr_min, rr_max = fg[0].min(), fg[0].max()
            cc_min, cc_max = fg[1].min(), fg[1].max()
            if rr_max-rr_min+1 == oh and cc_max-cc_min+1 == ow:
                crop = ga[rr_min:rr_max+1, cc_min:cc_max+1]
                if np.array_equal(crop, go):
                    fg_test = np.where(gi != bg)
                    if len(fg_test[0]) > 0:
                        return gi[fg_test[0].min():fg_test[0].max()+1, 
                                  fg_test[1].min():fg_test[1].max()+1].tolist()
        break  # 1つのtrainから学習
    
    return None


# ══════════════════════════════════════════════════════════════
# D. スケール変換（拡大/縮小）
# ══════════════════════════════════════════════════════════════

def scale_solve(train_pairs, test_input):
    """拡大/縮小"""
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        ih, iw = gi.shape
        oh, ow = go.shape
        
        if oh > ih and ow > iw:
            # 拡大
            if oh % ih == 0 and ow % iw == 0:
                sr, sc = oh // ih, ow // iw
                if sr == sc:
                    # 各セルをsr×sc倍
                    check = np.repeat(np.repeat(gi, sr, axis=0), sc, axis=1)
                    if np.array_equal(check, go):
                        ti = np.array(test_input)
                        result = np.repeat(np.repeat(ti, sr, axis=0), sc, axis=1)
                        return result.tolist()
        
        elif oh < ih and ow < iw:
            # 縮小
            if ih % oh == 0 and iw % ow == 0:
                sr, sc = ih // oh, iw // ow
                if sr == sc:
                    # 各sr×scブロックの代表値
                    result = np.zeros((oh, ow), dtype=int)
                    for r in range(oh):
                        for c in range(ow):
                            block = gi[r*sr:(r+1)*sr, c*sc:(c+1)*sc]
                            result[r, c] = Counter(block.flatten()).most_common(1)[0][0]
                    if np.array_equal(result, go):
                        ti = np.array(test_input)
                        th, tw = ti.shape
                        res = np.zeros((th//sr, tw//sc), dtype=int)
                        for r in range(th//sr):
                            for c in range(tw//sc):
                                block = ti[r*sr:(r+1)*sr, c*sc:(c+1)*sc]
                                res[r, c] = Counter(block.flatten()).most_common(1)[0][0]
                        return res.tolist()
        break
    
    return None


# ══════════════════════════════════════════════════════════════
# E. Separator → Panel操作
# ══════════════════════════════════════════════════════════════

def separator_solve(train_pairs, test_input):
    """区切り線でパネル分割 → 論理演算"""
    gi = np.array(test_input)
    h, w = gi.shape
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    # separator検出
    sep_color, panels = _split_panels(gi, bg)
    if sep_color is None or len(panels) < 2:
        return None
    
    # trainで同じseparator構造を確認
    for op_name, op_fn in [('xor', _panel_xor), ('and', _panel_and), 
                            ('or', _panel_or), ('diff', _panel_diff)]:
        ok = True
        for inp, out in train_pairs:
            ga = np.array(inp)
            bg_t = int(Counter(ga.flatten()).most_common(1)[0][0])
            sc, ps = _split_panels(ga, bg_t)
            if sc is None or len(ps) < 2:
                ok = False; break
            
            pred = op_fn(ps, bg_t)
            if pred is None or not grid_eq(pred, out):
                ok = False; break
        
        if ok:
            result = op_fn(panels, bg)
            if result is not None:
                return result
    
    return None


def _split_panels(g, bg):
    h, w = g.shape
    sep_color = None
    
    # 全行を埋める色を探す
    for r in range(h):
        vals = set(int(v) for v in g[r])
        if len(vals) == 1 and vals.pop() != bg:
            sep_color = int(g[r, 0])
            break
    
    # 全列を埋める色
    if sep_color is None:
        for c in range(w):
            vals = set(int(v) for v in g[:, c])
            if len(vals) == 1 and vals.pop() != bg:
                sep_color = int(g[0, c])
                break
    
    if sep_color is None:
        return None, []
    
    # separator行/列で分割
    sep_rows = [r for r in range(h) if all(g[r, c] == sep_color for c in range(w))]
    sep_cols = [c for c in range(w) if all(g[r, c] == sep_color for r in range(h))]
    
    panels = []
    row_bounds = [-1] + sep_rows + [h]
    col_bounds = [-1] + sep_cols + [w]
    
    for i in range(len(row_bounds) - 1):
        for j in range(len(col_bounds) - 1):
            r1 = row_bounds[i] + 1
            r2 = row_bounds[i + 1]
            c1 = col_bounds[j] + 1
            c2 = col_bounds[j + 1]
            if r1 < r2 and c1 < c2:
                panels.append(g[r1:r2, c1:c2])
    
    return sep_color, panels


def _panel_xor(panels, bg):
    if len(panels) < 2: return None
    p0, p1 = panels[0], panels[1]
    if p0.shape != p1.shape: return None
    result = np.full_like(p0, bg)
    for r in range(p0.shape[0]):
        for c in range(p0.shape[1]):
            v0 = p0[r,c] != bg
            v1 = p1[r,c] != bg
            if v0 != v1:
                result[r,c] = p0[r,c] if v0 else p1[r,c]
    return result.tolist()

def _panel_and(panels, bg):
    if len(panels) < 2: return None
    p0, p1 = panels[0], panels[1]
    if p0.shape != p1.shape: return None
    result = np.full_like(p0, bg)
    for r in range(p0.shape[0]):
        for c in range(p0.shape[1]):
            if p0[r,c] != bg and p1[r,c] != bg:
                result[r,c] = p0[r,c]
    return result.tolist()

def _panel_or(panels, bg):
    if len(panels) < 2: return None
    p0, p1 = panels[0], panels[1]
    if p0.shape != p1.shape: return None
    result = np.full_like(p0, bg)
    for r in range(p0.shape[0]):
        for c in range(p0.shape[1]):
            if p0[r,c] != bg:
                result[r,c] = p0[r,c]
            elif p1[r,c] != bg:
                result[r,c] = p1[r,c]
    return result.tolist()

def _panel_diff(panels, bg):
    if len(panels) < 2: return None
    p0, p1 = panels[0], panels[1]
    if p0.shape != p1.shape: return None
    result = np.full_like(p0, bg)
    for r in range(p0.shape[0]):
        for c in range(p0.shape[1]):
            if p0[r,c] != bg and p1[r,c] == bg:
                result[r,c] = p0[r,c]
    return result.tolist()


# ══════════════════════════════════════════════════════════════
# F. テンプレートマッチング（柄の類似）
# ══════════════════════════════════════════════════════════════

def template_match_solve(train_pairs, test_input):
    """小さいパターン（テンプレート）を見つけて操作"""
    gi = np.array(test_input)
    h, w = gi.shape
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    # trainで「小さいオブジェクト=テンプレート、大きいオブジェクト=キャンバス」
    for inp, out in train_pairs:
        ga = np.array(inp)
        bg_t = int(Counter(ga.flatten()).most_common(1)[0][0])
        objs = _objects(ga, bg_t)
        
        if len(objs) < 2:
            continue
        
        sorted_objs = sorted(objs, key=lambda o: o['size'])
        template = sorted_objs[0]  # 最小
        canvas = sorted_objs[-1]   # 最大
        
        # テンプレートをキャンバス内で探す
        tr, tc, _, _ = template['bbox']
        th = template['grid'].shape[0]
        tw = template['grid'].shape[1]
        
        cr, cc, cr2, cc2 = canvas['bbox']
        
        # 出力でテンプレートの位置がどう変わったか
        go = np.array(out)
        # ... 複雑すぎるのでスキップ
        break
    
    return None


# ══════════════════════════════════════════════════════════════
# G. 最大/最小オブジェクト抽出
# ══════════════════════════════════════════════════════════════

def extract_object_solve(train_pairs, test_input):
    """特定のオブジェクトだけを抽出"""
    for strategy in ['largest', 'smallest', 'most_colors', 'unique_color']:
        ok = True
        for inp, out in train_pairs:
            pred = _extract_by_strategy(inp, strategy)
            if pred is None or not grid_eq(pred, out):
                ok = False
                break
        
        if ok:
            return _extract_by_strategy(test_input, strategy)
    
    return None

def _extract_by_strategy(grid, strategy):
    g = np.array(grid)
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    objs = _objects(g, bg)
    
    if not objs:
        return None
    
    if strategy == 'largest':
        obj = max(objs, key=lambda o: o['size'])
    elif strategy == 'smallest':
        obj = min(objs, key=lambda o: o['size'])
    elif strategy == 'most_colors':
        obj = max(objs, key=lambda o: len(o['colors']))
    elif strategy == 'unique_color':
        color_counts = Counter(o['color'] for o in objs)
        unique = [o for o in objs if color_counts[o['color']] == 1]
        if not unique: return None
        obj = unique[0]
    else:
        return None
    
    r1, c1, r2, c2 = obj['bbox']
    return g[r1:r2+1, c1:c2+1].tolist()


# ══════════════════════════════════════════════════════════════
# H. Dedup（重複行/列の削除）
# ══════════════════════════════════════════════════════════════

def dedup_solve(train_pairs, test_input):
    """重複行/列を削除"""
    for mode in ['rows', 'cols', 'both']:
        ok = True
        for inp, out in train_pairs:
            pred = _dedup(inp, mode)
            if pred is None or not grid_eq(pred, out):
                ok = False
                break
        if ok:
            return _dedup(test_input, mode)
    return None

def _dedup(grid, mode):
    g = np.array(grid)
    
    if mode in ('rows', 'both'):
        seen = []
        unique_rows = []
        for r in range(g.shape[0]):
            row = tuple(int(v) for v in g[r])
            if row not in seen:
                seen.append(row)
                unique_rows.append(g[r])
        g = np.array(unique_rows) if unique_rows else g
    
    if mode in ('cols', 'both'):
        seen = []
        unique_cols = []
        for c in range(g.shape[1]):
            col = tuple(int(v) for v in g[:, c])
            if col not in seen:
                seen.append(col)
                unique_cols.append(g[:, c])
        if unique_cols:
            g = np.array(unique_cols).T
    
    return g.tolist()


# ══════════════════════════════════════════════════════════════
# I. Rotate/Flip全探索
# ══════════════════════════════════════════════════════════════

def rotate_flip_solve(train_pairs, test_input):
    """単純な回転/反転"""
    transforms = [
        ('rot90', lambda g: np.rot90(g, 1)),
        ('rot180', lambda g: np.rot90(g, 2)),
        ('rot270', lambda g: np.rot90(g, 3)),
        ('flip_h', lambda g: np.fliplr(g)),
        ('flip_v', lambda g: np.flipud(g)),
        ('transpose', lambda g: g.T),
        ('transpose_rot90', lambda g: np.rot90(g.T, 1)),
    ]
    
    for name, fn in transforms:
        ok = True
        for inp, out in train_pairs:
            pred = fn(np.array(inp)).tolist()
            if not grid_eq(pred, out):
                ok = False
                break
        if ok:
            return fn(np.array(test_input)).tolist()
    
    return None


# ══════════════════════════════════════════════════════════════
# J. タイリング（繰り返し配置）
# ══════════════════════════════════════════════════════════════

def tile_solve(train_pairs, test_input):
    """入力をタイル状に繰り返して出力"""
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        ih, iw = gi.shape
        oh, ow = go.shape
        
        if oh >= ih and ow >= iw and oh % ih == 0 and ow % iw == 0:
            rr, rc = oh // ih, ow // iw
            check = np.tile(gi, (rr, rc))
            if np.array_equal(check, go):
                ti = np.array(test_input)
                return np.tile(ti, (rr, rc)).tolist()
        break
    return None


# ══════════════════════════════════════════════════════════════
# K. 交互塗り
# ══════════════════════════════════════════════════════════════

def alternating_fill_solve(train_pairs, test_input):
    """行/列を交互に別色で塗る"""
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        if gi.shape != go.shape: return None
    
    gi = np.array(test_input)
    h, w = gi.shape
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    # trainの出力で行ごとに異なるパターンか
    for inp, out in train_pairs:
        go = np.array(out)
        ha, wa = go.shape
        
        # 偶数行と奇数行の色が違うか
        even_vals = set(int(v) for r in range(0, ha, 2) for v in go[r])
        odd_vals = set(int(v) for r in range(1, ha, 2) for v in go[r])
        
        if even_vals and odd_vals and not (even_vals & odd_vals - {bg}):
            # 交互パターン
            even_color = (even_vals - {bg}).pop() if even_vals - {bg} else bg
            odd_color = (odd_vals - {bg}).pop() if odd_vals - {bg} else bg
            
            result = gi.copy()
            for r in range(h):
                for c in range(w):
                    if gi[r, c] != bg:
                        result[r, c] = even_color if r % 2 == 0 else odd_color
            return result.tolist()
        break
    
    return None


# ══════════════════════════════════════════════════════════════
# マスターリスト
# ══════════════════════════════════════════════════════════════

EXTRA_SOLVERS = [
    ('gravity', gravity_solve),
    ('connect', connect_solve),
    ('crop', crop_solve),
    ('scale', scale_solve),
    ('separator', separator_solve),
    ('extract_object', extract_object_solve),
    ('dedup', dedup_solve),
    ('rotate_flip', rotate_flip_solve),
    ('tile', tile_solve),
    ('alternating_fill', alternating_fill_solve),
]


def extra_solve(train_pairs, test_input):
    """追加ソルバー群"""
    for name, solver in EXTRA_SOLVERS:
        try:
            ok = True
            for inp, out in train_pairs:
                pred = solver(train_pairs, inp)
                if pred is None or not grid_eq(pred, out):
                    ok = False
                    break
            if ok:
                result = solver(train_pairs, test_input)
                if result is not None:
                    return result, name
        except Exception:
            continue
    return None, None


if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    
    split = 'evaluation' if sys.argv[1] == '--eval' else 'training'
    data_dir = Path(f'/tmp/arc-agi-2/data/{split}')
    
    existing = set()
    try:
        with open('arc_cross_engine_v9.log') as f:
            for line in f:
                m = re.search(r'✓.*?([0-9a-f]{8})', line)
                if m: existing.add(m.group(1))
    except: pass
    
    solved = []
    solver_hits = Counter()
    
    for tf in sorted(data_dir.glob('*.json')):
        tid = tf.stem
        with open(tf) as f: task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti, to = task['test'][0]['input'], task['test'][0].get('output')
        
        result, name = extra_solve(tp, ti)
        if result and to and grid_eq(result, to):
            solved.append(tid)
            solver_hits[name] += 1
            tag = 'NEW' if tid not in existing else ''
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t in solved if t not in existing]
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
    print(f'Solver hits: {dict(solver_hits)}')
