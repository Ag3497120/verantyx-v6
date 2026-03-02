"""
arc/cross_elementary.py — Crossの基礎記憶: 小学生の数学 + リバーシ + ソリティア

=== Crossが知っていること ===

【小学生の数学】
- 数える（何個あるか）
- 比べる（大きい/小さい、同じ/違う）
- 並べる（順番、ソート）
- 偶数/奇数
- 対称（左右同じ、上下同じ、回転して同じ）
- 図形（正方形、長方形、L字、十字）
- パターンの繰り返し（1,2,1,2,1,2...）
- 穴埋め（抜けてるところを埋める）

【リバーシ】
- 同色で挟んだら間を塗る（8方向）
- 挟めるか全方向チェック
- 最終的に全部同じ色になる方向に進む

【ソリティア】
- 色の順番でカードを並べる
- 同じスート(色)をまとめる
- 空いた場所に移動できる
- 完成したら取り除く
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label


def _bg(g):
    return int(Counter(np.array(g).flatten()).most_common(1)[0][0])

def _objs(g, bg, conn=8):
    struct = np.ones((3,3), dtype=int) if conn == 8 else np.array([[0,1,0],[1,1,1],[0,1,0]])
    mask = (np.array(g) != bg).astype(int)
    labeled, n = scipy_label(mask, structure=struct)
    objs = []
    for i in range(1, n+1):
        cells = list(zip(*np.where(labeled == i)))
        colors = [int(g[r,c]) for r,c in cells]
        r_min = min(r for r,c in cells); c_min = min(c for r,c in cells)
        r_max = max(r for r,c in cells); c_max = max(c for r,c in cells)
        objs.append({
            'cells': cells, 'size': len(cells),
            'color': Counter(colors).most_common(1)[0][0],
            'colors': set(colors),
            'bbox': (r_min, c_min, r_max, c_max),
            'bh': r_max - r_min + 1, 'bw': c_max - c_min + 1,
            'shape': frozenset((r-r_min, c-c_min) for r,c in cells),
        })
    return objs


# ══════════════════════════════════════════════════════════════
# 小学生の知恵: 基本操作
# ══════════════════════════════════════════════════════════════

def count_and_output(train_pairs, test_input):
    """数える: オブジェクト数/色数を出力"""
    from arc.grid import grid_eq
    
    for count_what in ['n_objects', 'n_colors', 'n_fg_cells']:
        ok = True
        for inp, out in train_pairs:
            ga = np.array(inp); go = np.array(out)
            bg = _bg(ga)
            
            if count_what == 'n_objects':
                val = len(_objs(ga, bg))
            elif count_what == 'n_colors':
                val = len(set(int(v) for v in ga.flatten()) - {bg})
            elif count_what == 'n_fg_cells':
                val = int((ga != bg).sum())
            
            # 出力が1x1でその値か
            if go.shape == (1, 1) and int(go[0, 0]) == val:
                continue
            else:
                ok = False; break
        
        if ok:
            gi = np.array(test_input)
            bg = _bg(gi)
            if count_what == 'n_objects': val = len(_objs(gi, bg))
            elif count_what == 'n_colors': val = len(set(int(v) for v in gi.flatten()) - {bg})
            elif count_what == 'n_fg_cells': val = int((gi != bg).sum())
            return [[val]]
    
    return None


def compare_and_select(train_pairs, test_input):
    """比べる: 最大/最小/特異オブジェクトを抽出"""
    from arc.grid import grid_eq
    
    for strategy in ['largest', 'smallest', 'most_colors', 'unique_color',
                     'tallest', 'widest', 'most_frequent_color']:
        ok = True
        for inp, out in train_pairs:
            p = _select(inp, strategy)
            if p is None or not grid_eq(p, out):
                ok = False; break
        if ok:
            r = _select(test_input, strategy)
            if r is not None: return r
    return None

def _select(grid, strategy):
    g = np.array(grid)
    bg = _bg(g)
    objs = _objs(g, bg)
    if not objs: return None
    
    if strategy == 'largest': obj = max(objs, key=lambda o: o['size'])
    elif strategy == 'smallest': obj = min(objs, key=lambda o: o['size'])
    elif strategy == 'most_colors': obj = max(objs, key=lambda o: len(o['colors']))
    elif strategy == 'tallest': obj = max(objs, key=lambda o: o['bh'])
    elif strategy == 'widest': obj = max(objs, key=lambda o: o['bw'])
    elif strategy == 'unique_color':
        cc = Counter(o['color'] for o in objs)
        unique = [o for o in objs if cc[o['color']] == 1]
        if not unique: return None
        obj = unique[0]
    elif strategy == 'most_frequent_color':
        cc = Counter(o['color'] for o in objs)
        target = cc.most_common(1)[0][0]
        obj = max([o for o in objs if o['color'] == target], key=lambda o: o['size'])
    else:
        return None
    
    r1, c1, r2, c2 = obj['bbox']
    return g[r1:r2+1, c1:c2+1].tolist()


def sort_objects(train_pairs, test_input):
    """並べる: オブジェクトをサイズ/色で並べ替え"""
    from arc.grid import grid_eq
    
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        bg = _bg(gi)
        objs = _objs(gi, bg)
        if len(objs) < 2: return None
        
        # 出力でオブジェクトの順番が変わっているか
        objs_out = _objs(go, bg)
        if len(objs) != len(objs_out): return None
        
        # サイズ順に並んでいるか（出力の行位置で判定）
        out_order = sorted(objs_out, key=lambda o: o['bbox'][0])
        
        for sort_key in ['size', 'color', 'bh', 'bw']:
            sorted_in = sorted(objs, key=lambda o: o[sort_key])
            # 色が一致するか
            if [o['color'] for o in sorted_in] == [o['color'] for o in out_order]:
                # test に適用
                ti = np.array(test_input)
                bg_t = _bg(ti)
                test_objs = _objs(ti, bg_t)
                sorted_test = sorted(test_objs, key=lambda o: o[sort_key])
                
                # 出力グリッド構築
                max_w = max(o['bw'] for o in sorted_test)
                total_h = sum(o['bh'] for o in sorted_test) + len(sorted_test) - 1
                result = np.full((total_h, max_w), bg_t, dtype=int)
                r_pos = 0
                for obj in sorted_test:
                    r1, c1, r2, c2 = obj['bbox']
                    patch = ti[r1:r2+1, c1:c2+1]
                    result[r_pos:r_pos+obj['bh'], :obj['bw']] = patch
                    r_pos += obj['bh'] + 1
                
                if grid_eq(result.tolist(), out):
                    return result.tolist()
        break
    return None


def even_odd_rule(train_pairs, test_input):
    """偶数/奇数: 位置やサイズの偶奇で処理を分ける"""
    from arc.grid import grid_eq
    
    gi = np.array(test_input)
    h, w = gi.shape
    bg = _bg(gi)
    
    # train: 偶数行と奇数行で色が異なるか
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        
        even_changes = {}
        odd_changes = {}
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                if ga[r,c] != go[r,c]:
                    if r % 2 == 0:
                        even_changes[int(ga[r,c])] = int(go[r,c])
                    else:
                        odd_changes[int(ga[r,c])] = int(go[r,c])
        
        if even_changes or odd_changes:
            result = gi.copy()
            for r in range(h):
                for c in range(w):
                    v = int(gi[r,c])
                    if r % 2 == 0 and v in even_changes:
                        result[r,c] = even_changes[v]
                    elif r % 2 == 1 and v in odd_changes:
                        result[r,c] = odd_changes[v]
            
            ok = True
            for inp2, out2 in train_pairs:
                ga2 = np.array(inp2)
                res2 = ga2.copy()
                for r in range(ga2.shape[0]):
                    for c in range(ga2.shape[1]):
                        v = int(ga2[r,c])
                        if r % 2 == 0 and v in even_changes:
                            res2[r,c] = even_changes[v]
                        elif r % 2 == 1 and v in odd_changes:
                            res2[r,c] = odd_changes[v]
                if not grid_eq(res2.tolist(), out2):
                    ok = False; break
            if ok:
                return result.tolist()
        break
    return None


def symmetry_complete(train_pairs, test_input):
    """対称: 左右/上下/回転対称に補完"""
    from arc.grid import grid_eq
    
    gi = np.array(test_input)
    h, w = gi.shape
    bg = _bg(gi)
    
    for sym in ['lr', 'ud', 'lr_ud', 'rot90', 'rot180']:
        ok = True
        for inp, out in train_pairs:
            p = _sym_apply(inp, sym)
            if p is None or not grid_eq(p, out):
                ok = False; break
        if ok:
            r = _sym_apply(test_input, sym)
            if r: return r
    return None

def _sym_apply(grid, sym):
    g = np.array(grid)
    h, w = g.shape
    bg = _bg(g)
    result = g.copy()
    changed = False
    
    if sym == 'lr':
        for r in range(h):
            for c in range(w):
                mc = w - 1 - c
                if result[r,c] == bg and result[r,mc] != bg:
                    result[r,c] = result[r,mc]; changed = True
    elif sym == 'ud':
        for r in range(h):
            mr = h - 1 - r
            for c in range(w):
                if result[r,c] == bg and result[mr,c] != bg:
                    result[r,c] = result[mr,c]; changed = True
    elif sym == 'lr_ud':
        for r in range(h):
            for c in range(w):
                mc = w-1-c; mr = h-1-r
                if result[r,c] == bg:
                    if result[r,mc] != bg: result[r,c] = result[r,mc]; changed = True
                    elif result[mr,c] != bg: result[r,c] = result[mr,c]; changed = True
                    elif result[mr,mc] != bg: result[r,c] = result[mr,mc]; changed = True
    elif sym == 'rot180':
        for r in range(h):
            for c in range(w):
                mr, mc = h-1-r, w-1-c
                if result[r,c] == bg and result[mr,mc] != bg:
                    result[r,c] = result[mr,mc]; changed = True
    
    return result.tolist() if changed else None


def pattern_repeat(train_pairs, test_input):
    """パターンの繰り返し: 周期的に塗る"""
    from arc.grid import grid_eq
    
    gi = np.array(test_input)
    h, w = gi.shape
    bg = _bg(gi)
    
    # 行方向の周期
    for period in range(1, h//2+1):
        if h % period != 0: continue
        tile = gi[:period, :]
        result = np.tile(tile, (h//period, 1))
        if not np.array_equal(result, gi):
            ok = True
            for inp, out in train_pairs:
                ga = np.array(inp)
                if ga.shape[0] % period != 0: ok = False; break
                pred = np.tile(ga[:period, :], (ga.shape[0]//period, 1))
                if not grid_eq(pred.tolist(), out): ok = False; break
            if ok: return result.tolist()
    
    # 列方向
    for period in range(1, w//2+1):
        if w % period != 0: continue
        tile = gi[:, :period]
        result = np.tile(tile, (1, w//period))
        if not np.array_equal(result, gi):
            ok = True
            for inp, out in train_pairs:
                ga = np.array(inp)
                if ga.shape[1] % period != 0: ok = False; break
                pred = np.tile(ga[:, :period], (1, ga.shape[1]//period))
                if not grid_eq(pred.tolist(), out): ok = False; break
            if ok: return result.tolist()
    
    return None


def hole_fill(train_pairs, test_input):
    """穴埋め: BGの穴を周囲の色で埋める"""
    from arc.grid import grid_eq
    
    def _apply(grid):
        g = np.array(grid)
        h, w = g.shape
        bg = _bg(g)
        result = g.copy()
        changed = False
        
        for r in range(h):
            for c in range(w):
                if g[r,c] != bg: continue
                # 4近傍の非BG色
                nb = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w and g[nr,nc] != bg:
                        nb.append(int(g[nr,nc]))
                # 3方向以上が同色なら埋める
                if len(nb) >= 3:
                    mc = Counter(nb).most_common(1)[0][0]
                    result[r,c] = mc; changed = True
        
        return result.tolist() if changed else None
    
    ok = True
    for inp, out in train_pairs:
        p = _apply(inp)
        if p is None or not grid_eq(p, out):
            ok = False; break
    return _apply(test_input) if ok else None


# ══════════════════════════════════════════════════════════════
# リバーシ: 挟んで塗る（拡張版）
# ══════════════════════════════════════════════════════════════

def reversi_full(train_pairs, test_input):
    """リバーシ全方向挟み塗り（距離制限なし + 1ステップ/反復）"""
    from arc.grid import grid_eq
    
    for mode in ['immediate', 'distance_any', 'iterative']:
        ok = True
        for inp, out in train_pairs:
            p = _reversi(inp, mode)
            if p is None or not grid_eq(p, out):
                ok = False; break
        if ok:
            return _reversi(test_input, mode)
    return None

def _reversi(grid, mode):
    g = np.array(grid).copy()
    h, w = g.shape
    bg = _bg(g)
    orig = g.copy()
    
    max_iter = 1 if mode != 'iterative' else 20
    
    for _ in range(max_iter):
        changed = False
        for r in range(h):
            for c in range(w):
                if g[r,c] != bg: continue
                
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                    # 方向1: 最初の非BG
                    nr, nc = r+dr, c+dc
                    c1 = None
                    while 0<=nr<h and 0<=nc<w:
                        if g[nr,nc] != bg:
                            c1 = int(g[nr,nc]); break
                        if mode == 'immediate': break
                        nr += dr; nc += dc
                    
                    # 逆方向
                    nr, nc = r-dr, c-dc
                    c2 = None
                    while 0<=nr<h and 0<=nc<w:
                        if g[nr,nc] != bg:
                            c2 = int(g[nr,nc]); break
                        if mode == 'immediate': break
                        nr -= dr; nc -= dc
                    
                    if c1 is not None and c2 is not None and c1 == c2:
                        g[r,c] = c1; changed = True; break
        
        if not changed: break
    
    return g.tolist() if not np.array_equal(g, orig) else None


# ══════════════════════════════════════════════════════════════
# ソリティア: 色の整列・分類
# ══════════════════════════════════════════════════════════════

def solitaire_gather(train_pairs, test_input):
    """ソリティア: 同色を集めて並べる"""
    from arc.grid import grid_eq
    
    # 行ソート
    def _row_sort(grid):
        g = np.array(grid)
        bg = _bg(g)
        result = g.copy()
        changed = False
        for r in range(g.shape[0]):
            vals = [int(v) for v in g[r] if v != bg]
            vals.sort()
            ci = 0
            for c in range(g.shape[1]):
                if g[r,c] != bg:
                    if result[r,c] != vals[ci]:
                        result[r,c] = vals[ci]; changed = True
                    ci += 1
        return result.tolist() if changed else None
    
    # 列ソート
    def _col_sort(grid):
        g = np.array(grid)
        bg = _bg(g)
        result = g.copy()
        changed = False
        for c in range(g.shape[1]):
            vals = [int(v) for v in g[:,c] if v != bg]
            vals.sort()
            ri = 0
            for r in range(g.shape[0]):
                if g[r,c] != bg:
                    if result[r,c] != vals[ri]:
                        result[r,c] = vals[ri]; changed = True
                    ri += 1
        return result.tolist() if changed else None
    
    # 色ごとにグループ化して行に配置
    def _color_group(grid):
        g = np.array(grid)
        bg = _bg(g)
        h, w = g.shape
        color_cells = defaultdict(list)
        for r in range(h):
            for c in range(w):
                if g[r,c] != bg:
                    color_cells[int(g[r,c])].append((r,c))
        
        result = np.full_like(g, bg)
        row = 0
        for color in sorted(color_cells.keys()):
            for i, (_, oc) in enumerate(sorted(color_cells[color])):
                if row < h:
                    result[row, i % w] = color
            row += 1
        
        return result.tolist() if not np.array_equal(result, g) else None
    
    for fn in [_row_sort, _col_sort, _color_group]:
        ok = True
        for inp, out in train_pairs:
            p = fn(inp)
            if p is None or not grid_eq(p, out):
                ok = False; break
        if ok:
            return fn(test_input)
    
    return None


# ══════════════════════════════════════════════════════════════
# マスターソルバー
# ══════════════════════════════════════════════════════════════

ALL_ELEMENTARY = [
    ('reversi_full', reversi_full),
    ('solitaire_gather', solitaire_gather),
    ('symmetry_complete', symmetry_complete),
    ('hole_fill', hole_fill),
    ('count_and_output', count_and_output),
    ('compare_and_select', compare_and_select),
    ('even_odd_rule', even_odd_rule),
    ('pattern_repeat', pattern_repeat),
    ('sort_objects', sort_objects),
]


def elementary_solve(train_pairs, test_input):
    from arc.grid import grid_eq
    for name, solver in ALL_ELEMENTARY:
        try:
            result = solver(train_pairs, test_input)
            if result is not None:
                return result, name
        except Exception:
            continue
    return None, None


if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    from arc.grid import grid_eq
    
    split = 'evaluation' if '--eval' in sys.argv else 'training'
    data_dir = Path(f'/tmp/arc-agi-2/data/{split}')
    
    existing = set()
    with open('arc_v82.log') as f:
        for line in f:
            m = re.search(r'✓.*?([0-9a-f]{8})', line)
            if m: existing.add(m.group(1))
    synth = set(f.stem for f in Path('synth_results').glob('*.py'))
    all_existing = existing | synth
    
    solved = []
    for tf in sorted(data_dir.glob('*.json')):
        tid = tf.stem
        with open(tf) as f: task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti, to = task['test'][0]['input'], task['test'][0].get('output')
        
        result, name = elementary_solve(tp, ti)
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_existing else ''
            solved.append((tid, name, tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tag in solved if tag == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
