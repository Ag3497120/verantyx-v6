"""
arc/game_rules_v2.py — ゲームルール追加パック

=== 追加ゲーム ===
1. ソリティア: 色をソートして整列
2. ピクロス/ノノグラム: 行列ヒントで塗る
3. 倉庫番(Sokoban): 壁にぶつかるまで押す
4. Lights Out: クリックで十字反転
5. 2048: 同色隣接を合体
6. Connect Four: 列に落として積む
7. パイプパズル: 隣接セル間の接続
8. Water Sort: 色を列ごとに分離
9. ピンボール: 壁で反射しながら線を引く
10. ドミノ: 隣接ペアのルール
11. スネーク: 連結パスを伸ばす
12. Flood It: 起点から色を広げる
13. 非ゲーム: 射影(行/列の畳み込み)
14. 非ゲーム: 近傍多数決
15. 非ゲーム: 対角コピー
"""

import numpy as np
from typing import Optional, List
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label

def _bg(g): return int(Counter(np.array(g).flatten()).most_common(1)[0][0])
def _objs8(g, bg):
    struct = np.ones((3,3), dtype=int)
    mask = (np.array(g) != bg).astype(int)
    labeled, n = scipy_label(mask, structure=struct)
    objs = []
    for i in range(1, n+1):
        cells = list(zip(*np.where(labeled == i)))
        objs.append({'cells': cells, 'size': len(cells), 'color': int(g[cells[0]])})
    return objs


# 1. ソリティア: オブジェクトをサイズ/色で並び替え
def solitaire_sort(train_pairs, test_input):
    from arc.grid import grid_eq
    gi = np.array(test_input)
    bg = _bg(gi)
    h, w = gi.shape
    
    # trainから並べ替えルールを学習: 色の行位置が変わる
    for sort_key in ['color_asc', 'color_desc', 'size_asc', 'size_desc', 'row_sort', 'col_sort']:
        ok = True
        for inp, out in train_pairs:
            p = _solitaire_apply(inp, sort_key)
            if p is None or not grid_eq(p, out):
                ok = False; break
        if ok:
            return _solitaire_apply(test_input, sort_key)
    return None

def _solitaire_apply(grid, key):
    g = np.array(grid)
    h, w = g.shape
    bg = _bg(g)
    
    if key == 'row_sort':
        result = g.copy()
        for r in range(h):
            row = sorted([int(v) for v in g[r] if v != bg])
            ci = 0
            for c in range(w):
                if result[r,c] != bg:
                    result[r,c] = row[ci] if ci < len(row) else bg
                    ci += 1
        return result.tolist() if not np.array_equal(result, g) else None
    
    elif key == 'col_sort':
        result = g.copy()
        for c in range(w):
            col = sorted([int(v) for v in g[:,c] if v != bg])
            ri = 0
            for r in range(h):
                if result[r,c] != bg:
                    result[r,c] = col[ri] if ri < len(col) else bg
                    ri += 1
        return result.tolist() if not np.array_equal(result, g) else None
    
    return None


# 2. Sokoban: 壁にぶつかるまで押す（特定色が壁）
def sokoban_push(train_pairs, test_input):
    from arc.grid import grid_eq
    
    for direction in ['down', 'up', 'left', 'right']:
        ok = True
        for inp, out in train_pairs:
            p = _sokoban_apply(inp, direction)
            if p is None or not grid_eq(p, out):
                ok = False; break
        if ok:
            return _sokoban_apply(test_input, direction)
    return None

def _sokoban_apply(grid, direction):
    g = np.array(grid)
    h, w = g.shape
    bg = _bg(g)
    
    # 最も多い非BG色 = 壁
    colors = Counter(int(v) for v in g.flatten() if v != bg)
    if len(colors) < 2: return None
    wall_color = colors.most_common(1)[0][0]
    
    result = g.copy()
    dr, dc = {'down':(1,0),'up':(-1,0),'left':(0,-1),'right':(0,1)}[direction]
    
    moved = False
    # 方向に応じた順序で処理
    rows = range(h-1,-1,-1) if dr > 0 else range(h) if dr < 0 else range(h)
    cols = range(w-1,-1,-1) if dc > 0 else range(w) if dc < 0 else range(w)
    
    for r in rows:
        for c in cols:
            v = int(result[r,c])
            if v == bg or v == wall_color: continue
            # この方向に壁orエッジまで移動
            nr, nc = r, c
            while True:
                nnr, nnc = nr+dr, nc+dc
                if not (0<=nnr<h and 0<=nnc<w): break
                if result[nnr,nnc] != bg: break
                nr, nc = nnr, nnc
            if (nr, nc) != (r, c):
                result[nr, nc] = v
                result[r, c] = bg
                moved = True
    
    return result.tolist() if moved else None


# 3. Lights Out: 十字パターンの反転
def lights_out(train_pairs, test_input):
    from arc.grid import grid_eq
    
    gi = np.array(test_input)
    bg = _bg(gi)
    h, w = gi.shape
    
    # trainの差分が十字パターンか
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        diff = np.where(ga != go)
        if len(diff[0]) == 0: return None
        
        # 差分セルが十字パターンを形成するか
        dr = diff[0] - diff[0].mean()
        dc = diff[1] - diff[1].mean()
        # 十字 = 同じ行or同じ列
        cross = all((abs(r) < 0.5 or abs(c) < 0.5) for r, c in zip(dr, dc))
        if not cross:
            return None
    
    return None  # TODO: 実装


# 4. 2048: 同色隣接を合体（色+1に）
def merge_2048(train_pairs, test_input):
    from arc.grid import grid_eq
    
    for direction in ['down', 'up', 'left', 'right']:
        ok = True
        for inp, out in train_pairs:
            p = _merge_apply(inp, direction)
            if p is None or not grid_eq(p, out):
                ok = False; break
        if ok:
            return _merge_apply(test_input, direction)
    return None

def _merge_apply(grid, direction):
    g = np.array(grid)
    h, w = g.shape
    bg = _bg(g)
    result = np.full_like(g, bg)
    changed = False
    
    if direction in ('left', 'right'):
        for r in range(h):
            vals = [int(v) for v in g[r] if v != bg]
            if direction == 'right': vals = vals[::-1]
            merged = []
            i = 0
            while i < len(vals):
                if i+1 < len(vals) and vals[i] == vals[i+1]:
                    merged.append(vals[i] + 1)
                    i += 2; changed = True
                else:
                    merged.append(vals[i]); i += 1
            if direction == 'right': merged = merged[::-1]
            if direction == 'left':
                for i, v in enumerate(merged): result[r, i] = v
            else:
                for i, v in enumerate(merged): result[r, w-1-i] = v
    else:
        for c in range(w):
            vals = [int(g[r,c]) for r in range(h) if g[r,c] != bg]
            if direction == 'down': vals = vals[::-1]
            merged = []
            i = 0
            while i < len(vals):
                if i+1 < len(vals) and vals[i] == vals[i+1]:
                    merged.append(vals[i] + 1)
                    i += 2; changed = True
                else:
                    merged.append(vals[i]); i += 1
            if direction == 'down': merged = merged[::-1]
            if direction == 'up':
                for i, v in enumerate(merged): result[i, c] = v
            else:
                for i, v in enumerate(merged): result[h-1-i, c] = v
    
    return result.tolist() if changed else None


# 5. Flood It: 起点の色を隣接に広げる(1ステップ)
def flood_expand(train_pairs, test_input):
    from arc.grid import grid_eq
    
    def _apply(grid, steps=1):
        g = np.array(grid)
        h, w = g.shape
        bg = _bg(g)
        result = g.copy()
        
        for _ in range(steps):
            new = result.copy()
            for r in range(h):
                for c in range(w):
                    if result[r,c] == bg: continue
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and result[nr,nc] == bg:
                            new[nr,nc] = result[r,c]
            result = new
        
        return result.tolist() if not np.array_equal(result, g) else None
    
    for steps in range(1, 6):
        ok = True
        for inp, out in train_pairs:
            p = _apply(inp, steps)
            if p is None or not grid_eq(p, out):
                ok = False; break
        if ok:
            return _apply(test_input, steps)
    return None


# 6. スネーク: 連結パスを端から伸ばす
def snake_extend(train_pairs, test_input):
    # TODO
    return None


# 7. ピンボール: 壁に反射して4方向に線を引く
def pinball_reflect(train_pairs, test_input):
    from arc.grid import grid_eq
    
    def _apply(grid):
        g = np.array(grid).copy()
        h, w = g.shape
        bg = _bg(g)
        objs = _objs8(g, bg)
        single = [o for o in objs if o['size'] == 1]
        if not single: return None
        
        changed = False
        for o in single:
            r0, c0 = o['cells'][0]
            color = o['color']
            # 4対角方向に反射あり線引き
            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                r, c = r0+dr, c0+dc
                cdr, cdc = dr, dc
                for _ in range(max(h,w)*2):
                    if not (0<=r<h and 0<=c<w): break
                    if g[r,c] != bg and g[r,c] != color:
                        # 壁に当たった→反射
                        # 水平壁チェック
                        nr, nc = r+cdr, c
                        if 0<=nr<h and g[nr,c] == bg:
                            cdc = -cdc  # 水平反射
                        else:
                            cdr = -cdr  # 垂直反射
                        break
                    if g[r,c] == bg:
                        g[r,c] = color; changed = True
                    r += cdr; c += cdc
        
        return g.tolist() if changed else None
    
    ok = True
    for inp, out in train_pairs:
        p = _apply(inp)
        if p is None or not grid_eq(p, out):
            ok = False; break
    return _apply(test_input) if ok else None


# 8. 近傍多数決: 各セルを8近傍の最頻色に変換
def majority_vote(train_pairs, test_input):
    from arc.grid import grid_eq
    
    def _apply(grid, include_self=True):
        g = np.array(grid)
        h, w = g.shape
        bg = _bg(g)
        result = g.copy()
        changed = False
        
        for r in range(h):
            for c in range(w):
                nb = []
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0:
                            if include_self: nb.append(int(g[r,c]))
                            continue
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w:
                            nb.append(int(g[nr,nc]))
                non_bg = [v for v in nb if v != bg]
                if non_bg:
                    majority = Counter(non_bg).most_common(1)[0][0]
                    if result[r,c] != majority:
                        result[r,c] = majority; changed = True
        
        return result.tolist() if changed else None
    
    for include_self in [True, False]:
        ok = True
        for inp, out in train_pairs:
            p = _apply(inp, include_self)
            if p is None or not grid_eq(p, out):
                ok = False; break
        if ok:
            return _apply(test_input, include_self)
    return None


# 9. 行/列射影: 行or列を畳み込んで1行/列にする
def projection_fold(train_pairs, test_input):
    from arc.grid import grid_eq
    
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        oh, ow = go.shape
        ih, iw = gi.shape
        bg = _bg(gi)
        
        # 出力が1行 = 列方向のOR射影
        if oh == 1 and ow == iw:
            def _apply(grid):
                g = np.array(grid)
                bg2 = _bg(g)
                result = np.full((1, g.shape[1]), bg2)
                for c in range(g.shape[1]):
                    for r in range(g.shape[0]):
                        if g[r,c] != bg2:
                            result[0,c] = int(g[r,c]); break
                return result.tolist()
            
            ok = all(grid_eq(_apply(i), o) for i, o in train_pairs)
            if ok: return _apply(test_input)
        
        # 出力が1列 = 行方向のOR射影
        if ow == 1 and oh == ih:
            def _apply(grid):
                g = np.array(grid)
                bg2 = _bg(g)
                result = np.full((g.shape[0], 1), bg2)
                for r in range(g.shape[0]):
                    for c in range(g.shape[1]):
                        if g[r,c] != bg2:
                            result[r,0] = int(g[r,c]); break
                return result.tolist()
            
            ok = all(grid_eq(_apply(i), o) for i, o in train_pairs)
            if ok: return _apply(test_input)
        break
    
    return None


# 10. 対角コピー/ミラー
def diagonal_mirror(train_pairs, test_input):
    from arc.grid import grid_eq
    
    for mode in ['main_diag', 'anti_diag', 'both']:
        ok = True
        for inp, out in train_pairs:
            p = _diag_apply(inp, mode)
            if p is None or not grid_eq(p, out):
                ok = False; break
        if ok:
            return _diag_apply(test_input, mode)
    return None

def _diag_apply(grid, mode):
    g = np.array(grid)
    h, w = g.shape
    if h != w: return None
    bg = _bg(g)
    result = g.copy()
    changed = False
    
    for r in range(h):
        for c in range(w):
            if mode in ('main_diag', 'both'):
                if result[r,c] == bg and result[c,r] != bg:
                    result[r,c] = result[c,r]; changed = True
            if mode in ('anti_diag', 'both'):
                ar, ac = h-1-c, w-1-r
                if result[r,c] == bg and result[ar,ac] != bg:
                    result[r,c] = result[ar,ac]; changed = True
    
    return result.tolist() if changed else None


# 11. パターン繰り返し検出(周期)
def periodic_fill(train_pairs, test_input):
    from arc.grid import grid_eq
    
    gi = np.array(test_input)
    h, w = gi.shape
    bg = _bg(gi)
    
    # 行方向の周期
    for period in range(1, h//2+1):
        if h % period != 0: continue
        pattern = gi[:period, :]
        ok_p = True
        for i in range(1, h//period):
            chunk = gi[i*period:(i+1)*period, :]
            # chunkでBGのところはpatternで埋める
            mask = chunk != bg
            if mask.any() and not np.array_equal(chunk[mask], pattern[mask]):
                ok_p = False; break
        if ok_p:
            result = np.tile(pattern, (h//period, 1))
            if not np.array_equal(result, gi):
                # train検証
                ok = True
                for inp, out in train_pairs:
                    ga = np.array(inp); ha = ga.shape[0]
                    if ha % period != 0: ok = False; break
                    pat = ga[:period, :]
                    pred = np.tile(pat, (ha//period, 1))
                    if not grid_eq(pred.tolist(), out): ok = False; break
                if ok:
                    return result.tolist()
    
    return None


ALL_V2_SOLVERS = [
    ('solitaire_sort', solitaire_sort),
    ('sokoban_push', sokoban_push),
    ('merge_2048', merge_2048),
    ('flood_expand', flood_expand),
    ('pinball_reflect', pinball_reflect),
    ('majority_vote', majority_vote),
    ('projection_fold', projection_fold),
    ('diagonal_mirror', diagonal_mirror),
    ('periodic_fill', periodic_fill),
]


def game_v2_solve(train_pairs, test_input):
    from arc.grid import grid_eq
    for name, solver in ALL_V2_SOLVERS:
        try:
            result = solver(train_pairs, test_input)
            if result is not None:
                ok = True
                for inp, out in train_pairs:
                    p = solver(train_pairs, inp)
                    if p is None or not grid_eq(p, out):
                        ok = False; break
                if ok:
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
        
        result, name = game_v2_solve(tp, ti)
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_existing else ''
            solved.append((tid, name, tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tag in solved if tag == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
