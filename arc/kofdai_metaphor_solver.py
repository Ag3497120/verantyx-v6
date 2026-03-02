"""
arc/kofdai_metaphor_solver.py — kofdaiの比喩をソルバーに変換

各比喩 → 具体的な操作パターン
"""

import numpy as np
from typing import Optional, List, Tuple
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
        colors = [int(g[r,c]) for r,c in cells]
        r_min = min(r for r,c in cells); c_min = min(c for r,c in cells)
        r_max = max(r for r,c in cells); c_max = max(c for r,c in cells)
        objs.append({
            'cells': cells, 'size': len(cells),
            'color': Counter(colors).most_common(1)[0][0],
            'colors': set(colors),
            'bbox': (r_min, c_min, r_max, c_max),
            'shape': frozenset((r-r_min, c-c_min) for r,c in cells),
        })
    return objs


# 17. 最大枠の内部crop
def largest_frame_crop(train_pairs, test_input):
    """一番大きい囲みの中から内部の要素を取り出す"""
    from arc.grid import grid_eq
    
    def _apply(grid):
        g = np.array(grid)
        h, w = g.shape
        bg = _bg(g)
        objs = _objs8(g, bg)
        if not objs: return None
        
        # 最大オブジェクト(=枠)を探す
        largest = max(objs, key=lambda o: o['size'])
        r1, c1, r2, c2 = largest['bbox']
        
        # 内部をcrop (枠を1pxスキップ)
        inner = g[r1+1:r2, c1+1:c2]
        if inner.size == 0:
            # 枠が1px厚じゃないかも → bbox全体をcrop
            inner = g[r1:r2+1, c1:c2+1]
        
        return inner.tolist()
    
    ok = True
    for inp, out in train_pairs:
        p = _apply(inp)
        if p is None or not grid_eq(p, out):
            ok = False; break
    return _apply(test_input) if ok else None


# 2. 鍵差し込み: 特定形状を別の位置に嵌め込み、既存を押し出す
def key_insert(train_pairs, test_input):
    """水色の鍵を差し込むことによる既存の押し出しと位置固定"""
    # TODO: 複雑、後で
    return None


# 3. QR抽出: 集合パターンから一色取り出し
def qr_extract(train_pairs, test_input):
    """QRコード読み取り: 集合部分から一色取り出してまとめる"""
    from arc.grid import grid_eq
    
    # trainの出力サイズからパターンサイズを推測
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        oh, ow = go.shape
        bg_i = _bg(gi)
        
        # 入力内で出力サイズと同じbboxのオブジェクト群を探す
        objs = _objs8(gi, bg_i)
        
        # 出力に含まれる色
        out_colors = set(int(v) for v in go.flatten()) - {bg_i}
        
        # 各色のパターンを抽出してoutputと比較
        for color in out_colors:
            mask = (gi == color)
            if not mask.any(): continue
            rows, cols = np.where(mask)
            r1, c1 = rows.min(), cols.min()
            sub = gi[r1:r1+oh, c1:c1+ow]
            # この部分だけ取り出す
        break
    
    return None


# 4. 色順移動: 色の並び順で車を移動
def color_order_move(train_pairs, test_input):
    """青黄赤の並び順に着目して車を移動させるイメージ"""
    from arc.grid import grid_eq
    
    # trainで色の出現順が移動距離/方向を決めるか
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        bg = _bg(ga)
        
        objs_in = _objs8(ga, bg)
        objs_out = _objs8(go, bg)
        
        # 各オブジェクトの移動量を色でソート
        moves = []
        for oi in objs_in:
            for oo in objs_out:
                if oi['color'] == oo['color'] and oi['shape'] == oo['shape']:
                    dr = oo['bbox'][0] - oi['bbox'][0]
                    dc = oo['bbox'][1] - oi['bbox'][1]
                    moves.append((oi['color'], dr, dc))
                    break
        
        if moves:
            # 色順にソートして移動量が等差数列か
            moves.sort(key=lambda x: x[0])
        break
    
    return None


# 5. 重なり=量、単独=境界
def overlap_quantity(train_pairs, test_input):
    """重なりブロック=たくさん、単独ブロック=境界線"""
    from arc.grid import grid_eq
    
    def _apply(grid):
        g = np.array(grid)
        h, w = g.shape
        bg = _bg(g)
        objs = _objs8(g, bg)
        if not objs: return None
        
        result = g.copy()
        changed = False
        
        # 2色が重なるセルを検出（同じ位置に複数色 = ない）
        # → 隣接する2つのオブジェクトが重なる領域を探す
        # → 重なり = 2つのオブジェクトのbboxが重なる部分
        
        for i, o1 in enumerate(objs):
            for o2 in objs[i+1:]:
                r1 = max(o1['bbox'][0], o2['bbox'][0])
                c1 = max(o1['bbox'][1], o2['bbox'][1])
                r2 = min(o1['bbox'][2], o2['bbox'][2])
                c2 = min(o1['bbox'][3], o2['bbox'][3])
                
                if r1 <= r2 and c1 <= c2:
                    # 重なりあり → この領域を塗る
                    for r in range(r1, r2+1):
                        for c in range(c1, c2+1):
                            if result[r,c] == bg:
                                result[r,c] = o1['color']
                                changed = True
        
        return result.tolist() if changed else None
    
    ok = True
    for inp, out in train_pairs:
        p = _apply(inp)
        if p is None or not grid_eq(p, out):
            ok = False; break
    return _apply(test_input) if ok else None


# 6. AR中心集約: 同色を中心にまとめる
def ar_center_gather(train_pairs, test_input):
    """同じ色を中心にまとめるARコードパターン"""
    # TODO
    return None


# 10. 正方形/長方形で塗り分け
def shape_type_recolor(train_pairs, test_input):
    """色で正方形と長方形を作っての塗り分け"""
    from arc.grid import grid_eq
    
    # trainからオブジェクトの形状タイプ(正方形/長方形)→色のマッピングを学習
    shape_color_map = {}
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        bg = _bg(ga)
        objs = _objs8(ga, bg)
        
        for obj in objs:
            r1, c1, r2, c2 = obj['bbox']
            bh, bw = r2-r1+1, c2-c1+1
            is_square = (bh == bw)
            is_filled = (obj['size'] == bh * bw)
            
            # 出力での色
            out_color = int(go[obj['cells'][0]])
            
            key = ('square' if is_square else 'rect', is_filled)
            if key not in shape_color_map:
                shape_color_map[key] = out_color
    
    if not shape_color_map: return None
    
    def _apply(grid):
        g = np.array(grid)
        bg2 = _bg(g)
        objs = _objs8(g, bg2)
        result = g.copy()
        changed = False
        for obj in objs:
            r1, c1, r2, c2 = obj['bbox']
            bh, bw = r2-r1+1, c2-c1+1
            is_square = (bh == bw)
            is_filled = (obj['size'] == bh * bw)
            key = ('square' if is_square else 'rect', is_filled)
            if key in shape_color_map:
                nc = shape_color_map[key]
                for r, c in obj['cells']:
                    if result[r,c] != nc:
                        result[r,c] = nc; changed = True
        return result.tolist() if changed else None
    
    ok = True
    for inp, out in train_pairs:
        p = _apply(inp)
        if p is None or not grid_eq(p, out):
            ok = False; break
    return _apply(test_input) if ok else None


# 11. 同形同向検出
def same_shape_direction(train_pairs, test_input):
    """同じ向きかつ同じ形の検出"""
    from arc.grid import grid_eq
    
    def _apply(grid, train_out=None):
        g = np.array(grid)
        bg = _bg(g)
        objs = _objs8(g, bg)
        
        # 形でグループ化(向きも含む = shapeそのまま)
        shape_groups = defaultdict(list)
        for obj in objs:
            shape_groups[obj['shape']].append(obj)
        
        # 最も多いグループ = 「同じ形」
        if not shape_groups: return None
        
        # trainの出力との差分で「マーク方法」を学習
        if train_out is not None:
            go = np.array(train_out)
            result = g.copy()
            # 出力で色が変わったオブジェクトを特定
            for obj in objs:
                r, c = obj['cells'][0]
                if go[r,c] != g[r,c]:
                    # このオブジェクトの色が変わった
                    for rr, cc in obj['cells']:
                        result[rr, cc] = int(go[rr, cc])
            return result.tolist()
        
        return None
    
    # trainで学習
    return None


# 12. 三枚おろし→重ね
def three_layer_overlay(train_pairs, test_input):
    """三枚おろしにしたものを重ねる。色を正方形から選んで形を選んで周りの色を選ぶ"""
    from arc.grid import grid_eq
    
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        bg = _bg(gi)
        ih, iw = gi.shape
        oh, ow = go.shape
        
        # 入力を3分割（水平or垂直）して重ねる
        if ih % 3 == 0 and oh == ih // 3:
            ph = ih // 3
            p1 = gi[:ph, :]
            p2 = gi[ph:2*ph, :]
            p3 = gi[2*ph:, :]
            
            # OR合成
            result = np.full_like(p1, bg)
            for r in range(ph):
                for c in range(iw):
                    vals = [int(p1[r,c]), int(p2[r,c]), int(p3[r,c])]
                    non_bg = [v for v in vals if v != bg]
                    if non_bg:
                        result[r,c] = non_bg[0]  # 最初の非BG
            
            if grid_eq(result.tolist(), out):
                # test適用
                ti = np.array(test_input)
                th = ti.shape[0] // 3
                t1, t2, t3 = ti[:th,:], ti[th:2*th,:], ti[2*th:,:]
                res = np.full_like(t1, bg)
                for r in range(th):
                    for c in range(ti.shape[1]):
                        vals = [int(t1[r,c]), int(t2[r,c]), int(t3[r,c])]
                        non_bg = [v for v in vals if v != bg]
                        if non_bg: res[r,c] = non_bg[0]
                return res.tolist()
        
        if iw % 3 == 0 and ow == iw // 3:
            pw = iw // 3
            p1 = gi[:, :pw]
            p2 = gi[:, pw:2*pw]
            p3 = gi[:, 2*pw:]
            
            result = np.full_like(p1, bg)
            for r in range(ih):
                for c in range(pw):
                    vals = [int(p1[r,c]), int(p2[r,c]), int(p3[r,c])]
                    non_bg = [v for v in vals if v != bg]
                    if non_bg: result[r,c] = non_bg[0]
            
            if grid_eq(result.tolist(), out):
                ti = np.array(test_input)
                tw = ti.shape[1] // 3
                t1, t2, t3 = ti[:,:tw], ti[:,tw:2*tw], ti[:,2*tw:]
                res = np.full_like(t1, bg)
                for r in range(ti.shape[0]):
                    for c in range(tw):
                        vals = [int(t1[r,c]), int(t2[r,c]), int(t3[r,c])]
                        non_bg = [v for v in vals if v != bg]
                        if non_bg: res[r,c] = non_bg[0]
                return res.tolist()
        break
    
    return None


# 13. 色重ね模様: 複数パネルの論理合成
def color_overlay(train_pairs, test_input):
    """色の重ね合わせによる模様の付けかた"""
    from arc.grid import grid_eq
    
    for n_panels in [2, 3, 4]:
        for axis in ['h', 'v']:
            for op in ['or', 'xor', 'and', 'first_nonbg']:
                ok = True
                for inp, out in train_pairs:
                    p = _overlay_apply(inp, n_panels, axis, op)
                    if p is None or not grid_eq(p, out):
                        ok = False; break
                if ok:
                    return _overlay_apply(test_input, n_panels, axis, op)
    return None

def _overlay_apply(grid, n, axis, op):
    g = np.array(grid)
    h, w = g.shape
    bg = _bg(g)
    
    if axis == 'h' and h % n == 0:
        ph = h // n
        panels = [g[i*ph:(i+1)*ph, :] for i in range(n)]
    elif axis == 'v' and w % n == 0:
        pw = w // n
        panels = [g[:, i*pw:(i+1)*pw] for i in range(n)]
    else:
        return None
    
    # 全パネルが同サイズか
    shapes = set(p.shape for p in panels)
    if len(shapes) != 1: return None
    ph, pw = panels[0].shape
    
    result = np.full((ph, pw), bg, dtype=int)
    for r in range(ph):
        for c in range(pw):
            vals = [int(p[r,c]) for p in panels]
            non_bg = [v for v in vals if v != bg]
            
            if op == 'or':
                result[r,c] = non_bg[0] if non_bg else bg
            elif op == 'xor':
                if len(non_bg) == 1: result[r,c] = non_bg[0]
            elif op == 'and':
                if len(non_bg) == len(vals): result[r,c] = non_bg[0]
            elif op == 'first_nonbg':
                result[r,c] = non_bg[0] if non_bg else bg
    
    return result.tolist()


# 14. 色別壁衝突
def color_wall_collision(train_pairs, test_input):
    """それぞれの色の壁に衝突させる"""
    from arc.grid import grid_eq
    
    # trainで「移動するオブジェクト」と「壁」を区別
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        bg = _bg(ga)
        
        # before/after差分
        moved_from = set()
        moved_to = set()
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                if ga[r,c] != bg and go[r,c] == bg:
                    moved_from.add((r,c))
                elif ga[r,c] == bg and go[r,c] != bg:
                    moved_to.add((r,c))
        
        if not moved_from or not moved_to:
            return None
        break
    
    return None


# 15. 上部パターンで塗りつぶし決定
def top_pattern_fill(train_pairs, test_input):
    """上部について埋まっている色を元に塗りつぶしを決定する"""
    from arc.grid import grid_eq
    
    def _apply(grid):
        g = np.array(grid)
        h, w = g.shape
        bg = _bg(g)
        result = g.copy()
        changed = False
        
        # 各列の最上の非BG色で下を塗る
        for c in range(w):
            top_color = None
            for r in range(h):
                if g[r,c] != bg:
                    top_color = int(g[r,c])
                    break
            if top_color is not None:
                for r in range(h):
                    if result[r,c] == bg:
                        result[r,c] = top_color
                        changed = True
        
        return result.tolist() if changed else None
    
    ok = True
    for inp, out in train_pairs:
        if not grid_eq(_apply(inp), out):
            ok = False; break
    return _apply(test_input) if ok else None


# 16. 向きを合わせて嵌める
def orient_and_insert(train_pairs, test_input):
    """周りに散らばってる要素を向きを合わせて入れる"""
    # TODO: 複雑
    return None


ALL_METAPHOR_SOLVERS = [
    ('largest_frame_crop', largest_frame_crop),
    ('shape_type_recolor', shape_type_recolor),
    ('three_layer_overlay', three_layer_overlay),
    ('color_overlay', color_overlay),
    ('top_pattern_fill', top_pattern_fill),
    ('overlap_quantity', overlap_quantity),
    ('solitaire_sort', lambda tp, ti: None),  # placeholder
]


def metaphor_solve(train_pairs, test_input):
    from arc.grid import grid_eq
    for name, solver in ALL_METAPHOR_SOLVERS:
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
        
        result, name = metaphor_solve(tp, ti)
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_existing else ''
            solved.append((tid, name, tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tag in solved if tag == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
