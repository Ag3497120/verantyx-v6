"""
arc/cross_vision.py — 視覚経験エンジン

人間の視覚認識プロセス:
1. 形を見る → 何に見えるか（家、虫、道、壁...）
2. 見えたものから操作を推測（家→重力、道→つなぐ、壁→仕切り）
3. 推測した操作を検証（trainで合うか）

形→意味の辞書 + 意味→操作の辞書
"""

import numpy as np
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label
from arc.cross_repair import period_repair_solve_v2


def _bg(g): return int(Counter(np.array(g).flatten()).most_common(1)[0][0])

def _objs(g, bg, conn=8):
    struct=np.ones((3,3),dtype=int) if conn==8 else np.array([[0,1,0],[1,1,1],[0,1,0]])
    mask=(np.array(g)!=bg).astype(int); labeled,n=scipy_label(mask,structure=struct)
    objs=[]
    for i in range(1,n+1):
        cells=list(zip(*np.where(labeled==i)))
        colors=[int(g[r,c]) for r,c in cells]
        r1=min(r for r,c in cells);c1=min(c for r,c in cells)
        r2=max(r for r,c in cells);c2=max(c for r,c in cells)
        bh,bw=r2-r1+1,c2-c1+1
        is_rect=len(cells)==bh*bw
        rel=frozenset((r-r1,c-c1) for r,c in cells)
        objs.append({'cells':cells,'size':len(cells),
            'color':Counter(colors).most_common(1)[0][0],'colors':set(colors),
            'bbox':(r1,c1,r2,c2),'bh':bh,'bw':bw,'is_rect':is_rect,
            'shape':rel,'center':((r1+r2)/2,(c1+c2)/2),'fill_ratio':len(cells)/(bh*bw)})
    return objs

def grid_eq(a,b):
    a,b=np.array(a),np.array(b)
    return a.shape==b.shape and np.array_equal(a,b)


# ══════════════════════════════════════════════════════════════
# 形→意味 辞書（人間の視覚経験）
# ══════════════════════════════════════════════════════════════

def see_shape(obj, grid, bg):
    """形を見て「何に見えるか」を返す"""
    g = np.array(grid)
    h, w = g.shape
    bh, bw = obj['bh'], obj['bw']
    r1, c1, r2, c2 = obj['bbox']
    cells = set(obj['cells'])
    fill = obj['fill_ratio']
    
    meanings = []
    
    # ■ 壁/仕切り: 1行or1列を完全に占める
    if bh == 1 and bw >= w * 0.7:
        meanings.append(('wall_h', 0.95))  # 水平の壁
    if bw == 1 and bh >= h * 0.7:
        meanings.append(('wall_v', 0.95))  # 垂直の壁
    
    # ■ 枠/箱: 矩形の辺だけ（中に何かを入れる）
    if bh >= 3 and bw >= 3 and not obj['is_rect']:
        border = sum(1 for r,c in cells if r==r1 or r==r2 or c==c1 or c==c2)
        if border / obj['size'] > 0.65:
            meanings.append(('box', 0.85))
    
    # ■ 塊/ブロック: 塗りつぶされた矩形
    if obj['is_rect'] and bh >= 2 and bw >= 2:
        meanings.append(('block', 0.8))
    
    # ■ 点/種: 1-2セル（小さい存在）
    if obj['size'] <= 2:
        meanings.append(('seed', 0.9))
    
    # ■ 線/道: 細長い（つなぐ役割）
    aspect = max(bh, bw) / (min(bh, bw) + 0.001)
    if aspect >= 3 and fill > 0.6:
        meanings.append(('path', 0.7 + 0.1 * min(aspect/10, 1)))
    
    # ■ 十字/交差点: 4方向に伸びてる
    cr, cc = int(round(obj['center'][0])), int(round(obj['center'][1]))
    arms = 0
    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        for step in range(1, max(bh, bw)):
            if (cr+dr*step, cc+dc*step) in cells:
                arms += 1; break
    if arms >= 3 and not obj['is_rect']:
        meanings.append(('crossroad', 0.8))
    
    # ■ L字/角: 2方向に伸びてる
    if arms == 2 and not obj['is_rect'] and fill < 0.5:
        meanings.append(('corner', 0.7))
    
    # ■ 対称形/蝶: 左右or上下対称
    shape_list = list(obj['shape'])
    max_r = max(r for r,c in shape_list) if shape_list else 0
    max_c = max(c for r,c in shape_list) if shape_list else 0
    lr_sym = obj['shape'] == frozenset((r, max_c-c) for r,c in shape_list)
    ud_sym = obj['shape'] == frozenset((max_r-r, c) for r,c in shape_list)
    if lr_sym and ud_sym and obj['size'] >= 3:
        meanings.append(('butterfly', 0.7))
    elif lr_sym or ud_sym:
        meanings.append(('mirror', 0.6))
    
    # ■ コの字/受け皿: 3辺だけ
    if bh >= 3 and bw >= 3 and not obj['is_rect']:
        top = sum(1 for c in range(c1,c2+1) if (r1,c) in cells)
        bot = sum(1 for c in range(c1,c2+1) if (r2,c) in cells)
        left = sum(1 for r in range(r1,r2+1) if (r,c1) in cells)
        right = sum(1 for r in range(r1,r2+1) if (r,c2) in cells)
        sides = [top/bw, bot/bw, left/bh, right/bh]
        open_sides = sum(1 for s in sides if s < 0.3)
        if open_sides == 1 and sum(1 for s in sides if s > 0.7) >= 2:
            meanings.append(('cup', 0.75))
    
    # ■ 散らばり: 同色で離れた位置に複数
    # （個々のオブジェクトではなくグリッド全体で判定するので別処理）
    
    # ■ 大きさによる役割
    all_objs = _objs(g, bg)
    if all_objs:
        max_size = max(o['size'] for o in all_objs)
        if obj['size'] == max_size and len(all_objs) > 1:
            meanings.append(('ground', 0.5))  # 最大=地面/背景的存在
        if obj['size'] == min(o['size'] for o in all_objs) and obj['size'] < max_size * 0.1:
            meanings.append(('insect', 0.5))  # 最小=虫的な小さい存在
    
    # 位置による意味
    if r2 >= h - 2 and bw >= w * 0.5:
        meanings.append(('floor', 0.6))
    if r1 <= 1 and bw >= w * 0.5:
        meanings.append(('ceiling', 0.6))
    
    return sorted(meanings, key=lambda x: -x[1])


def see_scene(grid):
    """グリッド全体を見て「何の場面か」を推定"""
    g = np.array(grid)
    h, w = g.shape
    bg = _bg(g)
    objs = _objs(g, bg)
    
    if not objs:
        return {'scene': 'empty', 'objs': [], 'meanings': {}, 'bg': bg}
    
    # 各オブジェクトの意味
    obj_meanings = {}
    for i, obj in enumerate(objs):
        meanings = see_shape(obj, grid, bg)
        obj_meanings[i] = meanings
    
    # 全体の場面
    all_meanings = set()
    for ms in obj_meanings.values():
        for m, _ in ms:
            all_meanings.add(m)
    
    scene = 'unknown'
    if 'wall_h' in all_meanings or 'wall_v' in all_meanings:
        scene = 'divided'  # 壁で仕切られた場面
    elif 'box' in all_meanings:
        scene = 'contained'  # 箱に入った場面
    elif 'seed' in all_meanings and len(objs) >= 3:
        scene = 'scattered'  # 種が散らばった場面
    elif 'path' in all_meanings:
        scene = 'connected'  # 道でつながった場面
    elif 'crossroad' in all_meanings:
        scene = 'intersection'  # 交差点の場面
    elif 'block' in all_meanings and len(objs) >= 2:
        scene = 'blocks'  # ブロック配置の場面
    elif 'floor' in all_meanings:
        scene = 'gravity'  # 重力がある場面
    
    return {'scene': scene, 'objs': objs, 'meanings': obj_meanings, 'bg': bg}


# ══════════════════════════════════════════════════════════════
# 意味→操作 辞書
# ══════════════════════════════════════════════════════════════

def infer_operations(scene_in, scene_out, gi, go):
    """入出力の場面の差分から操作を推測"""
    ops = []
    
    ga = np.array(gi)
    go_a = np.array(go)
    bg = scene_in['bg']
    
    if ga.shape != go_a.shape:
        # サイズ変化
        if go_a.size < ga.size:
            ops.append('extract')  # 何かを取り出す
        else:
            ops.append('expand')  # 何かを広げる
        return ops
    
    # 同サイズ
    added = np.sum((ga == bg) & (go_a != bg))
    removed = np.sum((ga != bg) & (go_a == bg))
    changed = np.sum((ga != go_a) & (ga != bg) & (go_a != bg))
    
    if added > 0 and removed == 0 and changed == 0:
        # 追加のみ
        if scene_in['scene'] == 'divided':
            ops.append('fill_between_walls')
        elif scene_in['scene'] == 'contained':
            ops.append('fill_inside_box')
            ops.append('pattern_complete')
        elif scene_in['scene'] == 'scattered':
            ops.append('grow_seeds')
        elif scene_in['scene'] == 'connected':
            ops.append('extend_path')
        elif scene_in['scene'] == 'intersection':
            ops.append('spread_from_cross')
        else:
            ops.append('add_pattern')
        ops.append('pattern_complete')
    
    elif removed > 0 and added == 0 and changed == 0:
        ops.append('remove_something')
    
    elif changed > 0 and added == 0 and removed == 0:
        ops.append('recolor')
        ops.append('error_correct')
    
    elif added > 0 and removed > 0:
        ops.append('move_objects')
    
    return ops


# ══════════════════════════════════════════════════════════════
# 操作の実装
# ══════════════════════════════════════════════════════════════

def _op_fill_inside_box(tp, ti):
    """箱の中を塗る"""
    for fill_color_mode in ['same_as_box', 'inner_minority', 'inner_majority']:
        ok = True
        for inp, out in tp:
            p = _apply_fill_box(inp, fill_color_mode)
            if p is None or not grid_eq(p, out): ok = False; break
        if ok:
            return _apply_fill_box(ti, fill_color_mode)
    return None

def _apply_fill_box(grid, mode):
    g = np.array(grid).copy(); h, w = g.shape; bg = _bg(g)
    objs = _objs(g, bg)
    changed = False
    
    for obj in objs:
        meanings = see_shape(obj, grid, bg)
        if not any(m == 'box' for m, _ in meanings): continue
        
        r1, c1, r2, c2 = obj['bbox']
        fc = obj['color']
        
        # 内部のBGセルを塗る
        inner_colors = [int(g[r, c]) for r in range(r1+1, r2) for c in range(c1+1, c2) 
                       if g[r,c] != bg and g[r,c] != fc]
        
        if mode == 'same_as_box':
            fill = fc
        elif mode == 'inner_minority':
            if inner_colors:
                fill = Counter(inner_colors).most_common()[-1][0]
            else: continue
        elif mode == 'inner_majority':
            if inner_colors:
                fill = Counter(inner_colors).most_common(1)[0][0]
            else: fill = fc
        else:
            fill = fc
        
        for r in range(r1+1, r2):
            for c in range(c1+1, c2):
                if g[r, c] == bg:
                    g[r, c] = fill; changed = True
    
    return g.tolist() if changed else None


def _op_grow_seeds(tp, ti):
    """種から成長する"""
    # 種（1-2セル）から十字方向/全方向に成長
    for growth_mode in ['cross', 'flood_to_boundary', 'line_to_same']:
        ok = True
        for inp, out in tp:
            p = _apply_grow(inp, growth_mode, tp)
            if p is None or not grid_eq(p, out): ok = False; break
        if ok:
            return _apply_grow(ti, growth_mode, tp)
    return None

def _apply_grow(grid, mode, tp):
    g = np.array(grid).copy(); h, w = g.shape; bg = _bg(g)
    objs = _objs(g, bg)
    seeds = [o for o in objs if o['size'] <= 2]
    
    if not seeds: return None
    changed = False
    
    if mode == 'cross':
        # 種から十字方向に端まで伸びる
        for seed in seeds:
            for r, c in seed['cells']:
                color = int(g[r, c])
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nr, nc = r+dr, c+dc
                    while 0 <= nr < h and 0 <= nc < w and g[nr, nc] == bg:
                        g[nr, nc] = color; changed = True
                        nr += dr; nc += dc
    
    elif mode == 'flood_to_boundary':
        # 種からflood fill（他の色にぶつかるまで）
        from scipy.ndimage import label as sl
        for seed in seeds:
            sr, sc = seed['cells'][0]
            color = int(g[sr, sc])
            queue = [(sr, sc)]
            visited = {(sr, sc)}
            while queue:
                r, c = queue.pop(0)
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w and (nr,nc) not in visited and g[nr,nc]==bg:
                        visited.add((nr,nc))
                        g[nr,nc] = color; changed = True
                        queue.append((nr,nc))
    
    elif mode == 'line_to_same':
        # 種から同色の他のオブジェクトに向かって線を引く
        color_groups = defaultdict(list)
        for obj in objs:
            color_groups[obj['color']].append(obj)
        
        for color, group in color_groups.items():
            if len(group) < 2: continue
            pts = [(int(round(o['center'][0])), int(round(o['center'][1]))) for o in group]
            for i, (r1, c1) in enumerate(pts):
                for r2, c2 in pts[i+1:]:
                    if r1 == r2:
                        for c in range(min(c1,c2)+1, max(c1,c2)):
                            if g[r1, c] == bg: g[r1, c] = color; changed = True
                    elif c1 == c2:
                        for r in range(min(r1,r2)+1, max(r1,r2)):
                            if g[r, c1] == bg: g[r, c1] = color; changed = True
    
    return g.tolist() if changed else None


def _op_fill_between_walls(tp, ti):
    """壁と壁の間を塗る（パネル演算）"""
    from arc.cross_qr_deep import _panel_ops
    return _panel_ops(tp, ti)


def _op_spread_from_cross(tp, ti):
    """十字の中心から拡散"""
    ok = True
    for inp, out in tp:
        p = _apply_grow(inp, 'cross', tp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok:
        return _apply_grow(ti, 'cross', tp)
    return None


def _op_extend_path(tp, ti):
    """道を伸ばす"""
    ok = True
    for inp, out in tp:
        p = _apply_grow(inp, 'line_to_same', tp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok:
        return _apply_grow(ti, 'line_to_same', tp)
    return None


def _op_extract(tp, ti):
    """何かを取り出す（crop系）"""
    from arc.cross_qr_deep import _framed_crop, _odd_extract, decode_odd_deep
    from arc.cross_qr import detect_anchors, build_connections, StructureCode
    
    r = _framed_crop(tp, ti)
    if r: return r
    
    r = _odd_extract(tp, ti)
    if r: return r
    
    return None


def _op_recolor(tp, ti):
    """色を変える"""
    # 全セルに一貫した色マップ
    m = {}
    for inp, out in tp:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                iv, ov = int(ga[r,c]), int(go[r,c])
                if iv != ov:
                    if iv in m and m[iv] != ov: return None
                    m[iv] = ov
    if not m: return None
    gi = np.array(ti); result = gi.copy()
    for o, n in m.items(): result[gi==o] = n
    return result.tolist()


def _op_move_objects(tp, ti):
    """オブジェクトを移動"""
    from arc.cross_brain_v2 import gravity_variants, object_translate, rotate_transform
    for solver in [gravity_variants, object_translate, rotate_transform]:
        try:
            r = solver(tp, ti)
            if r: return r
        except: pass
    return None


# ══════════════════════════════════════════════════════════════
# 視覚経験ソルバー
# ══════════════════════════════════════════════════════════════

def cross_vision_solve(train_pairs, test_input):
    """視覚経験ソルバー: 場面を見る→操作を推測→一つ一つ検証"""
    
    # 場面を見る
    scenes_in = [see_scene(inp) for inp, _ in train_pairs]
    scenes_out = [see_scene(out) for _, out in train_pairs]
    
    # 操作を推測
    all_ops = []
    for i, (inp, out) in enumerate(train_pairs):
        ops = infer_operations(scenes_in[i], scenes_out[i], inp, out)
        all_ops.append(set(ops))
    
    common_ops = all_ops[0] if all_ops else set()
    for ops in all_ops[1:]:
        common_ops &= ops
    
    # 全操作のリスト（関数を直接参照）
    # 全操作を遅延参照で取得
    import sys
    mod = sys.modules[__name__]
    solver_names = [
        'fill_inside_box', 'grow_seeds', 'fill_between_walls',
        'spread_from_cross', 'extend_path', 'extract', 'recolor',
        'move_objects', 'period_repair_2d', 'pattern_in_box',
        'symmetry_complete', 'object_gravity', 'flood_enclosed',
        'majority_fill', 'mirror_across_axis', 'extract_by_size',
        'period_repair', 'general_error_correct',
    ]
    all_solvers = []
    for sn in solver_names:
        fn = getattr(mod, f'_op_{sn}', None)
        if fn is not None:
            all_solvers.append((sn, fn))
    
    # 推測した操作を優先的に試す
    for op_name, op_fn in all_solvers:
        if op_name not in common_ops:
            continue
        try:
            result = op_fn(train_pairs, test_input)
            if result is not None:
                ok = True
                for inp, out in train_pairs:
                    p = op_fn(train_pairs, inp)
                    if p is None or not grid_eq(p, out):
                        ok = False; break
                if ok:
                    return result, f'vision:{op_name}'
        except:
            pass
    
    # 推測に関係なく全操作を試す
    for op_name, op_fn in all_solvers:
        if op_name in common_ops:
            continue
        try:
            result = op_fn(train_pairs, test_input)
            if result is not None:
                ok = True
                for inp, out in train_pairs:
                    p = op_fn(train_pairs, inp)
                    if p is None or not grid_eq(p, out):
                        ok = False; break
                if ok:
                    return result, f'vision:{op_name}(fallback)'
        except:
            pass
    
    # Brain V2フォールバック
    # 2D周期修復
    from arc.cross_repair import period_repair_solve_v2
    result, name = period_repair_solve_v2(train_pairs, test_input)
    if result is not None:
        return result, name
    
    # Brain V2フォールバック
    from arc.cross_brain_v2 import cross_brain_v2_solve
    return cross_brain_v2_solve(train_pairs, test_input)




# ══════════════════════════════════════════════════════════════
# パターン補完（QRエラー訂正）
# ══════════════════════════════════════════════════════════════

def _op_pattern_complete(tp, ti):
    """繰り返しパターンの穴を埋める"""
    results = []
    
    # A. 行ごとの周期パターン補完
    r = _row_period_complete(tp, ti)
    if r: return r
    
    # B. 列ごとの周期パターン補完
    r = _col_period_complete(tp, ti)
    if r: return r
    
    # C. 2Dタイル周期補完
    r = _tile_period_complete(tp, ti)
    if r: return r
    
    return None

def _find_period(seq, bg):
    """配列から周期を見つける（BGでないセルのみで判定）"""
    n = len(seq)
    for period in range(1, n//2 + 1):
        ok = True
        for i in range(n):
            if seq[i] != bg and seq[i % period] != bg:
                # 両方非BGなら一致すべき
                # ただし周期の「テンプレ」は非BGセルから作る
                pass
        # 別アプローチ: 非BGのパターンだけ抽出
        non_bg = [(i, seq[i]) for i in range(n) if seq[i] != bg]
        if len(non_bg) < 2: continue
        
        positions = [i for i, _ in non_bg]
        # 位置の差が周期の倍数か
        diffs = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        if not diffs: continue
        
        gcd = diffs[0]
        for d in diffs[1:]:
            from math import gcd as _gcd
            gcd = _gcd(gcd, d)
        
        if gcd >= 1:
            # この周期で全非BGが一貫するか
            template = {}
            consistent = True
            for i, v in non_bg:
                pos = i % gcd
                if pos in template:
                    if template[pos] != v: consistent = False; break
                else:
                    template[pos] = v
            
            if consistent and template:
                return gcd, template
    return None, None

def _row_period_complete(tp, ti):
    """各行の周期パターンを補完"""
    for inp, out in tp:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        bg = _bg(ga)
        h, w = ga.shape
        
        # 各行で変更があるか
        changed_rows = [r for r in range(h) if not np.array_equal(ga[r], go[r])]
        if not changed_rows: return None
        
        # 変更行の出力に周期パターンがあるか
        for r in changed_rows:
            out_row = [int(v) for v in go[r]]
            period, template = _find_period(out_row, bg)
            if period is None: return None
        break
    
    def apply(grid):
        g = np.array(grid).copy()
        h2, w2 = g.shape; bg2 = _bg(g)
        changed = False
        for r in range(h2):
            row = [int(v) for v in g[r]]
            period, template = _find_period(row, bg2)
            if period and template:
                for c in range(w2):
                    pos = c % period
                    if pos in template and g[r, c] == bg2:
                        g[r, c] = template[pos]; changed = True
        return g.tolist() if changed else None
    
    ok = True
    for inp, out in tp:
        p = apply(inp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok: return apply(ti)
    return None

def _col_period_complete(tp, ti):
    """各列の周期パターンを補完"""
    def apply(grid):
        g = np.array(grid).copy()
        h2, w2 = g.shape; bg2 = _bg(g)
        changed = False
        for c in range(w2):
            col = [int(v) for v in g[:, c]]
            period, template = _find_period(col, bg2)
            if period and template:
                for r in range(h2):
                    pos = r % period
                    if pos in template and g[r, c] == bg2:
                        g[r, c] = template[pos]; changed = True
        return g.tolist() if changed else None
    
    ok = True
    for inp, out in tp:
        p = apply(inp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok: return apply(ti)
    return None

def _tile_period_complete(tp, ti):
    """2D周期パターン補完"""
    from math import gcd as _gcd
    
    def apply(grid):
        g = np.array(grid).copy()
        h, w = g.shape; bg2 = _bg(g)
        
        # 非BGセルの位置
        non_bg = [(r, c, int(g[r,c])) for r in range(h) for c in range(w) if g[r,c] != bg2]
        if len(non_bg) < 3: return None
        
        # 行と列それぞれの周期を見つける
        rows = sorted(set(r for r,_,_ in non_bg))
        cols = sorted(set(c for _,c,_ in non_bg))
        
        if len(rows) >= 2:
            row_diffs = [rows[i+1]-rows[i] for i in range(len(rows)-1)]
            row_period = row_diffs[0]
            for d in row_diffs[1:]: row_period = _gcd(row_period, d)
        else:
            row_period = h
        
        if len(cols) >= 2:
            col_diffs = [cols[i+1]-cols[i] for i in range(len(cols)-1)]
            col_period = col_diffs[0]
            for d in col_diffs[1:]: col_period = _gcd(col_period, d)
        else:
            col_period = w
        
        if row_period >= h and col_period >= w: return None
        
        # テンプレート構築
        template = {}
        consistent = True
        for r, c, v in non_bg:
            key = (r % row_period, c % col_period)
            if key in template:
                if template[key] != v: consistent = False; break
            else:
                template[key] = v
        
        if not consistent or not template: return None
        
        # 穴埋め
        changed = False
        for r in range(h):
            for c in range(w):
                key = (r % row_period, c % col_period)
                if key in template and g[r,c] == bg2:
                    g[r,c] = template[key]; changed = True
        
        return g.tolist() if changed else None
    
    ok = True
    for inp, out in tp:
        p = apply(inp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok: return apply(ti)
    return None


def _op_row_error_correct(tp, ti):
    """行ごとのパターンエラー訂正: 周期パターンの中のノイズを修正"""
    
    def correct_row(row):
        """行の周期パターンを見つけてエラーを訂正"""
        n = len(row)
        best_period = None
        best_errors = n  # 最小エラー数
        best_corrected = None
        
        for period in range(2, n//2 + 1):
            # この周期でテンプレートを「多数決」で決定
            template = []
            for p in range(period):
                vals = [row[i] for i in range(p, n, period)]
                mc = Counter(vals).most_common(1)[0][0]
                template.append(mc)
            
            # エラー数
            errors = 0
            corrected = list(row)
            for i in range(n):
                expected = template[i % period]
                if row[i] != expected:
                    errors += 1
                    corrected[i] = expected
            
            # 最小エラー（ただしエラーが0は「何も変わらない」ので除外しない）
            if 0 < errors < best_errors and errors <= n * 0.2:  # 20%以内のエラー
                best_errors = errors
                best_period = period
                best_corrected = corrected
        
        return best_corrected
    
    def apply(grid):
        g = np.array(grid).copy()
        h, w = g.shape
        changed = False
        
        for r in range(h):
            row = [int(v) for v in g[r]]
            corrected = correct_row(row)
            if corrected and corrected != row:
                for c in range(w):
                    g[r, c] = corrected[c]
                changed = True
        
        return g.tolist() if changed else None
    
    ok = True
    for inp, out in tp:
        p = apply(inp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok:
        return apply(ti)
    
    # 列方向も試す
    def apply_col(grid):
        g = np.array(grid).copy()
        h, w = g.shape
        changed = False
        for c in range(w):
            col = [int(g[r, c]) for r in range(h)]
            corrected = correct_row(col)
            if corrected and corrected != col:
                for r in range(h):
                    g[r, c] = corrected[r]
                changed = True
        return g.tolist() if changed else None
    
    ok = True
    for inp, out in tp:
        p = apply_col(inp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok:
        return apply_col(ti)
    
    # 行+列の両方
    def apply_both(grid):
        p = apply(grid)
        if p: return apply_col(p) or p
        return apply_col(grid)
    
    ok = True
    for inp, out in tp:
        p = apply_both(inp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok:
        return apply_both(ti)
    
    return None


def _op_qr_error_correct(tp, ti):
    """QR的エラー訂正: 枠を検出→内部の周期パターンを修正"""
    
    def correct_segment(seg):
        """セグメントの周期パターンを多数決で修正"""
        n = len(seg)
        if n < 2: return seg, False
        
        best = None; best_err = n
        for period in range(1, n//2 + 1):
            if n % period != 0 and period > 1:
                # 周期で割り切れなくてもOK
                pass
            template = []
            for p in range(period):
                vals = [seg[i] for i in range(p, n, period)]
                mc = Counter(vals).most_common(1)[0][0]
                template.append(mc)
            
            errors = sum(1 for i in range(n) if seg[i] != template[i % period])
            
            if 0 < errors <= max(1, n * 0.15) and errors < best_err:
                best_err = errors
                best = [template[i % period] for i in range(n)]
        
        if best:
            return best, True
        return seg, False
    
    def find_frame_cols(row):
        """行内の「枠色」列を検出（行の両端にある色 or 出現回数が少ない色）"""
        n = len(row)
        if n < 3: return [], row
        
        # 両端の色が同じなら枠色候補
        if row[0] == row[-1]:
            frame_color = row[0]
            # 枠色の列を見つける
            frame_cols = [i for i in range(n) if row[i] == frame_color 
                         and (i == 0 or i == n-1)]
            # 枠の直後に別の枠色がある場合
            inner_start = next((i for i in range(n) if row[i] != frame_color), 0)
            inner_end = n - 1 - next((i for i in range(n-1, -1, -1) if row[n-1-i] != frame_color), 0)
            
            if inner_start < inner_end:
                # 2層目の枠?
                if inner_start > 0 and inner_end < n:
                    inner2_color = row[inner_start]
                    if row[inner_end] == inner2_color:
                        # 2層枠: frame -> frame2 -> data -> frame2 -> frame
                        data_start = inner_start + 1
                        data_end = inner_end
                        return list(range(0, inner_start)) + list(range(inner_end+1, n)), \
                               row[data_start:data_end]
                
                return [i for i in range(inner_start)] + [i for i in range(inner_end+1, n)], \
                       row[inner_start:inner_end+1]
        
        return [], row
    
    def apply(grid):
        g = np.array(grid).copy()
        h, w = g.shape
        any_changed = False
        
        # 行ごとに枠を検出→内部を周期修正
        for r in range(h):
            row = [int(v) for v in g[r]]
            frame_cols, inner = find_frame_cols(row)
            
            corrected, changed = correct_segment(list(inner))
            if changed:
                # 内部を復元
                inner_idx = [i for i in range(w) if i not in frame_cols]
                for i, ci in enumerate(inner_idx):
                    if i < len(corrected):
                        g[r, ci] = corrected[i]
                any_changed = True
        
        # 列方向も同様
        for c in range(w):
            col = [int(g[r, c]) for r in range(h)]
            frame_rows, inner = find_frame_cols(col)
            
            corrected, changed = correct_segment(list(inner))
            if changed:
                inner_idx = [i for i in range(h) if i not in frame_rows]
                for i, ri in enumerate(inner_idx):
                    if i < len(corrected):
                        g[ri, c] = corrected[i]
                any_changed = True
        
        return g.tolist() if any_changed else None
    
    ok = True
    for inp, out in tp:
        p = apply(inp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok:
        return apply(ti)
    
    # 行だけ
    def apply_row_only(grid):
        g = np.array(grid).copy(); h, w = g.shape
        any_changed = False
        for r in range(h):
            row = [int(v) for v in g[r]]
            frame_cols, inner = find_frame_cols(row)
            corrected, changed = correct_segment(list(inner))
            if changed:
                inner_idx = [i for i in range(w) if i not in frame_cols]
                for i, ci in enumerate(inner_idx):
                    if i < len(corrected): g[r, ci] = corrected[i]
                any_changed = True
        return g.tolist() if any_changed else None
    
    ok = True
    for inp, out in tp:
        p = apply_row_only(inp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok:
        return apply_row_only(ti)
    
    return None



# ══════════════════════════════════════════════════════════════
# 精密操作: 枠認識→内部操作分離
# ══════════════════════════════════════════════════════════════

def _op_pattern_in_box(tp, ti):
    """箱の中のパターンを補完（枠を分離してから周期検出）"""
    bg0 = _bg(np.array(tp[0][0]))
    
    def find_box_interior(grid):
        """箱の内部領域を返す（r1,c1,r2,c2のリスト）"""
        g = np.array(grid)
        h, w = g.shape
        bg = _bg(g)
        objs = _objs(g, bg)
        interiors = []
        
        for obj in objs:
            r1,c1,r2,c2 = obj['bbox']
            if obj['bh'] < 3 or obj['bw'] < 3: continue
            if obj['is_rect']: continue
            
            # 辺上のセル割合
            cells_set = set(obj['cells'])
            border = sum(1 for r,c in cells_set if r==r1 or r==r2 or c==c1 or c==c2)
            if border / obj['size'] < 0.5: continue
            
            # 枠色
            fc = obj['color']
            
            # 内部の行範囲・列範囲
            # 各行で枠色の最左と最右を見つけ、その間が内部
            for r in range(r1, r2+1):
                row_frame = [c for c in range(c1, c2+1) if (r,c) in cells_set]
                if len(row_frame) >= 2:
                    inner_c1, inner_c2 = min(row_frame)+1, max(row_frame)-1
                    if inner_c2 >= inner_c1:
                        interiors.append((r, inner_c1, r, inner_c2, fc))
        
        # もっとシンプルに: 同色で囲まれた領域全体
        for obj in objs:
            if not obj['is_rect']: continue
            if obj['bh'] < 3 or obj['bw'] < 3: continue
            r1,c1,r2,c2 = obj['bbox']
            interiors.append((r1+1, c1+1, r2-1, c2-1, obj['color']))
        
        return interiors
    
    def complete_period_in_region(g, r1, c1, r2, c2, frame_color):
        """指定領域内の周期パターンを補完"""
        changed = False
        bg = _bg(g)
        
        for r in range(r1, r2+1):
            # この行の領域内セル
            segment = [(c, int(g[r,c])) for c in range(c1, c2+1)]
            if not segment: continue
            
            # 非フレーム色のパターンを抽出
            values = [v for _, v in segment]
            
            # 周期を探す
            for period in range(1, len(values)//2+1):
                # テンプレート構築（非BGかつ非frame）
                template = {}
                consistent = True
                for i, v in enumerate(values):
                    pos = i % period
                    if v != bg:
                        if pos in template:
                            if template[pos] != v:
                                consistent = False; break
                        else:
                            template[pos] = v
                
                if consistent and len(template) >= 1:
                    # BGセルを埋められるか
                    can_fill = False
                    for i, v in enumerate(values):
                        pos = i % period
                        if v == bg and pos in template:
                            can_fill = True; break
                    
                    if can_fill:
                        for i, (c, v) in enumerate(segment):
                            pos = i % period
                            if v == bg and pos in template:
                                g[r, c] = template[pos]
                                changed = True
                        break
        
        # 列方向も
        for c in range(c1, c2+1):
            segment = [(r, int(g[r,c])) for r in range(r1, r2+1)]
            values = [v for _, v in segment]
            
            for period in range(1, len(values)//2+1):
                template = {}
                consistent = True
                for i, v in enumerate(values):
                    pos = i % period
                    if v != bg:
                        if pos in template:
                            if template[pos] != v:
                                consistent = False; break
                        else:
                            template[pos] = v
                
                if consistent and len(template) >= 1:
                    can_fill = False
                    for i, v in enumerate(values):
                        pos = i % period
                        if v == bg and pos in template:
                            can_fill = True; break
                    
                    if can_fill:
                        for i, (r, v) in enumerate(segment):
                            pos = i % period
                            if v == bg and pos in template:
                                g[r, c] = template[pos]
                                changed = True
                        break
        
        return changed
    
    def apply(grid):
        g = np.array(grid).copy()
        h, w = g.shape
        bg = _bg(g)
        
        # 箱の内部を見つける
        interiors = find_box_interior(grid)
        
        changed = False
        for r1, c1, r2, c2, fc in interiors:
            if complete_period_in_region(g, r1, c1, r2, c2, fc):
                changed = True
        
        # 箱がなくても全体で周期補完を試す
        if not changed:
            if complete_period_in_region(g, 0, 0, h-1, w-1, -1):
                changed = True
        
        return g.tolist() if changed else None
    
    ok = True
    for inp, out in tp:
        p = apply(inp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok: return apply(ti)
    return None


def _op_symmetry_complete(tp, ti):
    """対称性を検出して補完する"""
    
    def apply(grid):
        g = np.array(grid).copy()
        h, w = g.shape
        bg = _bg(g)
        
        best = None; best_fill = 0
        
        for mode in ['lr', 'ud', 'lr_ud', 'rot90', 'rot180']:
            test = g.copy()
            fill_count = 0
            conflict = False
            
            if mode == 'lr':
                for r in range(h):
                    for c in range(w):
                        mc = w-1-c
                        if test[r,c] != bg and test[r,mc] == bg:
                            test[r,mc] = test[r,c]; fill_count += 1
                        elif test[r,c] != bg and test[r,mc] != bg and test[r,c] != test[r,mc]:
                            conflict = True; break
                    if conflict: break
            
            elif mode == 'ud':
                for r in range(h):
                    for c in range(w):
                        mr = h-1-r
                        if test[r,c] != bg and test[mr,c] == bg:
                            test[mr,c] = test[r,c]; fill_count += 1
                        elif test[r,c] != bg and test[mr,c] != bg and test[r,c] != test[mr,c]:
                            conflict = True; break
                    if conflict: break
            
            elif mode == 'lr_ud':
                for r in range(h):
                    for c in range(w):
                        mirrors = [(r, w-1-c), (h-1-r, c), (h-1-r, w-1-c)]
                        for mr, mc in mirrors:
                            if test[r,c] != bg and test[mr,mc] == bg:
                                test[mr,mc] = test[r,c]; fill_count += 1
                            elif test[r,c] != bg and test[mr,mc] != bg and test[r,c] != test[mr,mc]:
                                conflict = True; break
                        if conflict: break
                    if conflict: break
            
            elif mode == 'rot180':
                for r in range(h):
                    for c in range(w):
                        mr, mc = h-1-r, w-1-c
                        if test[r,c] != bg and test[mr,mc] == bg:
                            test[mr,mc] = test[r,c]; fill_count += 1
                        elif test[r,c] != bg and test[mr,mc] != bg and test[r,c] != test[mr,mc]:
                            conflict = True; break
                    if conflict: break
            
            elif mode == 'rot90':
                if h != w: continue
                for r in range(h):
                    for c in range(w):
                        rotations = [(c, h-1-r), (h-1-r, w-1-c), (w-1-c, r)]
                        for mr, mc in rotations:
                            if test[r,c] != bg and test[mr,mc] == bg:
                                test[mr,mc] = test[r,c]; fill_count += 1
                            elif test[r,c] != bg and test[mr,mc] != bg and test[r,c] != test[mr,mc]:
                                conflict = True; break
                        if conflict: break
                    if conflict: break
            
            if not conflict and fill_count > 0:
                if best is None or fill_count > best_fill:
                    best = test; best_fill = fill_count
        
        return best.tolist() if best is not None else None
    
    ok = True
    for inp, out in tp:
        p = apply(inp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok: return apply(ti)
    return None


def _op_object_gravity(tp, ti):
    """オブジェクトを重力方向に落とす"""
    for direction in ['down', 'up', 'left', 'right']:
        ok = True
        for inp, out in tp:
            p = _apply_gravity(inp, direction)
            if p is None or not grid_eq(p, out): ok = False; break
        if ok: return _apply_gravity(ti, direction)
    return None

def _apply_gravity(grid, direction):
    g = np.array(grid).copy()
    h, w = g.shape
    bg = _bg(g)
    
    if direction == 'down':
        for c in range(w):
            vals = [int(g[r,c]) for r in range(h) if g[r,c] != bg]
            g[:, c] = bg
            for i, v in enumerate(reversed(vals)):
                g[h-1-i, c] = v
    elif direction == 'up':
        for c in range(w):
            vals = [int(g[r,c]) for r in range(h) if g[r,c] != bg]
            g[:, c] = bg
            for i, v in enumerate(vals):
                g[i, c] = v
    elif direction == 'right':
        for r in range(h):
            vals = [int(g[r,c]) for c in range(w) if g[r,c] != bg]
            g[r, :] = bg
            for i, v in enumerate(reversed(vals)):
                g[r, w-1-i] = v
    elif direction == 'left':
        for r in range(h):
            vals = [int(g[r,c]) for c in range(w) if g[r,c] != bg]
            g[r, :] = bg
            for i, v in enumerate(vals):
                g[r, i] = v
    
    if np.array_equal(g, np.array(grid)): return None
    return g.tolist()


def _op_flood_enclosed(tp, ti):
    """閉じた領域をflood fill"""
    def apply(grid):
        g = np.array(grid).copy()
        h, w = g.shape
        bg = _bg(g)
        
        # BGの連結成分でエッジに触れないものを塗る
        from scipy.ndimage import label as sl
        bg_mask = (g == bg).astype(int)
        labeled, n = sl(bg_mask)
        
        edge_labels = set()
        for r in [0, h-1]:
            for c in range(w):
                if labeled[r,c] > 0: edge_labels.add(labeled[r,c])
        for c in [0, w-1]:
            for r in range(h):
                if labeled[r,c] > 0: edge_labels.add(labeled[r,c])
        
        changed = False
        for label_id in range(1, n+1):
            if label_id in edge_labels: continue
            # この閉じた領域を囲む色は？
            region = list(zip(*np.where(labeled == label_id)))
            neighbor_colors = set()
            for r, c in region:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w and g[nr,nc] != bg:
                        neighbor_colors.add(int(g[nr,nc]))
            
            if len(neighbor_colors) == 1:
                fill_color = neighbor_colors.pop()
                for r, c in region:
                    g[r,c] = fill_color
                    changed = True
        
        return g.tolist() if changed else None
    
    ok = True
    for inp, out in tp:
        p = apply(inp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok: return apply(ti)
    return None


def _op_majority_fill(tp, ti):
    """各オブジェクト/領域を最頻色で塗りつぶす"""
    def apply(grid):
        g = np.array(grid).copy()
        h, w = g.shape
        bg = _bg(g)
        objs = _objs(g, bg)
        changed = False
        
        for obj in objs:
            if len(obj['colors']) <= 1: continue
            majority = obj['color']  # most common
            for r, c in obj['cells']:
                if int(g[r,c]) != majority:
                    g[r,c] = majority
                    changed = True
        
        return g.tolist() if changed else None
    
    ok = True
    for inp, out in tp:
        p = apply(inp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok: return apply(ti)
    return None


def _op_mirror_across_axis(tp, ti):
    """軸オブジェクト（壁/線）を基準にミラー"""
    for inp, out in tp:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        bg = _bg(ga)
        h, w = ga.shape
        
        # 水平軸を探す
        for r in range(h):
            is_axis = all(ga[r,c] != bg for c in range(w))
            if not is_axis: continue
            
            # この行を軸にミラー
            def mirror_h(grid, axis_r):
                g = np.array(grid).copy()
                hh, ww = g.shape; bg2 = _bg(g)
                changed = False
                for rr in range(hh):
                    mr = 2*axis_r - rr
                    if mr < 0 or mr >= hh or mr == rr: continue
                    for cc in range(ww):
                        if g[rr,cc] != bg2 and g[mr,cc] == bg2:
                            g[mr,cc] = g[rr,cc]; changed = True
                return g.tolist() if changed else None
            
            ok = True
            for i2, o2 in tp:
                ga2 = np.array(i2); bg2 = _bg(ga2)
                axes = [rr for rr in range(ga2.shape[0]) if all(ga2[rr,cc]!=bg2 for cc in range(ga2.shape[1]))]
                if not axes: ok=False; break
                p = mirror_h(i2, axes[0])
                if p is None or not grid_eq(p, o2): ok=False; break
            if ok:
                gi = np.array(ti); bg_t = _bg(gi)
                axes_t = [rr for rr in range(gi.shape[0]) if all(gi[rr,cc]!=bg_t for cc in range(gi.shape[1]))]
                if axes_t: return mirror_h(ti, axes_t[0])
        
        # 垂直軸
        for c in range(w):
            is_axis = all(ga[r,c] != bg for r in range(h))
            if not is_axis: continue
            
            def mirror_v(grid, axis_c):
                g = np.array(grid).copy()
                hh, ww = g.shape; bg2 = _bg(g)
                changed = False
                for cc in range(ww):
                    mc = 2*axis_c - cc
                    if mc < 0 or mc >= ww or mc == cc: continue
                    for rr in range(hh):
                        if g[rr,cc] != bg2 and g[rr,mc] == bg2:
                            g[rr,mc] = g[rr,cc]; changed = True
                return g.tolist() if changed else None
            
            ok = True
            for i2, o2 in tp:
                ga2 = np.array(i2); bg2 = _bg(ga2)
                axes = [cc for cc in range(ga2.shape[1]) if all(ga2[rr,cc]!=bg2 for rr in range(ga2.shape[0]))]
                if not axes: ok=False; break
                p = mirror_v(i2, axes[0])
                if p is None or not grid_eq(p, o2): ok=False; break
            if ok:
                gi = np.array(ti); bg_t = _bg(gi)
                axes_t = [cc for cc in range(gi.shape[1]) if all(gi[rr,cc]!=bg_t for rr in range(gi.shape[0]))]
                if axes_t: return mirror_v(ti, axes_t[0])
        break
    return None


def _op_extract_by_size(tp, ti):
    """特定サイズのオブジェクトを抽出"""
    for criterion in ['smallest', 'largest', 'median', 'most_common_shape', 'unique_color']:
        ok = True
        for inp, out in tp:
            p = _apply_extract(inp, criterion)
            if p is None or not grid_eq(p, out): ok = False; break
        if ok: return _apply_extract(ti, criterion)
    return None

def _apply_extract(grid, criterion):
    g = np.array(grid); bg = _bg(g)
    objs = _objs(g, bg)
    if not objs: return None
    
    target = None
    if criterion == 'smallest':
        target = min(objs, key=lambda o: o['size'])
    elif criterion == 'largest':
        target = max(objs, key=lambda o: o['size'])
    elif criterion == 'median':
        objs_sorted = sorted(objs, key=lambda o: o['size'])
        target = objs_sorted[len(objs_sorted)//2]
    elif criterion == 'most_common_shape':
        from collections import Counter as C2
        sc = C2(o['shape'] for o in objs)
        if len(sc) >= 2:
            least = sc.most_common()[-1][0]
            cands = [o for o in objs if o['shape'] == least]
            if cands: target = cands[0]
    elif criterion == 'unique_color':
        from collections import Counter as C2
        cc = C2(o['color'] for o in objs)
        uniq = [o for o in objs if cc[o['color']] == 1]
        if uniq: target = uniq[0]
    
    if target is None: return None
    r1,c1,r2,c2 = target['bbox']
    return g[r1:r2+1, c1:c2+1].tolist()


# ══════════════════════════════════════════════════════════════
# 精密操作: 周期修復（QRエラー訂正の本質）
# ══════════════════════════════════════════════════════════════

def _find_best_period_fix(seq):
    """多数決ベースの周期修復: 最適周期でエラーセルを修正"""
    n = len(seq)
    if n < 2: return seq, False
    
    best = None
    for period in range(1, max(n//2+1, 2)):
        template = defaultdict(list)
        for i in range(n):
            template[i % period].append(seq[i])
        
        tmpl = {}
        errors = 0
        for pos, vals in template.items():
            c = Counter(vals)
            majority = c.most_common(1)[0][0]
            tmpl[pos] = majority
            errors += sum(1 for v in vals if v != majority)
        
        # 周期1（全部同じ）はスキップ（意味のある周期のみ）
        if period == 1 and len(set(tmpl.values())) == 1:
            continue
        
        if 0 < errors <= max(n * 0.35, 1):
            if best is None or errors < best[1]:
                best = (period, errors, tmpl)
    
    if best is None: return seq, False
    
    period, _, tmpl = best
    fixed = list(seq)
    changed = False
    for i in range(n):
        if fixed[i] != tmpl[i % period]:
            fixed[i] = tmpl[i % period]
            changed = True
    return fixed, changed


def _op_period_repair(tp, ti):
    """枠の中の周期パターンを修復（QRエラー訂正）"""
    
    def apply(grid):
        g = np.array(grid).copy()
        h, w = g.shape
        bg = _bg(g)
        changed = False
        
        # 枠色を検出（各行の端の非BG色）
        frame_color = None
        for r in range(h):
            for c in [0, w-1]:
                v = int(g[r, c])
                if v != bg:
                    frame_color = v; break
            if frame_color is not None: break
        
        # 各行で枠色に挟まれた内部を周期修復
        for r in range(h):
            row = [int(v) for v in g[r]]
            
            # 枠色の位置
            frame_positions = [c for c in range(w) if row[c] == frame_color] if frame_color is not None else []
            
            if len(frame_positions) >= 2:
                # 枠間の内部
                inner_start = frame_positions[0] + 1
                inner_end = frame_positions[-1]
                if inner_end > inner_start:
                    inner = row[inner_start:inner_end]
                    fixed, did_change = _find_best_period_fix(inner)
                    if did_change:
                        for i, v in enumerate(fixed):
                            g[r, inner_start + i] = v
                        changed = True
            else:
                # 枠なし: 行全体を修復
                fixed, did_change = _find_best_period_fix(row)
                if did_change:
                    for c, v in enumerate(fixed):
                        g[r, c] = v
                    changed = True
        
        # 各列も
        for c in range(w):
            col = [int(g[r, c]) for r in range(h)]
            
            frame_positions = [r for r in range(h) if col[r] == frame_color] if frame_color is not None else []
            
            if len(frame_positions) >= 2:
                inner_start = frame_positions[0] + 1
                inner_end = frame_positions[-1]
                if inner_end > inner_start:
                    inner = col[inner_start:inner_end]
                    fixed, did_change = _find_best_period_fix(inner)
                    if did_change:
                        for i, v in enumerate(fixed):
                            g[inner_start + i, c] = v
                        changed = True
            else:
                fixed, did_change = _find_best_period_fix(col)
                if did_change:
                    for r, v in enumerate(fixed):
                        g[r, c] = v
                    changed = True
        
        return g.tolist() if changed else None
    
    ok = True
    for inp, out in tp:
        p = apply(inp)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok: return apply(ti)
    return None


def _op_periodic_repair_general(tp, ti):
    """一般化された周期修正: 入出力の差分が周期パターンのエラー修正かどうか"""
    
    def fix_periodic(seg):
        """周期の多数決修正"""
        n = len(seg)
        if n < 3: return None
        best = None; be = n
        for p in range(1, n//2+1):
            t = [Counter([seg[i] for i in range(pp, n, p)]).most_common(1)[0][0] for pp in range(p)]
            e = sum(1 for i in range(n) if seg[i] != t[i%p])
            if 0 < e <= max(1, int(n*0.2)) and e < be:
                be = e; best = [t[i%p] for i in range(n)]
        return best
    
    # Step 1: trainから「変更されたセルの行を特定」して
    # その行全体 or 行のサブセグメントで周期修正が成立するか
    
    for inp, out in tp:
        gi, go = np.array(inp), np.array(out)
        if gi.shape != go.shape: return None
        
        # 変更行
        changed_rows = [r for r in range(gi.shape[0]) if not np.array_equal(gi[r], go[r])]
        changed_cols = [c for c in range(gi.shape[1]) if not np.array_equal(gi[:,c], go[:,c])]
        
        if not changed_rows and not changed_cols: return None
        
        # 変更行の各セグメント（同色の連続区間で区切る）で周期修正を試す
        break
    
    # アプローチ: 各trainの変更行を特定→その行を周期修正→合うかチェック
    # テストでは「周期修正で変わる行」を修正対象にする
    
    def apply_row_repair(grid, changed_rows_hint=None):
        g = np.array(grid).copy(); h, w = g.shape
        changed = False
        
        for r in range(h):
            if changed_rows_hint is not None and r not in changed_rows_hint:
                continue
            
            row = [int(v) for v in g[r]]
            fixed = fix_periodic(row)
            if fixed:
                for c in range(w):
                    if g[r,c] != fixed[c]:
                        g[r,c] = fixed[c]; changed = True
        
        return g.tolist() if changed else None
    
    def apply_col_repair(grid, changed_cols_hint=None):
        g = np.array(grid).copy(); h, w = g.shape
        changed = False
        
        for c in range(w):
            if changed_cols_hint is not None and c not in changed_cols_hint:
                continue
            
            col = [int(g[r2,c]) for r2 in range(h)]
            fixed = fix_periodic(col)
            if fixed:
                for r2 in range(h):
                    if g[r2,c] != fixed[r2]:
                        g[r2,c] = fixed[r2]; changed = True
        
        return g.tolist() if changed else None
    
    # 行修正のみ
    ok = True
    for inp, out in tp:
        gi, go = np.array(inp), np.array(out)
        cr = set(r for r in range(gi.shape[0]) if not np.array_equal(gi[r], go[r]))
        p = apply_row_repair(inp, cr)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok:
        # test: 周期エラーがある行を自動検出
        return apply_row_repair(ti)
    
    # 列修正のみ
    ok = True
    for inp, out in tp:
        gi, go = np.array(inp), np.array(out)
        cc = set(c for c in range(gi.shape[1]) if not np.array_equal(gi[:,c], go[:,c]))
        p = apply_col_repair(inp, cc)
        if p is None or not grid_eq(p, out): ok = False; break
    if ok:
        return apply_col_repair(ti)
    
    # 行+列
    ok = True
    for inp, out in tp:
        gi, go = np.array(inp), np.array(out)
        cr = set(r for r in range(gi.shape[0]) if not np.array_equal(gi[r], go[r]))
        p = apply_row_repair(inp, cr)
        if p:
            cc = set(c for c in range(gi.shape[1]) if not np.array_equal(np.array(p)[:,c], go[:,c]))
            p2 = apply_col_repair(p, cc)
            if p2 and grid_eq(p2, out): continue
        ok = False; break
    if ok:
        p = apply_row_repair(ti)
        if p: 
            p2 = apply_col_repair(p)
            return p2 or p
    
    return None



def _op_period_repair_2d(tp, ti):
    """2D周期修復: パネル→枠剥ぎ→多数決修復（短い周期優先）"""
    
    def find_panels(grid):
        g=np.array(grid); h,w=g.shape
        h_seps=[]; v_seps=[]
        for color in set(int(v) for v in g.flatten()):
            for r in range(h):
                if all(g[r,c]==color for c in range(w)): h_seps.append(r)
            for c in range(w):
                if all(g[r,c]==color for r in range(h)): v_seps.append(c)
        h_seps=sorted(set(h_seps)); v_seps=sorted(set(v_seps))
        rb=[]; prev=-1
        for s in h_seps:
            if s-prev>1: rb.append((prev+1,s-1))
            prev=s
        if prev<h-1: rb.append((prev+1,h-1))
        if not rb: rb=[(0,h-1)]
        cb=[]; prev=-1
        for s in v_seps:
            if s-prev>1: cb.append((prev+1,s-1))
            prev=s
        if prev<w-1: cb.append((prev+1,w-1))
        if not cb: cb=[(0,w-1)]
        return [(r1,c1,r2,c2) for r1,r2 in rb for c1,c2 in cb]
    
    def repair_flat(region):
        p=np.array(region); ph,pw=p.shape
        if ph<1 or pw<1: return None
        candidates=[]
        for pr in range(1,ph+1):
            for pc in range(1,pw+1):
                if pr==1 and pc==1: continue
                td=defaultdict(list)
                for r in range(ph):
                    for c in range(pw):
                        td[(r%pr,c%pc)].append(int(p[r,c]))
                tmpl={}; errors=0
                for k,vals in td.items():
                    maj=Counter(vals).most_common(1)[0][0]; tmpl[k]=maj
                    errors+=sum(1 for v in vals if v!=maj)
                if 0<errors<=max(ph*pw*0.25,2):
                    # スコア: 短い周期を強く優先
                    # errors_per_cell正規化 + 周期長ペナルティ
                    score = errors / max(ph*pw, 1) + (pr*pc) / (ph*pw) * 0.5
                    fixed=p.copy()
                    for r in range(ph):
                        for c in range(pw):
                            fixed[r,c]=tmpl[(r%pr,c%pc)]
                    candidates.append((score, pr, pc, errors, fixed))
        
        if not candidates: return None
        candidates.sort()
        return candidates[0][4]
    
    def repair_region(panel):
        p=np.array(panel); h,w=p.shape
        if h<1 or w<1: return None
        if h>=3 and w>=3:
            top=set(int(v) for v in p[0]); bot=set(int(v) for v in p[-1])
            left=set(int(p[r,0]) for r in range(h)); right=set(int(p[r,-1]) for r in range(h))
            if len(top)==1 and top==bot and top==left and top==right:
                inner=p[1:-1,1:-1]
                fixed_inner=repair_flat(inner)
                if fixed_inner is not None:
                    result=p.copy(); result[1:-1,1:-1]=fixed_inner
                    return result
        fixed=repair_flat(p)
        return fixed
    
    def apply(grid):
        g=np.array(grid).copy()
        panels=find_panels(grid)
        changed=False
        for r1,c1,r2,c2 in panels:
            fixed=repair_region(g[r1:r2+1,c1:c2+1])
            if fixed is not None:
                g[r1:r2+1,c1:c2+1]=np.array(fixed); changed=True
        if not changed:
            fixed=repair_region(g)
            if fixed is not None: return np.array(fixed).tolist()
        return g.tolist() if changed else None
    
    ok=True
    for inp,out in tp:
        p=apply(inp)
        if p is None or not grid_eq(p,out): ok=False; break
    if ok: return apply(ti)
    return None



def _op_general_error_correct(tp, ti):
    """汎用QRエラー訂正: 複数の修復戦略を統合
    
    1. 2D多数決修復（パネルあり/なし）
    2. 行/列ごとの周期修復（枠検出付き）
    3. 対称性ベースの修復
    4. テンプレートマッチング修復（同じパターンの領域間で多数決）
    """
    
    # Strategy A: 行ごとの周期多数決修復（training問題の広範なエラーに対応）
    def fix_periodic_row(seq):
        """行の周期パターンを多数決で修正（複数周期を試す）"""
        n = len(seq)
        if n < 3: return None
        best = None; best_score = 999
        for p in range(2, min(n//2+1, 16)):  # 周期上限を拡大
            tmpl = []
            for pp in range(p):
                vals = [seq[i] for i in range(pp, n, p)]
                tmpl.append(Counter(vals).most_common(1)[0][0])
            errors = sum(1 for i in range(n) if seq[i] != tmpl[i%p])
            if 0 < errors <= max(1, int(n*0.3)):  # 30%まで許容
                # スコア: エラー少＋周期短を優先
                score = errors + p * 0.1
                if score < best_score:
                    best_score = score
                    best = [tmpl[i%p] for i in range(n)]
        return best
    
    # A1: 全行修復
    def apply_all_row(grid):
        g=np.array(grid).copy();h,w=g.shape;changed=False
        for r in range(h):
            row=[int(v) for v in g[r]]
            fixed=fix_periodic_row(row)
            if fixed:
                for c in range(w):
                    if g[r,c]!=fixed[c]: g[r,c]=fixed[c];changed=True
        return g.tolist() if changed else None
    
    ok=True
    for inp,out in tp:
        p=apply_all_row(inp)
        if p is not None and grid_eq(p,out): continue
        ok=False;break
    if ok:
        r=apply_all_row(ti)
        if r: return r
    
    # A2: 全列修復
    def apply_all_col(grid):
        g=np.array(grid).copy();h,w=g.shape;changed=False
        for c in range(w):
            col=[int(g[r,c]) for r in range(h)]
            fixed=fix_periodic_row(col)
            if fixed:
                for r in range(h):
                    if g[r,c]!=fixed[r]: g[r,c]=fixed[r];changed=True
        return g.tolist() if changed else None
    
    ok=True
    for inp,out in tp:
        p=apply_all_col(inp)
        if p is not None and grid_eq(p,out): continue
        ok=False;break
    if ok:
        r=apply_all_col(ti)
        if r: return r
    
    # A3: 行→列の2パス修復
    def apply_row_then_col(grid):
        p=apply_all_row(grid)
        if p: 
            p2=apply_all_col(p)
            return p2 if p2 else p
        return apply_all_col(grid)
    
    ok=True
    for inp,out in tp:
        p=apply_row_then_col(inp)
        if p is not None and grid_eq(p,out): continue
        ok=False;break
    if ok:
        r=apply_row_then_col(ti)
        if r: return r
    
    # Strategy B: テンプレートマッチング修復
    # 同じパターンが複数箇所にある場合、多数決で修復
    def apply_template_repair(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g)
        # パネル分割
        h_seps=[];v_seps=[]
        for r2 in range(h):
            vals=set(int(v) for v in g[r2])
            if len(vals)==1 and vals.pop()!=bg: h_seps.append(r2)
        for c2 in range(w):
            vals=set(int(g[r2,c2]) for r2 in range(h))
            if len(vals)==1 and vals.pop()!=bg: v_seps.append(c2)
        
        if not h_seps and not v_seps: return None
        
        rows=[-1]+sorted(h_seps)+[h]; cols=[-1]+sorted(v_seps)+[w]
        panels=[]
        for i in range(len(rows)-1):
            for j in range(len(cols)-1):
                r1,r2_=rows[i]+1,rows[i+1]; c1,c2_=cols[j]+1,cols[j+1]
                if r2_>r1 and c2_>c1:
                    panels.append((r1,c1,r2_,c2_,g[r1:r2_,c1:c2_].copy()))
        
        if len(panels)<2: return None
        shapes=set(p[4].shape for p in panels)
        if len(shapes)!=1: return None
        
        ph,pw=panels[0][4].shape
        # 各位置で多数決
        template=np.zeros((ph,pw),dtype=int)
        for r in range(ph):
            for c in range(pw):
                vals=[int(p[4][r,c]) for p in panels]
                template[r,c]=Counter(vals).most_common(1)[0][0]
        
        # テンプレートと異なるセルを修正
        changed=False
        for r1,c1,r2_,c2_,panel in panels:
            for r in range(ph):
                for c in range(pw):
                    if panel[r,c]!=template[r,c]:
                        g[r1+r,c1+c]=template[r,c];changed=True
        
        return g.tolist() if changed else None
    
    ok=True
    for inp,out in tp:
        p=apply_template_repair(inp)
        if p is not None and grid_eq(p,out): continue
        ok=False;break
    if ok:
        r=apply_template_repair(ti)
        if r: return r
    
    # Strategy C: 対称性ベース修復
    # ほぼ対称なグリッドの非対称セルを修復
    for sym_mode in ['lr','ud','rot180']:
        def apply_sym_repair(grid, mode=sym_mode):
            g=np.array(grid).copy();h2,w2=g.shape;changed=False
            if mode=='lr':
                for r in range(h2):
                    for c in range(w2//2):
                        mc=w2-1-c
                        if g[r,c]!=g[r,mc]:
                            # 多数決: 左右でどちらが多いか
                            left_count=sum(1 for rr in range(h2) if g[rr,c]==g[r,c] and g[rr,mc]==g[r,c])
                            right_count=sum(1 for rr in range(h2) if g[rr,c]==g[r,mc] and g[rr,mc]==g[r,mc])
                            if right_count>left_count: g[r,c]=g[r,mc];changed=True
                            elif left_count>right_count: g[r,mc]=g[r,c];changed=True
            elif mode=='ud':
                for r in range(h2//2):
                    mr=h2-1-r
                    for c in range(w2):
                        if g[r,c]!=g[mr,c]:
                            top_count=sum(1 for cc in range(w2) if g[r,cc]==g[r,c] and g[mr,cc]==g[r,c])
                            bot_count=sum(1 for cc in range(w2) if g[r,cc]==g[mr,c] and g[mr,cc]==g[mr,c])
                            if bot_count>top_count: g[r,c]=g[mr,c];changed=True
                            elif top_count>bot_count: g[mr,c]=g[r,c];changed=True
            elif mode=='rot180':
                for r in range(h2):
                    for c in range(w2):
                        mr,mc=h2-1-r,w2-1-c
                        if (r*w2+c)>=(mr*w2+mc): continue
                        if g[r,c]!=g[mr,mc]:
                            g[mr,mc]=g[r,c];changed=True
            return g.tolist() if changed else None
        
        ok=True
        for inp,out in tp:
            p=apply_sym_repair(inp)
            if p is not None and grid_eq(p,out): continue
            ok=False;break
        if ok:
            r=apply_sym_repair(ti)
            if r: return r
    
    # Strategy D: 枠内部の2D周期修復（既存_op_period_repair_2dとは別の実装）
    def apply_frame_period_repair(grid):
        g=np.array(grid).copy();h2,w2=g.shape;bg2=_bg(g)
        # 枠色検出: 4辺で最頻の非BG色
        edge_colors=[]
        for c in range(w2): edge_colors.append(int(g[0,c]));edge_colors.append(int(g[h2-1,c]))
        for r in range(h2): edge_colors.append(int(g[r,0]));edge_colors.append(int(g[r,w2-1]))
        edge_nb=[v for v in edge_colors if v!=bg2]
        if not edge_nb: return None
        frame_color=Counter(edge_nb).most_common(1)[0][0]
        
        # 枠の内側を特定
        r1=0;r2=h2-1;c1=0;c2=w2-1
        while r1<h2 and all(g[r1,c]==frame_color or g[r1,c]==bg2 for c in range(w2)): r1+=1
        while r2>=0 and all(g[r2,c]==frame_color or g[r2,c]==bg2 for c in range(w2)): r2-=1
        while c1<w2 and all(g[r,c1]==frame_color or g[r,c1]==bg2 for r in range(h2)): c1+=1
        while c2>=0 and all(g[r,c2]==frame_color or g[r,c2]==bg2 for r in range(h2)): c2-=1
        
        if r1>=r2 or c1>=c2: return None
        inner=g[r1:r2+1,c1:c2+1]
        ih,iw=inner.shape
        
        # 内部の2D周期修復
        changed=False
        for pr in range(1,ih//2+1):
            for pc in range(1,iw//2+1):
                tmpl={};errors=0
                for r in range(ih):
                    for c in range(iw):
                        k=(r%pr,c%pc);v=int(inner[r,c])
                        if k not in tmpl: tmpl[k]=[]
                        tmpl[k].append(v)
                majority_tmpl={}
                for k,vals in tmpl.items():
                    maj=Counter(vals).most_common(1)[0][0]
                    majority_tmpl[k]=maj
                    errors+=sum(1 for v in vals if v!=maj)
                
                if 0<errors<=max(ih*iw*0.2,1):
                    fixed=inner.copy()
                    for r in range(ih):
                        for c in range(iw):
                            expected=majority_tmpl[(r%pr,c%pc)]
                            if fixed[r,c]!=expected:
                                fixed[r,c]=expected;changed=True
                    if changed:
                        g[r1:r2+1,c1:c2+1]=fixed
                        return g.tolist()
        return None
    
    ok=True
    for inp,out in tp:
        p=apply_frame_period_repair(inp)
        if p is not None and grid_eq(p,out): continue
        ok=False;break
    if ok:
        r=apply_frame_period_repair(ti)
        if r: return r
    
    return None



if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    
    split = 'evaluation' if '--eval' in sys.argv else 'training'
    data_dir = Path(f'/tmp/arc-agi-2/data/{split}')
    
    existing = set()
    with open('arc_v82.log') as f:
        for l in f:
            m = re.search(r'✓.*?([0-9a-f]{8})', l)
            if m: existing.add(m.group(1))
    synth = set(f.stem for f in Path('synth_results').glob('*.py'))
    all_e = existing | synth
    
    solved = []
    scene_stats = Counter()
    op_stats = Counter()
    
    for tf in sorted(data_dir.glob('*.json')):
        tid = tf.stem
        with open(tf) as f: task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti, to = task['test'][0]['input'], task['test'][0].get('output')
        
        # 場面統計
        s = see_scene(tp[0][0])
        scene_stats[s['scene']] += 1
        
        result, name = cross_vision_solve(tp, ti)
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_e else ''
            solved.append((tid, name, tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tg in solved if tg == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
    
    # 場面統計
    print('\n場面分布:')
    for s, c in scene_stats.most_common():
        print(f'  {s}: {c}')
    
    # ソルバー統計
    solver_stats = Counter()
    for _, name, _ in solved:
        solver_stats[name] += 1
    print('\nソルバー分布:')
    for s, c in solver_stats.most_common():
        print(f'  {s}: {c}')

