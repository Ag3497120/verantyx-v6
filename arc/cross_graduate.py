"""
arc/cross_graduate.py — フル記憶: 大学院生レベルの知識

断片記憶(十字くん)が「これっぽい」と勘で選ぶ → フル記憶から高度な手法を取り出す

=== 知識体系 ===
1. グラフ理論: 連結成分、最短経路、二部グラフ、マッチング
2. 群論: 対称群（回転・反転・スライド）、不変量
3. 画像処理: テンプレートマッチング、モルフォロジー、畳み込み
4. 制約充足: バックトラック、弧整合性
5. 文法推論: パターン→ルール抽出、一般化
6. 位相: 連結性、穴の数、境界検出
7. 信号処理: 周期検出、FFT的パターン分解
8. 組合せ: 全探索、枝刈り、対称性除去
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Set, Any
from collections import Counter, defaultdict, deque
from scipy.ndimage import label as scipy_label
from itertools import product

def _bg(g): return int(Counter(np.array(g).flatten()).most_common(1)[0][0])

def _objs(g, bg, conn=8):
    struct = np.ones((3,3),dtype=int) if conn==8 else np.array([[0,1,0],[1,1,1],[0,1,0]])
    mask = (np.array(g)!=bg).astype(int)
    labeled, n = scipy_label(mask, structure=struct)
    objs = []
    for i in range(1, n+1):
        cells = list(zip(*np.where(labeled==i)))
        colors = [int(g[r,c]) for r,c in cells]
        r1=min(r for r,c in cells); c1=min(c for r,c in cells)
        r2=max(r for r,c in cells); c2=max(c for r,c in cells)
        objs.append({
            'cells': cells, 'size': len(cells),
            'color': Counter(colors).most_common(1)[0][0],
            'colors': set(colors),
            'bbox': (r1,c1,r2,c2), 'bh': r2-r1+1, 'bw': c2-c1+1,
            'shape': frozenset((r-r1,c-c1) for r,c in cells),
            'patch': np.array(g)[r1:r2+1, c1:c2+1].copy(),
        })
    return objs


# ══════════════════════════════════════════════════════════════
# 1. テンプレートマッチング（画像処理）
# ══════════════════════════════════════════════════════════════

def template_match_and_apply(train_pairs, test_input):
    """trainの入出力差分からテンプレート→変換ルールを学習し適用"""
    from arc.grid import grid_eq
    
    gi = np.array(test_input)
    bg = _bg(gi)
    
    # 各trainからルールを抽出: 特定パターン周辺の変換
    rules = []
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: continue
        
        # 差分セル
        diff_mask = (ga != go)
        if not diff_mask.any(): continue
        
        # 差分の周辺コンテキストからルール化
        diff_cells = list(zip(*np.where(diff_mask)))
        for r, c in diff_cells:
            # 3x3近傍のinputパターン → output色
            ctx = []
            for dr in [-1, 0, 1]:
                row = []
                for dc in [-1, 0, 1]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<ga.shape[0] and 0<=nc<ga.shape[1]:
                        row.append(int(ga[nr, nc]))
                    else:
                        row.append(-1)
                ctx.append(tuple(row))
            rules.append((tuple(ctx), int(go[r, c])))
    
    if not rules: return None
    
    # ルールの一貫性チェック: 同じコンテキスト → 同じ出力
    rule_map = {}
    consistent = True
    for ctx, out_color in rules:
        if ctx in rule_map:
            if rule_map[ctx] != out_color:
                consistent = False; break
        rule_map[ctx] = out_color
    
    if not consistent or not rule_map: return None
    
    # test に適用
    h, w = gi.shape
    result = gi.copy()
    changed = False
    for r in range(h):
        for c in range(w):
            ctx = []
            for dr in [-1, 0, 1]:
                row = []
                for dc in [-1, 0, 1]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w:
                        row.append(int(gi[nr, nc]))
                    else:
                        row.append(-1)
                ctx.append(tuple(row))
            ctx = tuple(ctx)
            if ctx in rule_map and rule_map[ctx] != int(gi[r, c]):
                result[r, c] = rule_map[ctx]
                changed = True
    
    if not changed: return None
    
    ok = True
    for inp, out in train_pairs:
        ga = np.array(inp)
        pred = ga.copy()
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                ctx = []
                for dr in [-1, 0, 1]:
                    row = []
                    for dc in [-1, 0, 1]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<ga.shape[0] and 0<=nc<ga.shape[1]:
                            row.append(int(ga[nr, nc]))
                        else:
                            row.append(-1)
                    ctx.append(tuple(row))
                ctx = tuple(ctx)
                if ctx in rule_map:
                    pred[r, c] = rule_map[ctx]
        if not grid_eq(pred.tolist(), out):
            ok = False; break
    
    return result.tolist() if ok else None


# ══════════════════════════════════════════════════════════════
# 2. 制約充足: オブジェクト間の関係推論
# ══════════════════════════════════════════════════════════════

def relational_reasoning(train_pairs, test_input):
    """オブジェクト間の空間関係→操作ルールを推論"""
    from arc.grid import grid_eq
    
    # 各trainでオブジェクトペアの関係を分析
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        bg = _bg(ga)
        
        objs_in = _objs(ga, bg)
        objs_out = _objs(go, bg)
        
        if len(objs_in) < 2: return None
        
        # TODO: 高度な関係推論
        break
    
    return None


# ══════════════════════════════════════════════════════════════
# 3. 周期・対称性の高度検出（群論）
# ══════════════════════════════════════════════════════════════

def symmetry_group_complete(train_pairs, test_input):
    """対称群の検出と補完（回転+反転の全組み合わせ）"""
    from arc.grid import grid_eq
    
    def detect_symmetries(g, bg):
        """グリッドが持つ対称性を検出"""
        h, w = g.shape
        syms = set()
        
        # 水平反転
        if h == w or True:
            if np.array_equal(g, g[:, ::-1]): syms.add('lr')
        # 垂直反転
        if np.array_equal(g, g[::-1, :]): syms.add('ud')
        # 180度回転
        if np.array_equal(g, g[::-1, ::-1]): syms.add('rot180')
        # 主対角線
        if h == w and np.array_equal(g, g.T): syms.add('diag')
        # 反対角線
        if h == w and np.array_equal(g, g[::-1, ::-1].T): syms.add('anti_diag')
        # 90度回転
        if h == w and np.array_equal(g, np.rot90(g, 1)): syms.add('rot90')
        
        return syms
    
    def apply_symmetry(g, bg, target_sym):
        """対称性を使って補完"""
        h, w = g.shape
        result = g.copy()
        changed = False
        
        transforms = []
        if 'lr' in target_sym:
            transforms.append(lambda r,c: (r, w-1-c))
        if 'ud' in target_sym:
            transforms.append(lambda r,c: (h-1-r, c))
        if 'rot180' in target_sym:
            transforms.append(lambda r,c: (h-1-r, w-1-c))
        if 'diag' in target_sym:
            transforms.append(lambda r,c: (c, r))
        if 'rot90' in target_sym:
            transforms.append(lambda r,c: (c, h-1-r))
            transforms.append(lambda r,c: (h-1-r, w-1-c))
            transforms.append(lambda r,c: (w-1-c, r))
        
        for r in range(h):
            for c in range(w):
                if result[r,c] != bg: continue
                for tf in transforms:
                    try:
                        rr, cc = tf(r, c)
                        if 0<=rr<h and 0<=cc<w and result[rr,cc] != bg:
                            result[r,c] = result[rr,cc]
                            changed = True
                            break
                    except: pass
        
        return result if changed else None
    
    # trainから目標対称性を推定
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        bg = _bg(ga)
        
        in_syms = detect_symmetries(ga, bg)
        out_syms = detect_symmetries(go, bg)
        
        # 出力が持つべき対称性
        target = out_syms - in_syms
        if not target and out_syms:
            target = out_syms
        
        if target:
            ok = True
            for inp2, out2 in train_pairs:
                p = apply_symmetry(np.array(inp2), _bg(np.array(inp2)), target)
                if p is None or not grid_eq(p.tolist(), out2):
                    ok = False; break
            if ok:
                gi = np.array(test_input)
                p = apply_symmetry(gi, _bg(gi), target)
                return p.tolist() if p is not None else None
        break
    
    return None


# ══════════════════════════════════════════════════════════════
# 4. モルフォロジー演算（膨張・収縮・エッジ）
# ══════════════════════════════════════════════════════════════

def morphology_ops(train_pairs, test_input):
    """膨張/収縮/エッジ検出"""
    from arc.grid import grid_eq
    
    def dilate(g, bg, steps=1):
        result = g.copy(); h, w = g.shape
        for _ in range(steps):
            new = result.copy()
            for r in range(h):
                for c in range(w):
                    if result[r,c] != bg: continue
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and result[nr,nc] != bg:
                            new[r,c] = result[nr,nc]; break
            result = new
        return result if not np.array_equal(result, g) else None
    
    def erode(g, bg):
        result = g.copy(); h, w = g.shape; ch = False
        for r in range(h):
            for c in range(w):
                if g[r,c] == bg: continue
                edge = False
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if not (0<=nr<h and 0<=nc<w) or g[nr,nc] == bg:
                        edge = True; break
                if edge: result[r,c] = bg; ch = True
        return result if ch else None
    
    def edge_detect(g, bg):
        result = np.full_like(g, bg); h, w = g.shape; ch = False
        for r in range(h):
            for c in range(w):
                if g[r,c] == bg: continue
                edge = False
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if not (0<=nr<h and 0<=nc<w) or g[nr,nc] == bg:
                        edge = True; break
                if edge: result[r,c] = g[r,c]; ch = True
        return result if ch else None
    
    def interior(g, bg):
        result = np.full_like(g, bg); h, w = g.shape; ch = False
        for r in range(h):
            for c in range(w):
                if g[r,c] == bg: continue
                is_interior = True
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if not (0<=nr<h and 0<=nc<w) or g[nr,nc] == bg:
                        is_interior = False; break
                if is_interior: result[r,c] = g[r,c]; ch = True
        return result if ch else None
    
    for op_name, op_fn in [('dilate1', lambda g,bg: dilate(g,bg,1)),
                            ('dilate2', lambda g,bg: dilate(g,bg,2)),
                            ('erode', erode),
                            ('edge', edge_detect),
                            ('interior', interior)]:
        ok = True
        for inp, out in train_pairs:
            ga = np.array(inp); bg = _bg(ga)
            p = op_fn(ga, bg)
            if p is None or not grid_eq(p.tolist(), out):
                ok = False; break
        if ok:
            gi = np.array(test_input); bg = _bg(gi)
            p = op_fn(gi, bg)
            return p.tolist() if p is not None else None
    
    return None


# ══════════════════════════════════════════════════════════════
# 5. パターン分割・セパレータ検出
# ══════════════════════════════════════════════════════════════

def separator_split(train_pairs, test_input):
    """セパレータ線でグリッドを分割→パネルごとの操作"""
    from arc.grid import grid_eq
    
    gi = np.array(test_input)
    h, w = gi.shape
    bg = _bg(gi)
    
    # 水平セパレータ検出（全セル同色の行）
    def find_h_seps(g):
        seps = []
        for r in range(g.shape[0]):
            vals = set(int(v) for v in g[r])
            if len(vals) == 1 and vals.pop() != _bg(g):
                seps.append(r)
        return seps
    
    # 垂直セパレータ検出
    def find_v_seps(g):
        seps = []
        for c in range(g.shape[1]):
            vals = set(int(v) for v in g[:, c])
            if len(vals) == 1 and vals.pop() != _bg(g):
                seps.append(c)
        return seps
    
    h_seps = find_h_seps(gi)
    v_seps = find_v_seps(gi)
    
    if not h_seps and not v_seps:
        return None
    
    # セパレータでパネルに分割
    def split_panels(g, h_seps, v_seps):
        rows = [-1] + h_seps + [g.shape[0]]
        cols = [-1] + v_seps + [g.shape[1]]
        panels = []
        for i in range(len(rows)-1):
            for j in range(len(cols)-1):
                r1, r2 = rows[i]+1, rows[i+1]
                c1, c2 = cols[j]+1, cols[j+1]
                if r2 > r1 and c2 > c1:
                    panels.append(g[r1:r2, c1:c2])
        return panels
    
    # TODO: パネル間の関係推論
    return None


# ══════════════════════════════════════════════════════════════
# 6. オブジェクト変換学習（形状→形状のマッピング）
# ══════════════════════════════════════════════════════════════

def object_transform_learn(train_pairs, test_input):
    """オブジェクトの入力形状→出力形状のルールを学習"""
    from arc.grid import grid_eq
    
    # trainで各オブジェクトがどう変換されたか学習
    transform_rules = []
    
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        bg = _bg(ga)
        
        objs_in = _objs(ga, bg)
        objs_out = _objs(go, bg)
        
        for oi in objs_in:
            # 同じ位置のoutputオブジェクトを探す
            best = None
            for oo in objs_out:
                # bbox重複チェック
                r1 = max(oi['bbox'][0], oo['bbox'][0])
                c1 = max(oi['bbox'][1], oo['bbox'][1])
                r2 = min(oi['bbox'][2], oo['bbox'][2])
                c2 = min(oi['bbox'][3], oo['bbox'][3])
                if r1 <= r2 and c1 <= c2:
                    overlap = (r2-r1+1) * (c2-c1+1)
                    if best is None or overlap > best[0]:
                        best = (overlap, oo)
            
            if best:
                _, oo = best
                # 変換ルール: 色変化、サイズ変化、形状変化
                rule = {
                    'in_color': oi['color'],
                    'out_color': oo['color'],
                    'in_size': oi['size'],
                    'out_size': oo['size'],
                    'color_changed': oi['color'] != oo['color'],
                    'size_ratio': oo['size'] / max(oi['size'], 1),
                }
                transform_rules.append(rule)
    
    if not transform_rules: return None
    
    # 一貫したルールがあるか
    color_map = {}
    consistent = True
    for rule in transform_rules:
        if rule['color_changed']:
            ic, oc = rule['in_color'], rule['out_color']
            if ic in color_map and color_map[ic] != oc:
                consistent = False; break
            color_map[ic] = oc
    
    if not consistent or not color_map: return None
    
    # 色変換を適用
    gi = np.array(test_input)
    result = gi.copy()
    changed = False
    for old, new in color_map.items():
        mask = (gi == old)
        if mask.any():
            result[mask] = new
            changed = True
    
    if not changed: return None
    
    ok = True
    for inp2, out2 in train_pairs:
        ga2 = np.array(inp2)
        pred = ga2.copy()
        for o, n in color_map.items():
            pred[ga2 == o] = n
        if not grid_eq(pred.tolist(), out2):
            ok = False; break
    
    return result.tolist() if ok else None


# ══════════════════════════════════════════════════════════════
# 7. 近傍ルール学習（セルオートマトン汎用）
# ══════════════════════════════════════════════════════════════

def neighbor_rule_learn(train_pairs, test_input):
    """近傍パターン→出力色のルールを自動学習"""
    from arc.grid import grid_eq
    
    # 抽象化: 近傍を(自色, 非BG近傍数, 最頻近傍色)で表現
    rules = {}
    consistent = True
    
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        bg = _bg(ga)
        h, w = ga.shape
        
        for r in range(h):
            for c in range(w):
                self_color = int(ga[r,c])
                nb_colors = []
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w:
                            nb_colors.append(int(ga[nr,nc]))
                
                n_nonbg = sum(1 for v in nb_colors if v != bg)
                nb_nonbg = [v for v in nb_colors if v != bg]
                most_common = Counter(nb_nonbg).most_common(1)[0][0] if nb_nonbg else bg
                
                key = (self_color, n_nonbg, most_common)
                out_color = int(go[r,c])
                
                if key in rules and rules[key] != out_color:
                    consistent = False; break
                rules[key] = out_color
            if not consistent: break
        if not consistent: break
    
    if not consistent or not rules: return None
    
    # apply
    gi = np.array(test_input)
    h, w = gi.shape
    bg = _bg(gi)
    result = gi.copy()
    changed = False
    
    for r in range(h):
        for c in range(w):
            self_color = int(gi[r,c])
            nb_colors = []
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr==0 and dc==0: continue
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w:
                        nb_colors.append(int(gi[nr,nc]))
            
            n_nonbg = sum(1 for v in nb_colors if v != bg)
            nb_nonbg = [v for v in nb_colors if v != bg]
            most_common = Counter(nb_nonbg).most_common(1)[0][0] if nb_nonbg else bg
            
            key = (self_color, n_nonbg, most_common)
            if key in rules and rules[key] != int(gi[r,c]):
                result[r,c] = rules[key]
                changed = True
    
    if not changed: return None
    
    ok = True
    for inp, out in train_pairs:
        ga = np.array(inp); bg2 = _bg(ga)
        pred = ga.copy()
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                sc = int(ga[r,c])
                nbc = []
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        nr, nc = r+dr, c+dc
                        if 0<=nr<ga.shape[0] and 0<=nc<ga.shape[1]:
                            nbc.append(int(ga[nr,nc]))
                nn = sum(1 for v in nbc if v != bg2)
                nbn = [v for v in nbc if v != bg2]
                mc = Counter(nbn).most_common(1)[0][0] if nbn else bg2
                key = (sc, nn, mc)
                if key in rules: pred[r,c] = rules[key]
        if not grid_eq(pred.tolist(), out):
            ok = False; break
    
    return result.tolist() if ok else None


# ══════════════════════════════════════════════════════════════
# 8. 条件付きオブジェクト操作（大学院レベル）
# ══════════════════════════════════════════════════════════════

def conditional_object_op(train_pairs, test_input):
    """オブジェクトの属性(色/サイズ/形)に基づく条件分岐操作"""
    from arc.grid import grid_eq
    
    # 全trainの入出力オブジェクトを分析
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: continue
        bg = _bg(ga)
        
        objs_in = _objs(ga, bg)
        
        # 各オブジェクトが出力で消えた/残った/色変わった
        for obj in objs_in:
            r, c = obj['cells'][0]
            out_color = int(go[r, c])
            if out_color == bg:
                # 消された
                pass
            elif out_color != obj['color']:
                # 色変わった
                pass
        break
    
    return None


# ══════════════════════════════════════════════════════════════
# 9. 汎用パネル比較（差分/共通/XOR）
# ══════════════════════════════════════════════════════════════

def panel_comparison(train_pairs, test_input):
    """グリッドを2+分割して比較→結果を出力"""
    from arc.grid import grid_eq
    
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        oh, ow = go.shape
        ih, iw = gi.shape
        bg = _bg(gi)
        
        # 2分割（水平/垂直）
        for n in [2, 3, 4]:
            for ax in ['h', 'v']:
                if ax == 'h' and ih % n == 0:
                    ph = ih // n
                    if ph != oh or iw != ow: continue
                    panels = [gi[i*ph:(i+1)*ph, :] for i in range(n)]
                elif ax == 'v' and iw % n == 0:
                    pw = iw // n
                    if ih != oh or pw != ow: continue
                    panels = [gi[:, i*pw:(i+1)*pw] for i in range(n)]
                else: continue
                
                # 各合成演算を試す
                for op in ['and', 'or', 'xor', 'diff', 'first_nonbg', 'last_nonbg',
                           'max', 'min', 'majority']:
                    result = _combine_panels(panels, bg, op)
                    if result is not None and grid_eq(result.tolist(), out):
                        # test適用
                        ti = np.array(test_input)
                        if ax == 'h' and ti.shape[0] % n == 0:
                            tph = ti.shape[0] // n
                            tpanels = [ti[i*tph:(i+1)*tph, :] for i in range(n)]
                        elif ax == 'v' and ti.shape[1] % n == 0:
                            tpw = ti.shape[1] // n
                            tpanels = [ti[:, i*tpw:(i+1)*tpw] for i in range(n)]
                        else: continue
                        
                        # 全train検証
                        ok = True
                        for i2, o2 in train_pairs:
                            ga2 = np.array(i2)
                            if ax == 'h' and ga2.shape[0] % n == 0:
                                p2 = [ga2[i*ga2.shape[0]//n:(i+1)*ga2.shape[0]//n, :] for i in range(n)]
                            elif ax == 'v' and ga2.shape[1] % n == 0:
                                p2 = [ga2[:, i*ga2.shape[1]//n:(i+1)*ga2.shape[1]//n] for i in range(n)]
                            else: ok = False; break
                            r2 = _combine_panels(p2, _bg(ga2), op)
                            if r2 is None or not grid_eq(r2.tolist(), o2):
                                ok = False; break
                        
                        if ok:
                            tr = _combine_panels(tpanels, _bg(ti), op)
                            return tr.tolist() if tr is not None else None
        break
    
    return None

def _combine_panels(panels, bg, op):
    ph, pw = panels[0].shape
    result = np.full((ph, pw), bg, dtype=int)
    
    for r in range(ph):
        for c in range(pw):
            vals = [int(p[r,c]) for p in panels]
            nb = [v for v in vals if v != bg]
            
            if op == 'and':
                result[r,c] = nb[0] if len(nb) == len(vals) else bg
            elif op == 'or':
                result[r,c] = nb[0] if nb else bg
            elif op == 'xor':
                result[r,c] = nb[0] if len(nb) == 1 else bg
            elif op == 'diff':
                if len(nb) == 1: result[r,c] = nb[0]
                elif len(set(nb)) > 1: result[r,c] = nb[-1]
            elif op == 'first_nonbg':
                result[r,c] = nb[0] if nb else bg
            elif op == 'last_nonbg':
                result[r,c] = nb[-1] if nb else bg
            elif op == 'max':
                result[r,c] = max(nb) if nb else bg
            elif op == 'min':
                result[r,c] = min(nb) if nb else bg
            elif op == 'majority':
                if nb:
                    result[r,c] = Counter(nb).most_common(1)[0][0]
    
    return result


# ══════════════════════════════════════════════════════════════
# マスター: 断片記憶の判断でフル記憶を呼び出す
# ══════════════════════════════════════════════════════════════

# 断片の勘 → フル記憶の知識マッピング
INTUITION_TO_KNOWLEDGE = {
    # 断片の勘名 → フル記憶の関数リスト
    'リバーシ': [template_match_and_apply, neighbor_rule_learn],
    'リバーシ大会': [template_match_and_apply, neighbor_rule_learn],
    '折り紙': [symmetry_group_complete],
    '鏡': [symmetry_group_complete],
    '穴埋め': [neighbor_rule_learn, morphology_ops],
    '囲み塗り': [morphology_ops],
    '三枚おろし': [panel_comparison],
    '色変え': [object_transform_learn, neighbor_rule_learn],
    '仲間はずれ': [conditional_object_op],
    '大きい方': [],
    '小さい方': [],
    '数える': [],
    'テトリス': [],
    '回す': [],
    'タイル貼り': [],
    'ソリティア': [],
    '虫眼鏡': [],
    '砲台': [],
    '壁押し': [],
    '額縁crop': [],
    '点つなぎ': [],
    '繰り返し': [],
    '偶数奇数': [neighbor_rule_learn],
}

# 全フル記憶関数
ALL_GRADUATE = [
    ('template_match', template_match_and_apply),
    ('symmetry_group', symmetry_group_complete),
    ('morphology', morphology_ops),
    ('neighbor_rule', neighbor_rule_learn),
    ('panel_comparison', panel_comparison),
    ('object_transform', object_transform_learn),
]


def graduate_solve(train_pairs, test_input, intuition_hint=None):
    """
    断片記憶の勘(intuition_hint)に基づいてフル記憶を検索
    hintがなければ全部試す
    """
    from arc.grid import grid_eq
    
    # 優先順位付きリスト
    if intuition_hint and intuition_hint in INTUITION_TO_KNOWLEDGE:
        priority = INTUITION_TO_KNOWLEDGE[intuition_hint]
        others = [fn for _, fn in ALL_GRADUATE if fn not in priority]
        order = priority + others
    else:
        order = [fn for _, fn in ALL_GRADUATE]
    
    for fn in order:
        try:
            result = fn(train_pairs, test_input)
            if result is not None:
                return result, fn.__name__
        except:
            continue
    
    return None, None


def two_layer_solve(train_pairs, test_input):
    """
    二層記憶アーキテクチャ:
    1. 断片記憶(十字くん)で勘を働かせる
    2. フル記憶(大学院)で実行
    3. 断片で直接解けたらそれを使う
    """
    from arc.grid import grid_eq
    from arc.cross_childhood import CrossChildhood
    
    child = CrossChildhood()
    
    # まず断片記憶で解けるか試す
    result, intuition = child.solve(train_pairs, test_input)
    if result is not None:
        return result, f'断片:{intuition}'
    
    # 断片の勘だけ取得（解けなくても勘は使える）
    from arc.cross_childhood import _overlay
    overlay_info = [_overlay(inp, out) for inp, out in train_pairs]
    
    best_intuition = None
    best_score = 0
    for exp in child.experiences:
        try:
            score = exp.recognize(train_pairs, overlay_info) if hasattr(exp, 'recognize') else 0
        except:
            score = 0
        if score > best_score:
            best_score = score
            best_intuition = exp.name
    
    # フル記憶で解く（勘をヒントに）
    result, method = graduate_solve(train_pairs, test_input, best_intuition)
    if result is not None:
        return result, f'フル:{method}(勘:{best_intuition})'
    
    return None, None


if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    from arc.grid import grid_eq
    
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
    for tf in sorted(data_dir.glob('*.json')):
        tid = tf.stem
        with open(tf) as f: task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti, to = task['test'][0]['input'], task['test'][0].get('output')
        
        result, name = two_layer_solve(tp, ti)
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_e else ''
            solved.append((tid, name, tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tg in solved if tg == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
    print('\n手法別:')
    for name, cnt in Counter(n for _,n,_ in solved).most_common():
        print(f'  {name}: {cnt}')
