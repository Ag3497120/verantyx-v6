"""
arc/kofdai_full_memory.py — kofdaiのフル記憶層

断片記憶（ラベル） → フル記憶（具体的アルゴリズム）

=== kofdaiが言った操作パターン ===

1. リバーシ: 同色2点の間を塗りつぶす（水平/垂直/対角）
2. マリオジャンプ: オブジェクトを移動（方向・距離を学習）
3. 対称性復元: ミラー/回転で欠損を補完
4. チェック柄: 市松模様の検出と補完
5. 縞模様: ストライプパターンの検出と補完
6. 100マス計算: 行×列→セル値の関数的パターン
7. 直角・辺の等しさ: L字/矩形の検出
8. 多面体→入れ子: オブジェクト内にパターンをstamp
9. 対角線→中心: 中心点を基準にした変換
10. 交互パターン: 偶数/奇数行列で異なる処理
11. 記号に見える: 全体を俯瞰して形状認識
12. 立体展開: 3D展開図的な配置
13. 柄の類似: テンプレートマッチング
14. 顔に見える: パレイドリア（穴=目）
15. 偶数/奇数パターン: 偶奇ベースの条件分岐
16. 増減パターン: 等差/等比数列的な変化
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label
from arc.grid import grid_eq, grid_shape, most_common_color


def _objects(g, bg):
    mask = (g != bg).astype(int)
    labeled, n = scipy_label(mask)
    objs = []
    for i in range(1, n + 1):
        cells = list(zip(*np.where(labeled == i)))
        colors = [int(g[r, c]) for r, c in cells]
        objs.append({
            'cells': cells, 'size': len(cells),
            'color': Counter(colors).most_common(1)[0][0],
            'colors': set(colors),
            'bbox': (min(r for r,c in cells), min(c for r,c in cells),
                     max(r for r,c in cells), max(c for r,c in cells)),
        })
    return objs


# ══════════════════════════════════════════════════════════════
# 1. リバーシ — 同色2点間の塗りつぶし
# ══════════════════════════════════════════════════════════════
# リバーシのルール: 自分の石で相手の石を挟むと裏返せる
# ARC版: 同色2点の間のbgセルを、その色 or 別の色で塗りつぶす

def reversi_solve(train_pairs, test_input):
    """リバーシ的塗りつぶし: 同色点の間を埋める"""
    # trainから「どの色の間を何色で塗るか」を学習
    rules = []
    
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        if gi.shape != go.shape:
            return None
        bg = int(Counter(gi.flatten()).most_common(1)[0][0])
        h, w = gi.shape
        
        # 追加されたセルを特定
        added = list(zip(*np.where((gi == bg) & (go != bg))))
        if not added:
            continue
        
        # 各追加セルが「どの2点の間にあるか」を特定
        for r, c in added:
            fill_color = int(go[r, c])
            
            # 水平方向: 左右に同色点があるか
            left = right = None
            for cc in range(c-1, -1, -1):
                if gi[r, cc] != bg:
                    left = (r, cc, int(gi[r, cc]))
                    break
            for cc in range(c+1, w):
                if gi[r, cc] != bg:
                    right = (r, cc, int(gi[r, cc]))
                    break
            
            if left and right:
                rules.append({
                    'dir': 'h',
                    'left_color': left[2],
                    'right_color': right[2],
                    'fill_color': fill_color,
                    'same_color': left[2] == right[2],
                })
            
            # 垂直方向
            top = bottom = None
            for rr in range(r-1, -1, -1):
                if gi[rr, c] != bg:
                    top = (rr, c, int(gi[rr, c]))
                    break
            for rr in range(r+1, h):
                if gi[rr, c] != bg:
                    bottom = (rr, c, int(gi[rr, c]))
                    break
            
            if top and bottom:
                rules.append({
                    'dir': 'v',
                    'left_color': top[2],
                    'right_color': bottom[2],
                    'fill_color': fill_color,
                    'same_color': top[2] == bottom[2],
                })
    
    if not rules:
        return None
    
    # ルールの一般化: 最も多いパターン
    # パターン1: 同色2点間を同色で塗る
    same_color_fill = sum(1 for r in rules if r['same_color'] and r['fill_color'] == r['left_color'])
    # パターン2: 同色2点間を別色で塗る
    # パターン3: 異色2点間を何かで塗る
    
    gi = np.array(test_input)
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    h, w = gi.shape
    result = [row[:] for row in test_input]
    changed = False
    
    if same_color_fill > len(rules) * 0.5:
        # パターン1: 同色間を同色で
        # 水平
        for r in range(h):
            fg_pos = [(c, int(gi[r, c])) for c in range(w) if gi[r, c] != bg]
            for i, (c1, col1) in enumerate(fg_pos):
                for c2, col2 in fg_pos[i+1:]:
                    if col1 == col2:
                        for c in range(c1+1, c2):
                            if result[r][c] == bg:
                                result[r][c] = col1
                                changed = True
        # 垂直
        for c in range(w):
            fg_pos = [(r, int(gi[r, c])) for r in range(h) if gi[r, c] != bg]
            for i, (r1, col1) in enumerate(fg_pos):
                for r2, col2 in fg_pos[i+1:]:
                    if col1 == col2:
                        for r in range(r1+1, r2):
                            if result[r][c] == bg:
                                result[r][c] = col1
                                changed = True
    
    return result if changed else None


# ══════════════════════════════════════════════════════════════
# 2. マリオジャンプ — オブジェクト移動
# ══════════════════════════════════════════════════════════════

def mario_solve(train_pairs, test_input):
    """マリオが動く: オブジェクトの平行移動"""
    movements = []
    
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        if gi.shape != go.shape:
            return None
        bg = int(Counter(gi.flatten()).most_common(1)[0][0])
        
        objs_in = _objects(gi, bg)
        objs_out = _objects(go, bg)
        
        # 各入力オブジェクトの出力での位置を探す（色・形で一致）
        for oi in objs_in:
            shape_in = set((r - oi['bbox'][0], c - oi['bbox'][1]) for r, c in oi['cells'])
            
            for oo in objs_out:
                if oi['color'] != oo['color'] or oi['size'] != oo['size']:
                    continue
                shape_out = set((r - oo['bbox'][0], c - oo['bbox'][1]) for r, c in oo['cells'])
                if shape_in == shape_out:
                    dr = oo['bbox'][0] - oi['bbox'][0]
                    dc = oo['bbox'][1] - oi['bbox'][1]
                    movements.append((dr, dc))
                    break
    
    if not movements:
        return None
    
    # 全例で同じ移動？
    if len(set(movements)) != 1:
        return None
    
    dr, dc = movements[0]
    if dr == 0 and dc == 0:
        return None
    
    gi = np.array(test_input)
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    h, w = gi.shape
    result = np.full_like(gi, bg)
    objs = _objects(gi, bg)
    
    for obj in objs:
        for r, c in obj['cells']:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                result[nr, nc] = gi[r, c]
    
    return result.tolist()


# ══════════════════════════════════════════════════════════════
# 3. 対称性復元
# ══════════════════════════════════════════════════════════════

def symmetry_restore_solve(train_pairs, test_input):
    """対称性で欠損を復元（0934a4d8型）"""
    gi = np.array(test_input)
    h, w = gi.shape
    
    # マスク色を検出（全train共通の「塗りつぶし色」）
    mask_color = _find_mask_color(train_pairs, test_input)
    if mask_color is None:
        return None
    
    mask = (gi == mask_color)
    if not mask.any():
        return None
    
    # 8の矩形領域
    p = list(zip(*np.where(mask)))
    r_min, r_max = min(r for r, c in p), max(r for r, c in p)
    c_min, c_max = min(c for r, c in p), max(c for r, c in p)
    
    # 対称軸を発見（行ペア・列ペアの一致から）
    row_axis = _find_symmetry_axis(gi, mask_color, 'row')
    col_axis = _find_symmetry_axis(gi, mask_color, 'col')
    
    if row_axis is None and col_axis is None:
        return None
    
    # 復元
    out_h = r_max - r_min + 1
    out_w = c_max - c_min + 1
    result = np.zeros((out_h, out_w), dtype=int)
    
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            # 優先度: 行+列対称 → 行対称のみ → 列対称のみ
            val = None
            
            if row_axis is not None and col_axis is not None:
                sr = int(2 * row_axis - r + 0.5)
                sc = int(2 * col_axis - c + 0.5)
                if 0 <= sr < h and 0 <= sc < w and gi[sr, sc] != mask_color:
                    val = int(gi[sr, sc])
            
            if val is None and row_axis is not None:
                sr = int(2 * row_axis - r + 0.5)
                if 0 <= sr < h and gi[sr, c] != mask_color:
                    val = int(gi[sr, c])
            
            if val is None and col_axis is not None:
                sc = int(2 * col_axis - c + 0.5)
                if 0 <= sc < w and gi[r, sc] != mask_color:
                    val = int(gi[r, sc])
            
            # 行内対称
            if val is None and col_axis is not None:
                sc = int(2 * col_axis - c + 0.5)
                if 0 <= sc < w and gi[r, sc] != mask_color:
                    val = int(gi[r, sc])
            
            # 列内対称  
            if val is None and row_axis is not None:
                sr = int(2 * row_axis - r + 0.5)
                if 0 <= sr < h and gi[sr, c] != mask_color:
                    val = int(gi[sr, c])
            
            if val is None:
                val = 0  # 最後の手段
            
            result[r - r_min, c - c_min] = val
    
    return result.tolist()


def _find_mask_color(train_pairs, test_input):
    """マスク色（全例に1つだけ存在する特別な色）"""
    gi = np.array(test_input)
    test_colors = set(int(v) for v in gi.flatten())
    
    for color in test_colors:
        # テストでこの色が矩形領域を形成
        mask = (gi == color)
        if not mask.any():
            continue
        p = list(zip(*np.where(mask)))
        r_min, r_max = min(r for r, c in p), max(r for r, c in p)
        c_min, c_max = min(c for r, c in p), max(c for r, c in p)
        expected = (r_max - r_min + 1) * (c_max - c_min + 1)
        if mask.sum() == expected:
            # train全例でもこの色が矩形を形成
            all_rect = True
            for inp, out in train_pairs:
                ga = np.array(inp)
                ma = (ga == color)
                if not ma.any():
                    all_rect = False
                    break
                pa = list(zip(*np.where(ma)))
                ra = (max(r for r,c in pa) - min(r for r,c in pa) + 1)
                ca = (max(c for r,c in pa) - min(c for r,c in pa) + 1)
                if ma.sum() != ra * ca:
                    all_rect = False
                    break
                # 出力サイズ = マスクサイズ
                go = np.array(out)
                if go.shape != (ra, ca):
                    all_rect = False
                    break
            
            if all_rect:
                return color
    
    return None


def _find_symmetry_axis(g, mask_color, direction):
    """行/列ペアの一致から対称軸を発見"""
    h, w = g.shape
    
    if direction == 'row':
        pair_axes = []
        for r1 in range(h):
            for r2 in range(r1 + 1, h):
                m = (g[r1] != mask_color) & (g[r2] != mask_color)
                if m.sum() < max(w // 2, 3):
                    continue
                if (g[r1][m] == g[r2][m]).all():
                    pair_axes.append((r1 + r2) / 2)
        
        if pair_axes:
            return Counter(pair_axes).most_common(1)[0][0]
    
    elif direction == 'col':
        pair_axes = []
        for c1 in range(w):
            for c2 in range(c1 + 1, w):
                m = (g[:, c1] != mask_color) & (g[:, c2] != mask_color)
                if m.sum() < max(h // 2, 3):
                    continue
                if (g[:, c1][m] == g[:, c2][m]).all():
                    pair_axes.append((c1 + c2) / 2)
        
        if pair_axes:
            return Counter(pair_axes).most_common(1)[0][0]
    
    return None


# ══════════════════════════════════════════════════════════════
# 4. チェック柄
# ══════════════════════════════════════════════════════════════

def checker_solve(train_pairs, test_input):
    """チェック柄（市松模様）: (r+c)%2 で色を決定"""
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        if gi.shape != go.shape:
            return None
    
    gi = np.array(test_input)
    h, w = gi.shape
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    # trainから市松パターンを学習
    for inp, out in train_pairs:
        ga = np.array(inp)
        go = np.array(out)
        ha, wa = ga.shape
        
        even_colors = set()
        odd_colors = set()
        for r in range(ha):
            for c in range(wa):
                if go[r, c] != bg:
                    if (r + c) % 2 == 0:
                        even_colors.add(int(go[r, c]))
                    else:
                        odd_colors.add(int(go[r, c]))
        
        if len(even_colors) == 1 and len(odd_colors) == 1:
            ec = even_colors.pop()
            oc = odd_colors.pop()
            if ec != oc:
                # 市松パターン確認
                result = gi.copy()
                for r in range(h):
                    for c in range(w):
                        if gi[r, c] != bg:
                            if (r + c) % 2 == 0:
                                result[r, c] = ec
                            else:
                                result[r, c] = oc
                return result.tolist()
    
    return None


# ══════════════════════════════════════════════════════════════
# 5. 縞模様
# ══════════════════════════════════════════════════════════════

def stripe_solve(train_pairs, test_input):
    """縞模様: 行/列ごとに同じ色"""
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        if gi.shape != go.shape:
            return None
    
    # trainの出力が縞模様か
    for inp, out in train_pairs:
        go = np.array(out)
        h, w = go.shape
        
        # 横縞: 各行が1色
        h_stripe = all(len(set(int(v) for v in go[r])) == 1 for r in range(h))
        if h_stripe:
            row_colors = [int(go[r, 0]) for r in range(h)]
            # 周期を検出
            for period in range(1, h + 1):
                if h % period == 0:
                    if all(row_colors[r] == row_colors[r % period] for r in range(h)):
                        pattern = row_colors[:period]
                        gi = np.array(test_input)
                        ht, wt = gi.shape
                        result = np.zeros((ht, wt), dtype=int)
                        for r in range(ht):
                            result[r, :] = pattern[r % period]
                        return result.tolist()
    
    return None


# ══════════════════════════════════════════════════════════════
# 6. 100マス計算 — 行×列→値の関数
# ══════════════════════════════════════════════════════════════

def grid_function_solve(train_pairs, test_input):
    """行ラベル×列ラベル→セル値の関数的パターン"""
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        if gi.shape != go.shape:
            return None
    
    gi = np.array(test_input)
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    h, w = gi.shape
    
    # 行の「ラベル」= その行の色パターン、列の「ラベル」= その列の色パターン
    # trainで出力セルが「行の何か × 列の何か」の関数か
    
    # シンプル版: 出力色 = 入力の行ラベル色 or 列ラベル色
    # 行0が列ラベル、列0が行ラベルのパターン
    for inp, out in train_pairs:
        ga = np.array(inp)
        go = np.array(out)
        ha, wa = ga.shape
        
        # 1行目・1列目がヘッダか
        if ha > 2 and wa > 2:
            # go[r,c] = f(ga[0,c], ga[r,0]) ?
            # 色の組み合わせ → 出力色のマッピング
            mapping = {}
            ok = True
            for r in range(1, ha):
                for c in range(1, wa):
                    key = (int(ga[0, c]), int(ga[r, 0]))
                    val = int(go[r, c])
                    if key in mapping:
                        if mapping[key] != val:
                            ok = False
                            break
                    else:
                        mapping[key] = val
                if not ok:
                    break
            
            if ok and mapping:
                # testに適用
                result = [row[:] for row in test_input]
                for r in range(1, h):
                    for c in range(1, w):
                        key = (int(gi[0, c]), int(gi[r, 0]))
                        if key in mapping:
                            result[r][c] = mapping[key]
                return result
    
    return None


# ══════════════════════════════════════════════════════════════
# 7. 入れ子 — オブジェクト内にパターンをstamp
# ══════════════════════════════════════════════════════════════

def stamp_inside_solve(train_pairs, test_input):
    """大きいオブジェクトの中に小さいパターンをstamp"""
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        if gi.shape != go.shape:
            return None
    
    gi = np.array(test_input)
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    # trainで「追加されたセル」が既存オブジェクトの内部にあるか
    for inp, out in train_pairs:
        ga = np.array(inp)
        go = np.array(out)
        bg_t = int(Counter(ga.flatten()).most_common(1)[0][0])
        
        added = list(zip(*np.where((ga == bg_t) & (go != bg_t))))
        if not added:
            continue
        
        # 追加セルがオブジェクトのbbox内部か
        objs = _objects(ga, bg_t)
        for obj in objs:
            r1, c1, r2, c2 = obj['bbox']
            inside = [(r, c) for r, c in added if r1 <= r <= r2 and c1 <= c <= c2]
            if len(inside) > 0:
                # このオブジェクトの内部にstampされてる
                # stamp色を学習
                stamp_color = int(go[inside[0][0], inside[0][1]])
                
                # 内部のbgセルを全部stamp
                h, w = gi.shape
                result = [row[:] for row in test_input]
                test_objs = _objects(gi, bg)
                changed = False
                for tobj in test_objs:
                    tr1, tc1, tr2, tc2 = tobj['bbox']
                    for r in range(tr1, tr2 + 1):
                        for c in range(tc1, tc2 + 1):
                            if gi[r, c] == bg:
                                result[r][c] = stamp_color
                                changed = True
                
                if changed:
                    return result
    
    return None


# ══════════════════════════════════════════════════════════════
# 8. 対角線→中心基準変換
# ══════════════════════════════════════════════════════════════

def center_transform_solve(train_pairs, test_input):
    """中心点を基準にした変換（十字、ダイヤモンド等）"""
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        if gi.shape != go.shape:
            return None
    
    gi = np.array(test_input)
    h, w = gi.shape
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    # 中心の色が特別か
    cr, cc = h // 2, w // 2
    if gi[cr, cc] == bg:
        return None
    
    center_color = int(gi[cr, cc])
    
    # trainで中心色のセルから十字/対角に何が起きてるか
    for inp, out in train_pairs:
        ga = np.array(inp)
        go = np.array(out)
        ha, wa = ga.shape
        cra, cca = ha // 2, wa // 2
        
        if ga[cra, cca] == bg:
            continue
        
        # 中心から放射状に追加されたか
        added = list(zip(*np.where((ga == bg) & (go != bg))))
        if not added:
            continue
        
        # 十字パターン（同じ行/列）
        cross = [(r, c) for r, c in added if r == cra or c == cca]
        if len(cross) == len(added):
            # 十字パターン
            fill_color = int(go[added[0][0], added[0][1]])
            result = [row[:] for row in test_input]
            changed = False
            for r in range(h):
                if gi[r, cc] == bg:
                    result[r][cc] = fill_color
                    changed = True
            for c in range(w):
                if gi[cr, c] == bg:
                    result[cr][c] = fill_color
                    changed = True
            if changed:
                return result
    
    return None


# ══════════════════════════════════════════════════════════════
# 9. 偶数/奇数パターン
# ══════════════════════════════════════════════════════════════

def parity_solve(train_pairs, test_input):
    """偶数行/奇数行で異なる処理"""
    for inp, out in train_pairs:
        if np.array(inp).shape != np.array(out).shape:
            return None
    
    # trainで偶数行と奇数行の変化パターンが違うか
    even_rule = {}  # (in_color) → out_color for even rows
    odd_rule = {}   # (in_color) → out_color for odd rows
    
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        h, w = gi.shape
        
        for r in range(h):
            for c in range(w):
                ic, oc = int(gi[r, c]), int(go[r, c])
                if ic == oc:
                    continue
                
                if r % 2 == 0:
                    if ic in even_rule:
                        if even_rule[ic] != oc:
                            return None
                    else:
                        even_rule[ic] = oc
                else:
                    if ic in odd_rule:
                        if odd_rule[ic] != oc:
                            return None
                    else:
                        odd_rule[ic] = oc
    
    if not even_rule and not odd_rule:
        return None
    
    gi = np.array(test_input)
    h, w = gi.shape
    result = [row[:] for row in test_input]
    
    for r in range(h):
        for c in range(w):
            ic = int(gi[r, c])
            rule = even_rule if r % 2 == 0 else odd_rule
            if ic in rule:
                result[r][c] = rule[ic]
    
    return result


# ══════════════════════════════════════════════════════════════
# 10. 囲み塗り（fill_enclosed）
# ══════════════════════════════════════════════════════════════

def fill_enclosed_solve(train_pairs, test_input):
    """囲まれた領域を塗りつぶす"""
    gi = np.array(test_input)
    h, w = gi.shape
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    # fill色をtrainから学習（複数色対応）
    fill_colors = set()
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape:
            return None
        bg_t = int(Counter(ga.flatten()).most_common(1)[0][0])
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                if ga[r, c] == bg_t and go[r, c] != bg_t:
                    fill_colors.add(int(go[r, c]))
    
    if not fill_colors:
        return None
    
    fill_color = fill_colors.pop() if len(fill_colors) == 1 else None
    if fill_color is None:
        return None
    
    # flood fill: 外部bgを除外
    visited = np.zeros((h, w), dtype=bool)
    queue = []
    for r in range(h):
        for c in [0, w-1]:
            if gi[r, c] == bg and not visited[r, c]:
                queue.append((r, c)); visited[r, c] = True
    for c in range(w):
        for r in [0, h-1]:
            if gi[r, c] == bg and not visited[r, c]:
                queue.append((r, c)); visited[r, c] = True
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and gi[nr, nc] == bg:
                visited[nr, nc] = True
                queue.append((nr, nc))
    
    result = [row[:] for row in test_input]
    changed = False
    for r in range(h):
        for c in range(w):
            if gi[r, c] == bg and not visited[r, c]:
                result[r][c] = fill_color
                changed = True
    
    return result if changed else None


# ══════════════════════════════════════════════════════════════
# 11. 色マッピング
# ══════════════════════════════════════════════════════════════

def color_map_solve(train_pairs, test_input):
    """単純な色置換"""
    mapping = {}
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        if gi.shape != go.shape:
            return None
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                ic, oc = int(gi[r, c]), int(go[r, c])
                if ic in mapping:
                    if mapping[ic] != oc:
                        return None
                else:
                    mapping[ic] = oc
    
    if not mapping or all(k == v for k, v in mapping.items()):
        return None
    
    gi = np.array(test_input)
    result = gi.copy()
    for ic, oc in mapping.items():
        result[gi == ic] = oc
    return result.tolist()


# ══════════════════════════════════════════════════════════════
# 12. 対称性補完（欠損塗りではなく出力を対称にする）
# ══════════════════════════════════════════════════════════════

def symmetry_fill_solve(train_pairs, test_input):
    """入力の非対称部分を対称に補完"""
    gi = np.array(test_input)
    h, w = gi.shape
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    # trainの出力が対称か
    for axis in ['h', 'v', 'hv']:
        all_sym = True
        for _, out in train_pairs:
            go = np.array(out)
            if axis in ('h', 'hv'):
                if not np.array_equal(go, go[:, ::-1]):
                    if axis == 'h':
                        all_sym = False
                        break
            if axis in ('v', 'hv'):
                if not np.array_equal(go, go[::-1, :]):
                    if axis == 'v':
                        all_sym = False
                        break
            if axis == 'hv':
                if not (np.array_equal(go, go[:, ::-1]) and np.array_equal(go, go[::-1, :])):
                    all_sym = False
                    break
        
        if all_sym:
            result = gi.copy()
            if axis in ('h', 'hv'):
                for r in range(h):
                    for c in range(w):
                        mc = w - 1 - c
                        if result[r, c] == bg and result[r, mc] != bg:
                            result[r, c] = result[r, mc]
            if axis in ('v', 'hv'):
                for r in range(h):
                    mr = h - 1 - r
                    for c in range(w):
                        if result[r, c] == bg and result[mr, c] != bg:
                            result[r, c] = result[mr, c]
            
            if not np.array_equal(result, gi):
                return result.tolist()
    
    return None


# ══════════════════════════════════════════════════════════════
# 13. 抽象近傍リカラー（NB rule）
# ══════════════════════════════════════════════════════════════

def abstract_nb_solve(train_pairs, test_input):
    """抽象化された近傍(S/B/O) → 出力色"""
    for inp, out in train_pairs:
        if np.array(inp).shape != np.array(out).shape:
            return None
    
    gi = np.array(test_input)
    h, w = gi.shape
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    # 4近傍と8近傍の両方を試す
    for nb_dirs in [
        [(-1,0),(1,0),(0,-1),(0,1)],
        [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    ]:
        rules = {}
        consistent = True
        
        for inp, out in train_pairs:
            ga, go = np.array(inp), np.array(out)
            bg_t = int(Counter(ga.flatten()).most_common(1)[0][0])
            ha, wa = ga.shape
            
            for r in range(ha):
                for c in range(wa):
                    center = int(ga[r, c])
                    nb = []
                    for dr, dc in nb_dirs:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < ha and 0 <= nc < wa:
                            v = int(ga[nr, nc])
                            if v == bg_t: nb.append('B')
                            elif v == center: nb.append('S')
                            else: nb.append('O')
                        else:
                            nb.append('X')
                    
                    key = ('B' if center == bg_t else 'F', tuple(nb))
                    out_c = int(go[r, c])
                    
                    if out_c == center: out_role = 'SAME'
                    elif out_c == bg_t: out_role = 'BG'
                    else: out_role = out_c
                    
                    if key in rules:
                        if rules[key] != out_role:
                            consistent = False
                            break
                    else:
                        rules[key] = out_role
                if not consistent:
                    break
            if not consistent:
                break
        
        if not consistent or not rules:
            continue
        
        # 適用
        result = [row[:] for row in test_input]
        for r in range(h):
            for c in range(w):
                center = int(gi[r, c])
                nb = []
                for dr, dc in nb_dirs:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w:
                        v = int(gi[nr, nc])
                        if v == bg: nb.append('B')
                        elif v == center: nb.append('S')
                        else: nb.append('O')
                    else:
                        nb.append('X')
                
                key = ('B' if center == bg else 'F', tuple(nb))
                if key in rules:
                    role = rules[key]
                    if role == 'SAME': result[r][c] = center
                    elif role == 'BG': result[r][c] = bg
                    else: result[r][c] = role
        
        # train検証
        ok = True
        for inp, out in train_pairs:
            ga = np.array(inp)
            go = np.array(out)
            bg_t = int(Counter(ga.flatten()).most_common(1)[0][0])
            ha, wa = ga.shape
            
            pred = [row[:] for row in inp]
            for r in range(ha):
                for c in range(wa):
                    center = int(ga[r, c])
                    nb = []
                    for dr, dc in nb_dirs:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < ha and 0 <= nc < wa:
                            v = int(ga[nr, nc])
                            if v == bg_t: nb.append('B')
                            elif v == center: nb.append('S')
                            else: nb.append('O')
                        else:
                            nb.append('X')
                    
                    key = ('B' if center == bg_t else 'F', tuple(nb))
                    if key in rules:
                        role = rules[key]
                        if role == 'SAME': pred[r][c] = center
                        elif role == 'BG': pred[r][c] = bg_t
                        else: pred[r][c] = role
            
            if pred != out:
                ok = False
                break
        
        if ok:
            return result
    
    return None


# ══════════════════════════════════════════════════════════════
# マスターソルバー: 断片→フル記憶→試行
# ══════════════════════════════════════════════════════════════

# 全フル記憶ソルバーを優先度順に登録
ALL_SOLVERS = [
    ('symmetry_restore', symmetry_restore_solve),
    ('color_map', color_map_solve),
    ('reversi', reversi_solve),
    ('fill_enclosed', fill_enclosed_solve),
    ('symmetry_fill', symmetry_fill_solve),
    ('abstract_nb', abstract_nb_solve),
    ('mario', mario_solve),
    ('parity', parity_solve),
    ('grid_function', grid_function_solve),
    ('checker', checker_solve),
    ('stripe', stripe_solve),
    ('stamp_inside', stamp_inside_solve),
    ('center_transform', center_transform_solve),
]


def full_memory_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """
    kofdaiの断片→フル記憶パイプライン:
    
    1. 断片記憶（kofdai_brain）で第一印象を得る
    2. 第一印象に合うフル記憶ソルバーを優先的に試行
    3. 解けなかったら全ソルバーをbrute force
    """
    from arc.kofdai_brain import kofdai_first_glance, kofdai_compare
    
    # Step 1: 断片記憶からのルーティング
    impressions = kofdai_first_glance(test_input)
    comparisons = []
    for inp, out in train_pairs:
        comparisons.extend(kofdai_compare(inp, out))
    
    all_frags = impressions + comparisons
    
    # 断片のactionからソルバーへのマッピング
    ACTION_TO_SOLVER = {
        'fill_between_points': 'reversi',
        'connect_or_fill_between': 'reversi',
        'try_move_object': 'mario',
        'move_objects': 'mario',
        'check_symmetry': 'symmetry_fill',
        'fill_pattern': 'fill_enclosed',
        'recolor_rule': 'color_map',
        'apply_checker_rule': 'checker',
        'apply_parity_rule': 'parity',
        'crop_or_extract': 'symmetry_restore',  # 切り取り系
        'stamp_inside': 'stamp_inside',
        'use_center_as_anchor': 'center_transform',
    }
    
    # 優先ソルバーリスト
    priority_solvers = []
    for frag in all_frags:
        solver_name = ACTION_TO_SOLVER.get(frag.action)
        if solver_name and solver_name not in priority_solvers:
            priority_solvers.append(solver_name)
    
    # Step 2: 優先ソルバーを先に試行
    tried = set()
    for name in priority_solvers:
        tried.add(name)
        for sname, solver in ALL_SOLVERS:
            if sname == name:
                try:
                    # train検証
                    ok = True
                    for inp, out in train_pairs:
                        pred = solver(train_pairs, inp)
                        if pred is None or not grid_eq(pred, out):
                            ok = False
                            break
                    
                    if ok:
                        result = solver(train_pairs, test_input)
                        if result is not None:
                            return result
                except Exception:
                    continue
    
    # Step 3: 全ソルバーをbrute force
    for name, solver in ALL_SOLVERS:
        if name in tried:
            continue
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
                    return result
        except Exception:
            continue
    
    return None


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    
    if sys.argv[1] in ('--eval', '--train'):
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
            
            r = full_memory_solve(tp, ti)
            if r and to and grid_eq(r, to):
                solved.append(tid)
                tag = 'NEW' if tid not in existing else ''
                # どのソルバーで解けたか
                for name, solver in ALL_SOLVERS:
                    try:
                        ok = True
                        for inp, out in tp:
                            pred = solver(tp, inp)
                            if pred is None or not grid_eq(pred, out):
                                ok = False
                                break
                        if ok:
                            res = solver(tp, ti)
                            if res and grid_eq(res, to):
                                solver_hits[name] += 1
                                print(f'  ✓ {tid} [{name}] {tag}')
                                break
                    except:
                        continue
        
        total = len(list(data_dir.glob('*.json')))
        new = [t for t in solved if t not in existing]
        print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
        print(f'Solver hits: {dict(solver_hits)}')
