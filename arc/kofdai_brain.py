"""
arc/kofdai_brain.py — kofdaiの思考パターンをコード化した断片記憶

kofdaiの脳がARC問題を見たときに発火する連想パターン:

=== 数字・色の感覚 ===
- 奇数/偶数を即座に認識
- 何ずつ増減しているか
- 横・縦から切ってパターンを見る
- 100マス計算的な格子思考

=== 図形の感覚 ===
- 直角はどこか
- 辺の等しさ（イコール）
- 多面体 → 別の図形が入りそう
- 対角線 → 中心
- 辺の本数

=== パズル解法の思考 ===
- 交互パターンを頭の中で塗ってみる
- リバーシ連想: 2点間を塗りつぶせる
- 俯瞰して記号に見えないか
- チェック柄 or 縞模様
- 立体展開の可能性
- 柄の類似性

=== ARC固有 ===
- ドット → マリオ → 「跳ねる」「動かす」
- 穴2つ → 顔に見える（パレイドリア）
- 「ここに動かしたらこの形できそう」
"""

import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from collections import Counter, defaultdict
from dataclasses import dataclass
from scipy.ndimage import label as scipy_label


@dataclass(frozen=True)
class KofdaiFragment:
    """kofdaiの脳が発火させる断片"""
    trigger: str    # 何が引き金か
    thought: str    # 何を連想するか
    action: str     # 何をすべきか


# ──────────────────────────────────────────────────────────────
# kofdaiの目: グリッドを見たときの第一印象
# ──────────────────────────────────────────────────────────────

def kofdai_first_glance(grid) -> List[KofdaiFragment]:
    """kofdaiがグリッドをパッと見たときの断片"""
    g = np.array(grid)
    h, w = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    frags = []
    
    # ─── 「正方形か？辺が等しいか？」 ───
    if h == w:
        frags.append(KofdaiFragment('square_grid', '正方形だ', 'check_symmetry'))
    
    # ─── 「チェック柄か縞模様か」 ───
    checker = _detect_checker(g, bg)
    if checker:
        frags.append(KofdaiFragment('checker', 'チェック柄だ', 'apply_checker_rule'))
    
    stripe_dir = _detect_stripe(g, bg)
    if stripe_dir:
        frags.append(KofdaiFragment(f'stripe_{stripe_dir}', f'{stripe_dir}の縞模様', 'apply_stripe_rule'))
    
    # ─── 「交互に色が塗られているか」 ───
    alternating = _detect_alternating(g, bg)
    if alternating:
        frags.append(KofdaiFragment('alternating', '交互パターン、100マス計算的', 'fill_alternating'))
    
    # ─── 「穴が2つ → 顔に見える」（パレイドリア）───
    holes = _count_holes(g, bg)
    if holes == 2:
        frags.append(KofdaiFragment('face', '顔に見える！穴が目だ', 'mark_face_features'))
    
    # ─── 「対角線を引いて中心はどこか」 ───
    center_r, center_c = h // 2, w // 2
    center_color = int(g[center_r, center_c])
    if center_color != bg:
        frags.append(KofdaiFragment('center_marked', '中心に何かある', 'use_center_as_anchor'))
    
    # ─── 「直角はどこか」 ───
    corners = _detect_right_angles(g, bg)
    if corners:
        frags.append(KofdaiFragment('right_angles', f'直角が{len(corners)}個', 'use_corners'))
    
    # ─── 「マリオっぽい」（ドットキャラクター）───
    objs = _get_objects(g, bg)
    for obj in objs:
        if 4 <= obj['size'] <= 20:
            frags.append(KofdaiFragment('mario', 'マリオっぽいキャラがいる', 'try_move_object'))
            break
    
    # ─── 「何本辺があるか」 ───
    n_edges = _count_color_boundaries(g)
    if n_edges < 10:
        frags.append(KofdaiFragment('few_edges', 'シンプルな形', 'simple_transform'))
    elif n_edges > 50:
        frags.append(KofdaiFragment('many_edges', '複雑な模様', 'pattern_based'))
    
    return frags


def kofdai_compare(inp, out) -> List[KofdaiFragment]:
    """kofdaiが入力と出力を見比べたときの断片"""
    gi = np.array(inp)
    go = np.array(out)
    frags = []
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    if gi.shape != go.shape:
        if go.size > gi.size:
            frags.append(KofdaiFragment('grew', '大きくなった', 'tile_or_expand'))
        else:
            frags.append(KofdaiFragment('shrank', '小さくなった、切り取り', 'crop_or_extract'))
        return frags
    
    diff = gi != go
    
    # ─── 「リバーシだ！2点間を塗りつぶせる」 ───
    reversi = _detect_reversi_fill(gi, go, bg)
    if reversi:
        frags.append(KofdaiFragment('reversi', 'リバーシ！2点間を塗りつぶし', 'fill_between_points'))
    
    # ─── 「交互に塗ってみる」 ───
    added_cells = list(zip(*np.where((gi == bg) & (go != bg))))
    removed_cells = list(zip(*np.where((gi != bg) & (go == bg))))
    recolored_cells = list(zip(*np.where((gi != bg) & (go != bg) & diff)))
    
    if added_cells and not removed_cells and not recolored_cells:
        frags.append(KofdaiFragment('only_added', '足しただけ', 'fill_pattern'))
        
        # 追加セルが既存セルの間か
        if _added_cells_between_existing(gi, go, bg, added_cells):
            frags.append(KofdaiFragment('fill_between', '間を埋めた！リバーシ的', 'connect_or_fill_between'))
    
    if not added_cells and not removed_cells and recolored_cells:
        frags.append(KofdaiFragment('only_recolored', '色を変えただけ', 'recolor_rule'))
    
    if removed_cells and not added_cells:
        frags.append(KofdaiFragment('only_removed', '消しただけ', 'filter_or_mask'))
    
    # ─── 「ここに動かしたらこの形できそう」 ───
    if _detect_object_movement(gi, go, bg):
        frags.append(KofdaiFragment('moved', 'マリオが動いた！', 'move_objects'))
    
    # ─── 「偶数/奇数の並び」 ───
    if _detect_parity_pattern(gi, go, bg):
        frags.append(KofdaiFragment('parity', '偶数奇数のパターン', 'apply_parity_rule'))
    
    # ─── 「何ずつ大きくなっている」 ───
    progression = _detect_progression(gi, go, bg)
    if progression:
        frags.append(KofdaiFragment('progression', f'{progression}ずつ変化', 'apply_progression'))
    
    # ─── 「多面体 → 別の図形が入りそう」 ───
    if _detect_containment(gi, go, bg):
        frags.append(KofdaiFragment('containment', '中に入れた！', 'stamp_inside'))
    
    return frags


# ──────────────────────────────────────────────────────────────
# 検出関数群
# ──────────────────────────────────────────────────────────────

def _get_objects(g, bg):
    mask = (g != bg).astype(int)
    labeled, n = scipy_label(mask)
    objs = []
    for i in range(1, n + 1):
        cells = list(zip(*np.where(labeled == i)))
        colors = [int(g[r, c]) for r, c in cells]
        r_min = min(r for r, c in cells)
        r_max = max(r for r, c in cells)
        c_min = min(c for r, c in cells)
        c_max = max(c for r, c in cells)
        objs.append({
            'cells': cells, 'size': len(cells),
            'color': Counter(colors).most_common(1)[0][0],
            'colors': set(colors),
            'bbox': (r_min, c_min, r_max, c_max),
            'center': ((r_min + r_max) / 2, (c_min + c_max) / 2),
        })
    return objs


def _detect_checker(g, bg):
    """チェック柄検出"""
    h, w = g.shape
    if h < 2 or w < 2:
        return False
    
    # 2色以上で市松模様
    match = 0
    total = 0
    for r in range(h - 1):
        for c in range(w - 1):
            total += 1
            if g[r, c] != g[r, c+1] and g[r, c] != g[r+1, c]:
                match += 1
    
    return total > 0 and match / total > 0.7


def _detect_stripe(g, bg):
    """縞模様検出"""
    h, w = g.shape
    
    # 横縞
    h_stripe = True
    for r in range(h):
        if len(set(int(v) for v in g[r])) > 1:
            h_stripe = False
            break
    if h_stripe and h >= 2:
        return 'horizontal'
    
    # 縦縞
    v_stripe = True
    for c in range(w):
        if len(set(int(v) for v in g[:, c])) > 1:
            v_stripe = False
            break
    if v_stripe and w >= 2:
        return 'vertical'
    
    return None


def _detect_alternating(g, bg):
    """交互パターン（行ごと or 列ごと）"""
    h, w = g.shape
    if h < 3:
        return False
    
    # 行が交互か
    for period in [2, 3]:
        if h >= period * 2:
            ok = True
            for r in range(period, h):
                if not np.array_equal(g[r], g[r % period]):
                    ok = False
                    break
            if ok:
                return True
    
    return False


def _count_holes(g, bg):
    """背景の「穴」の数（オブジェクト内部の背景領域）"""
    h, w = g.shape
    
    # 外部bg: 境界から到達可能
    visited = np.zeros((h, w), dtype=bool)
    queue = []
    for r in range(h):
        for c in [0, w-1]:
            if g[r, c] == bg and not visited[r, c]:
                queue.append((r, c)); visited[r, c] = True
    for c in range(w):
        for r in [0, h-1]:
            if g[r, c] == bg and not visited[r, c]:
                queue.append((r, c)); visited[r, c] = True
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] == bg:
                visited[nr, nc] = True
                queue.append((nr, nc))
    
    # 内部bg領域をカウント
    inner_bg = (g == bg) & (~visited)
    if not inner_bg.any():
        return 0
    
    labeled, n = scipy_label(inner_bg.astype(int))
    return n


def _detect_right_angles(g, bg):
    """直角（L字型コーナー）の検出"""
    h, w = g.shape
    corners = []
    
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if g[r, c] != bg:
                # L字: 上と右がfg、右上がbg（など4パターン）
                patterns = [
                    (g[r-1,c] != bg and g[r,c+1] != bg and g[r-1,c+1] == bg),
                    (g[r-1,c] != bg and g[r,c-1] != bg and g[r-1,c-1] == bg),
                    (g[r+1,c] != bg and g[r,c+1] != bg and g[r+1,c+1] == bg),
                    (g[r+1,c] != bg and g[r,c-1] != bg and g[r+1,c-1] == bg),
                ]
                if any(patterns):
                    corners.append((r, c))
    
    return corners


def _count_color_boundaries(g):
    """色の境界線の数"""
    h, w = g.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if c + 1 < w and g[r, c] != g[r, c+1]:
                count += 1
            if r + 1 < h and g[r, c] != g[r+1, c]:
                count += 1
    return count


def _detect_reversi_fill(gi, go, bg):
    """リバーシ的パターン: 2つの同色点の間が塗りつぶされた"""
    h, w = gi.shape
    added = list(zip(*np.where((gi == bg) & (go != bg))))
    if not added:
        return False
    
    # 追加セルが直線上にあるか
    if len(added) < 2:
        return False
    
    # 同じ行に追加された場合
    rows = Counter(r for r, c in added)
    for r, count in rows.items():
        if count >= 2:
            cols = sorted(c for rr, c in added if rr == r)
            if cols[-1] - cols[0] == len(cols) - 1:  # 連続
                # 両端に同色fgがあるか
                c_min, c_max = cols[0], cols[-1]
                if c_min > 0 and c_max < w - 1:
                    if gi[r, c_min - 1] != bg and gi[r, c_max + 1] != bg:
                        return True
    
    return False


def _added_cells_between_existing(gi, go, bg, added_cells):
    """追加セルが既存セルの「間」にあるか"""
    for r, c in added_cells:
        # 左右に非bgがあるか
        left_fg = any(gi[r, cc] != bg for cc in range(c) if gi[r, cc] != bg)
        right_fg = any(gi[r, cc] != bg for cc in range(c+1, gi.shape[1]) if gi[r, cc] != bg)
        if left_fg and right_fg:
            return True
        
        # 上下に非bgがあるか
        top_fg = any(gi[rr, c] != bg for rr in range(r) if gi[rr, c] != bg)
        bot_fg = any(gi[rr, c] != bg for rr in range(r+1, gi.shape[0]) if gi[rr, c] != bg)
        if top_fg and bot_fg:
            return True
    
    return False


def _detect_object_movement(gi, go, bg):
    """オブジェクトが移動したか（マリオが跳ねた）"""
    objs_in = _get_objects(gi, bg)
    objs_out = _get_objects(go, bg)
    
    if not objs_in or not objs_out:
        return False
    
    # 同じ色・サイズのオブジェクトが位置だけ変わってるか
    for oi in objs_in:
        for oo in objs_out:
            if (oi['color'] == oo['color'] and 
                oi['size'] == oo['size'] and 
                oi['center'] != oo['center']):
                return True
    
    return False


def _detect_parity_pattern(gi, go, bg):
    """偶数/奇数行・列で異なる変換"""
    h, w = gi.shape
    if gi.shape != go.shape or h < 4:
        return False
    
    even_changes = 0
    odd_changes = 0
    for r in range(h):
        row_diff = sum(1 for c in range(w) if gi[r, c] != go[r, c])
        if r % 2 == 0:
            even_changes += row_diff
        else:
            odd_changes += row_diff
    
    total = even_changes + odd_changes
    if total > 0:
        ratio = min(even_changes, odd_changes) / total
        if ratio < 0.1:  # ほぼ片方だけ変化
            return True
    
    return False


def _detect_progression(gi, go, bg):
    """増減パターン"""
    objs_in = _get_objects(gi, bg)
    objs_out = _get_objects(go, bg)
    
    if len(objs_in) >= 2:
        sizes = sorted([o['size'] for o in objs_in])
        if len(sizes) >= 3:
            diffs = [sizes[i+1] - sizes[i] for i in range(len(sizes)-1)]
            if len(set(diffs)) == 1 and diffs[0] != 0:
                return diffs[0]
    
    return None


def _detect_containment(gi, go, bg):
    """入れ子構造: 出力でオブジェクトが別のオブジェクトの中に"""
    objs_out = _get_objects(go, bg)
    
    for i, o1 in enumerate(objs_out):
        for j, o2 in enumerate(objs_out):
            if i == j:
                continue
            r1, c1, r2, c2 = o1['bbox']
            r3, c3, r4, c4 = o2['bbox']
            # o2がo1の中に完全に含まれるか
            if r3 > r1 and r4 < r2 and c3 > c1 and c4 < c2:
                return True
    
    return False


# ──────────────────────────────────────────────────────────────
# kofdai脳ベースのソルバー
# ──────────────────────────────────────────────────────────────

def kofdai_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """kofdaiの思考プロセスを再現してARC問題を解く"""
    
    # Step 1: 第一印象（全体を見る）
    input_impressions = kofdai_first_glance(test_input)
    
    # Step 2: train例を見比べる
    comparison_frags = []
    for inp, out in train_pairs:
        comparison_frags.extend(kofdai_compare(inp, out))
    
    # Step 3: 断片から操作を選択して実行
    all_frags = input_impressions + comparison_frags
    
    # 断片のaction集計（kofdaiが「これやってみよう」と思う順）
    action_votes = Counter()
    for f in all_frags:
        action_votes[f.action] += 1
    
    # 優先度順に試行
    for action, _ in action_votes.most_common():
        result = _execute_kofdai_action(action, train_pairs, test_input)
        if result is not None:
            # train検証
            ok = True
            for inp, out in train_pairs:
                pred = _execute_kofdai_action(action, train_pairs, inp)
                if pred is None or pred != out:
                    ok = False
                    break
            if ok:
                return result
    
    return None


def _execute_kofdai_action(action, train_pairs, grid):
    """kofdaiの思考アクションを実行"""
    g = np.array(grid)
    h, w = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    
    if action == 'fill_between_points' or action == 'connect_or_fill_between':
        return _action_fill_between(grid, bg)
    
    elif action == 'fill_pattern':
        return _action_fill_pattern(train_pairs, grid, bg)
    
    elif action == 'check_symmetry':
        return _action_symmetry_fill(grid, bg)
    
    elif action == 'recolor_rule':
        return _action_recolor(train_pairs, grid, bg)
    
    elif action == 'move_objects' or action == 'try_move_object':
        return _action_move_objects(train_pairs, grid, bg)
    
    elif action == 'apply_checker_rule':
        return _action_checker_fill(train_pairs, grid, bg)
    
    elif action == 'crop_or_extract':
        return _action_extract(train_pairs, grid, bg)
    
    elif action == 'stamp_inside':
        return _action_stamp_inside(train_pairs, grid, bg)
    
    elif action == 'fill_alternating':
        return _action_fill_alternating(train_pairs, grid, bg)
    
    elif action == 'filter_or_mask':
        return _action_filter(train_pairs, grid, bg)
    
    elif action == 'apply_parity_rule':
        return _action_parity(train_pairs, grid, bg)
    
    return None


# ──── アクション実装 ────

def _action_fill_between(grid, bg):
    """リバーシ的: 同色点の間を塗りつぶす"""
    g = np.array(grid)
    h, w = g.shape
    result = g.copy()
    changed = False
    
    # 各行で同色点の間を塗る
    for r in range(h):
        fg_positions = [(c, int(g[r, c])) for c in range(w) if g[r, c] != bg]
        for i, (c1, color1) in enumerate(fg_positions):
            for c2, color2 in fg_positions[i+1:]:
                if color1 == color2:
                    for c in range(c1 + 1, c2):
                        if result[r, c] == bg:
                            result[r, c] = color1
                            changed = True
    
    # 各列でも
    for c in range(w):
        fg_positions = [(r, int(g[r, c])) for r in range(h) if g[r, c] != bg]
        for i, (r1, color1) in enumerate(fg_positions):
            for r2, color2 in fg_positions[i+1:]:
                if color1 == color2:
                    for r in range(r1 + 1, r2):
                        if result[r, c] == bg:
                            result[r, c] = color1
                            changed = True
    
    return result.tolist() if changed else None


def _action_fill_pattern(train_pairs, grid, bg):
    """パターン塗りつぶし: train例から塗りパターンを学習"""
    # 追加されたセルの位置パターンを学習
    g = np.array(grid)
    h, w = g.shape
    
    # train例で「どこを塗ったか」のルール
    # 囲まれた領域を塗る
    visited = np.zeros((h, w), dtype=bool)
    queue = []
    for r in range(h):
        for c in [0, w-1]:
            if g[r, c] == bg and not visited[r, c]:
                queue.append((r, c)); visited[r, c] = True
    for c in range(w):
        for r in [0, h-1]:
            if g[r, c] == bg and not visited[r, c]:
                queue.append((r, c)); visited[r, c] = True
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] == bg:
                visited[nr, nc] = True
                queue.append((nr, nc))
    
    # fill色をtrainから
    fill_color = None
    for inp, out in train_pairs:
        gi = np.array(inp)
        go = np.array(out)
        if gi.shape != go.shape:
            continue
        bg_t = int(Counter(gi.flatten()).most_common(1)[0][0])
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                if gi[r, c] == bg_t and go[r, c] != bg_t:
                    fill_color = int(go[r, c])
                    break
            if fill_color is not None:
                break
    
    if fill_color is None:
        return None
    
    result = [row[:] for row in grid]
    changed = False
    for r in range(h):
        for c in range(w):
            if g[r, c] == bg and not visited[r, c]:
                result[r][c] = fill_color
                changed = True
    
    return result if changed else None


def _action_symmetry_fill(grid, bg):
    """対称性補完"""
    g = np.array(grid)
    h, w = g.shape
    result = g.copy()
    changed = False
    
    # 4方向対称を試す
    for r in range(h):
        for c in range(w):
            if result[r, c] == bg:
                # 水平ミラー
                mc = w - 1 - c
                if 0 <= mc < w and result[r, mc] != bg:
                    result[r, c] = result[r, mc]
                    changed = True
                    continue
                # 垂直ミラー
                mr = h - 1 - r
                if 0 <= mr < h and result[mr, c] != bg:
                    result[r, c] = result[mr, c]
                    changed = True
    
    return result.tolist() if changed else None


def _action_recolor(train_pairs, grid, bg):
    """色変えルール"""
    # 入力色→出力色のマッピング
    color_map = {}
    for inp, out in train_pairs:
        gi = np.array(inp)
        go = np.array(out)
        if gi.shape != go.shape:
            return None
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                ic, oc = int(gi[r, c]), int(go[r, c])
                if ic != oc:
                    if ic in color_map:
                        if color_map[ic] != oc:
                            return None  # 矛盾
                    else:
                        color_map[ic] = oc
    
    if not color_map:
        return None
    
    g = np.array(grid)
    result = g.copy()
    for old_c, new_c in color_map.items():
        result[g == old_c] = new_c
    
    return result.tolist()


def _action_move_objects(train_pairs, grid, bg):
    """オブジェクト移動（マリオジャンプ）"""
    # trainから移動ベクトルを学習
    movements = []
    
    for inp, out in train_pairs:
        gi = np.array(inp)
        go = np.array(out)
        if gi.shape != go.shape:
            return None
        bg_t = int(Counter(gi.flatten()).most_common(1)[0][0])
        
        objs_in = _get_objects(gi, bg_t)
        objs_out = _get_objects(go, bg_t)
        
        for oi in objs_in:
            for oo in objs_out:
                if oi['color'] == oo['color'] and oi['size'] == oo['size']:
                    dr = oo['center'][0] - oi['center'][0]
                    dc = oo['center'][1] - oi['center'][1]
                    if abs(dr) > 0.5 or abs(dc) > 0.5:
                        movements.append((dr, dc, oi['color']))
    
    if not movements:
        return None
    
    # 最も一般的な移動を適用
    # 全オブジェクトに同じ移動を適用
    g = np.array(grid)
    result = np.full_like(g, bg)
    objs = _get_objects(g, bg)
    
    dr, dc = movements[0][0], movements[0][1]
    dr, dc = int(round(dr)), int(round(dc))
    
    h, w = g.shape
    for obj in objs:
        for r, c in obj['cells']:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                result[nr, nc] = g[r, c]
    
    # 背景要素を保持
    for r in range(h):
        for c in range(w):
            if g[r, c] == bg and result[r, c] == bg:
                pass
            elif g[r, c] != bg and result[r, c] == bg:
                pass  # 元のオブジェクトは消える
    
    return result.tolist()


def _action_checker_fill(train_pairs, grid, bg):
    """チェック柄ルール"""
    return None  # TODO


def _action_extract(train_pairs, grid, bg):
    """切り出し/抽出"""
    objs = _get_objects(np.array(grid), bg)
    if not objs:
        return None
    
    # trainの出力サイズから何を抽出すべきか推測
    for inp, out in train_pairs:
        go = np.array(out)
        oh, ow = go.shape
        
        # 最大オブジェクトのbboxが出力サイズと一致?
        for obj in sorted(objs, key=lambda o: -o['size']):
            r1, c1, r2, c2 = obj['bbox']
            if r2 - r1 + 1 == oh and c2 - c1 + 1 == ow:
                g = np.array(grid)
                return g[r1:r2+1, c1:c2+1].tolist()
    
    return None


def _action_stamp_inside(train_pairs, grid, bg):
    """中に入れる（入れ子）"""
    return None  # TODO


def _action_fill_alternating(train_pairs, grid, bg):
    """交互パターン塗り"""
    return None  # TODO


def _action_filter(train_pairs, grid, bg):
    """フィルタ/マスク"""
    return None  # TODO


def _action_parity(train_pairs, grid, bg):
    """偶数/奇数ルール"""
    return None  # TODO


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    from arc.grid import grid_eq
    
    if sys.argv[1] == '--explain':
        tf = sys.argv[2]
        with open(tf) as f:
            task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti = task['test'][0]['input']
        
        print("=== kofdaiの第一印象 ===")
        for f in kofdai_first_glance(ti):
            print(f"  {f.trigger}: 「{f.thought}」→ {f.action}")
        
        print("\n=== train比較 ===")
        for i, (inp, out) in enumerate(tp):
            print(f"\n  Train {i}:")
            for f in kofdai_compare(inp, out):
                print(f"    {f.trigger}: 「{f.thought}」→ {f.action}")
    
    elif sys.argv[1] in ('--eval', '--train'):
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
        for tf in sorted(data_dir.glob('*.json')):
            tid = tf.stem
            with open(tf) as f: task = json.load(f)
            tp = [(e['input'], e['output']) for e in task['train']]
            ti, to = task['test'][0]['input'], task['test'][0].get('output')
            
            r = kofdai_solve(tp, ti)
            if r and to and grid_eq(r, to):
                solved.append(tid)
                tag = 'NEW' if tid not in existing else ''
                print(f'  ✓ {tid} {tag}')
        
        total = len(list(data_dir.glob('*.json')))
        new = [t for t in solved if t not in existing]
        print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
