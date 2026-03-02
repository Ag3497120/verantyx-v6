"""
arc/cross_puzzle_engine.py — Cross × パズル推論エンジン

Cross入れ子構造:
  Level 0: 断片記憶（6軸の独立観測）
  Level 1: 6軸融合 → 仮説生成
  Level 2: 仮説検証 → 失敗分析 → 軸重み調整 → 次の仮説
  Level N: 永遠に繰り返し（総和が層ごとに変わる）

6軸:
  1. 色軸 (color) — 色の統計、マッピング、出現頻度
  2. 空間軸 (spatial) — 変更セルの位置、距離、方向
  3. オブジェクト軸 (object) — オブジェクトの役割、サイズ、形
  4. パターン軸 (pattern) — old→new のマッピングルール
  5. 関係軸 (relation) — トリガー、近傍、包含
  6. スコープ軸 (scope) — 局所/中距離/大域
"""

import numpy as np
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label


def _bg(g): return int(Counter(np.array(g).flatten()).most_common(1)[0][0])

def _objs(g, bg):
    struct = np.ones((3, 3), dtype=int)
    mask = (np.array(g) != bg).astype(int)
    labeled, n = scipy_label(mask, structure=struct)
    objs = []
    for i in range(1, n + 1):
        cells = list(zip(*np.where(labeled == i)))
        colors = [int(g[r, c]) for r, c in cells]
        r1 = min(r for r, c in cells); c1 = min(c for r, c in cells)
        r2 = max(r for r, c in cells); c2 = max(c for r, c in cells)
        bh, bw = r2 - r1 + 1, c2 - c1 + 1
        objs.append({
            'cells': set(cells), 'size': len(cells),
            'color': Counter(colors).most_common(1)[0][0],
            'colors': set(colors),
            'bbox': (r1, c1, r2, c2), 'bh': bh, 'bw': bw,
            'is_rect': len(cells) == bh * bw,
            'shape': frozenset((r - r1, c - c1) for r, c in cells),
            'center': ((r1 + r2) / 2, (c1 + c2) / 2),
        })
    return objs

def grid_eq(a, b):
    a, b = np.array(a), np.array(b)
    return a.shape == b.shape and np.array_equal(a, b)


# ══════════════════════════════════════════════════════════════
# Level 0: 6軸の断片記憶（差分観測）
# ══════════════════════════════════════════════════════════════

class DiffCross:
    """入出力ペアの差分を6軸で記述"""

    def __init__(self, inp, out):
        self.ga = np.array(inp)
        self.go = np.array(out)
        self.same_size = self.ga.shape == self.go.shape
        self.bg = _bg(self.ga)
        self.h, self.w = self.ga.shape

        if self.same_size:
            self.changed = [(r, c, int(self.ga[r, c]), int(self.go[r, c]))
                           for r in range(self.h) for c in range(self.w)
                           if self.ga[r, c] != self.go[r, c]]
        else:
            self.changed = []

        # 6軸の観測
        self.color = self._observe_color()
        self.spatial = self._observe_spatial()
        self.object = self._observe_object()
        self.pattern = self._observe_pattern()
        self.relation = self._observe_relation()
        self.scope = self._observe_scope()

    def _observe_color(self):
        """色軸: 色の統計と変化"""
        colors_in = Counter(int(v) for v in self.ga.flatten())
        if not self.same_size:
            colors_out = Counter(int(v) for v in self.go.flatten())
            return {'in': dict(colors_in), 'out': dict(colors_out),
                    'type': 'size_change'}

        # 各色の出現変化
        colors_out = Counter(int(v) for v in self.go.flatten())
        diff = {}
        for c in set(list(colors_in) + list(colors_out)):
            d = colors_out.get(c, 0) - colors_in.get(c, 0)
            if d != 0: diff[c] = d

        # 少数派色（トリガー候補）
        non_bg = {c: v for c, v in colors_in.items() if c != self.bg}
        rare_colors = sorted(non_bg.items(), key=lambda x: x[1])

        return {
            'diff': diff,
            'rare': [c for c, _ in rare_colors[:3]],
            'n_colors': len(non_bg),
            'unchanged_colors': [c for c in non_bg if c not in diff or diff[c] == 0],
        }

    def _observe_spatial(self):
        """空間軸: 変更セルの位置パターン"""
        if not self.changed:
            return {'n_changed': 0, 'type': 'none'}

        rows = set(r for r, c, _, _ in self.changed)
        cols = set(c for r, c, _, _ in self.changed)

        # 連結成分
        mask = np.zeros((self.h, self.w), dtype=int)
        for r, c, _, _ in self.changed:
            mask[r, c] = 1
        labeled, n_comp = scipy_label(mask, structure=np.ones((3, 3), dtype=int))

        components = []
        for i in range(1, n_comp + 1):
            cells = list(zip(*np.where(labeled == i)))
            r1 = min(r for r, c in cells); c1 = min(c for r, c in cells)
            r2 = max(r for r, c in cells); c2 = max(c for r, c in cells)
            components.append({
                'cells': cells, 'size': len(cells),
                'bbox': (r1, c1, r2, c2),
                'center': ((r1 + r2) / 2, (c1 + c2) / 2),
            })

        return {
            'n_changed': len(self.changed),
            'pct_changed': len(self.changed) / (self.h * self.w),
            'n_components': n_comp,
            'components': components,
            'row_spread': len(rows) / self.h,
            'col_spread': len(cols) / self.w,
        }

    def _observe_object(self):
        """オブジェクト軸: オブジェクトの役割"""
        objs = _objs(self.ga, self.bg)
        if not objs:
            return {'n_objects': 0, 'roles': {}}

        changed_set = set((r, c) for r, c, _, _ in self.changed)

        roles = []
        for i, obj in enumerate(objs):
            n_changed = len(obj['cells'] & changed_set)
            roles.append({
                'idx': i, 'size': obj['size'],
                'color': obj['color'], 'colors': obj['colors'],
                'n_changed': n_changed,
                'pct_changed': n_changed / obj['size'] if obj['size'] else 0,
                'is_rect': obj['is_rect'],
                'bbox': obj['bbox'],
                'role': 'affected' if n_changed > 0 else 'static',
            })

        # 静的オブジェクトの色（トリガー候補）
        static_colors = set()
        for r in roles:
            if r['role'] == 'static':
                static_colors.update(r['colors'])

        return {
            'n_objects': len(objs),
            'roles': roles,
            'static_colors': static_colors,
            'affected_count': sum(1 for r in roles if r['role'] == 'affected'),
        }

    def _observe_pattern(self):
        """パターン軸: old→newのマッピング"""
        if not self.changed:
            return {'type': 'none', 'map': {}}

        val_map = defaultdict(Counter)
        for r, c, ov, nv in self.changed:
            val_map[ov][nv] += 1

        # 1対1マッピングか
        is_consistent = all(len(targets) == 1 for targets in val_map.values())
        simple_map = {ov: targets.most_common(1)[0][0]
                     for ov, targets in val_map.items()} if is_consistent else {}

        return {
            'type': 'consistent_recolor' if is_consistent else 'complex',
            'map': simple_map,
            'val_map': {k: dict(v) for k, v in val_map.items()},
            'n_patterns': len(val_map),
        }

    def _observe_relation(self):
        """関係軸: トリガーとの近傍関係"""
        if not self.changed:
            return {'triggers': {}, 'type': 'none'}

        # 変更セルの4近傍と8近傍の不変色を収集
        trigger_4 = Counter()
        trigger_8 = Counter()
        for r, c, ov, nv in self.changed:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.h and 0 <= nc < self.w:
                    v = int(self.ga[nr, nc])
                    if v != self.bg and v != ov and self.ga[nr, nc] == self.go[nr, nc]:
                        trigger_4[v] += 1
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.h and 0 <= nc < self.w:
                        v = int(self.ga[nr, nc])
                        if v != self.bg and v != ov and self.ga[nr, nc] == self.go[nr, nc]:
                            trigger_8[v] += 1

        # トリガー色候補（不変で変更セル近傍に出現）
        primary_trigger = trigger_8.most_common(1)[0][0] if trigger_8 else None

        # トリガーとの距離分布
        distances = {}
        if primary_trigger is not None:
            t_pos = [(r, c) for r in range(self.h) for c in range(self.w)
                    if self.ga[r, c] == primary_trigger]
            if t_pos:
                dists = []
                for r, c, _, _ in self.changed:
                    min_d = min(abs(r - tr) + abs(c - tc) for tr, tc in t_pos)
                    dists.append(min_d)
                distances = {
                    'min': min(dists), 'max': max(dists),
                    'mean': np.mean(dists), 'all': dists,
                }

        # 変更セルが同じオブジェクト内にトリガー色を含むか
        objs = _objs(self.ga, self.bg)
        same_obj_trigger = False
        if primary_trigger is not None:
            changed_set = set((r, c) for r, c, _, _ in self.changed)
            for obj in objs:
                has_trigger = primary_trigger in obj['colors']
                has_changed = bool(obj['cells'] & changed_set)
                if has_trigger and has_changed:
                    same_obj_trigger = True
                    break

        return {
            'trigger_4': dict(trigger_4.most_common(3)),
            'trigger_8': dict(trigger_8.most_common(3)),
            'primary_trigger': primary_trigger,
            'distances': distances,
            'same_obj_trigger': same_obj_trigger,
        }

    def _observe_scope(self):
        """スコープ軸: 変更の範囲"""
        if not self.changed:
            return {'type': 'none'}

        pct = len(self.changed) / (self.h * self.w)
        n_comp = self.spatial.get('n_components', 0) if isinstance(self.spatial, dict) else 0

        if pct < 0.05:
            scope = 'minimal'
        elif pct < 0.15:
            scope = 'local'
        elif pct < 0.4:
            scope = 'medium'
        else:
            scope = 'global'

        return {'type': scope, 'pct': pct, 'n_components': n_comp}


# ══════════════════════════════════════════════════════════════
# プリミティブDSL: 組み合わせ可能な操作
# ══════════════════════════════════════════════════════════════

def prim_recolor_where(grid, old_color, new_color, condition_fn):
    """条件を満たすセルの色を変える"""
    g = np.array(grid).copy()
    h, w = g.shape
    changed = False
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == old_color and condition_fn(g, r, c, h, w):
                g[r, c] = new_color
                changed = True
    return g.tolist() if changed else None


def prim_recolor_all(grid, color_map):
    """全セルに色マップを適用"""
    g = np.array(grid).copy()
    changed = False
    for old_c, new_c in color_map.items():
        mask = g == old_c
        if np.any(mask):
            g[mask] = new_c
            changed = True
    return g.tolist() if changed else None


def prim_fill_concavity(grid, target_color, fill_color, marker_color):
    """マーカー色クラスタの凹み（bbox内の穴）をfill_colorで埋める"""
    g = np.array(grid).copy()
    h, w = g.shape
    mask = (g == marker_color).astype(int)
    labeled, n = scipy_label(mask, structure=np.ones((3, 3), dtype=int))
    changed = False
    for i in range(1, n + 1):
        cells = set(zip(*np.where(labeled == i)))
        if len(cells) < 2: continue
        r1 = min(r for r, c in cells); c1 = min(c for r, c in cells)
        r2 = max(r for r, c in cells); c2 = max(c for r, c in cells)
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if (r, c) not in cells and int(g[r, c]) == target_color:
                    g[r, c] = fill_color
                    changed = True
    return g.tolist() if changed else None


def prim_recolor_in_same_object(grid, trigger_color, old_color, new_color):
    """トリガー色と同じオブジェクト内のold→new"""
    g = np.array(grid).copy()
    bg = _bg(g)
    objs = _objs(g, bg)
    changed = False
    for obj in objs:
        if trigger_color not in obj['colors']: continue
        for r, c in obj['cells']:
            if int(g[r, c]) == old_color:
                g[r, c] = new_color
                changed = True
    return g.tolist() if changed else None



def prim_recolor_by_marker_direction(grid, marker_color, old_color, new_color):
    """マーカー色クラスタのL字/凹みの『開いた方向』にあるold→new"""
    g = np.array(grid).copy()
    h, w = g.shape; bg = _bg(g)
    mask = (g == marker_color).astype(int)
    labeled, n = scipy_label(mask, structure=np.ones((3, 3), dtype=int))
    changed = False
    for i in range(1, n + 1):
        cells = set(zip(*np.where(labeled == i)))
        if len(cells) < 2: continue
        r1 = min(r for r, c in cells); c1 = min(c for r, c in cells)
        r2 = max(r for r, c in cells); c2 = max(c for r, c in cells)
        bh, bw = r2 - r1 + 1, c2 - c1 + 1
        # L字の「開いた角」= bboxの4隅のうちマーカー色でないもの
        corners = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
        open_corners = [p for p in corners if p not in cells]
        if not open_corners: continue
        # 各開いた角の外側方向にあるold_colorセルを変更
        for oc_r, oc_c in open_corners:
            # 外側方向
            dr = -1 if oc_r == r1 else 1
            dc = -1 if oc_c == c1 else 1
            # 角から外側に広がる三角/矩形領域
            for step in range(1, max(h, w)):
                for sr in range(step + 1):
                    for sc in range(step + 1 - sr):
                        nr = oc_r + dr * sr
                        nc = oc_c + dc * sc
                        if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == old_color:
                            g[nr, nc] = new_color; changed = True
    return g.tolist() if changed else None


def prim_residual_nb_pattern(grid, nb_map, diffs):
    """残差駆動: trainペア間で一致するnb8パターンのみ適用"""
    g = np.array(grid).copy()
    h, w = g.shape; changed = False
    for r in range(h):
        for c in range(w):
            nb = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    nb.append(int(g[nr, nc]) if 0 <= nr < h and 0 <= nc < w else -1)
            key = (int(g[r, c]), tuple(nb))
            if key in nb_map:
                g[r, c] = nb_map[key]; changed = True
    return g.tolist() if changed else None


def prim_recolor_enclosed_by(grid, enclosing_color, old_color, new_color):
    """enclosing_colorで囲まれた領域内のold→new（BFSで端に到達しない領域）"""
    g = np.array(grid).copy()
    h, w = g.shape; bg = _bg(g)
    # enclosing_colorを壁としてBFS
    visited = np.zeros((h, w), dtype=bool)
    # 端からBFSで到達可能なold_colorセル
    queue = []
    for r in range(h):
        for c in [0, w - 1]:
            if int(g[r, c]) == old_color and not visited[r, c]:
                visited[r, c] = True; queue.append((r, c))
        if r == 0 or r == h - 1:
            for c in range(w):
                if int(g[r, c]) == old_color and not visited[r, c]:
                    visited[r, c] = True; queue.append((r, c))
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                v = int(g[nr, nc])
                if v == old_color or v == bg:
                    visited[nr, nc] = True
                    if v == old_color:
                        queue.append((nr, nc))
    # 端から到達できないold_colorセル → 囲まれている → new_colorに
    changed = False
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == old_color and not visited[r, c]:
                g[r, c] = new_color; changed = True
    return g.tolist() if changed else None


def prim_leave_one_out_nb(train_pairs):
    """Leave-One-Out交差検証で一貫するnb8パターンのみ抽出"""
    from collections import defaultdict
    all_patterns = defaultdict(lambda: defaultdict(int))
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return {}
        h, w = ga.shape
        for r in range(h):
            for c in range(w):
                if ga[r, c] != go[r, c]:
                    nb = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            nb.append(int(ga[nr, nc]) if 0 <= nr < h and 0 <= nc < w else -1)
                    key = (int(ga[r, c]), tuple(nb))
                    all_patterns[key][int(go[r, c])] += 1
    # 一貫したパターンのみ
    consistent = {}
    for key, targets in all_patterns.items():
        if len(targets) == 1:
            nv, count = list(targets.items())[0]
            if count >= 2:  # 2回以上出現で信頼
                consistent[key] = nv
    return consistent



def prim_recolor_adjacent_to(grid, trigger_color, old_color, new_color, max_dist=1):
    """トリガー色から距離max_dist以内のold→new"""
    g = np.array(grid).copy()
    h, w = g.shape
    t_pos = [(r, c) for r in range(h) for c in range(w) if g[r, c] == trigger_color]
    if not t_pos: return None
    changed = False
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != old_color: continue
            min_d = min(abs(r - tr) + abs(c - tc) for tr, tc in t_pos)
            if min_d <= max_dist:
                g[r, c] = new_color
                changed = True
    return g.tolist() if changed else None


def prim_flood_from_color(grid, source_color, fill_color, bg_only=True):
    """source_colorの位置からflood fill"""
    g = np.array(grid).copy()
    h, w = g.shape; bg = _bg(g)
    changed = False
    sources = [(r, c) for r in range(h) for c in range(w) if g[r, c] == source_color]
    visited = set(sources)
    queue = list(sources)
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if (bg_only and g[nr, nc] == bg) or (not bg_only and g[nr, nc] != source_color):
                    visited.add((nr, nc))
                    g[nr, nc] = fill_color
                    changed = True
                    queue.append((nr, nc))
    return g.tolist() if changed else None


def prim_fill_between_pair(grid, color_a, color_b, fill_color):
    """色Aと色Bの間（同行or同列）をfill_colorで塗る"""
    g = np.array(grid).copy()
    h, w = g.shape; bg = _bg(g)
    changed = False
    a_pos = [(r, c) for r in range(h) for c in range(w) if g[r, c] == color_a]
    b_pos = [(r, c) for r in range(h) for c in range(w) if g[r, c] == color_b]
    for ar, ac in a_pos:
        for br, bc in b_pos:
            if ar == br:  # 同行
                for c in range(min(ac, bc) + 1, max(ac, bc)):
                    if g[ar, c] == bg:
                        g[ar, c] = fill_color; changed = True
            elif ac == bc:  # 同列
                for r in range(min(ar, br) + 1, max(ar, br)):
                    if g[r, ac] == bg:
                        g[r, ac] = fill_color; changed = True
    return g.tolist() if changed else None


def prim_crop_object(grid, selector='largest'):
    """オブジェクトをcrop"""
    g = np.array(grid); bg = _bg(g)
    objs = _objs(g, bg)
    if not objs: return None
    if selector == 'largest':
        obj = max(objs, key=lambda o: o['size'])
    elif selector == 'smallest':
        obj = min(objs, key=lambda o: o['size'])
    elif selector == 'unique_color':
        cc = Counter(o['color'] for o in objs)
        uniq = [o for o in objs if cc[o['color']] == 1]
        obj = uniq[0] if uniq else None
    elif selector == 'unique_shape':
        sc = Counter(o['shape'] for o in objs)
        uniq = [o for o in objs if sc[o['shape']] == 1]
        obj = uniq[0] if uniq else None
    else:
        return None
    if obj is None: return None
    r1, c1, r2, c2 = obj['bbox']
    return g[r1:r2 + 1, c1:c2 + 1].tolist()


def prim_panel_template_apply(grid):
    """パネル構造: 特殊パネル（テンプレ）を見つけて他パネルに適用"""
    g = np.array(grid)
    h, w = g.shape; bg = _bg(g)

    # セパレータ検出
    h_seps = []; v_seps = []
    for r in range(h):
        vals = set(int(v) for v in g[r])
        if len(vals) == 1 and vals.pop() != bg:
            h_seps.append(r)
    for c in range(w):
        vals = set(int(g[r, c]) for r in range(h))
        if len(vals) == 1 and vals.pop() != bg:
            v_seps.append(c)

    if not h_seps and not v_seps: return None

    # パネル分割
    rows = [-1] + sorted(h_seps) + [h]
    cols = [-1] + sorted(v_seps) + [w]
    panels = []
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            r1, r2 = rows[i] + 1, rows[i + 1]
            c1, c2 = cols[j] + 1, cols[j + 1]
            if r2 > r1 and c2 > c1:
                panels.append((r1, c1, r2, c2, g[r1:r2, c1:c2].copy()))

    if len(panels) < 2: return None

    # 特殊パネル検出: 他と異なる色分布を持つパネル
    shapes = set(p[4].shape for p in panels)
    if len(shapes) != 1: return None

    # 各パネルの非BG色分布
    panel_profiles = []
    for r1, c1, r2, c2, p in panels:
        non_bg = set(int(v) for v in p.flatten()) - {bg}
        panel_profiles.append(non_bg)

    # 色数が多いorユニークな色を持つパネル = テンプレ
    for i, (r1, c1, r2, c2, tmpl) in enumerate(panels):
        other_colors = set()
        for j, (_, _, _, _, p) in enumerate(panels):
            if j != i:
                other_colors.update(set(int(v) for v in p.flatten()) - {bg})
        tmpl_unique = panel_profiles[i] - other_colors
        if tmpl_unique:
            # テンプレのパターンをマスクとして抽出
            # テンプレ内でユニーク色のセル位置 = マスク
            mask_color = tmpl_unique.pop()
            ph, pw = tmpl.shape
            mask_pos = [(r, c) for r in range(ph) for c in range(pw) if tmpl[r, c] == mask_color]

            if not mask_pos: continue

            # 他パネルのマスク位置の色を変更
            result = g.copy()
            changed = False
            for j, (pr1, pc1, pr2, pc2, p) in enumerate(panels):
                if j == i: continue
                for mr, mc in mask_pos:
                    if 0 <= mr < p.shape[0] and 0 <= mc < p.shape[1]:
                        # マスク位置の色をマスク色に
                        result[pr1 + mr, pc1 + mc] = mask_color
                        changed = True
            if changed:
                return result.tolist()

    return None


def prim_recolor_by_object_property(grid, property_name, value_map):
    """オブジェクトの属性に基づいてrecolor"""
    g = np.array(grid).copy()
    bg = _bg(g)
    objs = _objs(g, bg)
    changed = False
    for obj in objs:
        if property_name == 'size':
            prop = obj['size']
        elif property_name == 'n_colors':
            prop = len(obj['colors'])
        elif property_name == 'is_rect':
            prop = obj['is_rect']
        else:
            continue
        if prop in value_map:
            nc = value_map[prop]
            for r, c in obj['cells']:
                if g[r, c] != nc:
                    g[r, c] = nc; changed = True
    return g.tolist() if changed else None




def prim_symmetry_complete(grid, mode='any'):
    """対称性を検出して欠けた部分を補完"""
    g = np.array(grid); h, w = g.shape; bg = _bg(g)
    
    flips = {
        'h': np.fliplr, 'v': np.flipud,
        'hv': lambda x: np.flipud(np.fliplr(x)),
    }
    if mode in flips:
        flipped = flips[mode](g)
        mask = (g == bg) & (flipped != bg)
        if np.any(mask):
            r = g.copy(); r[mask] = flipped[mask]
            return r.tolist()
        return None
    
    if mode == 'rot90' and h == w:
        for k in [1, 2, 3]:
            rotated = np.rot90(g, k)
            mask = (g == bg) & (rotated != bg)
            if np.any(mask):
                r = g.copy(); r[mask] = rotated[mask]
                return r.tolist()
    
    if mode == 'any':
        for m in ['h', 'v', 'hv', 'rot90']:
            r = prim_symmetry_complete(grid, m)
            if r is not None: return r
    return None


def prim_gravity(grid, direction='down'):
    """非BGセルを指定方向に落とす"""
    g = np.array(grid); h, w = g.shape; bg = _bg(g)
    result = np.full((h, w), bg, dtype=int)
    if direction == 'down':
        for c in range(w):
            nb = [int(g[r, c]) for r in range(h) if g[r, c] != bg]
            for i, v in enumerate(reversed(nb)): result[h-1-i, c] = v
    elif direction == 'up':
        for c in range(w):
            nb = [int(g[r, c]) for r in range(h) if g[r, c] != bg]
            for i, v in enumerate(nb): result[i, c] = v
    elif direction == 'left':
        for r in range(h):
            nb = [int(g[r, c]) for c in range(w) if g[r, c] != bg]
            for i, v in enumerate(nb): result[r, i] = v
    elif direction == 'right':
        for r in range(h):
            nb = [int(g[r, c]) for c in range(w) if g[r, c] != bg]
            for i, v in enumerate(reversed(nb)): result[r, w-1-i] = v
    return result.tolist()


def prim_fill_enclosed_bg(grid, fill_color=None):
    """端から到達できないBGセルを塗る"""
    g = np.array(grid); h, w = g.shape; bg = _bg(g)
    visited = np.zeros((h, w), dtype=bool)
    queue = []
    for r in range(h):
        for c in [0, w-1]:
            if g[r, c] == bg and not visited[r, c]:
                visited[r, c] = True; queue.append((r, c))
    for c in range(w):
        for r in [0, h-1]:
            if g[r, c] == bg and not visited[r, c]:
                visited[r, c] = True; queue.append((r, c))
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] == bg:
                visited[nr, nc] = True; queue.append((nr, nc))
    enclosed = [(r, c) for r in range(h) for c in range(w) if g[r, c] == bg and not visited[r, c]]
    if not enclosed: return None
    result = g.copy()
    if fill_color is not None:
        for r, c in enclosed: result[r, c] = fill_color
    else:
        from collections import Counter as C2
        for r, c in enclosed:
            nbs = [int(g[r+dr, c+dc]) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                   if 0<=r+dr<h and 0<=c+dc<w and g[r+dr, c+dc] != bg]
            if nbs: result[r, c] = C2(nbs).most_common(1)[0][0]
    return result.tolist()


def prim_mirror_half(grid, axis='vertical', keep='left'):
    """半分をミラーして全体を生成"""
    g = np.array(grid); h, w = g.shape
    result = g.copy()
    if axis == 'vertical':
        mid = w // 2
        if keep == 'left':
            for c in range(mid): result[:, w-1-c] = g[:, c]
        else:
            for c in range(mid): result[:, c] = g[:, w-1-c]
    elif axis == 'horizontal':
        mid = h // 2
        if keep == 'top':
            for r in range(mid): result[h-1-r, :] = g[r, :]
        else:
            for r in range(mid): result[r, :] = g[h-1-r, :]
    return result.tolist()


def prim_select_panel(grid, criterion='monochrome'):
    """パネル構造から条件に合うパネルを選択して出力"""
    g = np.array(grid); h, w = g.shape; bg = _bg(g)
    
    # セパレータ検出
    h_seps = []; v_seps = []
    for r in range(h):
        vals = [int(v) for v in g[r]]
        if len(set(vals)) == 1 and vals[0] != bg:
            h_seps.append(r)
    for c in range(w):
        vals = [int(g[r,c]) for r in range(h)]
        if len(set(vals)) == 1 and vals[0] != bg:
            v_seps.append(c)
    if not h_seps and not v_seps: return None
    
    rows = [-1] + sorted(h_seps) + [h]
    cols = [-1] + sorted(v_seps) + [w]
    panels = []
    for i in range(len(rows)-1):
        for j in range(len(cols)-1):
            r1, r2 = rows[i]+1, rows[i+1]
            c1, c2 = cols[j]+1, cols[j+1]
            if r2 > r1 and c2 > c1:
                panels.append(g[r1:r2, c1:c2].copy())
    if len(panels) < 2: return None
    
    if criterion == 'monochrome':
        mono = [(i,p) for i,p in enumerate(panels) 
                if len(set(int(v) for v in p.flatten()) - {bg}) == 1]
        if len(mono) == 1: return mono[0][1].tolist()
    
    elif criterion == 'most_non_bg':
        best = max(panels, key=lambda p: np.sum(p != bg))
        return best.tolist()
    
    elif criterion == 'least_non_bg':
        candidates = [p for p in panels if np.any(p != bg)]
        if candidates:
            best = min(candidates, key=lambda p: int(np.sum(p != bg)))
            return best.tolist()
    
    elif criterion == 'most_colors':
        best = max(panels, key=lambda p: len(set(int(v) for v in p.flatten()) - {bg}))
        return best.tolist()
    
    elif criterion == 'least_colors':
        candidates = [p for p in panels if np.any(p != bg)]
        if candidates:
            best = min(candidates, key=lambda p: len(set(int(v) for v in p.flatten()) - {bg}))
            return best.tolist()
    
    elif criterion == 'unique':
        # 他のパネルと異なる唯一のパネル
        sigs = [frozenset((r,c,int(p[r,c])) for r in range(p.shape[0]) 
                for c in range(p.shape[1]) if p[r,c]!=bg) for p in panels]
        from collections import Counter as C2
        sig_counts = C2(sigs)
        unique = [(i,p) for i,(p,s) in enumerate(zip(panels,sigs)) if sig_counts[s]==1]
        if len(unique) == 1: return unique[0][1].tolist()
    
    return None


def prim_panel_logic_op(grid, op='xor'):
    """パネル間の論理演算（2パネル限定）"""
    g = np.array(grid); h, w = g.shape; bg = _bg(g)
    h_seps = []; v_seps = []
    for r in range(h):
        vals = [int(v) for v in g[r]]
        if len(set(vals)) == 1 and vals[0] != bg:
            h_seps.append(r)
    for c in range(w):
        vals = [int(g[r,c]) for r in range(h)]
        if len(set(vals)) == 1 and vals[0] != bg:
            v_seps.append(c)
    if not h_seps and not v_seps: return None
    
    rows = [-1] + sorted(h_seps) + [h]
    cols = [-1] + sorted(v_seps) + [w]
    panels = []
    for i in range(len(rows)-1):
        for j in range(len(cols)-1):
            r1, r2 = rows[i]+1, rows[i+1]
            c1, c2 = cols[j]+1, cols[j+1]
            if r2 > r1 and c2 > c1:
                panels.append(g[r1:r2, c1:c2].copy())
    
    if len(panels) != 2: return None
    pa, pb = panels[0], panels[1]
    if pa.shape != pb.shape: return None
    ph, pw = pa.shape
    
    result = np.full((ph, pw), bg, dtype=int)
    for r in range(ph):
        for c in range(pw):
            a_on = pa[r,c] != bg
            b_on = pb[r,c] != bg
            if op == 'xor':
                if a_on and not b_on: result[r,c] = pa[r,c]
                elif b_on and not a_on: result[r,c] = pb[r,c]
            elif op == 'or_a':
                if a_on: result[r,c] = pa[r,c]
                elif b_on: result[r,c] = pb[r,c]
            elif op == 'or_b':
                if b_on: result[r,c] = pb[r,c]
                elif a_on: result[r,c] = pa[r,c]
            elif op == 'and_a':
                if a_on and b_on: result[r,c] = pa[r,c]
            elif op == 'and_b':
                if a_on and b_on: result[r,c] = pb[r,c]
            elif op == 'diff_a':
                if a_on and not b_on: result[r,c] = pa[r,c]
            elif op == 'diff_b':
                if b_on and not a_on: result[r,c] = pb[r,c]
    return result.tolist()



def prim_conditional_recolor(grid, color_map, condition_type, condition_params):
    """条件付きrecolor: 条件を満たすセルのみ色マップを適用"""
    g = np.array(grid).copy()
    h, w = g.shape
    bg = _bg(g)
    changed = False
    
    if condition_type == 'in_object_with_color':
        # 特定色を含むオブジェクト内のセルのみ
        target_color = condition_params['color']
        objs = _objs(g, bg)
        target_cells = set()
        for obj in objs:
            if target_color in obj['colors']:
                target_cells.update(obj['cells'])
        for r in range(h):
            for c in range(w):
                if (r,c) in target_cells and int(g[r,c]) in color_map:
                    g[r,c] = color_map[int(g[r,c])]; changed = True
    
    elif condition_type == 'distance_to_color':
        # 特定色からの距離が閾値以内
        target_color = condition_params['color']
        max_dist = condition_params['max_dist']
        t_pos = [(r,c) for r in range(h) for c in range(w) if g[r,c]==target_color]
        for r in range(h):
            for c in range(w):
                if int(g[r,c]) in color_map:
                    min_d = min((abs(r-tr)+abs(c-tc) for tr,tc in t_pos), default=999)
                    if min_d <= max_dist:
                        g[r,c] = color_map[int(g[r,c])]; changed = True
    
    elif condition_type == 'same_row_or_col_as_color':
        target_color = condition_params['color']
        t_rows = set(r for r in range(h) for c in range(w) if g[r,c]==target_color)
        t_cols = set(c for r in range(h) for c in range(w) if g[r,c]==target_color)
        for r in range(h):
            for c in range(w):
                if int(g[r,c]) in color_map and (r in t_rows or c in t_cols):
                    g[r,c] = color_map[int(g[r,c])]; changed = True
    
    elif condition_type == 'component_size':
        min_size = condition_params.get('min', 0)
        max_size = condition_params.get('max', 99999)
        for old_c in color_map:
            mask = (g == old_c).astype(int)
            labeled, n = scipy_label(mask, structure=np.ones((3,3), dtype=int))
            for i in range(1, n+1):
                cells = list(zip(*np.where(labeled == i)))
                if min_size <= len(cells) <= max_size:
                    for r,c in cells:
                        g[r,c] = color_map[old_c]; changed = True
    
    elif condition_type == 'not_on_border':
        for r in range(1, h-1):
            for c in range(1, w-1):
                if int(g[r,c]) in color_map:
                    g[r,c] = color_map[int(g[r,c])]; changed = True
    
    return g.tolist() if changed else None


def auto_discover_condition(train_pairs):
    """trainペアから色マップと最適な条件を自動発見"""
    # 色マップ抽出
    color_map = {}
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None, None, None
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                if ga[r,c] != go[r,c]:
                    ov, nv = int(ga[r,c]), int(go[r,c])
                    if ov in color_map and color_map[ov] != nv:
                        return None, None, None
                    color_map[ov] = nv
    
    if not color_map: return None, None, None
    
    bg = _bg(np.array(train_pairs[0][0]))
    
    # 各条件タイプを試す
    conditions_to_try = []
    
    # in_object_with_color: 各非BG色を試す
    all_colors = set()
    for inp, _ in train_pairs:
        ga = np.array(inp)
        all_colors.update(int(v) for v in ga.flatten())
    all_colors -= {bg}
    all_colors -= set(color_map.keys())
    
    for tc in all_colors:
        conditions_to_try.append(('in_object_with_color', {'color': tc}))
    
    # distance_to_color
    for tc in all_colors:
        for md in [1, 2, 3, 5]:
            conditions_to_try.append(('distance_to_color', {'color': tc, 'max_dist': md}))
    
    # same_row_or_col
    for tc in all_colors:
        conditions_to_try.append(('same_row_or_col_as_color', {'color': tc}))
    
    # component_size
    for sz in [1, 2, 3, 4, 5]:
        conditions_to_try.append(('component_size', {'min': sz, 'max': sz}))
        conditions_to_try.append(('component_size', {'min': 1, 'max': sz}))
    
    conditions_to_try.append(('not_on_border', {}))
    
    # 各条件をtrain全体で検証
    for ctype, cparams in conditions_to_try:
        ok = True
        for inp, out in train_pairs:
            pred = prim_conditional_recolor(inp, color_map, ctype, cparams)
            if pred is None or not grid_eq(np.array(pred), np.array(out)):
                ok = False
                break
        if ok:
            return color_map, ctype, cparams
    
    return None, None, None


# ══════════════════════════════════════════════════════════════
# Level 1-N: 仮説生成 + 検証 + 層遷移
# ══════════════════════════════════════════════════════════════

def _generate_hypotheses(diffs, weights=None):
    """DiffCross群から仮説（プリミティブの組み合わせ）を生成"""
    hypotheses = []

    if not diffs or not diffs[0].same_size:
        # サイズ変化 → crop系
        hypotheses.append(('crop_largest', lambda g: prim_crop_object(g, 'largest')))
        # パネル選択
        for crit in ['monochrome', 'most_non_bg', 'least_non_bg', 'most_colors', 'least_colors', 'unique']:
            hypotheses.append((f'panel_select:{crit}', lambda g, c=crit: prim_select_panel(g, c)))
        # パネル論理演算
        for op in ['xor', 'or_a', 'or_b', 'and_a', 'and_b', 'diff_a', 'diff_b']:
            hypotheses.append((f'panel_logic:{op}', lambda g, o=op: prim_panel_logic_op(g, o)))
        hypotheses.append(('crop_smallest', lambda g: prim_crop_object(g, 'smallest')))
        hypotheses.append(('crop_unique_color', lambda g: prim_crop_object(g, 'unique_color')))
        hypotheses.append(('crop_unique_shape', lambda g: prim_crop_object(g, 'unique_shape')))
        return hypotheses

    d0 = diffs[0]

    # === 軸の重み（デフォルトまたは前層からの調整） ===
    w = weights or {'color': 1.0, 'spatial': 1.0, 'object': 1.0,
                    'pattern': 1.0, 'relation': 1.0, 'scope': 1.0}

    # === パターン軸が支配的: 単純recolor ===
    if d0.pattern['type'] == 'consistent_recolor' and w.get('pattern', 1) > 0.5:
        cmap = d0.pattern['map']
        hypotheses.append(('recolor_map', lambda g, m=cmap: prim_recolor_all(g, m)))

    # === 関係軸が支配的: トリガーベース ===
    trigger = d0.relation.get('primary_trigger')
    if trigger is not None and w.get('relation', 1) > 0.3:
        # 変更色のペアを取得
        for ov, targets in d0.pattern.get('val_map', {}).items():
            nv = max(targets, key=targets.get)
            tc = trigger

            # 距離ベースの仮説群
            dists = d0.relation.get('distances', {})
            max_d = dists.get('max', 3) if dists else 3
            for md in range(1, min(max_d + 2, 8)):
                hypotheses.append((
                    f'adj_d{md}_{ov}→{nv}_trigger{tc}',
                    lambda g, t=tc, o=ov, n=nv, d=md: prim_recolor_adjacent_to(g, t, o, n, d)
                ))

            # 同じオブジェクト内
            if d0.relation.get('same_obj_trigger'):
                hypotheses.append((
                    f'same_obj_{ov}→{nv}_trigger{tc}',
                    lambda g, t=tc, o=ov, n=nv: prim_recolor_in_same_object(g, t, o, n)
                ))

            # 凹み埋め
            hypotheses.append((
                f'concavity_{ov}→{nv}_marker{tc}',
                lambda g, t=tc, o=ov, n=nv: prim_fill_concavity(g, o, n, t)
            ))

            # flood from trigger
            hypotheses.append((
                f'flood_{tc}→{nv}',
                lambda g, t=tc, n=nv: prim_flood_from_color(g, t, n, bg_only=True)
            ))

            # between pair
            hypotheses.append((
                f'between_{tc}_{ov}→{nv}',
                lambda g, t=tc, o=ov, n=nv: prim_fill_between_pair(g, t, t, n)
            ))

    # === オブジェクト軸: 静的色が特別な役割 ===
    if d0.object.get('static_colors') and w.get('object', 1) > 0.5:
        for sc in d0.object['static_colors']:
            if sc == d0.bg: continue
            for ov, targets in d0.pattern.get('val_map', {}).items():
                nv = max(targets, key=targets.get)
                # 静的色の近傍条件
                def make_cond(static_c):
                    def cond(g, r, c, h, w):
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == static_c:
                                return True
                        return False
                    return cond

                hypotheses.append((
                    f'near_static{sc}_{ov}→{nv}',
                    lambda g, o=ov, n=nv, s=sc: prim_recolor_where(g, o, n, make_cond(s))
                ))

    # === スコープ軸: 大域 → パネル/flood ===
    if d0.scope.get('type') in ('medium', 'global') and w.get('scope', 1) > 0.5:
        hypotheses.append(('panel_template', prim_panel_template_apply))

    # === 近傍パターン（8近傍の色配置→出力色） ===
    if w.get('spatial', 1) > 0.5:
        # 変更セルの8近傍パターンを収集
        nb_map = {}
        consistent = True
        for d in diffs:
            for r, c, ov, nv in d.changed:
                nb = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < d.h and 0 <= nc < d.w:
                            nb.append(int(d.ga[nr, nc]))
                        else:
                            nb.append(-1)
                key = (ov, tuple(nb))
                if key in nb_map and nb_map[key] != nv:
                    consistent = False; break
                nb_map[key] = nv
            if not consistent: break

        if consistent and nb_map and len(nb_map) <= 50:
            def make_nb_fn(pm):
                def fn(grid):
                    g = np.array(grid).copy()
                    h, w = g.shape; changed = False
                    for r in range(h):
                        for c in range(w):
                            nb = []
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    if dr == 0 and dc == 0: continue
                                    nr, nc = r + dr, c + dc
                                    nb.append(int(g[nr, nc]) if 0 <= nr < h and 0 <= nc < w else -1)
                            key = (int(g[r, c]), tuple(nb))
                            if key in pm:
                                g[r, c] = pm[key]; changed = True
                    return g.tolist() if changed else None
                return fn

            hypotheses.append(('nb8_pattern', make_nb_fn(nb_map)))

    # === 追加プリミティブ ===
    # マーカーの方向性ベース
    if trigger is not None and w.get('relation', 1) > 0.3:
        for ov, targets in d0.pattern.get('val_map', {}).items():
            nv = max(targets, key=targets.get)
            tc = trigger
            hypotheses.append((
                f'marker_dir_{ov}→{nv}_marker{tc}',
                lambda g, t=tc, o=ov, n=nv: prim_recolor_by_marker_direction(g, t, o, n)
            ))
            hypotheses.append((
                f'enclosed_{ov}→{nv}_by{tc}',
                lambda g, t=tc, o=ov, n=nv: prim_recolor_enclosed_by(g, t, o, n)
            ))

    # === 対称性/重力/塗りつぶし/ミラー ===
    for sm in ['h', 'v', 'hv', 'rot90', 'any']:
        hypotheses.append((f'sym:{sm}', lambda g, m=sm: prim_symmetry_complete(g, m)))
    for gd in ['down', 'up', 'left', 'right']:
        hypotheses.append((f'gravity:{gd}', lambda g, d=gd: prim_gravity(g, d)))
    hypotheses.append(('fill_enclosed', lambda g: prim_fill_enclosed_bg(g)))
    for fc in range(10):
        hypotheses.append((f'fill_enc:{fc}', lambda g, c=fc: prim_fill_enclosed_bg(g, c)))
    for ax, kp in [('vertical','left'),('vertical','right'),('horizontal','top'),('horizontal','bottom')]:
        hypotheses.append((f'mirror:{ax}:{kp}', lambda g, a=ax, k=kp: prim_mirror_half(g, a, k)))

    # === 条件付きrecolor (auto-discovered) ===
    cmap, ctype, cparams = auto_discover_condition(
        [(d.ga.tolist(), d.go.tolist()) for d in diffs])
    if cmap is not None and ctype is not None:
        hypotheses.insert(0, (
            f'cond_recolor:{ctype}',
            lambda g, m=cmap, t=ctype, p=cparams: prim_conditional_recolor(g, m, t, p)
        ))

    # LOO交差検証nb8
    loo_nb = prim_leave_one_out_nb([(d.ga.tolist(), d.go.tolist()) for d in diffs])
    if loo_nb:
        def make_loo_fn(pm):
            def fn(grid):
                g = np.array(grid).copy()
                h, w = g.shape; changed = False
                for r in range(h):
                    for c in range(w):
                        nb = []
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = r + dr, c + dc
                                nb.append(int(g[nr, nc]) if 0 <= nr < h and 0 <= nc < w else -1)
                        key = (int(g[r, c]), tuple(nb))
                        if key in pm:
                            g[r, c] = pm[key]; changed = True
                return g.tolist() if changed else None
            return fn
        hypotheses.append(('loo_nb8', make_loo_fn(loo_nb)))

    return hypotheses


def _analyze_failure(hypothesis_fn, train_pairs):
    """仮説の失敗パターンを分析 → 次の層の軸重み調整に使う"""
    failures = []
    for inp, out in train_pairs:
        pred = hypothesis_fn(inp)
        if pred is None:
            failures.append({'type': 'no_output', 'inp': inp, 'out': out})
            continue
        if grid_eq(pred, out):
            continue

        pa = np.array(pred)
        oa = np.array(out)
        ia = np.array(inp)

        if pa.shape != oa.shape:
            failures.append({'type': 'size_mismatch', 'pred_shape': pa.shape, 'out_shape': oa.shape})
            continue

        # 間違えたセルの分析
        wrong = [(r, c, int(pa[r, c]), int(oa[r, c]))
                for r in range(pa.shape[0]) for c in range(pa.shape[1])
                if pa[r, c] != oa[r, c]]

        # 過剰変更（変えすぎ）vs 不足変更（変え足りない）
        over = [(r, c, pv, ev) for r, c, pv, ev in wrong if ia[r, c] == ev]  # 変えたが元に戻すべきだった
        under = [(r, c, pv, ev) for r, c, pv, ev in wrong if ia[r, c] == pv]  # 変えるべきだったが変えなかった

        failures.append({
            'type': 'wrong_cells',
            'n_wrong': len(wrong),
            'n_over': len(over),
            'n_under': len(under),
            'over_ratio': len(over) / max(len(wrong), 1),
            'under_ratio': len(under) / max(len(wrong), 1),
        })

    return failures


def _adjust_weights(weights, failures):
    """失敗パターンから軸重みを調整"""
    w = dict(weights)

    if not failures:
        return w

    # 失敗タイプの集計
    over_total = sum(f.get('n_over', 0) for f in failures if f['type'] == 'wrong_cells')
    under_total = sum(f.get('n_under', 0) for f in failures if f['type'] == 'wrong_cells')
    no_output = sum(1 for f in failures if f['type'] == 'no_output')

    if no_output > 0:
        # 出力が出ない → 条件が厳しすぎ → relation/spatialの重みを下げ、pattern/objectを上げ
        w['relation'] *= 0.5
        w['spatial'] *= 0.5
        w['pattern'] = min(w['pattern'] * 1.5, 2.0)
        w['object'] = min(w['object'] * 1.5, 2.0)

    elif over_total > under_total:
        # 変えすぎ → 条件が緩すぎ → relation/scopeの重みを上げ
        w['relation'] = min(w['relation'] * 1.5, 2.0)
        w['scope'] = min(w['scope'] * 1.5, 2.0)
        w['spatial'] *= 0.7

    elif under_total > over_total:
        # 変え足りない → 条件が厳しすぎ → scopeを広げ
        w['scope'] *= 0.7
        w['spatial'] = min(w['spatial'] * 1.3, 2.0)

    else:
        # 質的に間違い → objectとcolorの重みを変える
        w['object'] = min(w['object'] * 1.3, 2.0)
        w['color'] = min(w['color'] * 1.3, 2.0)
        w['pattern'] *= 0.7

    return w


# ══════════════════════════════════════════════════════════════
# Master: Cross Puzzle Engine
# ══════════════════════════════════════════════════════════════

MAX_LAYERS = 5

def _cell_feats_light(ga, r, c):
    """軽量セル特徴"""
    h, w = ga.shape; bg = _bg(ga); v = int(ga[r, c])
    f = set()
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0<=nr<h and 0<=nc<w: f.add(f'n4:{int(ga[nr,nc])}')
    f.add(f'border:{r==0 or r==h-1 or c==0 or c==w-1}')
    mask = (ga==v).astype(int)
    labeled, _ = scipy_label(mask, structure=np.ones((3,3),dtype=int))
    f.add(f'csize:{int(np.sum(labeled==labeled[r,c]))}')
    obj_mask = (ga!=bg).astype(int)
    obj_lab, _ = scipy_label(obj_mask, structure=np.ones((3,3),dtype=int))
    if obj_lab[r,c] > 0:
        obj_cells = list(zip(*np.where(obj_lab==obj_lab[r,c])))
        for oc in set(int(ga[rr,cc]) for rr,cc in obj_cells):
            if oc != v: f.add(f'obj_has:{oc}')
    return f


def cross_puzzle_solve(train_pairs, test_input):
    """Cross入れ子構造のパズル推論エンジン (v3: 失敗蓄積型)"""

    diffs = []
    for inp, out in train_pairs:
        diffs.append(DiffCross(inp, out))

    weights = {'color': 1.0, 'spatial': 1.0, 'object': 1.0,
               'pattern': 1.0, 'relation': 1.0, 'scope': 1.0}

    # Level 0: 全プリミティブを試す
    hypotheses = _generate_hypotheses(diffs, weights)
    
    near_misses = []  # (wrong, name, fn)
    
    for name, h_fn in hypotheses:
        ok = True
        total_wrong = 0
        for inp, out in train_pairs:
            pred = h_fn(inp)
            if pred is None:
                ok = False; total_wrong += 999; break
            if not grid_eq(pred, out):
                ok = False
                pa, oa = np.array(pred), np.array(out)
                if pa.shape == oa.shape:
                    total_wrong += int(np.sum(pa != oa))
                else:
                    total_wrong += 999; break

        if ok:
            result = h_fn(test_input)
            if result is not None:
                return result, f'puzzle:L0:{name}'
        elif total_wrong < 80:
            near_misses.append((total_wrong, name, h_fn))

    # Level 1: 失敗蓄積 → 制約付き仮説
    near_misses.sort(key=lambda x: x[0])
    
    # 色マップ学習
    cmap = {}
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: continue
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                if ga[r,c] != go[r,c]:
                    ov, nv = int(ga[r,c]), int(go[r,c])
                    if ov not in cmap: cmap[ov] = nv

    for _, name, h_fn in near_misses[:5]:
        # over/underセルの特徴収集
        over_feats = []
        under_feats = []
        correct_feats = []
        
        for inp, out in train_pairs:
            pred = h_fn(inp)
            if pred is None: break
            ga, go, pa = np.array(inp), np.array(out), np.array(pred)
            if pa.shape != go.shape: break
            
            for r in range(pa.shape[0]):
                for c in range(pa.shape[1]):
                    if pa[r,c] != go[r,c]:
                        if ga[r,c] == go[r,c]:
                            over_feats.append(_cell_feats_light(ga, r, c))
                        elif ga[r,c] == pa[r,c]:
                            under_feats.append(_cell_feats_light(ga, r, c))
                    elif pa[r,c] != ga[r,c]:
                        correct_feats.append(_cell_feats_light(ga, r, c))
        
        if not over_feats and not under_feats:
            continue
        
        over_common = set.intersection(*over_feats) if over_feats else set()
        correct_common = set.intersection(*correct_feats) if correct_feats else set()
        
        # over除外候補: overにあってcorrectにない
        excl_cands = list(over_common - correct_common) if over_feats and correct_feats else list(over_common)
        
        # under追加候補
        under_common = set.intersection(*under_feats) if under_feats else set()
        all_over_union = set.union(*over_feats) if over_feats else set()
        incl_cands = list(under_common - all_over_union) if under_feats else []
        
        # 除外のみ
        for ef in excl_cands:
            def make_excl(bf, ef_):
                def fn(grid):
                    pred = bf(grid)
                    if pred is None: return None
                    go_ = np.array(grid); gp = np.array(pred)
                    h, w = gp.shape
                    for r in range(h):
                        for c in range(w):
                            if gp[r,c] != go_[r,c]:
                                if ef_ in _cell_feats_light(go_, r, c):
                                    gp[r,c] = go_[r,c]
                    return gp.tolist()
                return fn
            
            corrected = make_excl(h_fn, ef)
            ok = all(grid_eq(corrected(inp), out) for inp, out in train_pairs)
            if ok:
                result = corrected(test_input)
                if result is not None:
                    return result, f'puzzle:L1:{name}+excl({ef})'
        
        # 追加のみ
        for af in incl_cands:
            def make_incl(bf, af_, cm):
                def fn(grid):
                    pred = bf(grid)
                    if pred is None: return None
                    go_ = np.array(grid); gp = np.array(pred)
                    h, w = gp.shape
                    for r in range(h):
                        for c in range(w):
                            if gp[r,c] == go_[r,c] and int(go_[r,c]) in cm:
                                if af_ in _cell_feats_light(go_, r, c):
                                    gp[r,c] = cm[int(go_[r,c])]
                    return gp.tolist()
                return fn
            
            augmented = make_incl(h_fn, af, cmap)
            ok = all(grid_eq(augmented(inp), out) for inp, out in train_pairs)
            if ok:
                result = augmented(test_input)
                if result is not None:
                    return result, f'puzzle:L1:{name}+incl({af})'
        
        # 除外+追加の同時
        for ef in excl_cands[:3]:
            for af in incl_cands[:3]:
                def make_both(bf, ef_, af_, cm):
                    def fn(grid):
                        pred = bf(grid)
                        if pred is None: return None
                        go_ = np.array(grid); gp = np.array(pred)
                        h, w = gp.shape
                        for r in range(h):
                            for c in range(w):
                                if gp[r,c] != go_[r,c]:
                                    if ef_ in _cell_feats_light(go_, r, c):
                                        gp[r,c] = go_[r,c]
                                elif int(go_[r,c]) in cm:
                                    if af_ in _cell_feats_light(go_, r, c):
                                        gp[r,c] = cm[int(go_[r,c])]
                        return gp.tolist()
                    return fn
                
                both = make_both(h_fn, ef, af, cmap)
                ok = all(grid_eq(both(inp), out) for inp, out in train_pairs)
                if ok:
                    result = both(test_input)
                    if result is not None:
                        return result, f'puzzle:L1:{name}+excl({ef})+incl({af})'

    return None, None


if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path

    split = 'evaluation' if '--eval' in sys.argv else 'training'
    data_dir = Path(f'/tmp/arc-agi-2/data/{split}')

    existing = set()
    try:
        with open('arc_v82.log') as f:
            for l in f:
                m = re.search(r'✓.*?([0-9a-f]{8})', l)
                if m: existing.add(m.group(1))
    except: pass
    synth = set(f.stem for f in Path('synth_results').glob('*.py')) if Path('synth_results').exists() else set()
    all_e = existing | synth

    solved = []
    for tf in sorted(data_dir.glob('*.json')):
        tid = tf.stem
        with open(tf) as f: task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti, to = task['test'][0]['input'], task['test'][0].get('output')

        result, name = cross_puzzle_solve(tp, ti)
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_e else ''
            solved.append((tid[:8], name, tag))
            if tag:
                print(f'  ✓ {tid[:8]} [{name}] NEW')

    total = len(list(data_dir.glob('*.json')))
    new = [t for t, _, tg in solved if tg == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')

    from collections import Counter as C2
    solver_stats = C2()
    for _, name, _ in solved:
        solver_stats[name] += 1
    print('\nソルバー別:')
    for s, c in solver_stats.most_common():
        print(f'  {s}: {c}')
