"""
arc/cross_cut.py — 引き算ソルバー: 全接続Cross → カット → 翻訳

全セル間の6軸接続をグラフとして表現。
入力→出力の差分から「どの接続を切るか」のルールを学習。

CrossGraph:
  ノード = (r, c) 各セル
  エッジ = 6軸方向の接続 (上下左右+対角2方向)
  各エッジに属性: (色同士?, run長差, 境界距離差, etc.)

カットルール:
  エッジの属性パターン → cut / keep の判定
  train例から学習し、test入力に適用。

カット後:
  分離された連結成分 = Cross断片
  各断片の色 = 成分内の最頻色 or 出力色ルール
  断片群 → 出力grid
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Set, FrozenSet
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label


# ──────────────────────────────────────────────────────────────
# CrossGraph: 全接続グラフ表現
# ──────────────────────────────────────────────────────────────

# 6軸方向 (8方向にしてもよいが、kofdai設計に従い6軸)
# 上下左右 + 対角2方向 (↖↘)
DIRS_6 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]
DIR_NAMES = ['up', 'down', 'left', 'right', 'diag_ul', 'diag_dr']

# 8方向版
DIRS_8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]


class CrossGraph:
    """グリッドの全接続6軸Crossグラフ"""

    def __init__(self, grid: List[List[int]], use_8dir: bool = False):
        self.grid = grid
        self.g = np.array(grid, dtype=np.int8)
        self.h, self.w = self.g.shape
        self.bg = int(Counter(self.g.flatten()).most_common(1)[0][0])
        self.dirs = DIRS_8 if use_8dir else DIRS_6

        # エッジ: ((r1,c1),(r2,c2)) -> edge_features
        # edge_features: dict of attributes
        self.edges: Dict[Tuple, Dict] = {}
        self._build_edges()

    def _build_edges(self):
        g = self.g
        h, w = self.h, self.w

        for r in range(h):
            for c in range(w):
                color1 = int(g[r, c])
                for di, (dr, dc) in enumerate(self.dirs):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        color2 = int(g[nr, nc])

                        # 正規化キー (小さい方を先に)
                        if (r, c) < (nr, nc):
                            key = ((r, c), (nr, nc))
                        else:
                            key = ((nr, nc), (r, c))

                        if key in self.edges:
                            continue

                        self.edges[key] = {
                            'same_color': color1 == color2,
                            'color1': color1,
                            'color2': color2,
                            'both_bg': color1 == self.bg and color2 == self.bg,
                            'both_fg': color1 != self.bg and color2 != self.bg,
                            'one_bg': (color1 == self.bg) != (color2 == self.bg),
                            'dir_idx': di,
                        }

    def cut_edges(self, cut_fn) -> 'CutResult':
        """cut_fn(edge_key, edge_features) -> bool (True=cut)"""
        kept = set()
        cut = set()

        for key, feat in self.edges.items():
            if cut_fn(key, feat):
                cut.add(key)
            else:
                kept.add(key)

        return CutResult(self, kept, cut)


class CutResult:
    """カット結果: 連結成分の集合"""

    def __init__(self, graph: CrossGraph, kept_edges: Set, cut_edges: Set):
        self.graph = graph
        self.kept_edges = kept_edges
        self.cut_edges = cut_edges
        self.h, self.w = graph.h, graph.w

        # Union-Find で連結成分を計算
        self.components = self._find_components()

    def _find_components(self):
        parent = {}
        for r in range(self.h):
            for c in range(self.w):
                parent[(r, c)] = (r, c)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for (n1, n2) in self.kept_edges:
            union(n1, n2)

        # グループ化
        groups = defaultdict(set)
        for r in range(self.h):
            for c in range(self.w):
                root = find((r, c))
                groups[root].add((r, c))

        return list(groups.values())

    def to_grid(self, color_rule: str = 'majority') -> List[List[int]]:
        """カット結果をgridに翻訳"""
        g = self.graph.g
        bg = self.graph.bg
        result = np.full((self.h, self.w), bg, dtype=np.int8)

        for comp in self.components:
            if color_rule == 'majority':
                colors = [int(g[r, c]) for r, c in comp]
                color = Counter(colors).most_common(1)[0][0]
                for r, c in comp:
                    result[r, c] = color
            elif color_rule == 'keep':
                for r, c in comp:
                    result[r, c] = g[r, c]
            elif color_rule == 'majority_fg':
                colors = [int(g[r, c]) for r, c in comp if g[r, c] != bg]
                if colors:
                    color = Counter(colors).most_common(1)[0][0]
                    for r, c in comp:
                        result[r, c] = color
                else:
                    for r, c in comp:
                        result[r, c] = bg

        return result.tolist()


# ──────────────────────────────────────────────────────────────
# カットルール学習
# ──────────────────────────────────────────────────────────────

def _edge_feature_key(feat: Dict, level: str = 'full') -> tuple:
    """エッジ特徴をキーに変換"""
    if level == 'color_pair':
        return (feat['color1'], feat['color2'])
    elif level == 'same_color':
        return (feat['same_color'],)
    elif level == 'role':
        return (feat['both_bg'], feat['both_fg'], feat['one_bg'])
    elif level == 'role_dir':
        return (feat['both_bg'], feat['both_fg'], feat['one_bg'], feat['dir_idx'])
    elif level == 'full':
        return (feat['same_color'], feat['both_bg'], feat['both_fg'],
                feat['one_bg'], feat['dir_idx'])
    return (feat['same_color'],)


def learn_cut_rule(train_pairs: List[Tuple]) -> Optional[Dict]:
    """
    train例から「どのエッジを切るか」のルールを学習。

    入力gridのエッジグラフを構築し、出力gridと比較して
    「このエッジは切られるべきだったか」をラベル付け。
    エッジ特徴 → cut/keep のマッピングを学習。
    """
    if not all(len(inp) == len(out) and len(inp[0]) == len(out[0])
               for inp, out in train_pairs):
        return None

    # 各抽象化レベルでルールを試行
    for level in ['color_pair', 'role', 'role_dir', 'same_color', 'full']:
        for color_rule in ['keep', 'majority', 'majority_fg']:
            for use_8dir in [False, True]:
                rule = _try_cut_level(train_pairs, level, color_rule, use_8dir)
                if rule is not None:
                    return rule

    return None


def _try_cut_level(train_pairs, level, color_rule, use_8dir) -> Optional[Dict]:
    """特定の抽象化レベルでカットルールを試行"""
    cut_map = {}  # feature_key -> should_cut (bool)

    for inp, out in train_pairs:
        cg_in = CrossGraph(inp, use_8dir=use_8dir)
        g_out = np.array(out, dtype=np.int8)

        for key, feat in cg_in.edges.items():
            (r1, c1), (r2, c2) = key
            out_c1 = int(g_out[r1, c1])
            out_c2 = int(g_out[r2, c2])

            # 出力で異なる色 → このエッジは切られた
            # 出力で同じ色 → このエッジは保持された
            should_cut = (out_c1 != out_c2)

            fkey = _edge_feature_key(feat, level)
            if fkey in cut_map:
                if cut_map[fkey] != should_cut:
                    return None  # 矛盾
            else:
                cut_map[fkey] = should_cut

    if not cut_map:
        return None

    # 検証: train全例に適用して出力と一致するか
    for inp, out in train_pairs:
        cg = CrossGraph(inp, use_8dir=use_8dir)
        cr = cg.cut_edges(
            lambda k, f, cm=cut_map, lv=level:
                cm.get(_edge_feature_key(f, lv), False)
        )
        pred = cr.to_grid(color_rule)
        if pred != out:
            return None

    return {
        'cut_map': cut_map,
        'level': level,
        'color_rule': color_rule,
        'use_8dir': use_8dir,
    }


def apply_cut_rule(grid: List[List[int]], rule: Dict) -> List[List[int]]:
    """学習したカットルールを適用"""
    cut_map = rule['cut_map']
    level = rule['level']
    color_rule = rule['color_rule']
    use_8dir = rule['use_8dir']

    cg = CrossGraph(grid, use_8dir=use_8dir)
    cr = cg.cut_edges(
        lambda k, f: cut_map.get(_edge_feature_key(f, level), False)
    )
    return cr.to_grid(color_rule)


# ──────────────────────────────────────────────────────────────
# 高度なカットルール: 位置情報を含む
# ──────────────────────────────────────────────────────────────

def _positional_edge_key(key, feat, g, bg, level='pos_role'):
    """位置情報を含むエッジキー"""
    (r1, c1), (r2, c2) = key
    h, w = g.shape

    if level == 'pos_role':
        # 相対位置 (境界からの距離カテゴリ)
        def pos_cat(r, c):
            edge_r = min(r, h - 1 - r)
            edge_c = min(c, w - 1 - c)
            return ('E' if edge_r == 0 else 'I',
                    'E' if edge_c == 0 else 'I')

        return (feat['same_color'], feat['one_bg'],
                pos_cat(r1, c1), pos_cat(r2, c2))

    elif level == 'nb_context':
        # 周囲の色の多様性
        def nb_diversity(r, c):
            colors = set()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w:
                    colors.add(int(g[nr, nc]))
            return len(colors)

        return (feat['same_color'], feat['one_bg'],
                nb_diversity(r1, c1), nb_diversity(r2, c2))

    return (feat['same_color'],)


def learn_positional_cut(train_pairs) -> Optional[Dict]:
    """位置情報を含むカットルールの学習"""
    if not all(len(inp) == len(out) and len(inp[0]) == len(out[0])
               for inp, out in train_pairs):
        return None

    for level in ['pos_role', 'nb_context']:
        for color_rule in ['keep', 'majority', 'majority_fg']:
            for use_8dir in [False, True]:
                cut_map = {}
                consistent = True

                for inp, out in train_pairs:
                    g_in = np.array(inp, dtype=np.int8)
                    g_out = np.array(out, dtype=np.int8)
                    bg = int(Counter(g_in.flatten()).most_common(1)[0][0])
                    cg = CrossGraph(inp, use_8dir=use_8dir)

                    for key, feat in cg.edges.items():
                        (r1, c1), (r2, c2) = key
                        should_cut = int(g_out[r1, c1]) != int(g_out[r2, c2])
                        fkey = _positional_edge_key(key, feat, g_in, bg, level)

                        if fkey in cut_map:
                            if cut_map[fkey] != should_cut:
                                consistent = False
                                break
                        else:
                            cut_map[fkey] = should_cut

                    if not consistent:
                        break

                if not consistent or not cut_map:
                    continue

                # 検証
                ok = True
                for inp, out in train_pairs:
                    g_in = np.array(inp, dtype=np.int8)
                    bg = int(Counter(g_in.flatten()).most_common(1)[0][0])
                    cg = CrossGraph(inp, use_8dir=use_8dir)
                    cr = cg.cut_edges(
                        lambda k, f, cm=cut_map, lv=level, gi=g_in, b=bg:
                            cm.get(_positional_edge_key(k, f, gi, b, lv), False)
                    )
                    pred = cr.to_grid(color_rule)
                    if pred != out:
                        ok = False
                        break

                if ok:
                    return {
                        'cut_map': cut_map,
                        'level': level,
                        'color_rule': color_rule,
                        'use_8dir': use_8dir,
                        'positional': True,
                    }

    return None


def apply_positional_cut(grid, rule):
    g_in = np.array(grid, dtype=np.int8)
    bg = int(Counter(g_in.flatten()).most_common(1)[0][0])
    cut_map = rule['cut_map']
    level = rule['level']
    color_rule = rule['color_rule']
    use_8dir = rule['use_8dir']

    cg = CrossGraph(grid, use_8dir=use_8dir)
    cr = cg.cut_edges(
        lambda k, f: cut_map.get(
            _positional_edge_key(k, f, g_in, bg, level), False)
    )
    return cr.to_grid(color_rule)


# ──────────────────────────────────────────────────────────────
# 引き算ソルバー統合
# ──────────────────────────────────────────────────────────────

def cross_cut_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """引き算ソルバー"""
    # 1. 基本カットルール
    rule = learn_cut_rule(train_pairs)
    if rule is not None:
        result = apply_cut_rule(test_input, rule)
        # LOO検証
        if _loo_check(train_pairs, rule, positional=False):
            return result

    # 2. 位置情報付きカットルール
    rule = learn_positional_cut(train_pairs)
    if rule is not None:
        result = apply_positional_cut(test_input, rule)
        if _loo_check(train_pairs, rule, positional=True):
            return result

    return None


def _loo_check(train_pairs, rule, positional=False):
    """Leave-one-out検証"""
    if len(train_pairs) < 3:
        return True

    for i in range(len(train_pairs)):
        loo_train = train_pairs[:i] + train_pairs[i+1:]
        loo_in, loo_out = train_pairs[i]

        if positional:
            loo_rule = learn_positional_cut(loo_train)
            if loo_rule is None:
                return False
            pred = apply_positional_cut(loo_in, loo_rule)
        else:
            loo_rule = learn_cut_rule(loo_train)
            if loo_rule is None:
                return False
            pred = apply_cut_rule(loo_in, loo_rule)

        if pred != loo_out:
            return False

    return True


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python3 -m arc.cross_cut <task.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        task = json.load(f)

    tp = [(e['input'], e['output']) for e in task['train']]
    ti = task['test'][0]['input']
    to = task['test'][0].get('output')

    result = cross_cut_solve(tp, ti)
    if result:
        if to and result == to:
            print("✅ SOLVED!")
        else:
            print("⚠️ Solution generated")
    else:
        print("✗ No solution")
