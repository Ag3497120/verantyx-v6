"""
arc/cross_memory.py — Cross Engine の「記憶」（概念的原型データベース）

人間が生まれてから大人になるまでに獲得する直感的知識を
CrossEngineの操作選択に使えるようにコード化する。

=== 発達段階 ===

Stage 0: 乳児期 (0-1歳) — 物体の永続性、因果関係
  「物は消えない」「触ると動く」「落ちる」

Stage 1: 幼児期 (1-3歳) — 空間認識、分類
  「同じ形は同じグループ」「大きい小さい」「上下左右」

Stage 2: 幼稒期 (3-6歳) — パターン、対称性、数
  「鏡」「繰り返し」「数える」「色で分ける」

Stage 3: 児童期 (6-12歳) — 論理、規則、因果推論
  「もし〜なら〜」「ルールに従う」「例外を見つける」

Stage 4: 青年期 (12-18歳) — 抽象思考、メタ推論
  「パターンのパターン」「類推」「変換の本質を理解」

各段階で獲得する「原体験」を、gridの特徴から検出する関数と
推薦する操作のマッピングとして実装する。
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from scipy.ndimage import label as scipy_label


@dataclass
class Memory:
    """一つの原体験"""
    name: str
    stage: int          # 発達段階 0-4
    concept: str        # 概念名
    description: str    # 人間的説明
    detect: object      # 入力gridから検出する関数
    operations: list    # 推薦する操作のリスト
    confidence: float = 0.5


@dataclass
class MemoryActivation:
    """活性化された記憶"""
    memory: Memory
    strength: float     # 0-1
    evidence: dict      # 検出根拠


# ──────────────────────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────────────────────

def _objects(grid, bg=None):
    """連結成分オブジェクトを返す"""
    g = np.array(grid)
    if bg is None:
        bg = Counter(g.flatten()).most_common(1)[0][0]
    mask = (g != bg).astype(int)
    labeled, n = scipy_label(mask)
    objs = []
    for i in range(1, n + 1):
        cells = list(zip(*np.where(labeled == i)))
        colors = [int(g[r, c]) for r, c in cells]
        objs.append({
            'cells': cells,
            'color': Counter(colors).most_common(1)[0][0],
            'colors': set(colors),
            'size': len(cells),
            'bbox': (min(r for r, c in cells), min(c for r, c in cells),
                     max(r for r, c in cells), max(c for r, c in cells)),
        })
    return objs, bg


def _has_symmetry(g, axis):
    a = np.array(g)
    h, w = a.shape
    if axis == 'h':
        return np.array_equal(a, a[:, ::-1])
    elif axis == 'v':
        return np.array_equal(a, a[::-1, :])
    elif axis == 'rot2':
        return np.array_equal(a, np.rot90(a, 2))
    elif axis == 'rot4':
        r = np.rot90(a, 1)
        return a.shape == r.shape and np.array_equal(a, r)
    return False


def _find_separators(grid, bg):
    """separatorの行/列を検出"""
    g = np.array(grid)
    h, w = g.shape
    
    full_rows = []
    for r in range(h):
        vals = set(int(v) for v in g[r])
        if len(vals) == 1 and vals.pop() != bg:
            full_rows.append(r)
    
    full_cols = []
    for c in range(w):
        vals = set(int(v) for v in g[:, c])
        if len(vals) == 1 and vals.pop() != bg:
            full_cols.append(c)
    
    return full_rows, full_cols


def _grid_diff(inp, out):
    """入力と出力の差分を分析"""
    gi = np.array(inp)
    go = np.array(out)
    if gi.shape != go.shape:
        return {'same_size': False, 'size_ratio': (go.shape[0]/gi.shape[0], go.shape[1]/gi.shape[1])}
    
    diff = gi != go
    n_changed = int(diff.sum())
    bg = Counter(gi.flatten()).most_common(1)[0][0]
    
    added = int(((gi == bg) & (go != bg)).sum())
    removed = int(((gi != bg) & (go == bg)).sum())
    recolored = int(((gi != bg) & (go != bg) & diff).sum())
    
    return {
        'same_size': True,
        'n_changed': n_changed,
        'total': gi.size,
        'change_ratio': n_changed / gi.size if gi.size else 0,
        'added': added,
        'removed': removed,
        'recolored': recolored,
    }


# ──────────────────────────────────────────────────────────────
# Stage 0: 乳児期 — 物体の永続性、因果関係、物理
# ──────────────────────────────────────────────────────────────

def detect_gravity(inp, out, train_pairs):
    """物が下に落ちる — 重力の直感"""
    gi, go = np.array(inp), np.array(out)
    if gi.shape != go.shape:
        return None
    
    bg = Counter(gi.flatten()).most_common(1)[0][0]
    h, w = gi.shape
    
    # 各列で非bg色が下に移動しているか
    gravity_cols = 0
    total_cols = 0
    
    for c in range(w):
        in_colors = [int(gi[r, c]) for r in range(h) if gi[r, c] != bg]
        out_colors = [int(go[r, c]) for r in range(h) if go[r, c] != bg]
        
        if not in_colors:
            continue
        total_cols += 1
        
        # 出力で色が下に寄ってるか
        if sorted(in_colors) == sorted(out_colors):
            out_positions = [r for r in range(h) if go[r, c] != bg]
            if out_positions and out_positions[-1] == h - 1:
                gravity_cols += 1
    
    if total_cols > 0 and gravity_cols / total_cols > 0.5:
        return {'direction': 'down', 'ratio': gravity_cols / total_cols}
    
    return None


def detect_object_permanence(inp, out, train_pairs):
    """物体は消えない — 入力のオブジェクトが出力にも存在する"""
    objs_in, bg = _objects(inp)
    objs_out, _ = _objects(out)
    
    if not objs_in:
        return None
    
    # 入力の各オブジェクトの色が出力にもあるか
    in_colors = {o['color'] for o in objs_in}
    out_colors = {o['color'] for o in objs_out}
    
    preserved = in_colors & out_colors
    if len(preserved) == len(in_colors):
        return {'all_preserved': True, 'colors': preserved}
    
    return None


def detect_contact_cause(inp, out, train_pairs):
    """接触 → 変化（因果）: 2つのオブジェクトが隣接すると色が変わる"""
    gi, go = np.array(inp), np.array(out)
    if gi.shape != go.shape:
        return None
    
    objs, bg = _objects(inp)
    if len(objs) < 2:
        return None
    
    # 隣接オブジェクトペアで色変化が起きてるか
    diff = _grid_diff(inp, out)
    if diff['recolored'] > 0:
        return {'recolored': diff['recolored'], 'n_objects': len(objs)}
    
    return None


# ──────────────────────────────────────────────────────────────
# Stage 1: 幼児期 — 分類、空間、大小
# ──────────────────────────────────────────────────────────────

def detect_sorting(inp, out, train_pairs):
    """同じものを集める — 色や形でグループ化"""
    objs_in, bg = _objects(inp)
    objs_out, _ = _objects(out)
    
    if len(objs_in) < 2:
        return None
    
    # 出力でオブジェクトが色ごとにまとまってるか
    out_arr = np.array(out)
    colors = set(int(v) for v in out_arr.flatten() if v != bg)
    
    for color in colors:
        mask = (out_arr == color)
        labeled, n = scipy_label(mask.astype(int))
        if n == 1:  # 1色1かたまり → 集まってる
            continue
        else:
            return None
    
    if colors:
        return {'grouped_colors': colors}
    return None


def detect_size_relation(inp, out, train_pairs):
    """大きい/小さいの関係 — サイズで選択・操作"""
    objs, bg = _objects(inp)
    if len(objs) < 2:
        return None
    
    sizes = sorted([o['size'] for o in objs])
    if sizes[-1] > sizes[0] * 2:  # 明確なサイズ差
        return {'largest': sizes[-1], 'smallest': sizes[0], 'ratio': sizes[-1] / sizes[0]}
    return None


def detect_spatial_relation(inp, out, train_pairs):
    """上下左右の空間関係"""
    objs, bg = _objects(inp)
    if len(objs) < 2:
        return None
    
    # オブジェクト間の空間関係を検出
    relations = []
    for i, o1 in enumerate(objs):
        for j, o2 in enumerate(objs):
            if i >= j:
                continue
            r1, c1 = np.mean([r for r, c in o1['cells']]), np.mean([c for r, c in o1['cells']])
            r2, c2 = np.mean([r for r, c in o2['cells']]), np.mean([c for r, c in o2['cells']])
            
            if abs(r1 - r2) > abs(c1 - c2):
                relations.append('vertical')
            else:
                relations.append('horizontal')
    
    return {'relations': relations} if relations else None


# ──────────────────────────────────────────────────────────────
# Stage 2: 幼稒期 — パターン、対称性、数
# ──────────────────────────────────────────────────────────────

def detect_mirror(inp, out, train_pairs):
    """鏡 — 対称性の検出"""
    for axis in ['h', 'v', 'rot2', 'rot4']:
        if _has_symmetry(out, axis) and not _has_symmetry(inp, axis):
            return {'axis': axis, 'created': True}
        if _has_symmetry(out, axis) and _has_symmetry(inp, axis):
            return {'axis': axis, 'preserved': True}
    return None


def detect_repetition(inp, out, train_pairs):
    """繰り返し — タイリング、周期パターン"""
    gi, go = np.array(inp), np.array(out)
    
    if go.shape[0] > gi.shape[0] or go.shape[1] > gi.shape[1]:
        # 出力が大きい → タイリング?
        if go.shape[0] % gi.shape[0] == 0 and go.shape[1] % gi.shape[1] == 0:
            return {'type': 'tile', 'factor': (go.shape[0]//gi.shape[0], go.shape[1]//gi.shape[1])}
    
    # 出力内の周期
    h, w = go.shape
    for period in range(1, h // 2 + 1):
        if h % period != 0:
            continue
        if all(np.array_equal(go[r], go[r % period]) for r in range(period, h)):
            return {'type': 'row_period', 'period': period}
    
    return None


def detect_counting(inp, out, train_pairs):
    """数える — オブジェクト数→色/サイズの対応"""
    objs, bg = _objects(inp)
    gi, go = np.array(inp), np.array(out)
    
    if go.size < gi.size and len(objs) > 0:
        # 出力が小さい → 数え上げ/要約?
        n_objs = len(objs)
        if go.shape[0] * go.shape[1] <= n_objs * 2:
            return {'n_objects': n_objs, 'out_size': go.shape}
    
    return None


# ──────────────────────────────────────────────────────────────
# Stage 3: 児童期 — 論理、規則、条件分岐
# ──────────────────────────────────────────────────────────────

def detect_conditional(inp, out, train_pairs):
    """もし〜なら〜 — 条件付き変換"""
    diff = _grid_diff(inp, out)
    if not diff.get('same_size'):
        return None
    
    gi, go = np.array(inp), np.array(out)
    bg = Counter(gi.flatten()).most_common(1)[0][0]
    
    # 変化したセルに共通する条件を探す
    changed_cells = list(zip(*np.where(gi != go)))
    if not changed_cells:
        return None
    
    # 変化セルの近傍特徴
    h, w = gi.shape
    conditions = []
    
    for r, c in changed_cells:
        # 隣接に特定色があるか
        adj_colors = set()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                adj_colors.add(int(gi[nr, nc]))
        conditions.append(frozenset(adj_colors))
    
    # 共通条件
    common = conditions[0]
    for c in conditions[1:]:
        common = common & c
    
    if common - {bg}:
        return {'trigger_colors': common - {bg}, 'n_changed': len(changed_cells)}
    
    return None


def detect_separator_logic(inp, out, train_pairs):
    """区切り線 → パネル間の論理演算"""
    gi = np.array(inp)
    bg = Counter(gi.flatten()).most_common(1)[0][0]
    
    full_rows, full_cols = _find_separators(inp, bg)
    
    if full_rows or full_cols:
        return {
            'h_separators': len(full_rows),
            'v_separators': len(full_cols),
            'n_panels': (len(full_rows) + 1) * (len(full_cols) + 1),
        }
    return None


def detect_rule_exception(inp, out, train_pairs):
    """ルールの例外 — ほとんどのセルは同じルールだが一部だけ違う"""
    diff = _grid_diff(inp, out)
    if not diff.get('same_size'):
        return None
    
    if 0 < diff.get('change_ratio', 0) < 0.1:
        return {'few_changes': True, 'ratio': diff['change_ratio']}
    return None


# ──────────────────────────────────────────────────────────────
# Stage 4: 青年期 — 抽象思考、メタ推論、類推
# ──────────────────────────────────────────────────────────────

def detect_analogy(inp, out, train_pairs):
    """類推 — train例間の変換パターンの一貫性"""
    diffs = []
    for ti, to in train_pairs:
        d = _grid_diff(ti, to)
        diffs.append(d)
    
    # 全例で同じ変化率？
    if all(d.get('same_size') for d in diffs):
        ratios = [d.get('change_ratio', 0) for d in diffs]
        if max(ratios) - min(ratios) < 0.1:
            return {'consistent_change_ratio': np.mean(ratios)}
    
    return None


def detect_meta_pattern(inp, out, train_pairs):
    """パターンのパターン — 変換自体に規則性がある"""
    if len(train_pairs) < 3:
        return None
    
    # 例間でオブジェクト数が変化するパターン
    counts = []
    for ti, to in train_pairs:
        objs_in, _ = _objects(ti)
        objs_out, _ = _objects(to)
        counts.append((len(objs_in), len(objs_out)))
    
    # 入力オブジェクト数→出力オブジェクト数の関係
    if all(ci == co for ci, co in counts):
        return {'object_count': 'preserved'}
    
    diffs = [co - ci for ci, co in counts]
    if len(set(diffs)) == 1:
        return {'object_count_delta': diffs[0]}
    
    return None


# ──────────────────────────────────────────────────────────────
# 記憶データベース
# ──────────────────────────────────────────────────────────────

ALL_MEMORIES = [
    # Stage 0: 乳児期
    Memory('gravity', 0, 'physics', '物は下に落ちる',
           detect_gravity, ['gravity_down', 'gravity_up', 'gravity_left', 'gravity_right']),
    Memory('permanence', 0, 'physics', '物体は消えない、形を保つ',
           detect_object_permanence, ['preserve_objects', 'move', 'copy']),
    Memory('contact_cause', 0, 'causality', '接触すると変化する',
           detect_contact_cause, ['conditional_recolor', 'flood_fill', 'extend']),
    
    # Stage 1: 幼児期
    Memory('sorting', 1, 'classification', '同じものは同じグループ',
           detect_sorting, ['group_by_color', 'sort', 'cluster']),
    Memory('size_relation', 1, 'spatial', '大きいと小さいは違う役割',
           detect_size_relation, ['extract_largest', 'extract_smallest', 'size_filter']),
    Memory('spatial_relation', 1, 'spatial', '上下左右の位置関係',
           detect_spatial_relation, ['move', 'align', 'stack']),
    
    # Stage 2: 幼稒期
    Memory('mirror', 2, 'symmetry', '鏡に映すと左右が逆',
           detect_mirror, ['flip_h', 'flip_v', 'rotate', 'symmetry_fill']),
    Memory('repetition', 2, 'pattern', '同じものが繰り返される',
           detect_repetition, ['tile', 'repeat', 'periodic_fill']),
    Memory('counting', 2, 'number', '数えると数字になる',
           detect_counting, ['count_objects', 'summarize', 'reduce']),
    
    # Stage 3: 児童期
    Memory('conditional', 3, 'logic', 'もし〜なら〜する',
           detect_conditional, ['conditional_recolor', 'conditional_fill', 'if_then']),
    Memory('separator', 3, 'structure', '区切り線でパネルに分ける',
           detect_separator_logic, ['panel_xor', 'panel_and', 'panel_or', 'extract_panel']),
    Memory('exception', 3, 'logic', 'ほとんどは同じだが例外がある',
           detect_rule_exception, ['mark_exception', 'highlight_different']),
    
    # Stage 4: 青年期
    Memory('analogy', 4, 'abstraction', '同じ変換パターンが繰り返される',
           detect_analogy, ['apply_same_transform', 'generalize']),
    Memory('meta_pattern', 4, 'abstraction', '変換自体にパターンがある',
           detect_meta_pattern, ['compose_transforms', 'chain']),
]


# ──────────────────────────────────────────────────────────────
# 記憶活性化エンジン
# ──────────────────────────────────────────────────────────────

class MemoryEngine:
    """人間の発達段階に基づく概念記憶エンジン"""
    
    def __init__(self):
        self.memories = ALL_MEMORIES
    
    def activate(self, train_pairs) -> List[MemoryActivation]:
        """train例から記憶を活性化"""
        activations = []
        
        for mem in self.memories:
            total_strength = 0
            evidences = []
            
            for inp, out in train_pairs:
                try:
                    result = mem.detect(inp, out, train_pairs)
                    if result is not None:
                        total_strength += 1
                        evidences.append(result)
                except Exception:
                    continue
            
            if total_strength > 0:
                strength = total_strength / len(train_pairs)
                # 全例で活性化 → 高信頼
                if strength == 1.0:
                    strength = 1.0
                else:
                    strength *= 0.7  # 部分活性化は割引
                
                activations.append(MemoryActivation(
                    memory=mem,
                    strength=strength,
                    evidence=evidences[0] if evidences else {},
                ))
        
        # 強度順にソート
        activations.sort(key=lambda a: (-a.strength, a.memory.stage))
        return activations
    
    def recommend_operations(self, train_pairs) -> List[Tuple[str, float]]:
        """活性化された記憶から操作を推薦"""
        activations = self.activate(train_pairs)
        
        op_scores = defaultdict(float)
        for act in activations:
            for op in act.memory.operations:
                op_scores[op] = max(op_scores[op], act.strength)
        
        return sorted(op_scores.items(), key=lambda x: -x[1])
    
    def explain(self, train_pairs) -> str:
        """活性化された記憶を人間向けに説明"""
        activations = self.activate(train_pairs)
        
        if not activations:
            return "記憶: 活性化なし"
        
        lines = ["記憶活性化:"]
        for act in activations[:5]:
            stage_names = ['乳児期', '幼児期', '幼稒期', '児童期', '青年期']
            lines.append(
                f"  [{stage_names[act.memory.stage]}] {act.memory.description} "
                f"(強度={act.strength:.1%}, 推薦: {', '.join(act.memory.operations[:3])})"
            )
        
        return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────
# 記憶ベースソルバー: 活性化された概念→操作実行
# ──────────────────────────────────────────────────────────────

def memory_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """記憶活性化 → 推薦操作の実行 → train検証"""
    engine = MemoryEngine()
    activations = engine.activate(train_pairs)
    
    if not activations:
        return None
    
    # 推薦操作を試行
    from arc.cross6_brute import brute_solve  # 既存の操作実行器を流用
    
    # 活性化された記憶に基づいてbruteの操作を優先的に試行
    result = brute_solve(train_pairs, test_input)
    if result:
        return result
    
    # 記憶固有の操作を試行
    for act in activations:
        if act.strength < 0.5:
            continue
        
        for op_name in act.memory.operations:
            result = _try_memory_op(op_name, train_pairs, test_input)
            if result is not None:
                return result
    
    return None


def _try_memory_op(op_name, train_pairs, test_input):
    """記憶推薦の操作を実行"""
    gi = np.array(test_input)
    bg = Counter(gi.flatten()).most_common(1)[0][0]
    h, w = gi.shape
    
    if op_name == 'gravity_down':
        return _apply_gravity(test_input, bg, 'down')
    elif op_name == 'gravity_up':
        return _apply_gravity(test_input, bg, 'up')
    elif op_name == 'flip_h':
        return np.fliplr(gi).tolist()
    elif op_name == 'flip_v':
        return np.flipud(gi).tolist()
    elif op_name == 'symmetry_fill':
        return _symmetry_fill(test_input, bg)
    elif op_name == 'conditional_recolor':
        return _conditional_recolor(train_pairs, test_input)
    elif op_name == 'conditional_fill':
        return _conditional_fill(train_pairs, test_input)
    
    return None


def _apply_gravity(grid, bg, direction):
    g = np.array(grid)
    h, w = g.shape
    result = np.full_like(g, bg)
    
    for c in range(w):
        colors = [int(g[r, c]) for r in range(h) if g[r, c] != bg]
        if direction == 'down':
            for i, color in enumerate(reversed(colors)):
                result[h - 1 - i, c] = color
        elif direction == 'up':
            for i, color in enumerate(colors):
                result[i, c] = color
    
    return result.tolist()


def _symmetry_fill(grid, bg):
    g = np.array(grid)
    h, w = g.shape
    result = g.copy()
    
    # 水平対称
    for r in range(h):
        for c in range(w):
            mc = w - 1 - c
            if result[r, c] == bg and result[r, mc] != bg:
                result[r, c] = result[r, mc]
    
    # 垂直対称
    for r in range(h):
        mr = h - 1 - r
        for c in range(w):
            if result[r, c] == bg and result[mr, c] != bg:
                result[r, c] = result[mr, c]
    
    if np.array_equal(result, g):
        return None
    return result.tolist()


def _conditional_recolor(train_pairs, test_input):
    """条件付きリカラー: train例から条件を学習"""
    gi = np.array(test_input)
    h, w = gi.shape
    bg = Counter(gi.flatten()).most_common(1)[0][0]
    
    # 3x3近傍→出力色のマッピングをtrainから学習
    # 抽象的近傍（S/B/O）を使用
    rules = {}
    
    for inp, out in train_pairs:
        gin = np.array(inp)
        gout = np.array(out)
        if gin.shape != gout.shape:
            return None
        bg_t = Counter(gin.flatten()).most_common(1)[0][0]
        hi, wi = gin.shape
        
        for r in range(hi):
            for c in range(wi):
                if gin[r, c] == gout[r, c]:
                    continue  # 変化なし→スキップ
                
                center = int(gin[r, c])
                # 4近傍の抽象パターン
                nb = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < hi and 0 <= nc < wi:
                        v = int(gin[nr, nc])
                        if v == bg_t: nb.append('B')
                        elif v == center: nb.append('S')
                        else: nb.append('O')
                    else:
                        nb.append('X')
                
                center_role = 'B' if center == bg_t else 'F'
                key = (center_role, tuple(nb))
                new_color = int(gout[r, c])
                
                # 新色も抽象化
                if new_color == bg_t:
                    new_role = 'BG'
                else:
                    new_role = new_color  # 具体色
                
                if key in rules:
                    if rules[key] != new_role:
                        return None  # 矛盾
                else:
                    rules[key] = new_role
    
    if not rules:
        return None
    
    # test入力に適用
    result = [row[:] for row in test_input]
    changed = False
    
    for r in range(h):
        for c in range(w):
            center = int(gi[r, c])
            nb = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w:
                    v = int(gi[nr, nc])
                    if v == bg: nb.append('B')
                    elif v == center: nb.append('S')
                    else: nb.append('O')
                else:
                    nb.append('X')
            
            center_role = 'B' if center == bg else 'F'
            key = (center_role, tuple(nb))
            
            if key in rules:
                new_role = rules[key]
                if new_role == 'BG':
                    new_c = bg
                else:
                    new_c = new_role
                
                if new_c != int(gi[r, c]):
                    result[r][c] = new_c
                    changed = True
    
    # train検証
    for inp, out in train_pairs:
        gin = np.array(inp)
        gout = np.array(out)
        if gin.shape != gout.shape:
            return None
        bg_t = Counter(gin.flatten()).most_common(1)[0][0]
        
        pred = [row[:] for row in inp]
        hi, wi = gin.shape
        for r in range(hi):
            for c in range(wi):
                center = int(gin[r, c])
                nb = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < hi and 0 <= nc < wi:
                        v = int(gin[nr, nc])
                        if v == bg_t: nb.append('B')
                        elif v == center: nb.append('S')
                        else: nb.append('O')
                    else:
                        nb.append('X')
                center_role = 'B' if center == bg_t else 'F'
                key = (center_role, tuple(nb))
                if key in rules:
                    new_role = rules[key]
                    if new_role == 'BG':
                        pred[r][c] = bg_t
                    else:
                        pred[r][c] = new_role
        
        if pred != out:
            return None
    
    return result if changed else None


def _conditional_fill(train_pairs, test_input):
    """条件付きfill: 囲まれた領域を塗る"""
    gi = np.array(test_input)
    bg = Counter(gi.flatten()).most_common(1)[0][0]
    h, w = gi.shape
    
    # 全train例で「bgセルが非bgで囲まれてたら色が変わる」パターン
    fill_color = None
    
    for inp, out in train_pairs:
        gin = np.array(inp)
        gout = np.array(out)
        if gin.shape != gout.shape:
            return None
        bg_t = Counter(gin.flatten()).most_common(1)[0][0]
        hi, wi = gin.shape
        
        for r in range(hi):
            for c in range(wi):
                if gin[r, c] == bg_t and gout[r, c] != bg_t:
                    if fill_color is None:
                        fill_color = int(gout[r, c])
                    elif fill_color != int(gout[r, c]):
                        return None  # 複数の塗り色 → 複雑
    
    if fill_color is None:
        return None
    
    # 囲まれたbgセルを検出 (flood_fillで外部bgを除外)
    result = [row[:] for row in test_input]
    visited = np.zeros((h, w), dtype=bool)
    
    # 境界から到達可能なbgセルをマーク（外部）
    queue = []
    for r in range(h):
        for c in [0, w-1]:
            if gi[r, c] == bg and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True
    for c in range(w):
        for r in [0, h-1]:
            if gi[r, c] == bg and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True
    
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and gi[nr, nc] == bg:
                visited[nr, nc] = True
                queue.append((nr, nc))
    
    # 外部でないbgセルをfill
    changed = False
    for r in range(h):
        for c in range(w):
            if gi[r, c] == bg and not visited[r, c]:
                result[r][c] = fill_color
                changed = True
    
    if not changed:
        return None
    
    # train検証
    for inp, out in train_pairs:
        gin = np.array(inp)
        bg_t = Counter(gin.flatten()).most_common(1)[0][0]
        hi, wi = gin.shape
        
        pred = [row[:] for row in inp]
        vis = np.zeros((hi, wi), dtype=bool)
        q = []
        for r in range(hi):
            for c in [0, wi-1]:
                if gin[r, c] == bg_t and not vis[r, c]:
                    q.append((r, c)); vis[r, c] = True
        for c in range(wi):
            for r in [0, hi-1]:
                if gin[r, c] == bg_t and not vis[r, c]:
                    q.append((r, c)); vis[r, c] = True
        while q:
            r, c = q.pop(0)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < hi and 0 <= nc < wi and not vis[nr, nc] and gin[nr, nc] == bg_t:
                    vis[nr, nc] = True; q.append((nr, nc))
        for r in range(hi):
            for c in range(wi):
                if gin[r, c] == bg_t and not vis[r, c]:
                    pred[r][c] = fill_color
        
        if pred != out:
            return None
    
    return result


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json
    from pathlib import Path
    from arc.grid import grid_eq
    
    if sys.argv[1] == '--train':
        TRAIN_DIR = Path('/tmp/arc-agi-2/data/training')
        engine = MemoryEngine()
        
        solved = 0
        total = 0
        for tf in sorted(TRAIN_DIR.glob('*.json')):
            tid = tf.stem
            with open(tf) as f:
                task = json.load(f)
            tp = [(e['input'], e['output']) for e in task['train']]
            ti = task['test'][0]['input']
            to = task['test'][0].get('output')
            total += 1
            
            result = memory_solve(tp, ti)
            if result and to and grid_eq(result, to):
                solved += 1
                acts = engine.activate(tp)
                top = acts[0] if acts else None
                mem_name = top.memory.name if top else '?'
                print(f'✓ {tid} [{mem_name}]')
        
        print(f'\nMemory solve: {solved}/{total}')
    
    elif sys.argv[1] == '--explain':
        tf = sys.argv[2]
        with open(tf) as f:
            task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        engine = MemoryEngine()
        print(engine.explain(tp))
    
    else:
        tf = sys.argv[1]
        with open(tf) as f:
            task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti = task['test'][0]['input']
        to = task['test'][0].get('output')
        
        result = memory_solve(tp, ti)
        if result:
            print('✅ SOLVED!' if to and grid_eq(result, to) else '⚠️ Solution')
        else:
            print('✗ No solution')
