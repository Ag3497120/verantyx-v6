"""
arc/program_tree.py — ProgramTree: 条件付きプログラム木

直列DSL列から「条件付きプログラム木」へ拡張。
Cross構造上でプログラムを表現する。

ノード種類:
  - ApplyNode: 単一変換を適用
  - SequenceNode: 変換を順次適用（A → B → C）
  - ConditionNode: 条件分岐（if P then A else B）
  - LoopNode: 収束まで繰り返し（while changed: apply A）
  - SelectBestNode: 複数候補から最良を選択

CEGIS統合:
  - 訓練ペアを反例として使い、木を刈り込む
  - 部分マッチから条件分岐を自動推定
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple, Dict, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

Grid = List[List[int]]


# ============================================================
# Program Tree Nodes
# ============================================================

class ProgramNode(ABC):
    """プログラム木のノード基底クラス"""
    name: str = "node"

    @abstractmethod
    def apply(self, inp: Grid) -> Optional[Grid]:
        pass

    @abstractmethod
    def describe(self) -> str:
        pass


@dataclass
class ApplyNode(ProgramNode):
    """単一変換ノード"""
    name: str = ""
    apply_fn: Callable = None
    params: dict = field(default_factory=dict)

    def apply(self, inp: Grid) -> Optional[Grid]:
        try:
            return self.apply_fn(inp)
        except Exception:
            return None

    def describe(self) -> str:
        return f"Apply({self.name})"


@dataclass
class SequenceNode(ProgramNode):
    """順次適用: children[0] → children[1] → ... → children[n]"""
    name: str = "sequence"
    children: List[ProgramNode] = field(default_factory=list)

    def apply(self, inp: Grid) -> Optional[Grid]:
        x = inp
        for child in self.children:
            x = child.apply(x)
            if x is None:
                return None
        return x

    def describe(self) -> str:
        steps = " → ".join(c.describe() for c in self.children)
        return f"Seq({steps})"


@dataclass
class ConditionNode(ProgramNode):
    """条件分岐: if predicate(inp) then true_branch else false_branch"""
    name: str = "condition"
    predicate: Callable = None  # Grid -> bool
    predicate_name: str = ""
    true_branch: ProgramNode = None
    false_branch: ProgramNode = None

    def apply(self, inp: Grid) -> Optional[Grid]:
        try:
            if self.predicate(inp):
                return self.true_branch.apply(inp) if self.true_branch else None
            else:
                return self.false_branch.apply(inp) if self.false_branch else None
        except Exception:
            return None

    def describe(self) -> str:
        t = self.true_branch.describe() if self.true_branch else "None"
        f = self.false_branch.describe() if self.false_branch else "None"
        return f"If({self.predicate_name} ? {t} : {f})"


@dataclass
class LoopNode(ProgramNode):
    """収束まで繰り返し: while body changes grid, keep applying"""
    name: str = "loop"
    body: ProgramNode = None
    max_iterations: int = 10

    def apply(self, inp: Grid) -> Optional[Grid]:
        from arc.grid import grid_eq
        x = inp
        for _ in range(self.max_iterations):
            x_new = self.body.apply(x)
            if x_new is None:
                return x
            if grid_eq(x, x_new):
                return x  # converged
            x = x_new
        return x

    def describe(self) -> str:
        return f"Loop({self.body.describe()}, max={self.max_iterations})"


@dataclass
class SelectBestNode(ProgramNode):
    """複数候補から訓練スコア最良を選択（ランタイム選択）"""
    name: str = "select_best"
    candidates: List[ProgramNode] = field(default_factory=list)

    def apply(self, inp: Grid) -> Optional[Grid]:
        # ランタイムでは最初の成功候補を返す
        for cand in self.candidates:
            result = cand.apply(inp)
            if result is not None:
                return result
        return None

    def describe(self) -> str:
        return f"Best({[c.describe() for c in self.candidates]})"


# ============================================================
# Predicates Library (条件関数)
# ============================================================

def _count_colors(grid: Grid) -> int:
    """非bg色の数"""
    from arc.grid import most_common_color
    bg = most_common_color(grid)
    return len(set(int(v) for row in grid for v in row if v != bg))


def _count_objects(grid: Grid) -> int:
    """連結成分の数"""
    from scipy import ndimage
    from arc.grid import most_common_color
    arr = np.array(grid)
    bg = most_common_color(grid)
    _, n = ndimage.label(arr != bg)
    return n


def _has_symmetry(grid: Grid, axis: str = 'h') -> bool:
    """対称性チェック"""
    arr = np.array(grid)
    if axis == 'h':
        return np.array_equal(arr, arr[:, ::-1])
    elif axis == 'v':
        return np.array_equal(arr, arr[::-1, :])
    return False


def _is_square(grid: Grid) -> bool:
    return len(grid) == len(grid[0])


def _has_single_color_objects(grid: Grid) -> bool:
    """全オブジェクトが単色か"""
    from scipy import ndimage
    from arc.grid import most_common_color
    arr = np.array(grid)
    bg = most_common_color(grid)
    labeled, n = ndimage.label(arr != bg)
    for oid in range(1, n + 1):
        mask = labeled == oid
        colors = set(int(arr[r, c]) for r, c in zip(*np.where(mask)))
        if len(colors) > 1:
            return False
    return True


def _grid_density(grid: Grid) -> float:
    """非bg率"""
    from arc.grid import most_common_color
    arr = np.array(grid)
    bg = most_common_color(grid)
    return float(np.sum(arr != bg)) / arr.size


def _has_enclosed_regions(grid: Grid) -> bool:
    """閉じた領域があるか"""
    from scipy import ndimage
    from arc.grid import most_common_color
    arr = np.array(grid)
    bg = most_common_color(grid)
    bg_mask = arr == bg
    labeled, n = ndimage.label(bg_mask)
    if n <= 1:
        return False
    # border-connected bg regions
    border_labels = set()
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and bg_mask[r, c]:
                border_labels.add(labeled[r, c])
    return n > len(border_labels)


# All predicates as (name, fn) pairs
PREDICATES: List[Tuple[str, Callable]] = [
    ("n_colors==1", lambda g: _count_colors(g) == 1),
    ("n_colors==2", lambda g: _count_colors(g) == 2),
    ("n_colors>=3", lambda g: _count_colors(g) >= 3),
    ("n_objects==1", lambda g: _count_objects(g) == 1),
    ("n_objects==2", lambda g: _count_objects(g) == 2),
    ("n_objects>=3", lambda g: _count_objects(g) >= 3),
    ("is_square", _is_square),
    ("has_sym_h", lambda g: _has_symmetry(g, 'h')),
    ("has_sym_v", lambda g: _has_symmetry(g, 'v')),
    ("single_color_objs", _has_single_color_objects),
    ("density>0.5", lambda g: _grid_density(g) > 0.5),
    ("has_enclosed", _has_enclosed_regions),
]


# ============================================================
# CEGIS: Program Tree Synthesizer
# ============================================================

class ProgramTreeSynthesizer:
    """
    CEGIS-based synthesizer: 訓練ペアから ProgramTree を合成する。
    
    Strategy:
    1. 既存ピースから ApplyNode を生成
    2. 単一ノードで全train通過するものを探す
    3. 失敗パターンを分析して ConditionNode を生成
    4. 残差から SequenceNode を生成
    5. 収束パターンから LoopNode を生成
    """

    def __init__(self, pieces: List, train_pairs: List[Tuple[Grid, Grid]],
                 timeout: float = 2.0):
        self.pieces = pieces
        self.train_pairs = train_pairs
        self.timeout = timeout
        self._start_time = 0

    def synthesize(self) -> Optional[ProgramNode]:
        """メインエントリ: ProgramTreeを合成"""
        import time
        self._start_time = time.time()
        from arc.grid import grid_eq

        # Phase 1: 単一ノード（既存ピースで全train通過するもの）
        # → これは cross_engine が既にやっている。ここでは条件分岐を試す。

        # Phase 2: ConditionNode — ピースAがtrain[0,1]で成功、BがTrain[2]で成功
        cond_result = self._try_condition_split()
        if cond_result is not None:
            return cond_result

        # Phase 3: SequenceNode — ピースAで部分変換、残差をピースBで処理
        seq_result = self._try_sequence_from_residual()
        if seq_result is not None:
            return seq_result

        # Phase 4: LoopNode — ピースAを収束まで繰り返し
        loop_result = self._try_loop()
        if loop_result is not None:
            return loop_result

        return None

    def _timed_out(self) -> bool:
        import time
        return (time.time() - self._start_time) > self.timeout

    def _verify(self, node: ProgramNode) -> bool:
        """全trainペアで検証"""
        from arc.grid import grid_eq
        for inp, expected in self.train_pairs:
            result = node.apply(inp)
            if result is None or not grid_eq(result, expected):
                return False
        return True

    def _partial_score(self, node: ProgramNode) -> Tuple[int, int]:
        """(成功ペア数, 総ペア数)"""
        from arc.grid import grid_eq
        ok = 0
        for inp, expected in self.train_pairs:
            result = node.apply(inp)
            if result is not None and grid_eq(result, expected):
                ok += 1
        return ok, len(self.train_pairs)

    def _try_condition_split(self) -> Optional[ConditionNode]:
        """
        条件分岐合成: ピースAがtrain[i]で成功し、ピースBがtrain[j]で成功するとき、
        trainペアを分割する条件Pを見つけてConditionNode(P, A, B)を生成。
        """
        from arc.grid import grid_eq

        if len(self.train_pairs) < 2:
            return None

        # 各ピースの成功/失敗パターンを収集
        piece_results = []  # (piece, [True/False per train pair])
        for piece in self.pieces:
            if self._timed_out():
                return None
            results = []
            for inp, expected in self.train_pairs:
                try:
                    r = piece.apply(inp)
                    results.append(r is not None and grid_eq(r, expected))
                except Exception:
                    results.append(False)
            if any(results) and not all(results):
                piece_results.append((piece, results))

        if not piece_results:
            return None

        # ペア(A, B)で全trainペアをカバーできる組み合わせを探す
        n = len(self.train_pairs)
        for i, (piece_a, res_a) in enumerate(piece_results):
            if self._timed_out():
                return None
            for j, (piece_b, res_b) in enumerate(piece_results):
                if i == j:
                    continue
                # A と B で全ペアをカバーできるか?
                covered = all(res_a[k] or res_b[k] for k in range(n))
                if not covered:
                    continue

                # Aが成功するペアとBが成功するペアを分割
                a_set = {k for k in range(n) if res_a[k]}
                b_set = {k for k in range(n) if res_b[k] and k not in a_set}

                # この分割を説明する条件(predicate)を探す
                for pred_name, pred_fn in PREDICATES:
                    if self._timed_out():
                        return None
                    try:
                        pred_values = [pred_fn(self.train_pairs[k][0])
                                       for k in range(n)]
                    except Exception:
                        continue

                    # predicate=True → A成功, predicate=False → B成功
                    if (all(pred_values[k] for k in a_set) and
                            all(not pred_values[k] for k in b_set)):
                        node_a = ApplyNode(name=piece_a.name,
                                           apply_fn=piece_a.apply)
                        node_b = ApplyNode(name=piece_b.name,
                                           apply_fn=piece_b.apply)
                        cond = ConditionNode(
                            predicate=pred_fn,
                            predicate_name=pred_name,
                            true_branch=node_a,
                            false_branch=node_b,
                        )
                        if self._verify(cond):
                            return cond

                    # Reverse: predicate=True → B, predicate=False → A
                    if (all(pred_values[k] for k in b_set) and
                            all(not pred_values[k] for k in a_set)):
                        node_a = ApplyNode(name=piece_a.name,
                                           apply_fn=piece_a.apply)
                        node_b = ApplyNode(name=piece_b.name,
                                           apply_fn=piece_b.apply)
                        cond = ConditionNode(
                            predicate=pred_fn,
                            predicate_name=pred_name,
                            true_branch=node_b,
                            false_branch=node_a,
                        )
                        if self._verify(cond):
                            return cond

        return None

    def _try_sequence_from_residual(self) -> Optional[SequenceNode]:
        """
        残差ベースの合成: ピースAを適用した結果と期待出力の差分（残差）を
        別のピースBで埋める。A→Bのシーケンス。
        """
        from arc.grid import grid_eq

        for piece_a in self.pieces:
            if self._timed_out():
                return None

            # piece_a を全trainペアに適用して中間結果を得る
            mid_pairs = []
            all_mid_ok = True
            for inp, expected in self.train_pairs:
                mid = piece_a.apply(inp)
                if mid is None:
                    all_mid_ok = False
                    break
                mid_pairs.append((mid, expected))

            if not all_mid_ok:
                continue

            # 中間結果が既に正解なら単一ピースで十分（skip）
            if all(grid_eq(m, e) for m, e in mid_pairs):
                continue

            # 中間結果→期待出力を解くピースBを探す
            # NB系はtrain overfitしやすいので除外
            _nb_prefixes = ('cross_nb', 'structural_nb', 'abstract_nb')
            for piece_b in self.pieces:
                if any(piece_b.name.startswith(p) for p in _nb_prefixes):
                    continue
                if self._timed_out():
                    return None

                all_ok = True
                for mid, expected in mid_pairs:
                    result = piece_b.apply(mid)
                    if result is None or not grid_eq(result, expected):
                        all_ok = False
                        break

                if all_ok:
                    node_a = ApplyNode(name=piece_a.name,
                                       apply_fn=piece_a.apply)
                    node_b = ApplyNode(name=piece_b.name,
                                       apply_fn=piece_b.apply)
                    seq = SequenceNode(children=[node_a, node_b])
                    if self._verify(seq):
                        return seq

        return None

    def _try_loop(self) -> Optional[LoopNode]:
        """
        ループ合成: ピースAを収束まで繰り返し適用。
        cellular automataライクなルール（NB rule等）で有効。
        """
        from arc.grid import grid_eq

        for piece in self.pieces:
            if self._timed_out():
                return None

            # まず1回適用で失敗するが、N回繰り返すと成功するケース
            single_ok = True
            for inp, expected in self.train_pairs:
                r = piece.apply(inp)
                if r is None or not grid_eq(r, expected):
                    single_ok = False
                    break

            if single_ok:
                continue  # 単一適用で成功するなら不要

            # ループで試行
            body = ApplyNode(name=piece.name, apply_fn=piece.apply)
            for max_iter in [2, 3, 5, 10]:
                loop = LoopNode(body=body, max_iterations=max_iter)
                if self._verify(loop):
                    return loop

        return None
