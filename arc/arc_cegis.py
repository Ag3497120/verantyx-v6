"""
arc_cegis.py — ARC用CEGIS（Counter-Example Guided Inductive Synthesis）

訓練ペアを完璧なオラクルとして使い、変換候補を刈り込む。
HLEのCEGISLoopに相当するが、ARCでは訓練ペアが反例を無限に提供するため
CEGISが最も威力を発揮する。

フロー:
  1. 入出力ペアの構造差分から変換候補を生成
  2. 各候補を全訓練ペアで検証
  3. 不一致なら候補を除外
  4. 一致する候補があれば、チェーン化を試みる
  5. 全訓練ペアで一致する変換チェーンをテスト入力に適用
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
import time

from arc.grid_ir import GridDecomposer, GridIR, decompose_pair
from arc.transforms import (
    Transform, Grid,
    FIXED_TRANSFORMS,
    get_parametric_transforms,
)


@dataclass
class CEGISResult:
    """CEGIS の結果"""
    solved: bool
    prediction: Optional[np.ndarray] = None  # テスト出力の予測
    transform_chain: list[Transform] = field(default_factory=list)
    iterations: int = 0
    candidates_tested: int = 0
    elapsed_ms: float = 0.0
    method: str = ""  # "single" | "chain_2" | "chain_3" | "composite"


class ARCCEGISLoop:
    """
    ARC用CEGISループ

    訓練ペア = 完璧なオラクル（入力→出力が確定）
    候補変換を訓練ペアで検証し、全ペアで一致する変換を見つける。
    """

    def __init__(
        self,
        max_chain_len: int = 3,
        time_limit_ms: float = 30000.0,  # 30秒制限
    ):
        self.max_chain_len = max_chain_len
        self.time_limit_ms = time_limit_ms
        self.decomposer = GridDecomposer()

    def solve(
        self,
        train_pairs: list[dict],   # [{"input": [[...]], "output": [[...]]}]
        test_input: list[list[int]],
    ) -> CEGISResult:
        """
        ARC問題を解く。

        Args:
            train_pairs: 訓練ペア（入力/出力グリッドのリスト）
            test_input: テスト入力グリッド

        Returns:
            CEGISResult
        """
        t0 = time.time()

        # 訓練ペアをnumpy化
        train = [
            (np.array(p["input"], dtype=np.int8), np.array(p["output"], dtype=np.int8))
            for p in train_pairs
        ]
        test_in = np.array(test_input, dtype=np.int8)

        # 背景色を推定
        # ARCでは0(黒)が背景であることが多い。0が存在すれば0を背景とする。
        from collections import Counter
        all_colors = Counter()
        for inp, _ in train:
            all_colors.update(inp.flatten().tolist())
        if 0 in all_colors:
            bg = 0
        else:
            bg = all_colors.most_common(1)[0][0]

        # Phase 1: 構造差分分析
        pair_analyses = []
        for inp, out in train:
            analysis = decompose_pair(inp.tolist(), out.tolist())
            pair_analyses.append(analysis)

        # Phase 2: 候補変換の生成
        candidates = self._generate_candidates(train, bg, pair_analyses)
        total_candidates = len(candidates)

        # Phase 3: 単一変換で解けるか試す
        result = self._try_single_transforms(train, test_in, candidates, t0)
        if result.solved:
            result.candidates_tested = total_candidates
            return result

        # Phase 4: 2段チェーンを試す
        if self.max_chain_len >= 2:
            result = self._try_chain_2(train, test_in, candidates, t0)
            if result.solved:
                result.candidates_tested = total_candidates
                return result

        # Phase 5: 特殊合成変換を試す
        result = self._try_composite_transforms(train, test_in, pair_analyses, bg, t0)
        if result.solved:
            result.candidates_tested = total_candidates
            return result

        elapsed = (time.time() - t0) * 1000
        return CEGISResult(
            solved=False,
            iterations=total_candidates,
            candidates_tested=total_candidates,
            elapsed_ms=elapsed,
        )

    def _generate_candidates(
        self,
        train: list[tuple[Grid, Grid]],
        bg: int,
        analyses: list[dict],
    ) -> list[Transform]:
        """訓練ペアの特徴から変換候補を生成"""
        candidates = list(FIXED_TRANSFORMS)

        # 最初のペアからパラメトリック変換を生成
        inp0, out0 = train[0]
        candidates.extend(get_parametric_transforms(inp0, out0, bg))

        # 重複除去（名前+パラメータベース）
        seen = set()
        unique = []
        for c in candidates:
            key = repr(c)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

    def _try_single_transforms(
        self,
        train: list[tuple[Grid, Grid]],
        test_in: Grid,
        candidates: list[Transform],
        t0: float,
    ) -> CEGISResult:
        """単一変換で全訓練ペアに一致するものを探す"""
        for t in candidates:
            if (time.time() - t0) * 1000 > self.time_limit_ms:
                break

            all_match = True
            for inp, out in train:
                try:
                    result = t.apply(inp)
                    if result.shape != out.shape or not np.array_equal(result, out):
                        all_match = False
                        break
                except Exception:
                    all_match = False
                    break

            if all_match:
                try:
                    prediction = t.apply(test_in)
                    elapsed = (time.time() - t0) * 1000
                    return CEGISResult(
                        solved=True,
                        prediction=prediction,
                        transform_chain=[t],
                        iterations=1,
                        elapsed_ms=elapsed,
                        method="single",
                    )
                except Exception:
                    pass

        return CEGISResult(solved=False)

    def _try_chain_2(
        self,
        train: list[tuple[Grid, Grid]],
        test_in: Grid,
        candidates: list[Transform],
        t0: float,
    ) -> CEGISResult:
        """2段変換チェーンを試す"""
        # まず各候補の1段目結果をキャッシュ
        first_results: list[tuple[Transform, list[Optional[Grid]]]] = []
        for t1 in candidates:
            if (time.time() - t0) * 1000 > self.time_limit_ms:
                break
            results = []
            valid = True
            for inp, _ in train:
                try:
                    r = t1.apply(inp)
                    results.append(r)
                except Exception:
                    valid = False
                    break
            if valid:
                first_results.append((t1, results))

        # 2段目
        for t1, r1_list in first_results:
            if (time.time() - t0) * 1000 > self.time_limit_ms:
                break
            for t2 in candidates:
                if (time.time() - t0) * 1000 > self.time_limit_ms:
                    break
                all_match = True
                for i, (_, out) in enumerate(train):
                    try:
                        r2 = t2.apply(r1_list[i])
                        if r2.shape != out.shape or not np.array_equal(r2, out):
                            all_match = False
                            break
                    except Exception:
                        all_match = False
                        break

                if all_match:
                    try:
                        mid = t1.apply(test_in)
                        prediction = t2.apply(mid)
                        elapsed = (time.time() - t0) * 1000
                        return CEGISResult(
                            solved=True,
                            prediction=prediction,
                            transform_chain=[t1, t2],
                            iterations=2,
                            elapsed_ms=elapsed,
                            method="chain_2",
                        )
                    except Exception:
                        pass

        return CEGISResult(solved=False)

    def _try_composite_transforms(
        self,
        train: list[tuple[Grid, Grid]],
        test_in: Grid,
        analyses: list[dict],
        bg: int,
        t0: float,
    ) -> CEGISResult:
        """
        特殊な合成変換を試す:
        - 列/行の拡張（パターン繰り返し）
        - オブジェクト移動
        - マスク操作
        """
        # パターン繰り返し検出: 入力の非背景列を出力全体に繰り返す
        inp0, out0 = train[0]
        h_in, w_in = inp0.shape
        h_out, w_out = out0.shape

        # 縦縞繰り返し: 入力の各列パターンを出力幅に繰り返す
        if h_in == h_out:
            # 入力から非背景の列パターンを抽出
            fg_cols = []
            for c in range(w_in):
                if np.any(inp0[:, c] != bg):
                    fg_cols.append((c, inp0[:, c].copy()))

            if fg_cols and len(fg_cols) <= 5:
                # パターン = 前景列の相対位置と色
                # 周期 = 前景列間の間隔
                if len(fg_cols) >= 2:
                    intervals = [fg_cols[i+1][0] - fg_cols[i][0] for i in range(len(fg_cols)-1)]
                    if len(set(intervals)) == 1:
                        period = intervals[0]
                        # 周期で出力全体に繰り返す
                        try:
                            prediction_test = self._apply_column_repeat(
                                inp0, fg_cols, period, w_out, h_out, bg, test_in
                            )
                            if prediction_test is not None:
                                # 全訓練ペアで検証
                                all_ok = True
                                for inp, out in train:
                                    pred = self._apply_column_repeat(
                                        inp,
                                        [(c, inp[:, c].copy()) for c in range(inp.shape[1]) if np.any(inp[:, c] != bg)],
                                        period, out.shape[1], out.shape[0], bg, None
                                    )
                                    if pred is None or not np.array_equal(pred, out):
                                        all_ok = False
                                        break
                                if all_ok:
                                    elapsed = (time.time() - t0) * 1000
                                    return CEGISResult(
                                        solved=True,
                                        prediction=prediction_test,
                                        transform_chain=[Transform(name="column_repeat", fn=lambda g: g, params={"period": period})],
                                        elapsed_ms=elapsed,
                                        method="composite",
                                    )
                        except Exception:
                            pass

        # 色マッピング学習（訓練ペアの入出力から直接マッピングを構築）
        result_colormap = self._try_learned_color_map(train, test_in, t0)
        if result_colormap.solved:
            return result_colormap

        # 分割+ブール演算（区切り線で左右/上下に分割してAND/OR/XOR）
        result_split = self._try_split_boolean(train, test_in, bg, t0)
        if result_split.solved:
            return result_split

        # 自己テンソル積（各セルをグリッド全体で置換）
        result_tensor = self._try_self_tensor(train, test_in, bg, t0)
        if result_tensor.solved:
            return result_tensor

        # 「ポイント→縦線拡張→パターン繰り返し」パターン
        result_expand = self._try_point_expand_repeat(train, test_in, bg, t0)
        if result_expand.solved:
            return result_expand

        return CEGISResult(solved=False)

    def _try_learned_color_map(
        self,
        train: list[tuple[Grid, Grid]],
        test_in: Grid,
        t0: float,
    ) -> CEGISResult:
        """訓練ペアから色マッピングを直接学習して適用"""
        # 全ペアで同一の色マッピングが成立するか確認
        # 同サイズの場合のみ
        if not all(inp.shape == out.shape for inp, out in train):
            return CEGISResult(solved=False)

        # 最初のペアからマッピングを学習
        inp0, out0 = train[0]
        mapping = {}
        for r in range(inp0.shape[0]):
            for c in range(inp0.shape[1]):
                src = int(inp0[r, c])
                dst = int(out0[r, c])
                if src in mapping:
                    if mapping[src] != dst:
                        return CEGISResult(solved=False)  # 矛盾
                else:
                    mapping[src] = dst

        if not mapping:
            return CEGISResult(solved=False)

        # 全ペアで検証
        for inp, out in train[1:]:
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    src = int(inp[r, c])
                    dst = int(out[r, c])
                    if src in mapping:
                        if mapping[src] != dst:
                            return CEGISResult(solved=False)
                    else:
                        mapping[src] = dst

        # テスト入力に適用
        pred = test_in.copy()
        for src, dst in mapping.items():
            pred[test_in == src] = dst

        elapsed = (time.time() - t0) * 1000
        return CEGISResult(
            solved=True,
            prediction=pred,
            transform_chain=[Transform(name="learned_color_map", fn=lambda g: g, params={"mapping": mapping})],
            elapsed_ms=elapsed,
            method="composite",
        )

    def _try_split_boolean(
        self,
        train: list[tuple[Grid, Grid]],
        test_in: Grid,
        bg: int,
        t0: float,
    ) -> CEGISResult:
        """
        区切り線（単色列/行）で左右/上下に分割し、
        ブール演算（AND）で出力を生成する。
        例: 0520fde7 — 色5の列で分割 → 左右のAND → 重なり=2
        """
        inp0, out0 = train[0]
        h, w = inp0.shape
        h_out, w_out = out0.shape

        # 垂直分割（列が全て同じ色 = 区切り線）
        for split_c in range(1, w - 1):
            col = inp0[:, split_c]
            if len(set(col.tolist())) == 1 and int(col[0]) != bg:
                separator_color = int(col[0])
                left = inp0[:, :split_c]
                right = inp0[:, split_c + 1:]

                if left.shape != right.shape:
                    continue
                if left.shape != (h_out, w_out):
                    continue

                # AND: 両方が非bgなら特定色(2)、そうでなければbg
                # まず訓練ペアで法則を確認
                fg_left = (left != bg)
                fg_right = (right != bg)
                overlap = fg_left & fg_right

                # 出力で overlap 位置の色を確認
                overlap_colors = set()
                for r in range(h_out):
                    for c in range(w_out):
                        if overlap[r, c]:
                            overlap_colors.add(int(out0[r, c]))

                if len(overlap_colors) != 1:
                    continue
                result_color = overlap_colors.pop()

                # 非overlap位置がbgか確認
                non_overlap_ok = True
                for r in range(h_out):
                    for c in range(w_out):
                        if not overlap[r, c] and out0[r, c] != bg:
                            non_overlap_ok = False
                            break

                if not non_overlap_ok:
                    continue

                # 全訓練ペアで検証
                all_ok = True
                for inp, out in train[1:]:
                    s_col = inp[:, split_c]
                    if len(set(s_col.tolist())) != 1 or int(s_col[0]) != separator_color:
                        all_ok = False
                        break
                    l = inp[:, :split_c]
                    r_part = inp[:, split_c + 1:]
                    if l.shape != r_part.shape or l.shape != out.shape:
                        all_ok = False
                        break
                    pred = np.full_like(out, bg)
                    ol = (l != bg) & (r_part != bg)
                    pred[ol] = result_color
                    if not np.array_equal(pred, out):
                        all_ok = False
                        break

                if all_ok:
                    # テストに適用
                    t_col = test_in[:, split_c]
                    t_left = test_in[:, :split_c]
                    t_right = test_in[:, split_c + 1:]
                    pred = np.full((t_left.shape[0], t_left.shape[1]), bg, dtype=np.int8)
                    ol = (t_left != bg) & (t_right != bg)
                    pred[ol] = result_color
                    elapsed = (time.time() - t0) * 1000
                    return CEGISResult(
                        solved=True,
                        prediction=pred,
                        transform_chain=[Transform(name="split_and", fn=lambda g: g, params={"split": split_c, "color": result_color})],
                        elapsed_ms=elapsed,
                        method="composite",
                    )

        return CEGISResult(solved=False)

    def _try_self_tensor(
        self,
        train: list[tuple[Grid, Grid]],
        test_in: Grid,
        bg: int,
        t0: float,
    ) -> CEGISResult:
        """
        自己テンソル積: 各セルの値に基づいてグリッド自身をタイル配置。
        非bg → 自身のコピー、bg → 空ブロック。
        例: 007bbfb7 — 3x3入力 → 9x9出力
        """
        inp0, out0 = train[0]
        h_in, w_in = inp0.shape
        h_out, w_out = out0.shape

        # 出力サイズ = 入力サイズの2乗か確認
        if h_out != h_in * h_in or w_out != w_in * w_in:
            return CEGISResult(solved=False)

        def _apply_tensor(inp: Grid) -> Grid:
            h, w = inp.shape
            out = np.full((h * h, w * w), bg, dtype=np.int8)
            for r in range(h):
                for c in range(w):
                    if inp[r, c] != bg:
                        # このセルに元グリッドのコピーを配置
                        out[r*h:(r+1)*h, c*w:(c+1)*w] = inp
                    else:
                        out[r*h:(r+1)*h, c*w:(c+1)*w] = bg
            return out

        # 全訓練ペアで検証
        for inp, out in train:
            pred = _apply_tensor(inp)
            if pred.shape != out.shape or not np.array_equal(pred, out):
                return CEGISResult(solved=False)

        # テストに適用
        prediction = _apply_tensor(test_in)
        elapsed = (time.time() - t0) * 1000
        return CEGISResult(
            solved=True,
            prediction=prediction,
            transform_chain=[Transform(name="self_tensor", fn=lambda g: g)],
            elapsed_ms=elapsed,
            method="composite",
        )

    def _try_point_expand_repeat(
        self,
        train: list[tuple[Grid, Grid]],
        test_in: Grid,
        bg: int,
        t0: float,
    ) -> CEGISResult:
        """
        入力のポイント（単一ピクセル）を:
        1. 列全体に拡張
        2. パターンを右方向に繰り返す
        """
        def _apply_rule(inp: Grid, target_w: int, target_h: int) -> Optional[Grid]:
            h, w = inp.shape
            # 非背景ピクセルを見つける
            points = []
            for r in range(h):
                for c in range(w):
                    if inp[r, c] != bg:
                        points.append((r, c, int(inp[r, c])))
            if len(points) < 1:
                return None

            # 各ポイントの列位置だけ取得
            col_colors = {}  # col -> color
            for _, c, color in points:
                col_colors[c] = color

            if len(col_colors) < 1:
                return None

            cols = sorted(col_colors.keys())

            # パターン周期を推定
            if len(cols) >= 2:
                intervals = [cols[i+1] - cols[i] for i in range(len(cols)-1)]
                if len(set(intervals)) == 1:
                    period = intervals[0]
                else:
                    period = cols[-1] - cols[0] + intervals[0] if intervals else target_w
            else:
                period = target_w  # 1ポイントの場合は繰り返さない

            # 出力グリッド生成
            out = np.full((target_h, target_w), bg, dtype=np.int8)

            # パターンの全幅 = 最後の列 - 最初の列 + 間隔（次パターンの開始まで）
            pattern_len = cols[-1] - cols[0] + period if len(cols) >= 2 else period

            # 各繰り返しオフセットでパターンを配置
            rep = 0
            while True:
                offset = rep * pattern_len
                placed_any = False
                for c_orig, color in col_colors.items():
                    c_target = c_orig + offset
                    if c_target >= target_w:
                        continue
                    if 0 <= c_target < target_w:
                        for r in range(target_h):
                            out[r, c_target] = color
                        placed_any = True
                if not placed_any:
                    break
                rep += 1
                if pattern_len <= 0:
                    break

            return out

        # 全訓練ペアで検証
        for inp, out in train:
            pred = _apply_rule(inp, out.shape[1], out.shape[0])
            if pred is None or not np.array_equal(pred, out):
                return CEGISResult(solved=False)

        # テスト入力に適用
        # テスト出力サイズ = テスト入力サイズ（ARC-AGIの多くの問題）
        h_test, w_test = test_in.shape
        # 訓練ペアから出力サイズの傾向を推定
        if all(inp.shape == out.shape for inp, out in train):
            pred_test = _apply_rule(test_in, w_test, h_test)
        else:
            # サイズが変わる場合は最初の訓練ペアの比率から推定
            inp0, out0 = train[0]
            h_ratio = out0.shape[0] / inp0.shape[0]
            w_ratio = out0.shape[1] / inp0.shape[1]
            pred_test = _apply_rule(test_in, int(w_test * w_ratio), int(h_test * h_ratio))

        if pred_test is not None:
            elapsed = (time.time() - t0) * 1000
            return CEGISResult(
                solved=True,
                prediction=pred_test,
                transform_chain=[Transform(name="point_expand_repeat", fn=lambda g: g)],
                elapsed_ms=elapsed,
                method="composite",
            )
        return CEGISResult(solved=False)

    def _apply_column_repeat(
        self,
        inp: Grid,
        fg_cols: list[tuple[int, np.ndarray]],
        period: int,
        target_w: int,
        target_h: int,
        bg: int,
        test_in: Optional[Grid],
    ) -> Optional[Grid]:
        """列パターンを繰り返して出力グリッドを生成"""
        if not fg_cols:
            return None
        source = test_in if test_in is not None else inp
        h = source.shape[0]
        out = np.full((target_h, target_w), bg, dtype=np.int8)

        # source から前景列を取得
        src_fg_cols = []
        for c in range(source.shape[1]):
            if np.any(source[:, c] != bg):
                src_fg_cols.append((c, source[:, c].copy()))

        if not src_fg_cols:
            return None

        # 元のパターンの開始位置
        start_col = src_fg_cols[0][0]

        # 繰り返し
        for rep_start in range(start_col, target_w, period):
            for offset, col_data in enumerate(src_fg_cols):
                rel_offset = col_data[0] - src_fg_cols[0][0]
                target_col = rep_start + rel_offset
                if 0 <= target_col < target_w:
                    out[:min(h, target_h), target_col] = col_data[1][:target_h]

        return out


def solve_arc_task(task: dict) -> list[list[int]]:
    """
    ARC タスク（JSON形式）を解く。

    Args:
        task: {"train": [...], "test": [{"input": [[...]]}]}

    Returns:
        予測出力グリッド (2D list)、解けない場合は空リスト
    """
    cegis = ARCCEGISLoop(max_chain_len=2, time_limit_ms=30000)
    train_pairs = task["train"]
    test_input = task["test"][0]["input"]

    result = cegis.solve(train_pairs, test_input)

    if result.solved and result.prediction is not None:
        return result.prediction.tolist()
    return []
