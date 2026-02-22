"""
grid_ir.py — ARC グリッドの内部表現

グリッドを構造的に分解してIRに変換する。
HLEの RuleBasedDecomposer に相当。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from collections import Counter


@dataclass
class GridObject:
    """グリッド上の連結オブジェクト"""
    obj_id: int
    color: int
    cells: list[tuple[int, int]]  # (row, col)
    bbox: tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    shape: np.ndarray  # bboxに収まるマスク (bool)

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def centroid(self) -> tuple[float, float]:
        rs = [c[0] for c in self.cells]
        cs = [c[1] for c in self.cells]
        return (sum(rs) / len(rs), sum(cs) / len(cs))

    def is_rectangular(self) -> bool:
        return self.size == self.width * self.height

    def is_line(self) -> bool:
        return self.width == 1 or self.height == 1

    def is_single_pixel(self) -> bool:
        return self.size == 1


@dataclass
class Symmetry:
    """検出された対称性"""
    kind: str  # "horizontal" | "vertical" | "rotate_90" | "rotate_180" | "diagonal"
    axis: Optional[int] = None  # 対称軸の位置
    confidence: float = 1.0


@dataclass
class Pattern:
    """検出されたパターン"""
    kind: str  # "repeat_h" | "repeat_v" | "stripe" | "checkerboard" | "border" | "fill"
    params: dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class GridIR:
    """グリッドの内部表現 — Decomposerの出力"""
    grid: np.ndarray                              # 元グリッド
    height: int
    width: int
    colors: list[int]                             # 使用色リスト (背景含む)
    background_color: int                         # 背景色 (最頻色)
    foreground_colors: list[int]                  # 非背景色
    objects: list[GridObject] = field(default_factory=list)
    symmetries: list[Symmetry] = field(default_factory=list)
    patterns: list[Pattern] = field(default_factory=list)
    color_counts: dict[int, int] = field(default_factory=dict)
    non_zero_count: int = 0

    def summary(self) -> dict:
        return {
            "size": f"{self.height}x{self.width}",
            "colors": self.colors,
            "bg": self.background_color,
            "objects": len(self.objects),
            "symmetries": [s.kind for s in self.symmetries],
            "patterns": [p.kind for p in self.patterns],
        }


class GridDecomposer:
    """
    グリッドを構造分解してGridIRを生成する。

    ARC版の RuleBasedDecomposer。
    LLM不使用、完全ルールベース。
    """

    def decompose(self, grid: list[list[int]]) -> GridIR:
        """グリッド(2D list) → GridIR"""
        arr = np.array(grid, dtype=np.int8)
        h, w = arr.shape

        # 色分析
        color_counts = Counter(arr.flatten().tolist())
        colors = sorted(color_counts.keys())
        background_color = color_counts.most_common(1)[0][0]
        foreground_colors = [c for c in colors if c != background_color]
        non_zero = int(np.count_nonzero(arr != background_color))

        # オブジェクト検出（連結成分）
        objects = self._find_objects(arr, background_color)

        # 対称性検出
        symmetries = self._detect_symmetries(arr)

        # パターン検出
        patterns = self._detect_patterns(arr, background_color, objects)

        return GridIR(
            grid=arr,
            height=h,
            width=w,
            colors=colors,
            background_color=background_color,
            foreground_colors=foreground_colors,
            objects=objects,
            symmetries=symmetries,
            patterns=patterns,
            color_counts=dict(color_counts),
            non_zero_count=non_zero,
        )

    def _find_objects(self, grid: np.ndarray, bg: int) -> list[GridObject]:
        """4-連結の連結成分を検出"""
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        objects = []
        obj_id = 0

        for r in range(h):
            for c in range(w):
                if visited[r, c] or grid[r, c] == bg:
                    continue
                # BFS
                color = int(grid[r, c])
                cells = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                min_r = min(c[0] for c in cells)
                max_r = max(c[0] for c in cells)
                min_c = min(c[1] for c in cells)
                max_c = max(c[1] for c in cells)
                shape = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
                for cr, cc in cells:
                    shape[cr - min_r, cc - min_c] = True

                objects.append(GridObject(
                    obj_id=obj_id,
                    color=color,
                    cells=cells,
                    bbox=(min_r, min_c, max_r, max_c),
                    shape=shape,
                ))
                obj_id += 1

        return objects

    def _detect_symmetries(self, grid: np.ndarray) -> list[Symmetry]:
        """対称性を検出"""
        syms = []

        # 水平対称（上下反転）
        if np.array_equal(grid, grid[::-1, :]):
            syms.append(Symmetry(kind="horizontal", axis=grid.shape[0] // 2))

        # 垂直対称（左右反転）
        if np.array_equal(grid, grid[:, ::-1]):
            syms.append(Symmetry(kind="vertical", axis=grid.shape[1] // 2))

        # 180度回転対称
        if np.array_equal(grid, np.rot90(grid, 2)):
            syms.append(Symmetry(kind="rotate_180"))

        # 90度回転対称（正方形のみ）
        if grid.shape[0] == grid.shape[1]:
            if np.array_equal(grid, np.rot90(grid, 1)):
                syms.append(Symmetry(kind="rotate_90"))

            # 対角対称
            if np.array_equal(grid, grid.T):
                syms.append(Symmetry(kind="diagonal"))

        return syms

    def _detect_patterns(self, grid: np.ndarray, bg: int, objects: list[GridObject]) -> list[Pattern]:
        """パターンを検出"""
        patterns = []
        h, w = grid.shape

        # 縦縞パターン: 各列が同一
        col_uniform = all(len(set(grid[:, c].tolist())) == 1 for c in range(w))
        if col_uniform and w > 1:
            patterns.append(Pattern(kind="stripe_v"))

        # 横縞パターン: 各行が同一
        row_uniform = all(len(set(grid[r, :].tolist())) == 1 for r in range(h))
        if row_uniform and h > 1:
            patterns.append(Pattern(kind="stripe_h"))

        # ボーダーパターン: 外周が同一色で内部が異なる
        if h >= 3 and w >= 3:
            border_cells = (
                list(grid[0, :]) + list(grid[-1, :]) +
                list(grid[1:-1, 0]) + list(grid[1:-1, -1])
            )
            if len(set(border_cells)) == 1 and border_cells[0] != bg:
                patterns.append(Pattern(kind="border", params={"color": int(border_cells[0])}))

        # 繰り返しパターン（水平タイル）
        for tile_w in range(1, w // 2 + 1):
            if w % tile_w == 0:
                tile = grid[:, :tile_w]
                is_repeat = all(
                    np.array_equal(grid[:, i:i+tile_w], tile)
                    for i in range(tile_w, w, tile_w)
                )
                if is_repeat:
                    patterns.append(Pattern(kind="repeat_h", params={"period": tile_w}))
                    break

        # 繰り返しパターン（垂直タイル）
        for tile_h in range(1, h // 2 + 1):
            if h % tile_h == 0:
                tile = grid[:tile_h, :]
                is_repeat = all(
                    np.array_equal(grid[i:i+tile_h, :], tile)
                    for i in range(tile_h, h, tile_h)
                )
                if is_repeat:
                    patterns.append(Pattern(kind="repeat_v", params={"period": tile_h}))
                    break

        # チェッカーボードパターン
        if h >= 2 and w >= 2:
            is_checker = True
            c0 = int(grid[0, 0])
            c1 = int(grid[0, 1]) if w > 1 else c0
            if c0 != c1:
                for r in range(h):
                    for c in range(w):
                        expected = c0 if (r + c) % 2 == 0 else c1
                        if grid[r, c] != expected:
                            is_checker = False
                            break
                    if not is_checker:
                        break
                if is_checker:
                    patterns.append(Pattern(kind="checkerboard", params={"colors": [c0, c1]}))

        # 単一オブジェクト
        if len(objects) == 1:
            patterns.append(Pattern(kind="single_object"))
        elif len(objects) == 0:
            patterns.append(Pattern(kind="empty"))

        return patterns


def decompose_pair(input_grid: list[list[int]], output_grid: list[list[int]]) -> dict:
    """
    入力/出力ペアを分解して差分を分析する。
    CEGIS候補生成のヒントに使う。
    """
    dec = GridDecomposer()
    in_ir = dec.decompose(input_grid)
    out_ir = dec.decompose(output_grid)

    in_arr = np.array(input_grid, dtype=np.int8)
    out_arr = np.array(output_grid, dtype=np.int8)

    # サイズ変化
    size_changed = in_arr.shape != out_arr.shape

    # 色変化
    in_colors = set(in_ir.colors)
    out_colors = set(out_ir.colors)
    colors_added = out_colors - in_colors
    colors_removed = in_colors - out_colors

    # オブジェクト数変化
    obj_count_change = len(out_ir.objects) - len(in_ir.objects)

    # 同サイズなら差分マスク
    diff_mask = None
    diff_count = 0
    if not size_changed:
        diff_mask = (in_arr != out_arr)
        diff_count = int(diff_mask.sum())

    return {
        "input_ir": in_ir,
        "output_ir": out_ir,
        "size_changed": size_changed,
        "in_size": in_arr.shape,
        "out_size": out_arr.shape,
        "colors_added": colors_added,
        "colors_removed": colors_removed,
        "obj_count_change": obj_count_change,
        "diff_count": diff_count,
        "diff_mask": diff_mask,
    }
