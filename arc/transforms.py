"""
transforms.py — ARC 原子的変換ピース

HLEの PieceDB に相当。
各変換は Grid → Grid の純粋関数。
変換チェーンを組み合わせてARC問題を解く。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


Grid = np.ndarray  # (H, W) int8


@dataclass
class Transform:
    """原子的変換"""
    name: str
    fn: Callable[[Grid], Grid]
    params: dict = None
    description: str = ""

    def apply(self, grid: Grid) -> Grid:
        return self.fn(grid)

    def __repr__(self):
        if self.params:
            p = ",".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.name}({p})"
        return self.name


# ══════════════════════════════════════════════════════════════
# 幾何変換
# ══════════════════════════════════════════════════════════════

def rotate_90(grid: Grid) -> Grid:
    return np.rot90(grid, -1)  # 時計回り90度

def rotate_180(grid: Grid) -> Grid:
    return np.rot90(grid, 2)

def rotate_270(grid: Grid) -> Grid:
    return np.rot90(grid, 1)  # 反時計回り90度

def flip_horizontal(grid: Grid) -> Grid:
    return grid[:, ::-1]

def flip_vertical(grid: Grid) -> Grid:
    return grid[::-1, :]

def transpose(grid: Grid) -> Grid:
    return grid.T

def flip_diagonal(grid: Grid) -> Grid:
    """副対角線で反転"""
    return np.rot90(grid.T, 2)


# ══════════════════════════════════════════════════════════════
# 色変換
# ══════════════════════════════════════════════════════════════

def make_color_swap(a: int, b: int) -> Transform:
    def _swap(grid: Grid) -> Grid:
        out = grid.copy()
        out[grid == a] = b
        out[grid == b] = a
        return out
    return Transform(name="color_swap", fn=_swap, params={"a": a, "b": b})

def make_color_map(mapping: dict[int, int]) -> Transform:
    def _map(grid: Grid) -> Grid:
        out = grid.copy()
        for src, dst in mapping.items():
            out[grid == src] = dst
        return out
    return Transform(name="color_map", fn=_map, params={"mapping": mapping})

def make_fill_color(color: int) -> Transform:
    def _fill(grid: Grid) -> Grid:
        return np.full_like(grid, color)
    return Transform(name="fill_color", fn=_fill, params={"color": color})

def make_replace_color(old: int, new: int) -> Transform:
    def _replace(grid: Grid) -> Grid:
        out = grid.copy()
        out[grid == old] = new
        return out
    return Transform(name="replace_color", fn=_replace, params={"old": old, "new": new})


# ══════════════════════════════════════════════════════════════
# サイズ変換
# ══════════════════════════════════════════════════════════════

def make_scale(factor: int) -> Transform:
    def _scale(grid: Grid) -> Grid:
        return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)
    return Transform(name="scale", fn=_scale, params={"factor": factor})

def make_crop(r1: int, c1: int, r2: int, c2: int) -> Transform:
    def _crop(grid: Grid) -> Grid:
        return grid[r1:r2+1, c1:c2+1].copy()
    return Transform(name="crop", fn=_crop, params={"r1": r1, "c1": c1, "r2": r2, "c2": c2})

def make_pad(top: int, bottom: int, left: int, right: int, color: int = 0) -> Transform:
    def _pad(grid: Grid) -> Grid:
        return np.pad(grid, ((top, bottom), (left, right)), constant_values=color)
    return Transform(name="pad", fn=_pad, params={"top": top, "bottom": bottom, "left": left, "right": right, "color": color})


# ══════════════════════════════════════════════════════════════
# タイル・繰り返し変換
# ══════════════════════════════════════════════════════════════

def make_tile(rows: int, cols: int) -> Transform:
    def _tile(grid: Grid) -> Grid:
        return np.tile(grid, (rows, cols))
    return Transform(name="tile", fn=_tile, params={"rows": rows, "cols": cols})

def make_repeat_pattern_h(target_width: int) -> Transform:
    """水平方向にパターンを繰り返して target_width にする"""
    def _repeat(grid: Grid) -> Grid:
        h, w = grid.shape
        if w == 0:
            return grid
        repeats = (target_width + w - 1) // w
        tiled = np.tile(grid, (1, repeats))
        return tiled[:, :target_width]
    return Transform(name="repeat_h", fn=_repeat, params={"target_width": target_width})

def make_repeat_pattern_v(target_height: int) -> Transform:
    """垂直方向にパターンを繰り返して target_height にする"""
    def _repeat(grid: Grid) -> Grid:
        h, w = grid.shape
        if h == 0:
            return grid
        repeats = (target_height + h - 1) // h
        tiled = np.tile(grid, (repeats, 1))
        return tiled[:target_height, :]
    return Transform(name="repeat_v", fn=_repeat, params={"target_height": target_height})


# ══════════════════════════════════════════════════════════════
# オブジェクト操作
# ══════════════════════════════════════════════════════════════

def make_gravity(direction: str = "down", bg: int = 0) -> Transform:
    """オブジェクトを指定方向に落とす (gravity)"""
    def _gravity(grid: Grid) -> Grid:
        out = grid.copy()
        h, w = out.shape
        if direction == "down":
            for c in range(w):
                col = [out[r, c] for r in range(h) if out[r, c] != bg]
                for r in range(h):
                    out[r, c] = bg
                for i, v in enumerate(col):
                    out[h - len(col) + i, c] = v
        elif direction == "up":
            for c in range(w):
                col = [out[r, c] for r in range(h) if out[r, c] != bg]
                for r in range(h):
                    out[r, c] = bg
                for i, v in enumerate(col):
                    out[i, c] = v
        elif direction == "left":
            for r in range(h):
                row = [out[r, c] for c in range(w) if out[r, c] != bg]
                for c in range(w):
                    out[r, c] = bg
                for i, v in enumerate(row):
                    out[r, i] = v
        elif direction == "right":
            for r in range(h):
                row = [out[r, c] for c in range(w) if out[r, c] != bg]
                for c in range(w):
                    out[r, c] = bg
                for i, v in enumerate(row):
                    out[r, w - len(row) + i] = v
        return out
    return Transform(name="gravity", fn=_gravity, params={"direction": direction, "bg": bg})


def fill_enclosed(grid: Grid, bg: int = 0, fill_color: int = -1) -> Grid:
    """背景色で囲まれた領域を塗りつぶす"""
    h, w = grid.shape
    out = grid.copy()
    # BFSで外部背景を特定
    visited = np.zeros((h, w), dtype=bool)
    queue = []
    for r in range(h):
        for c in [0, w-1]:
            if grid[r, c] == bg and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True
    for c in range(w):
        for r in [0, h-1]:
            if grid[r, c] == bg and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True

    while queue:
        cr, cc = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == bg:
                visited[nr, nc] = True
                queue.append((nr, nc))

    # 未訪問の背景セル = 囲まれた領域
    fc = fill_color if fill_color >= 0 else _most_common_fg(grid, bg)
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg and not visited[r, c]:
                out[r, c] = fc
    return out


def _most_common_fg(grid: Grid, bg: int) -> int:
    from collections import Counter
    counts = Counter(grid.flatten().tolist())
    counts.pop(bg, None)
    if counts:
        return counts.most_common(1)[0][0]
    return 1


# ══════════════════════════════════════════════════════════════
# ブール演算
# ══════════════════════════════════════════════════════════════

def make_boolean_op(op: str, grid_b: Grid, bg: int = 0) -> Transform:
    """2つのグリッド間のブール演算 (and/or/xor)"""
    def _bool_op(grid_a: Grid) -> Grid:
        mask_a = (grid_a != bg)
        mask_b = (grid_b != bg)
        if op == "and":
            mask = mask_a & mask_b
        elif op == "or":
            mask = mask_a | mask_b
        elif op == "xor":
            mask = mask_a ^ mask_b
        elif op == "not":
            mask = ~mask_a
        else:
            mask = mask_a
        out = np.full_like(grid_a, bg)
        # ANDの場合はgrid_aの色を維持
        out[mask] = grid_a[mask] if mask_a[mask].any() else grid_b[mask]
        return out
    return Transform(name=f"bool_{op}", fn=_bool_op, params={"op": op})


# ══════════════════════════════════════════════════════════════
# 変換ピースDB（静的レジストリ）
# ══════════════════════════════════════════════════════════════

# 固定変換（パラメータ不要）
FIXED_TRANSFORMS: list[Transform] = [
    Transform(name="rotate_90", fn=rotate_90, description="時計回り90度"),
    Transform(name="rotate_180", fn=rotate_180, description="180度回転"),
    Transform(name="rotate_270", fn=rotate_270, description="反時計回り90度"),
    Transform(name="flip_h", fn=flip_horizontal, description="左右反転"),
    Transform(name="flip_v", fn=flip_vertical, description="上下反転"),
    Transform(name="transpose", fn=transpose, description="転置"),
    Transform(name="identity", fn=lambda g: g.copy(), description="恒等変換"),
]


def get_parametric_transforms(
    in_grid: Grid,
    out_grid: Optional[Grid] = None,
    bg: int = 0,
) -> list[Transform]:
    """
    入力/出力グリッドの特徴からパラメトリック変換の候補を生成する。
    """
    transforms = []
    h_in, w_in = in_grid.shape
    colors_in = set(int(c) for c in np.unique(in_grid))
    fg_colors = [c for c in colors_in if c != bg]

    # 色変換
    for i, c1 in enumerate(fg_colors):
        for c2 in fg_colors[i+1:]:
            transforms.append(make_color_swap(c1, c2))
    for c in fg_colors:
        transforms.append(make_replace_color(c, bg))
        transforms.append(make_replace_color(bg, c))

    # スケール (2x, 3x)
    for factor in [2, 3]:
        transforms.append(make_scale(factor))

    # タイル
    for r in [1, 2, 3]:
        for c in [1, 2, 3]:
            if r == 1 and c == 1:
                continue
            transforms.append(make_tile(r, c))

    # gravity
    for d in ["down", "up", "left", "right"]:
        transforms.append(make_gravity(d, bg))

    # fill_enclosed
    transforms.append(Transform(
        name="fill_enclosed",
        fn=lambda g, _bg=bg: fill_enclosed(g, _bg),
        params={"bg": bg},
    ))

    # Extended transforms
    transforms.append(make_flood_fill_enclosed(bg))
    transforms.append(make_sort_rows(bg, "color_count"))
    transforms.append(make_sort_rows(bg, "color_count_desc"))
    transforms.append(make_sort_cols(bg, "color_count"))
    transforms.append(make_sort_cols(bg, "color_count_desc"))
    transforms.append(make_mirror_complete('h'))
    transforms.append(make_mirror_complete('v'))
    transforms.append(make_extract_largest_object(bg))
    transforms.append(make_boolean_and(bg))
    transforms.append(make_boolean_or(bg))
    transforms.append(make_boolean_xor(bg))
    transforms.append(make_dedup_rows())
    transforms.append(make_dedup_cols())
    for c in fg_colors:
        transforms.append(make_remove_color(c, bg))

    # 出力サイズが分かっている場合のリサイズ系
    if out_grid is not None:
        h_out, w_out = out_grid.shape
        if w_out != w_in:
            transforms.append(make_repeat_pattern_h(w_out))
        if h_out != h_in:
            transforms.append(make_repeat_pattern_v(h_out))
        # crop候補
        if h_out <= h_in and w_out <= w_in:
            for r1 in range(h_in - h_out + 1):
                for c1 in range(w_in - w_out + 1):
                    transforms.append(make_crop(r1, c1, r1 + h_out - 1, c1 + w_out - 1))

    return transforms


# ============================================================
# Extended transforms for ARC-AGI-2
# ============================================================

def make_flood_fill_enclosed(bg: int) -> Transform:
    """Fill enclosed bg regions with the surrounding color."""
    def fn(grid: Grid) -> Grid:
        from scipy import ndimage
        arr = grid.copy() if isinstance(grid, np.ndarray) else np.array(grid, dtype=np.int8)
        bg_mask = arr == bg
        labeled, n = ndimage.label(bg_mask)
        h, w = arr.shape
        border_labels = set()
        for r in range(h):
            for c in range(w):
                if (r == 0 or r == h-1 or c == 0 or c == w-1) and bg_mask[r, c]:
                    border_labels.add(labeled[r, c])
        for lbl in range(1, n+1):
            if lbl in border_labels:
                continue
            mask = labeled == lbl
            # Find surrounding color
            dilated = ndimage.binary_dilation(mask)
            border = dilated & ~mask & ~bg_mask
            if border.any():
                colors = arr[border]
                fill_color = int(np.bincount(colors).argmax())
                arr[mask] = fill_color
        return arr
    return Transform(name="flood_fill_enclosed", fn=fn, params={"bg": bg})


def make_sort_rows(bg: int, key: str = "color_count") -> Transform:
    """Sort rows by a key."""
    def fn(grid: Grid) -> Grid:
        arr = grid.copy() if isinstance(grid, np.ndarray) else np.array(grid, dtype=np.int8)
        rows = list(arr)
        if key == "color_count":
            rows.sort(key=lambda r: sum(1 for v in r if v != bg))
        elif key == "color_count_desc":
            rows.sort(key=lambda r: -sum(1 for v in r if v != bg))
        return np.array(rows, dtype=np.int8)
    return Transform(name=f"sort_rows_{key}", fn=fn, params={"bg": bg, "key": key})


def make_sort_cols(bg: int, key: str = "color_count") -> Transform:
    """Sort columns by a key."""
    def fn(grid: Grid) -> Grid:
        arr = grid.copy() if isinstance(grid, np.ndarray) else np.array(grid, dtype=np.int8)
        cols = list(arr.T)
        if key == "color_count":
            cols.sort(key=lambda c: sum(1 for v in c if v != bg))
        elif key == "color_count_desc":
            cols.sort(key=lambda c: -sum(1 for v in c if v != bg))
        return np.array(cols, dtype=np.int8).T
    return Transform(name=f"sort_cols_{key}", fn=fn, params={"bg": bg, "key": key})


def make_mirror_complete(axis: str) -> Transform:
    """Complete partial symmetry by mirroring."""
    def fn(grid: Grid) -> Grid:
        arr = grid.copy() if isinstance(grid, np.ndarray) else np.array(grid, dtype=np.int8)
        h, w = arr.shape
        result = arr.copy()
        if axis == 'h':
            flipped = arr[:, ::-1]
        elif axis == 'v':
            flipped = arr[::-1, :]
        else:
            return arr
        # Fill bg cells with mirror
        bg = int(np.bincount(arr.flatten()).argmax())
        mask = result == bg
        result[mask] = flipped[mask]
        return result
    return Transform(name=f"mirror_complete_{axis}", fn=fn)


def make_extract_largest_object(bg: int) -> Transform:
    """Extract the largest connected component."""
    def fn(grid: Grid) -> Grid:
        from scipy import ndimage
        arr = grid.copy() if isinstance(grid, np.ndarray) else np.array(grid, dtype=np.int8)
        labeled, n = ndimage.label(arr != bg)
        if n == 0:
            return arr
        sizes = {i: int(np.sum(labeled == i)) for i in range(1, n+1)}
        largest = max(sizes, key=sizes.get)
        mask = labeled == largest
        rows, cols = np.where(mask)
        r0, c0 = rows.min(), cols.min()
        r1, c1 = rows.max(), cols.max()
        result = np.full((r1-r0+1, c1-c0+1), bg, dtype=np.int8)
        for r, c in zip(rows, cols):
            result[r-r0, c-c0] = arr[r, c]
        return result
    return Transform(name="extract_largest_object", fn=fn, params={"bg": bg})


def make_remove_color(color: int, bg: int) -> Transform:
    """Remove all cells of a specific color (set to bg)."""
    def fn(grid: Grid) -> Grid:
        arr = grid.copy() if isinstance(grid, np.ndarray) else np.array(grid, dtype=np.int8)
        arr[arr == color] = bg
        return arr
    return Transform(name=f"remove_color_{color}", fn=fn, params={"color": color, "bg": bg})


def make_boolean_and(bg: int) -> Transform:
    """Split grid in half horizontally and AND the two halves."""
    def fn(grid: Grid) -> Grid:
        arr = grid.copy() if isinstance(grid, np.ndarray) else np.array(grid, dtype=np.int8)
        h, w = arr.shape
        if h % 2 != 0:
            return arr
        top = arr[:h//2]
        bot = arr[h//2:]
        result = np.full_like(top, bg)
        result[(top != bg) & (bot != bg)] = top[(top != bg) & (bot != bg)]
        return result
    return Transform(name="boolean_and_h", fn=fn, params={"bg": bg})


def make_boolean_or(bg: int) -> Transform:
    """Split grid in half horizontally and OR the two halves."""
    def fn(grid: Grid) -> Grid:
        arr = grid.copy() if isinstance(grid, np.ndarray) else np.array(grid, dtype=np.int8)
        h, w = arr.shape
        if h % 2 != 0:
            return arr
        top = arr[:h//2]
        bot = arr[h//2:]
        result = top.copy()
        result[top == bg] = bot[top == bg]
        return result
    return Transform(name="boolean_or_h", fn=fn, params={"bg": bg})


def make_boolean_xor(bg: int) -> Transform:
    """Split grid in half horizontally and XOR the two halves."""
    def fn(grid: Grid) -> Grid:
        arr = grid.copy() if isinstance(grid, np.ndarray) else np.array(grid, dtype=np.int8)
        h, w = arr.shape
        if h % 2 != 0:
            return arr
        top = arr[:h//2]
        bot = arr[h//2:]
        result = np.full_like(top, bg)
        # XOR: cells that differ
        result[(top != bg) & (bot == bg)] = top[(top != bg) & (bot == bg)]
        result[(top == bg) & (bot != bg)] = bot[(top == bg) & (bot != bg)]
        return result
    return Transform(name="boolean_xor_h", fn=fn, params={"bg": bg})


def make_dedup_rows() -> Transform:
    """Remove duplicate rows."""
    def fn(grid: Grid) -> Grid:
        arr = grid.copy() if isinstance(grid, np.ndarray) else np.array(grid, dtype=np.int8)
        seen = set()
        rows = []
        for r in arr:
            key = tuple(r)
            if key not in seen:
                seen.add(key)
                rows.append(r)
        return np.array(rows, dtype=np.int8) if rows else arr
    return Transform(name="dedup_rows", fn=fn)


def make_dedup_cols() -> Transform:
    """Remove duplicate columns."""
    def fn(grid: Grid) -> Grid:
        arr = grid.copy() if isinstance(grid, np.ndarray) else np.array(grid, dtype=np.int8)
        seen = set()
        cols = []
        for c in arr.T:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                cols.append(c)
        return np.array(cols, dtype=np.int8).T if cols else arr
    return Transform(name="dedup_cols", fn=fn)
