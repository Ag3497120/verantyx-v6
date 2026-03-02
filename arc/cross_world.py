"""
arc/cross_world.py — CrossWorld: 6軸Cross全領域表現 + 操作エンジン

kofdai設計:
  入力/出力それぞれをCrossWorld（色レイヤー × 6軸記述子 × 連結塊）に変換。
  差分を「透明Cross」として表現し、操作ルールを学習。

CrossWorld:
  - color_layers[c] = bool mask (H×W)  — 色cのセルがどこにあるか
  - chunks[c] = List[Chunk]            — 色cの連結塊
  - cross6[r,c] = (8,) int16           — 8方向run length
  - boundary[r,c] = (4,) int16         — 4方向の境界距離
  - ray_color[r,c] = (4,) int8         — 4方向レイの最初の非bg色

TransparentCross（差分表現）:
  - added: 入力にない → 出力にある セル（透明Crossの「印」）
  - removed: 入力にある → 出力にない セル
  - recolored: 入力と出力で色が違うセル
  - unchanged: 入力と出力で同じセル

操作（Crossベース）:
  copy, move, fill, extend, mirror, gravity, stamp, connect, recolor,
  flood_fill, line_draw, pattern_complete, conditional_recolor
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Set, FrozenSet
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────
# Chunk: 同色連結領域
# ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    color: int
    cells: FrozenSet[Tuple[int, int]]
    is_bg: bool = False

    def __post_init__(self):
        if self.cells:
            rows = [r for r, c in self.cells]
            cols = [c for r, c in self.cells]
            self._r0, self._c0 = min(rows), min(cols)
            self._r1, self._c1 = max(rows), max(cols)
        else:
            self._r0 = self._c0 = self._r1 = self._c1 = 0
        self._shape_sig = None

    @property
    def size(self): return len(self.cells)
    @property
    def bbox(self): return (self._r0, self._c0, self._r1, self._c1)
    @property
    def bh(self): return self._r1 - self._r0 + 1
    @property
    def bw(self): return self._c1 - self._c0 + 1
    @property
    def center(self):
        rs = [r for r, c in self.cells]
        cs = [c for r, c in self.cells]
        return (sum(rs) / len(rs), sum(cs) / len(cs))

    @property
    def shape_sig(self) -> Tuple:
        if self._shape_sig is None:
            self._shape_sig = tuple(sorted(
                (r - self._r0, c - self._c0) for r, c in self.cells
            ))
        return self._shape_sig

    def translate(self, dr, dc):
        new_cells = frozenset((r + dr, c + dc) for r, c in self.cells)
        return Chunk(self.color, new_cells, self.is_bg)

    def recolor(self, new_color):
        return Chunk(new_color, self.cells, self.is_bg)

    def flip_h(self):
        new_cells = frozenset(
            (r, self._c0 + self._c1 - c) for r, c in self.cells
        )
        return Chunk(self.color, new_cells, self.is_bg)

    def flip_v(self):
        new_cells = frozenset(
            (self._r0 + self._r1 - r, c) for r, c in self.cells
        )
        return Chunk(self.color, new_cells, self.is_bg)

    def rot90(self):
        new_cells = frozenset(
            (self._r0 + (c - self._c0), self._c0 + (self.bh - 1 - (r - self._r0)))
            for r, c in self.cells
        )
        return Chunk(self.color, new_cells, self.is_bg)

    def rot180(self):
        new_cells = frozenset(
            (self._r0 + self._r1 - r, self._c0 + self._c1 - c)
            for r, c in self.cells
        )
        return Chunk(self.color, new_cells, self.is_bg)

    def to_subgrid(self, bg=0):
        sg = np.full((self.bh, self.bw), bg, dtype=np.int8)
        for r, c in self.cells:
            sg[r - self._r0, c - self._c0] = self.color
        return sg

    def __hash__(self):
        return hash((self.color, self.cells))

    def __eq__(self, other):
        return self.color == other.color and self.cells == other.cells


# ──────────────────────────────────────────────────────────────
# CrossWorld: グリッドの完全Cross表現
# ──────────────────────────────────────────────────────────────

class CrossWorld:
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.g = np.array(grid, dtype=np.int8)
        self.h, self.w = self.g.shape
        counts = Counter(int(x) for x in self.g.flatten())
        self.bg = counts.most_common(1)[0][0]
        self.colors = sorted(set(int(x) for x in self.g.flatten()))
        self.fg_colors = [c for c in self.colors if c != self.bg]

        # 色レイヤー
        self.color_layers: Dict[int, np.ndarray] = {}
        for c in self.colors:
            self.color_layers[c] = (self.g == c)

        # 連結塊
        self.chunks: Dict[int, List[Chunk]] = {}
        self.all_chunks: List[Chunk] = []
        self.fg_chunks: List[Chunk] = []
        self.bg_chunks: List[Chunk] = []
        for c in self.colors:
            labeled, n = scipy_label(self.g == c)
            clist = []
            for i in range(1, n + 1):
                cells = frozenset(zip(*np.where(labeled == i)))
                ch = Chunk(c, cells, is_bg=(c == self.bg))
                clist.append(ch)
            clist.sort(key=lambda ch: -ch.size)
            self.chunks[c] = clist
            self.all_chunks.extend(clist)
            if c == self.bg:
                self.bg_chunks.extend(clist)
            else:
                self.fg_chunks.extend(clist)
        self.fg_chunks.sort(key=lambda ch: -ch.size)

        # 6軸Cross (8方向 run length)
        self.cross6 = self._compute_runs()

        # 4方向レイ (最初に出会う非bg色, 距離)
        self.ray_color = np.full((self.h, self.w, 4), -1, dtype=np.int8)
        self.ray_dist = np.zeros((self.h, self.w, 4), dtype=np.int16)
        self._compute_rays()

        # 近傍情報
        self.nb8_colors = np.full((self.h, self.w, 8), -1, dtype=np.int8)
        self._compute_nb8()

    def _compute_runs(self):
        DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
        runs = np.zeros((self.h, self.w, 8), dtype=np.int16)
        g = self.g
        h, w = self.h, self.w
        for d, (dr, dc) in enumerate(DIRS):
            for r in range(h):
                for c in range(w):
                    color = g[r, c]
                    n = 0
                    r2, c2 = r + dr, c + dc
                    while 0 <= r2 < h and 0 <= c2 < w and g[r2, c2] == color:
                        n += 1; r2 += dr; c2 += dc
                    runs[r, c, d] = n
        return runs

    def _compute_rays(self):
        DIRS = [(-1,0),(1,0),(0,-1),(0,1)]
        g, h, w, bg = self.g, self.h, self.w, self.bg
        for d, (dr, dc) in enumerate(DIRS):
            for r in range(h):
                for c in range(w):
                    r2, c2 = r + dr, c + dc
                    dist = 1
                    while 0 <= r2 < h and 0 <= c2 < w:
                        if g[r2, c2] != bg:
                            self.ray_color[r, c, d] = g[r2, c2]
                            self.ray_dist[r, c, d] = dist
                            break
                        r2 += dr; c2 += dc; dist += 1

    def _compute_nb8(self):
        DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for d, (dr, dc) in enumerate(DIRS):
            for r in range(self.h):
                for c in range(self.w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.h and 0 <= nc < self.w:
                        self.nb8_colors[r, c, d] = self.g[nr, nc]

    def cell_features(self, r, c) -> Dict:
        """1セルの完全特徴量"""
        color = int(self.g[r, c])
        is_bg = color == self.bg
        fg_adj = sum(1 for d in range(8)
                     if self.nb8_colors[r, c, d] >= 0
                     and self.nb8_colors[r, c, d] != self.bg)
        nb_role = tuple(
            'F' if (self.nb8_colors[r, c, d] >= 0 and self.nb8_colors[r, c, d] != self.bg)
            else ('B' if self.nb8_colors[r, c, d] == self.bg else 'X')
            for d in range(8)
        )
        ray_first = tuple(int(self.ray_color[r, c, d]) for d in range(4))
        ray_d = tuple(int(self.ray_dist[r, c, d]) for d in range(4))
        between_h = ray_first[2] >= 0 and ray_first[3] >= 0
        between_v = ray_first[0] >= 0 and ray_first[1] >= 0
        return {
            'color': color, 'is_bg': is_bg,
            'cross6': tuple(int(x) for x in self.cross6[r, c]),
            'fg_adj': fg_adj, 'nb_role': nb_role,
            'ray_first': ray_first, 'ray_dist': ray_d,
            'between_h': between_h, 'between_v': between_v,
            'pos': (r, c),
        }


# ──────────────────────────────────────────────────────────────
# TransparentCross: 入力→出力の差分表現
# ──────────────────────────────────────────────────────────────

@dataclass
class TransparentCross:
    """差分をCross塊として表現"""
    added_cells: Dict[int, Set[Tuple[int,int]]]     # color -> added cell positions
    removed_cells: Dict[int, Set[Tuple[int,int]]]   # color -> removed cell positions
    recolored_cells: List[Tuple[int,int,int,int]]    # (r, c, from_color, to_color)
    unchanged_mask: np.ndarray                        # bool H×W

    # 追加セルを連結塊にまとめたもの
    added_chunks: List[Chunk] = field(default_factory=list)
    removed_chunks: List[Chunk] = field(default_factory=list)


def compute_delta(cw_in: CrossWorld, cw_out: CrossWorld) -> Optional[TransparentCross]:
    """入力→出力の差分をTransparentCrossとして計算"""
    gi, go = cw_in.g, cw_out.g
    if gi.shape != go.shape:
        return None  # サイズ変更は別処理

    h, w = gi.shape
    added = defaultdict(set)
    removed = defaultdict(set)
    recolored = []
    unchanged = np.ones((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            ic, oc = int(gi[r, c]), int(go[r, c])
            if ic != oc:
                unchanged[r, c] = False
                if ic == cw_in.bg:
                    added[oc].add((r, c))
                elif oc == cw_out.bg:
                    removed[ic].add((r, c))
                else:
                    recolored.append((r, c, ic, oc))
                    removed[ic].add((r, c))
                    added[oc].add((r, c))

    tc = TransparentCross(
        added_cells=dict(added),
        removed_cells=dict(removed),
        recolored_cells=recolored,
        unchanged_mask=unchanged,
    )

    # 追加/削除セルを塊にまとめる
    for color, cells in added.items():
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        labeled, n = scipy_label(mask)
        for i in range(1, n + 1):
            chunk_cells = frozenset(zip(*np.where(labeled == i)))
            tc.added_chunks.append(Chunk(color, chunk_cells))

    for color, cells in removed.items():
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        labeled, n = scipy_label(mask)
        for i in range(1, n + 1):
            chunk_cells = frozenset(zip(*np.where(labeled == i)))
            tc.removed_chunks.append(Chunk(color, chunk_cells))

    return tc


# ──────────────────────────────────────────────────────────────
# Cross操作: 個々の変換操作
# ──────────────────────────────────────────────────────────────

@dataclass
class CrossOp:
    """1つのCross操作"""
    op_type: str
    params: Dict = field(default_factory=dict)

    def __repr__(self):
        return f"CrossOp({self.op_type}, {self.params})"


def _find_chunk_containing(chunks: List[Chunk], cells: FrozenSet) -> Optional[Chunk]:
    """cellsを含む塊を探す"""
    for ch in chunks:
        if cells.issubset(ch.cells):
            return ch
        if len(cells & ch.cells) > len(cells) * 0.5:
            return ch
    return None


def _shape_match(sig_a, sig_b) -> Optional[str]:
    """2つの形状シグネチャが変換で一致するか"""
    if sig_a == sig_b:
        return 'identity'
    # flip_h
    if sig_a:
        max_c = max(c for r, c in sig_a)
        flipped_h = tuple(sorted((r, max_c - c) for r, c in sig_a))
        if flipped_h == sig_b:
            return 'flip_h'
    # flip_v
    if sig_a:
        max_r = max(r for r, c in sig_a)
        flipped_v = tuple(sorted((max_r - r, c) for r, c in sig_a))
        if flipped_v == sig_b:
            return 'flip_v'
    # rot180
    if sig_a:
        max_r = max(r for r, c in sig_a)
        max_c = max(c for r, c in sig_a)
        rot180 = tuple(sorted((max_r - r, max_c - c) for r, c in sig_a))
        if rot180 == sig_b:
            return 'rot180'
    # rot90
    if sig_a:
        max_r = max(r for r, c in sig_a)
        rot90 = tuple(sorted((c, max_r - r) for r, c in sig_a))
        if rot90 == sig_b:
            return 'rot90'
    # rot270
    if sig_a:
        max_c = max(c for r, c in sig_a)
        rot270 = tuple(sorted((max_c - c, r) for r, c in sig_a))
        if rot270 == sig_b:
            return 'rot270'
    return None


# ──────────────────────────────────────────────────────────────
# 操作検出器
# ──────────────────────────────────────────────────────────────

def detect_copy_ops(cw_in: CrossWorld, cw_out: CrossWorld,
                    tc: TransparentCross) -> List[CrossOp]:
    """コピー操作を検出: 入力の塊が出力に複製されている"""
    ops = []
    used_added = set()

    for ach in tc.added_chunks:
        if id(ach) in used_added:
            continue
        asig = ach.shape_sig

        for src in cw_in.fg_chunks:
            match_type = _shape_match(src.shape_sig, asig)
            if match_type is None:
                continue

            # 位置差
            dr = ach._r0 - src._r0
            dc = ach._c0 - src._c0

            ops.append(CrossOp('copy', {
                'src_color': src.color,
                'src_shape': src.shape_sig,
                'src_bbox': src.bbox,
                'dst_color': ach.color,
                'delta': (dr, dc),
                'transform': match_type,
                'dst_bbox': ach.bbox,
            }))
            used_added.add(id(ach))
            break

    return ops


def detect_move_ops(cw_in: CrossWorld, cw_out: CrossWorld,
                    tc: TransparentCross) -> List[CrossOp]:
    """移動操作: 入力の塊が消えて、出力に同形状で出現"""
    ops = []
    used_removed = set()
    used_added = set()

    for rch in tc.removed_chunks:
        if id(rch) in used_removed:
            continue
        rsig = rch.shape_sig

        for ach in tc.added_chunks:
            if id(ach) in used_added:
                continue
            match = _shape_match(rsig, ach.shape_sig)
            if match is None:
                continue

            dr = ach._r0 - rch._r0
            dc = ach._c0 - rch._c0
            ops.append(CrossOp('move', {
                'src_color': rch.color,
                'src_shape': rsig,
                'dst_color': ach.color,
                'delta': (dr, dc),
                'transform': match,
            }))
            used_removed.add(id(rch))
            used_added.add(id(ach))
            break

    return ops


def detect_flood_fill(cw_in: CrossWorld, cw_out: CrossWorld,
                      tc: TransparentCross) -> List[CrossOp]:
    """Flood fill: 閉じた領域が塗りつぶされた"""
    ops = []

    for color, cells in tc.added_cells.items():
        if not cells:
            continue
        # added cellsが入力で背景のCross塊の部分集合か？
        for bg_ch in cw_in.bg_chunks:
            if cells.issubset(bg_ch.cells):
                # この背景塊が完全に塗られた = flood fill
                ops.append(CrossOp('flood_fill', {
                    'fill_color': color,
                    'target_bg_chunk_bbox': bg_ch.bbox,
                    'target_bg_chunk_size': bg_ch.size,
                    'filled_size': len(cells),
                    'complete': len(cells) == bg_ch.size,
                }))
                break

    return ops


def detect_line_draw(cw_in: CrossWorld, cw_out: CrossWorld,
                     tc: TransparentCross) -> List[CrossOp]:
    """線引き: 追加セルが直線状"""
    ops = []

    for ach in tc.added_chunks:
        cells = sorted(ach.cells)
        if len(cells) < 2:
            continue

        # 水平線チェック
        rows = set(r for r, c in cells)
        cols = set(c for r, c in cells)
        if len(rows) == 1:
            r = rows.pop()
            min_c, max_c = min(cols), max(cols)
            if len(cells) == max_c - min_c + 1:
                ops.append(CrossOp('line_h', {
                    'color': ach.color, 'row': r,
                    'c_start': min_c, 'c_end': max_c,
                }))
                continue

        # 垂直線チェック
        if len(cols) == 1:
            c = cols.pop()
            min_r, max_r = min(rows), max(rows)
            if len(cells) == max_r - min_r + 1:
                ops.append(CrossOp('line_v', {
                    'color': ach.color, 'col': c,
                    'r_start': min_r, 'r_end': max_r,
                }))
                continue

        # 対角線チェック
        if len(cells) >= 2:
            dr = cells[1][0] - cells[0][0]
            dc = cells[1][1] - cells[0][1]
            if dr != 0 or dc != 0:
                is_diag = True
                for i in range(2, len(cells)):
                    if cells[i][0] - cells[i-1][0] != dr or cells[i][1] - cells[i-1][1] != dc:
                        is_diag = False
                        break
                if is_diag:
                    ops.append(CrossOp('line_diag', {
                        'color': ach.color,
                        'start': cells[0], 'end': cells[-1],
                        'direction': (dr, dc),
                    }))

    return ops


def detect_extend(cw_in: CrossWorld, cw_out: CrossWorld,
                  tc: TransparentCross) -> List[CrossOp]:
    """延長: 既存オブジェクトが一方向に伸びた"""
    ops = []

    for ach in tc.added_chunks:
        # added chunkが既存fg chunkに隣接してるか
        for src in cw_in.fg_chunks:
            if ach.color != src.color:
                continue
            # 隣接チェック
            adjacent = False
            for r, c in ach.cells:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    if (r+dr, c+dc) in src.cells:
                        adjacent = True
                        break
                if adjacent:
                    break

            if adjacent:
                # 延長方向を特定
                ach_center = ach.center
                src_center = src.center
                dr = ach_center[0] - src_center[0]
                dc = ach_center[1] - src_center[1]

                ops.append(CrossOp('extend', {
                    'src_color': src.color,
                    'src_bbox': src.bbox,
                    'direction': ('down' if dr > 0 else 'up' if dr < 0 else '',
                                  'right' if dc > 0 else 'left' if dc < 0 else ''),
                    'added_size': ach.size,
                }))
                break

    return ops


def detect_mirror(cw_in: CrossWorld, cw_out: CrossWorld,
                  tc: TransparentCross) -> List[CrossOp]:
    """対称化: 追加部分が既存部分の鏡像"""
    ops = []

    for ach in tc.added_chunks:
        asig = ach.shape_sig
        for src in cw_in.fg_chunks:
            if ach.color != src.color:
                continue

            # flip_h check
            match = _shape_match(src.shape_sig, asig)
            if match and match != 'identity':
                # 対称軸を計算
                axis_r = (src.center[0] + ach.center[0]) / 2
                axis_c = (src.center[1] + ach.center[1]) / 2

                ops.append(CrossOp('mirror', {
                    'src_color': src.color,
                    'src_bbox': src.bbox,
                    'transform': match,
                    'axis': (axis_r, axis_c),
                }))
                break

    return ops


def detect_gravity(cw_in: CrossWorld, cw_out: CrossWorld,
                   tc: TransparentCross) -> List[CrossOp]:
    """重力: 全非bg塊が一方向に移動"""
    for direction, (dr, dc) in [('down',(1,0)),('up',(-1,0)),('right',(0,1)),('left',(0,-1))]:
        # 全fg chunkについて、出力での位置を確認
        matched = True
        for src in cw_in.fg_chunks:
            # このchunkは出力のどこにある？
            found = False
            for dst in cw_out.fg_chunks:
                if src.color == dst.color and src.shape_sig == dst.shape_sig:
                    # 正しい方向に移動したか
                    actual_dr = dst._r0 - src._r0
                    actual_dc = dst._c0 - src._c0
                    if dr != 0 and dc == 0:
                        if actual_dc != 0:
                            continue
                        if (dr > 0 and actual_dr > 0) or (dr < 0 and actual_dr < 0) or actual_dr == 0:
                            found = True
                            break
                    elif dc != 0 and dr == 0:
                        if actual_dr != 0:
                            continue
                        if (dc > 0 and actual_dc > 0) or (dc < 0 and actual_dc < 0) or actual_dc == 0:
                            found = True
                            break
            if not found:
                matched = False
                break

        if matched and cw_in.fg_chunks:
            ops = [CrossOp('gravity', {'direction': direction})]
            return ops

    return []


def detect_conditional_recolor(cw_in: CrossWorld, cw_out: CrossWorld,
                               tc: TransparentCross) -> List[CrossOp]:
    """条件付き色変更: 近傍条件で色が変わる"""
    if not tc.recolored_cells:
        return []

    # 色変更セルの特徴を収集
    features = []
    for r, c, from_c, to_c in tc.recolored_cells:
        feat = cw_in.cell_features(r, c)
        features.append((feat, to_c))

    # 共通パターンを探す
    # fg_adj が同じか？
    fg_adjs = set(f['fg_adj'] for f, _ in features)
    if len(fg_adjs) == 1:
        return [CrossOp('conditional_recolor', {
            'condition': 'fg_adj',
            'value': fg_adjs.pop(),
            'to_color': features[0][1],  # assuming same target color
        })]

    return []


def detect_stamp(cw_in: CrossWorld, cw_out: CrossWorld,
                 tc: TransparentCross) -> List[CrossOp]:
    """スタンプ: 同じパターンが複数箇所に追加"""
    if len(tc.added_chunks) < 2:
        return []

    # 同形状の追加塊をグループ化
    shape_groups = defaultdict(list)
    for ach in tc.added_chunks:
        shape_groups[ach.shape_sig].append(ach)

    ops = []
    for sig, chunks in shape_groups.items():
        if len(chunks) >= 2:
            positions = [(ch._r0, ch._c0) for ch in chunks]
            ops.append(CrossOp('stamp', {
                'pattern_sig': sig,
                'color': chunks[0].color,
                'positions': positions,
                'count': len(chunks),
            }))

    return ops


def detect_pattern_complete(cw_in: CrossWorld, cw_out: CrossWorld,
                            tc: TransparentCross) -> List[CrossOp]:
    """パターン補完: 部分的なパターンを完成させる"""
    # 入力に対称性の欠けがあり、出力で補完されている
    ops = []

    # 水平対称チェック
    h, w = cw_out.h, cw_out.w
    go = cw_out.g
    if np.array_equal(go, go[:, ::-1]):
        gi = cw_in.g
        if not np.array_equal(gi, gi[:, ::-1]):
            ops.append(CrossOp('pattern_complete', {'symmetry': 'horizontal'}))

    # 垂直対称チェック
    if np.array_equal(go, go[::-1, :]):
        gi = cw_in.g
        if not np.array_equal(gi, gi[::-1, :]):
            ops.append(CrossOp('pattern_complete', {'symmetry': 'vertical'}))

    # 4回転対称
    if np.array_equal(go, np.rot90(go)):
        gi = cw_in.g
        if not np.array_equal(gi, np.rot90(gi)):
            ops.append(CrossOp('pattern_complete', {'symmetry': 'rot4'}))

    return ops


def detect_connect(cw_in: CrossWorld, cw_out: CrossWorld,
                   tc: TransparentCross) -> List[CrossOp]:
    """接続: 2つのオブジェクト間を線で繋ぐ"""
    ops = []

    for ach in tc.added_chunks:
        cells = sorted(ach.cells)
        if len(cells) < 2:
            continue

        # 線状かチェック
        rows = set(r for r, c in cells)
        cols = set(c for r, c in cells)
        is_line = (len(rows) == 1 or len(cols) == 1)
        if not is_line:
            continue

        # 両端に隣接するfgオブジェクトを探す
        endpoints = [cells[0], cells[-1]]
        connected = []
        for ep_r, ep_c in endpoints:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = ep_r + dr, ep_c + dc
                for fch in cw_in.fg_chunks:
                    if (nr, nc) in fch.cells:
                        connected.append(fch)
                        break

        if len(connected) >= 2:
            ops.append(CrossOp('connect', {
                'color': ach.color,
                'obj_a_color': connected[0].color,
                'obj_a_bbox': connected[0].bbox,
                'obj_b_color': connected[1].color,
                'obj_b_bbox': connected[1].bbox,
            }))

    return ops


# ──────────────────────────────────────────────────────────────
# ルール学習: 全train例の操作を分析して一貫ルールを抽出
# ──────────────────────────────────────────────────────────────

ALL_DETECTORS = [
    detect_copy_ops,
    detect_move_ops,
    detect_flood_fill,
    detect_line_draw,
    detect_extend,
    detect_mirror,
    detect_gravity,
    detect_conditional_recolor,
    detect_stamp,
    detect_pattern_complete,
    detect_connect,
]


def analyze_example(inp, out) -> Optional[Dict]:
    """1つのtrain例を分析"""
    cw_in = CrossWorld(inp)
    cw_out = CrossWorld(out)

    if cw_in.h != cw_out.h or cw_in.w != cw_out.w:
        return {'size_change': True, 'cw_in': cw_in, 'cw_out': cw_out}

    tc = compute_delta(cw_in, cw_out)
    if tc is None:
        return None

    all_ops = []
    for detector in ALL_DETECTORS:
        try:
            ops = detector(cw_in, cw_out, tc)
            all_ops.extend(ops)
        except Exception:
            continue

    return {
        'size_change': False,
        'cw_in': cw_in, 'cw_out': cw_out, 'tc': tc,
        'ops': all_ops,
        'n_changed': int((~tc.unchanged_mask).sum()),
        'n_added_chunks': len(tc.added_chunks),
        'n_removed_chunks': len(tc.removed_chunks),
    }


def learn_rule(train_pairs) -> Optional[Dict]:
    """全train例から一貫するCross操作ルールを学習"""
    analyses = []
    for inp, out in train_pairs:
        a = analyze_example(inp, out)
        if a is None:
            return None
        analyses.append(a)

    # サイズ変更タスクは別処理
    if any(a.get('size_change') for a in analyses):
        return _learn_size_rule(train_pairs, analyses)

    # 各例の操作タイプ集合
    op_types_per_example = []
    for a in analyses:
        types = Counter(op.op_type for op in a['ops'])
        op_types_per_example.append(types)

    # 全例で共通する操作タイプ
    if not op_types_per_example:
        return None

    common_types = set(op_types_per_example[0].keys())
    for types in op_types_per_example[1:]:
        common_types &= set(types.keys())

    if not common_types:
        return _learn_cell_rule(train_pairs, analyses)

    # 共通操作ごとにルールを試行
    for op_type in ['gravity', 'pattern_complete', 'flood_fill',
                    'copy', 'move', 'mirror', 'connect',
                    'line_h', 'line_v', 'extend', 'stamp',
                    'conditional_recolor']:
        if op_type not in common_types:
            continue

        rule = _try_op_rule(op_type, train_pairs, analyses)
        if rule is not None:
            return rule

    # Fallback: cell-level rule
    return _learn_cell_rule(train_pairs, analyses)


def _learn_size_rule(train_pairs, analyses):
    """サイズ変更ルール"""
    from arc.cross6_scale import size_change_solve
    return {'solver': 'size_change'}


def _learn_cell_rule(train_pairs, analyses):
    """セルレベルのCrossルール"""
    return {'solver': 'cell_level'}


def _try_op_rule(op_type, train_pairs, analyses) -> Optional[Dict]:
    """特定の操作タイプでルールを構築"""

    if op_type == 'gravity':
        directions = set()
        for a in analyses:
            for op in a['ops']:
                if op.op_type == 'gravity':
                    directions.add(op.params['direction'])
        if len(directions) == 1:
            return {'solver': 'gravity', 'direction': directions.pop()}

    if op_type == 'pattern_complete':
        symmetries = set()
        for a in analyses:
            for op in a['ops']:
                if op.op_type == 'pattern_complete':
                    symmetries.add(op.params['symmetry'])
        if len(symmetries) == 1:
            return {'solver': 'pattern_complete', 'symmetry': symmetries.pop()}

    if op_type == 'flood_fill':
        fill_colors = set()
        for a in analyses:
            for op in a['ops']:
                if op.op_type == 'flood_fill':
                    fill_colors.add(op.params['fill_color'])
        if len(fill_colors) == 1:
            return {'solver': 'flood_fill', 'fill_color': fill_colors.pop()}

    if op_type == 'connect':
        colors = set()
        for a in analyses:
            for op in a['ops']:
                if op.op_type == 'connect':
                    colors.add(op.params['color'])
        if len(colors) == 1:
            return {'solver': 'connect', 'color': colors.pop()}

    return None


# ──────────────────────────────────────────────────────────────
# 操作の適用
# ──────────────────────────────────────────────────────────────

def apply_gravity(grid, direction):
    """重力をセル単位で適用"""
    g = np.array(grid, dtype=np.int8)
    h, w = g.shape
    bg = Counter(g.flatten()).most_common(1)[0][0]

    if direction == 'down':
        for c in range(w):
            col = [g[r, c] for r in range(h) if g[r, c] != bg]
            for r in range(h):
                g[r, c] = bg
            for i, v in enumerate(reversed(col)):
                g[h - 1 - i, c] = v
    elif direction == 'up':
        for c in range(w):
            col = [g[r, c] for r in range(h) if g[r, c] != bg]
            for r in range(h):
                g[r, c] = bg
            for i, v in enumerate(col):
                g[i, c] = v
    elif direction == 'right':
        for r in range(h):
            row = [g[r, c] for c in range(w) if g[r, c] != bg]
            for c in range(w):
                g[r, c] = bg
            for i, v in enumerate(reversed(row)):
                g[r, w - 1 - i] = v
    elif direction == 'left':
        for r in range(h):
            row = [g[r, c] for c in range(w) if g[r, c] != bg]
            for c in range(w):
                g[r, c] = bg
            for i, v in enumerate(row):
                g[r, i] = v

    return g.tolist()


def apply_pattern_complete(grid, symmetry):
    """対称性補完"""
    g = np.array(grid, dtype=np.int8)
    h, w = g.shape
    bg = Counter(g.flatten()).most_common(1)[0][0]
    result = g.copy()

    if symmetry == 'horizontal':
        for r in range(h):
            for c in range(w):
                mirror_c = w - 1 - c
                if result[r, c] == bg and result[r, mirror_c] != bg:
                    result[r, c] = result[r, mirror_c]
                elif result[r, c] != bg and result[r, mirror_c] == bg:
                    result[r, mirror_c] = result[r, c]
    elif symmetry == 'vertical':
        for r in range(h):
            for c in range(w):
                mirror_r = h - 1 - r
                if result[r, c] == bg and result[mirror_r, c] != bg:
                    result[r, c] = result[mirror_r, c]
                elif result[r, c] != bg and result[mirror_r, c] == bg:
                    result[mirror_r, c] = result[r, c]
    elif symmetry == 'rot4':
        for r in range(h):
            for c in range(w):
                positions = [(r, c), (c, h-1-r), (h-1-r, w-1-c), (w-1-c, r)]
                vals = []
                for pr, pc in positions:
                    if 0 <= pr < h and 0 <= pc < w and result[pr, pc] != bg:
                        vals.append(result[pr, pc])
                if vals:
                    fill_val = Counter(vals).most_common(1)[0][0]
                    for pr, pc in positions:
                        if 0 <= pr < h and 0 <= pc < w and result[pr, pc] == bg:
                            result[pr, pc] = fill_val

    return result.tolist()


def apply_flood_fill(grid, fill_color):
    """閉じた背景領域をfill_colorで塗る"""
    g = np.array(grid, dtype=np.int8)
    h, w = g.shape
    bg = Counter(g.flatten()).most_common(1)[0][0]

    # 背景の連結成分
    bg_mask = (g == bg)
    labeled, n = scipy_label(bg_mask)

    # 境界に触れていない背景塊 = 閉じた領域
    border_labels = set()
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h-1 or c == 0 or c == w-1) and labeled[r, c] > 0:
                border_labels.add(labeled[r, c])

    result = g.copy()
    for i in range(1, n + 1):
        if i not in border_labels:
            result[labeled == i] = fill_color

    return result.tolist()


def apply_connect(grid, color):
    """同色オブジェクト間を線で接続"""
    g = np.array(grid, dtype=np.int8)
    h, w = g.shape
    bg = Counter(g.flatten()).most_common(1)[0][0]
    result = g.copy()

    # 全fg chunksを取得
    fg_mask = g != bg
    labeled, n = scipy_label(fg_mask)
    chunks = []
    for i in range(1, n + 1):
        cells = list(zip(*np.where(labeled == i)))
        if cells:
            cr = sum(r for r, c in cells) / len(cells)
            cc = sum(c for r, c in cells) / len(cells)
            chunks.append((i, cr, cc, cells))

    # 同色のペアを接続
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            li, ri, ci, cells_i = chunks[i]
            lj, rj, cj, cells_j = chunks[j]

            # 同じ行 or 同じ列に並んでいるか
            if abs(ri - rj) < 1:  # same row
                r = int(round(ri))
                c_start = min(int(ci), int(cj))
                c_end = max(int(ci), int(cj))
                if r < h:
                    for c in range(c_start, c_end + 1):
                        if result[r, c] == bg:
                            result[r, c] = color
            elif abs(ci - cj) < 1:  # same col
                c = int(round(ci))
                r_start = min(int(ri), int(rj))
                r_end = max(int(ri), int(rj))
                if c < w:
                    for r in range(r_start, r_end + 1):
                        if result[r, c] == bg:
                            result[r, c] = color

    return result.tolist()


def apply_rule(grid, rule):
    """学習したルールを適用"""
    solver = rule.get('solver')

    if solver == 'gravity':
        return apply_gravity(grid, rule['direction'])
    elif solver == 'pattern_complete':
        return apply_pattern_complete(grid, rule['symmetry'])
    elif solver == 'flood_fill':
        return apply_flood_fill(grid, rule['fill_color'])
    elif solver == 'connect':
        return apply_connect(grid, rule['color'])
    elif solver == 'size_change':
        return None  # handled by cross6_scale
    elif solver == 'cell_level':
        return None  # handled by cross6axis/fill

    return None


# ──────────────────────────────────────────────────────────────
# メインソルバー
# ──────────────────────────────────────────────────────────────

def cross_world_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """CrossWorld全操作ソルバー"""
    from arc.grid import grid_eq

    rule = learn_rule(train_pairs)
    if rule is None:
        return None

    solver = rule.get('solver')
    if solver in ('size_change', 'cell_level'):
        return None  # 別ソルバーに委譲

    result = apply_rule(test_input, rule)
    if result is None:
        return None

    # Train verify
    for inp, out in train_pairs:
        pred = apply_rule(inp, rule)
        if pred is None or not grid_eq(pred, out):
            return None

    return result
