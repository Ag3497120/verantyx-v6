"""
CrossUniverse3D — 立体十字構造の多段階マッピング

kofdaiの思想:
- 立体十字構造を囲む宇宙そのものも立体十字構造
- 「一番大きい」= 宇宙の広さ（占有体積）
- 「ユニークな形」= 構造内の情報配置の違い
- 構造の切断 = パネル/オブジェクト分離
- 平面→立体マッピングでパネル比較・オブジェクト選択が自然に実現

3次元構成:
  Z=0: ピクセルレベル（元の2Dグリッド）
  Z=1: オブジェクトレベル（connected components → 各オブジェクトが3D体積を占有）
  Z=2: パネルレベル（separator分割 → 各パネルが3D体積を占有）
  Z=3: グリッド全体のメタ特徴

各レベルで十字構造（6方向: ±x, ±y, ±z）のフローが走る。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set
from collections import Counter, deque
from scipy import ndimage

Grid = List[List[int]]


# ============================================================
# Level 0: ピクセル → オブジェクト抽出
# ============================================================

class Object3D:
    """立体cross構造上の1オブジェクト。2Dマスク + 3Dプロパティ。"""
    __slots__ = ['id', 'mask', 'color', 'colors', 'bbox', 'pixels',
                 'area', 'shape_sig', 'volume', 'centroid']
    
    def __init__(self, obj_id: int, mask: np.ndarray, grid: np.ndarray):
        self.id = obj_id
        self.mask = mask
        rows, cols = np.where(mask)
        self.pixels = list(zip(rows.tolist(), cols.tolist()))
        self.area = len(self.pixels)
        
        # Bounding box
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        self.bbox = (int(r0), int(c0), int(r1), int(c1))
        
        # Colors in this object
        self.colors = Counter(int(grid[r, c]) for r, c in self.pixels)
        self.color = self.colors.most_common(1)[0][0]  # dominant color
        
        # Centroid
        self.centroid = (rows.mean(), cols.mean())
        
        # Shape signature: normalized mask (shift to origin)
        norm_mask = mask[r0:r1+1, c0:c1+1]
        self.shape_sig = tuple(norm_mask.flatten().tolist())
        
        # Volume = area (in 3D: how much space this object occupies)
        self.volume = self.area


class Panel3D:
    """立体cross構造上の1パネル。separator分割された矩形領域。"""
    __slots__ = ['id', 'r0', 'c0', 'r1', 'c1', 'subgrid', 'objects',
                 'color_hist', 'shape_sig', 'volume']
    
    def __init__(self, panel_id: int, r0: int, c0: int, r1: int, c1: int, 
                 grid: np.ndarray, bg: int = 0):
        self.id = panel_id
        self.r0, self.c0, self.r1, self.c1 = r0, c0, r1, c1
        self.subgrid = grid[r0:r1, c0:c1].copy()
        self.volume = self.subgrid.size
        
        # Objects within this panel
        self.objects = []
        
        # Color histogram (excluding bg)
        flat = self.subgrid.flatten()
        self.color_hist = Counter(int(v) for v in flat if v != bg)
        
        # Shape signature: normalized content
        self.shape_sig = tuple(self.subgrid.flatten().tolist())


# ============================================================
# CrossUniverse3D: 立体cross構造マッピング
# ============================================================

class CrossUniverse3D:
    """
    入力グリッドを立体cross構造にマッピングする。
    
    マッピング手順:
    1. 2Dグリッドからオブジェクト抽出（connected components）
    2. Separator検出 → パネル分割
    3. 各オブジェクト/パネルに3Dプロパティ付与（体積=占有サイズ）
    4. 立体cross構造上でフロー制御
    
    クエリ可能な情報:
    - largest_object() → 最も体積の大きいオブジェクト
    - unique_object() → 形状がユニークなオブジェクト
    - odd_panel() → 他と異なるパネル
    - select_by_property(prop, mode) → プロパティに基づく選択
    """
    
    def __init__(self, grid: Grid, bg: int = 0):
        self.grid = np.array(grid)
        self.bg = bg
        self.h, self.w = self.grid.shape
        self.objects: List[Object3D] = []
        self.panels: List[Panel3D] = []
        self.sep_color: Optional[int] = None
        self.sep_rows: List[int] = []
        self.sep_cols: List[int] = []
        
        self._extract_objects()
        self._extract_panels()
    
    # ---- Object extraction ----
    
    def _extract_objects(self):
        """Connected component抽出。各色ごとに分離。"""
        self.objects = []
        obj_id = 0
        
        # Multi-color object detection
        nonbg = (self.grid != self.bg)
        labeled, n = ndimage.label(nonbg)
        
        for lbl in range(1, n + 1):
            mask = (labeled == lbl)
            if mask.sum() == 0:
                continue
            obj = Object3D(obj_id, mask, self.grid)
            self.objects.append(obj)
            obj_id += 1
        
        # Also per-color objects
        self._per_color_objects = {}
        for color in range(1, 10):
            color_mask = (self.grid == color)
            if not color_mask.any():
                continue
            labeled_c, n_c = ndimage.label(color_mask)
            objs = []
            for lbl in range(1, n_c + 1):
                mask = (labeled_c == lbl)
                objs.append(Object3D(obj_id, mask, self.grid))
                obj_id += 1
            self._per_color_objects[color] = objs
    
    # ---- Panel extraction ----
    
    def _extract_panels(self):
        """Separator行/列を検出してパネル分割。"""
        self.panels = []
        
        # Detect separator rows
        for r in range(self.h):
            row = self.grid[r]
            if len(set(row)) == 1 and row[0] != self.bg:
                self.sep_rows.append(r)
                if self.sep_color is None:
                    self.sep_color = int(row[0])
        
        # Detect separator cols
        for c in range(self.w):
            col = self.grid[:, c]
            if len(set(col)) == 1 and col[0] != self.bg:
                self.sep_cols.append(c)
                if self.sep_color is None:
                    self.sep_color = int(col[0])
        
        if not self.sep_rows and not self.sep_cols:
            # No separators: entire grid is one panel
            self.panels.append(Panel3D(0, 0, 0, self.h, self.w, self.grid, self.bg))
            return
        
        # Split by rows then cols
        row_boundaries = [0] + [r for r in self.sep_rows] + [self.h]
        col_boundaries = [0] + [c for c in self.sep_cols] + [self.w]
        
        panel_id = 0
        for i in range(len(row_boundaries) - 1):
            r0 = row_boundaries[i]
            r1 = row_boundaries[i + 1]
            # Skip separator rows
            if r0 in self.sep_rows:
                r0 += 1
            if r1 > r0:
                for j in range(len(col_boundaries) - 1):
                    c0 = col_boundaries[j]
                    c1 = col_boundaries[j + 1]
                    if c0 in self.sep_cols:
                        c0 += 1
                    if c1 > c0 and r1 > r0:
                        panel = Panel3D(panel_id, r0, c0, r1, c1, self.grid, self.bg)
                        self.panels.append(panel)
                        panel_id += 1
    
    # ---- 3D Cross queries ----
    
    def largest_object(self) -> Optional[Object3D]:
        """最大体積のオブジェクト"""
        if not self.objects:
            return None
        return max(self.objects, key=lambda o: o.volume)
    
    def smallest_object(self) -> Optional[Object3D]:
        """最小体積のオブジェクト"""
        if not self.objects:
            return None
        return min(self.objects, key=lambda o: o.volume)
    
    def unique_shape_object(self) -> Optional[Object3D]:
        """形状がユニークな（1回だけ出現する）オブジェクト"""
        if len(self.objects) < 2:
            return None
        shape_counts = Counter(o.shape_sig for o in self.objects)
        unique = [o for o in self.objects if shape_counts[o.shape_sig] == 1]
        return unique[0] if len(unique) == 1 else None
    
    def most_common_shape_object(self) -> Optional[Object3D]:
        """最も多く出現する形状のオブジェクト（1つ目）"""
        if len(self.objects) < 2:
            return None
        shape_counts = Counter(o.shape_sig for o in self.objects)
        most_common_shape = shape_counts.most_common(1)[0][0]
        return [o for o in self.objects if o.shape_sig == most_common_shape][0]
    
    def unique_color_object(self) -> Optional[Object3D]:
        """色がユニークなオブジェクト"""
        if len(self.objects) < 2:
            return None
        color_counts = Counter(o.color for o in self.objects)
        unique = [o for o in self.objects if color_counts[o.color] == 1]
        return unique[0] if len(unique) == 1 else None
    
    def odd_panel(self) -> Optional[Panel3D]:
        """他と異なるパネル（仲間外れ）"""
        if len(self.panels) < 3:
            return None
        sig_counts = Counter(p.shape_sig for p in self.panels)
        odd = [p for p in self.panels if sig_counts[p.shape_sig] == 1]
        return odd[0] if len(odd) == 1 else None
    
    def common_panel_pattern(self) -> Optional[np.ndarray]:
        """全パネルの共通パターン（AND的結合）"""
        if len(self.panels) < 2:
            return None
        
        # All panels must be same shape
        shapes = set(p.subgrid.shape for p in self.panels)
        if len(shapes) != 1:
            return None
        
        sh = list(shapes)[0]
        # Find cells that are the same across all panels
        result = self.panels[0].subgrid.copy()
        for p in self.panels[1:]:
            diff = (result != p.subgrid)
            result[diff] = self.bg
        
        return result
    
    def panel_diff(self) -> Optional[Tuple[Panel3D, np.ndarray]]:
        """パネル間の差分: 1つだけ違うパネルとその差分位置"""
        if len(self.panels) < 2:
            return None
        
        shapes = set(p.subgrid.shape for p in self.panels)
        if len(shapes) != 1:
            return None
        
        # Find the panel that differs from the majority
        sigs = Counter(p.shape_sig for p in self.panels)
        majority_sig = sigs.most_common(1)[0][0]
        diff_panels = [p for p in self.panels if p.shape_sig != majority_sig]
        
        if len(diff_panels) == 1:
            # Find the majority pattern
            majority_panel = [p for p in self.panels if p.shape_sig == majority_sig][0]
            diff_mask = (diff_panels[0].subgrid != majority_panel.subgrid)
            return diff_panels[0], diff_mask
        
        return None
    
    def select_objects_by_color(self, color: int) -> List[Object3D]:
        """特定色のオブジェクトを選択"""
        return [o for o in self.objects if o.color == color]
    
    def rank_objects_by_volume(self) -> List[Object3D]:
        """体積順にオブジェクトをランク付け（降順）"""
        return sorted(self.objects, key=lambda o: -o.volume)
    
    def extract_object_grid(self, obj: Object3D) -> np.ndarray:
        """オブジェクトのbounding box領域を抽出"""
        r0, c0, r1, c1 = obj.bbox
        return self.grid[r0:r1+1, c0:c1+1].copy()
    
    def extract_object_normalized(self, obj: Object3D) -> np.ndarray:
        """オブジェクトを正規化抽出（bgをbgのまま）"""
        r0, c0, r1, c1 = obj.bbox
        region = self.grid[r0:r1+1, c0:c1+1].copy()
        # Mask out pixels not belonging to this object
        obj_region = obj.mask[r0:r1+1, c0:c1+1]
        region[~obj_region] = self.bg
        return region


# ============================================================
# ルール学習: 立体cross構造 → 変換ルール
# ============================================================

def learn_3d_cross_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """訓練ペアから立体cross構造ベースのルールを学習する。"""
    
    if not train_pairs:
        return None
    
    bg = _detect_bg(train_pairs[0][0])
    
    rules_to_try = [
        _try_select_largest_object,
        _try_select_smallest_object,
        _try_select_unique_shape,
        _try_select_unique_color,
        _try_odd_panel_out,
        _try_common_panel_pattern,
        _try_extract_by_rank,
        _try_panel_overlay,
        _try_panel_xor,
        _try_select_by_color_count,
        _try_majority_panel,
    ]
    
    for rule_fn in rules_to_try:
        result = rule_fn(train_pairs, bg)
        if result is not None:
            return result
    
    return None


def apply_3d_cross_rule(inp: Grid, rule: Dict) -> Optional[Grid]:
    """学習した立体crossルールを適用"""
    bg = rule.get('bg', 0)
    universe = CrossUniverse3D(inp, bg)
    rule_type = rule['type']
    
    if rule_type == 'select_largest':
        obj = universe.largest_object()
        if obj is None: return None
        return universe.extract_object_grid(obj).tolist()
    
    elif rule_type == 'select_smallest':
        obj = universe.smallest_object()
        if obj is None: return None
        return universe.extract_object_grid(obj).tolist()
    
    elif rule_type == 'select_unique_shape':
        obj = universe.unique_shape_object()
        if obj is None: return None
        return universe.extract_object_grid(obj).tolist()
    
    elif rule_type == 'select_unique_color':
        obj = universe.unique_color_object()
        if obj is None: return None
        return universe.extract_object_grid(obj).tolist()
    
    elif rule_type == 'select_by_rank':
        rank = rule['rank']
        ranked = universe.rank_objects_by_volume()
        if rank >= len(ranked): return None
        obj = ranked[rank]
        return universe.extract_object_grid(obj).tolist()
    
    elif rule_type == 'select_by_color':
        color = rule['color']
        objs = universe.select_objects_by_color(color)
        if not objs: return None
        obj = max(objs, key=lambda o: o.volume)
        return universe.extract_object_grid(obj).tolist()
    
    elif rule_type == 'select_by_color_count':
        mode = rule['mode']  # 'max' or 'min'
        objs = universe.objects
        if not objs: return None
        if mode == 'max':
            obj = max(objs, key=lambda o: len(o.colors))
        else:
            obj = min(objs, key=lambda o: len(o.colors))
        return universe.extract_object_grid(obj).tolist()
    
    elif rule_type == 'odd_panel':
        panel = universe.odd_panel()
        if panel is None: return None
        return panel.subgrid.tolist()
    
    elif rule_type == 'common_panel':
        result = universe.common_panel_pattern()
        if result is None: return None
        return result.tolist()
    
    elif rule_type == 'majority_panel':
        if len(universe.panels) < 2: return None
        sig_counts = Counter(p.shape_sig for p in universe.panels)
        majority_sig = sig_counts.most_common(1)[0][0]
        panel = [p for p in universe.panels if p.shape_sig == majority_sig][0]
        return panel.subgrid.tolist()
    
    elif rule_type == 'panel_overlay':
        if len(universe.panels) < 2: return None
        shapes = set(p.subgrid.shape for p in universe.panels)
        if len(shapes) != 1: return None
        sh = list(shapes)[0]
        result = np.full(sh, bg, dtype=int)
        for p in universe.panels:
            mask = (p.subgrid != bg)
            result[mask] = p.subgrid[mask]
        return result.tolist()
    
    elif rule_type == 'panel_xor':
        if len(universe.panels) != 2: return None
        p0, p1 = universe.panels[0], universe.panels[1]
        if p0.subgrid.shape != p1.subgrid.shape: return None
        # XOR: cells that differ
        result = np.full(p0.subgrid.shape, bg, dtype=int)
        diff = (p0.subgrid != p1.subgrid)
        # Take from p0 where different and non-bg
        mask0 = diff & (p0.subgrid != bg)
        mask1 = diff & (p1.subgrid != bg)
        result[mask0] = p0.subgrid[mask0]
        result[mask1] = p1.subgrid[mask1]
        return result.tolist()
    
    elif rule_type == 'select_normalized':
        rank = rule.get('rank', 0)
        ranked = universe.rank_objects_by_volume()
        if rank >= len(ranked): return None
        obj = ranked[rank]
        return universe.extract_object_normalized(obj).tolist()
    
    return None


# ============================================================
# 個別ルール学習関数
# ============================================================

def _detect_bg(grid: Grid) -> int:
    a = np.array(grid)
    return int(Counter(a.flatten().tolist()).most_common(1)[0][0])


def _try_select_largest_object(train_pairs, bg):
    for inp, out in train_pairs:
        u = CrossUniverse3D(inp, bg)
        obj = u.largest_object()
        if obj is None: return None
        extracted = u.extract_object_grid(obj)
        if extracted.tolist() != out: return None
    return {'type': 'select_largest', 'bg': bg}


def _try_select_smallest_object(train_pairs, bg):
    for inp, out in train_pairs:
        u = CrossUniverse3D(inp, bg)
        obj = u.smallest_object()
        if obj is None: return None
        extracted = u.extract_object_grid(obj)
        if extracted.tolist() != out: return None
    return {'type': 'select_smallest', 'bg': bg}


def _try_select_unique_shape(train_pairs, bg):
    for inp, out in train_pairs:
        u = CrossUniverse3D(inp, bg)
        obj = u.unique_shape_object()
        if obj is None: return None
        extracted = u.extract_object_grid(obj)
        if extracted.tolist() != out:
            # Try normalized
            norm = u.extract_object_normalized(obj)
            if norm.tolist() != out: return None
    return {'type': 'select_unique_shape', 'bg': bg}


def _try_select_unique_color(train_pairs, bg):
    for inp, out in train_pairs:
        u = CrossUniverse3D(inp, bg)
        obj = u.unique_color_object()
        if obj is None: return None
        extracted = u.extract_object_grid(obj)
        if extracted.tolist() != out: return None
    return {'type': 'select_unique_color', 'bg': bg}


def _try_extract_by_rank(train_pairs, bg):
    """Try selecting object by volume rank (0=largest, 1=2nd largest, etc)"""
    max_rank = min(10, max(len(CrossUniverse3D(inp, bg).objects) for inp, _ in train_pairs))
    
    for rank in range(max_rank):
        ok = True
        for inp, out in train_pairs:
            u = CrossUniverse3D(inp, bg)
            ranked = u.rank_objects_by_volume()
            if rank >= len(ranked):
                ok = False; break
            obj = ranked[rank]
            extracted = u.extract_object_grid(obj)
            if extracted.tolist() != out:
                # Try normalized
                norm = u.extract_object_normalized(obj)
                if norm.tolist() != out:
                    ok = False; break
        if ok:
            return {'type': 'select_by_rank', 'rank': rank, 'bg': bg}
    
    return None


def _try_odd_panel_out(train_pairs, bg):
    for inp, out in train_pairs:
        u = CrossUniverse3D(inp, bg)
        panel = u.odd_panel()
        if panel is None: return None
        if panel.subgrid.tolist() != out: return None
    return {'type': 'odd_panel', 'bg': bg}


def _try_common_panel_pattern(train_pairs, bg):
    for inp, out in train_pairs:
        u = CrossUniverse3D(inp, bg)
        result = u.common_panel_pattern()
        if result is None: return None
        if result.tolist() != out: return None
    return {'type': 'common_panel', 'bg': bg}


def _try_panel_overlay(train_pairs, bg):
    for inp, out in train_pairs:
        u = CrossUniverse3D(inp, bg)
        if len(u.panels) < 2: return None
        shapes = set(p.subgrid.shape for p in u.panels)
        if len(shapes) != 1: return None
        sh = list(shapes)[0]
        result = np.full(sh, bg, dtype=int)
        for p in u.panels:
            mask = (p.subgrid != bg)
            result[mask] = p.subgrid[mask]
        if result.tolist() != out: return None
    return {'type': 'panel_overlay', 'bg': bg}


def _try_panel_xor(train_pairs, bg):
    for inp, out in train_pairs:
        u = CrossUniverse3D(inp, bg)
        if len(u.panels) != 2: return None
        p0, p1 = u.panels[0], u.panels[1]
        if p0.subgrid.shape != p1.subgrid.shape: return None
        result = np.full(p0.subgrid.shape, bg, dtype=int)
        diff = (p0.subgrid != p1.subgrid)
        mask0 = diff & (p0.subgrid != bg)
        mask1 = diff & (p1.subgrid != bg)
        result[mask0] = p0.subgrid[mask0]
        result[mask1] = p1.subgrid[mask1]
        if result.tolist() != out: return None
    return {'type': 'panel_xor', 'bg': bg}


def _try_select_by_color_count(train_pairs, bg):
    for mode in ['max', 'min']:
        ok = True
        for inp, out in train_pairs:
            u = CrossUniverse3D(inp, bg)
            if not u.objects:
                ok = False; break
            if mode == 'max':
                obj = max(u.objects, key=lambda o: len(o.colors))
            else:
                obj = min(u.objects, key=lambda o: len(o.colors))
            extracted = u.extract_object_grid(obj)
            if extracted.tolist() != out:
                ok = False; break
        if ok:
            return {'type': 'select_by_color_count', 'mode': mode, 'bg': bg}
    return None


def _try_majority_panel(train_pairs, bg):
    for inp, out in train_pairs:
        u = CrossUniverse3D(inp, bg)
        if len(u.panels) < 2: return None
        sig_counts = Counter(p.shape_sig for p in u.panels)
        majority_sig = sig_counts.most_common(1)[0][0]
        panel = [p for p in u.panels if p.shape_sig == majority_sig][0]
        if panel.subgrid.tolist() != out: return None
    return {'type': 'majority_panel', 'bg': bg}


# ============================================================
# CrossPiece生成
# ============================================================

def _try_panel_compact(train_pairs, bg):
    """パネルの非bg bounding boxを切り出して再配置"""
    for inp, out in train_pairs:
        a = np.array(inp)
        h, w = a.shape
        sep_rows = [r for r in range(h) if len(set(a[r])) == 1 and a[r,0] != bg]
        sep_cols = [c for c in range(w) if len(set(a[:,c])) == 1 and a[0,c] != bg]
        if not sep_rows and not sep_cols:
            return None
        
        row_bounds = [0] + sep_rows + [h]
        col_bounds = [0] + sep_cols + [w]
        
        panels_grid = []
        for i in range(len(row_bounds)-1):
            r0, r1 = row_bounds[i], row_bounds[i+1]
            if r0 in sep_rows: r0 += 1
            if r0 >= r1: continue
            row = []
            for j in range(len(col_bounds)-1):
                c0, c1 = col_bounds[j], col_bounds[j+1]
                if c0 in sep_cols: c0 += 1
                if c0 >= c1: continue
                row.append(a[r0:r1, c0:c1])
            if row:
                panels_grid.append(row)
        
        if not panels_grid:
            return None
        
        # Trim each panel to nonbg bbox
        trimmed = []
        trim_shape = None
        for row in panels_grid:
            trow = []
            for panel in row:
                nonbg_mask = (panel != bg)
                if nonbg_mask.any():
                    rows = np.where(nonbg_mask.any(axis=1))[0]
                    cols = np.where(nonbg_mask.any(axis=0))[0]
                    t = panel[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
                else:
                    t = np.full((1, 1), bg, dtype=int)
                if trim_shape is None:
                    trim_shape = t.shape
                elif t.shape != trim_shape:
                    return None
                trow.append(t)
            trimmed.append(trow)
        
        rows_assembled = [np.concatenate(trow, axis=1) for trow in trimmed]
        result = np.concatenate(rows_assembled, axis=0)
        if result.tolist() != out:
            return None
    
    return {'type': 'panel_compact', 'bg': bg}


def apply_panel_compact(inp, bg):
    """panel_compactの適用"""
    a = np.array(inp)
    h, w = a.shape
    sep_rows = [r for r in range(h) if len(set(a[r])) == 1 and a[r,0] != bg]
    sep_cols = [c for c in range(w) if len(set(a[:,c])) == 1 and a[0,c] != bg]
    
    row_bounds = [0] + sep_rows + [h]
    col_bounds = [0] + sep_cols + [w]
    
    panels_grid = []
    for i in range(len(row_bounds)-1):
        r0, r1 = row_bounds[i], row_bounds[i+1]
        if r0 in sep_rows: r0 += 1
        if r0 >= r1: continue
        row = []
        for j in range(len(col_bounds)-1):
            c0, c1 = col_bounds[j], col_bounds[j+1]
            if c0 in sep_cols: c0 += 1
            if c0 >= c1: continue
            row.append(a[r0:r1, c0:c1])
        if row:
            panels_grid.append(row)
    
    trimmed = []
    for row in panels_grid:
        trow = []
        for panel in row:
            nonbg_mask = (panel != bg)
            if nonbg_mask.any():
                rows = np.where(nonbg_mask.any(axis=1))[0]
                cols = np.where(nonbg_mask.any(axis=0))[0]
                t = panel[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
            else:
                t = np.full((1, 1), bg, dtype=int)
            trow.append(t)
        trimmed.append(trow)
    
    rows_assembled = [np.concatenate(trow, axis=1) for trow in trimmed]
    return np.concatenate(rows_assembled, axis=0).tolist()


def _try_invert_recolor(train_pairs, bg):
    """色反転: nonbg→bg, bg→学習色"""
    color_map = {}
    for inp, out in train_pairs:
        a_in = np.array(inp)
        a_out = np.array(out)
        if a_in.shape != a_out.shape:
            return None
        in_nonbg = (a_in != bg)
        out_nonbg = (a_out != bg)
        if not np.array_equal(in_nonbg, ~out_nonbg):
            return None
        in_colors = set(int(v) for v in a_in[in_nonbg].flatten())
        out_colors = set(int(v) for v in a_out[~in_nonbg].flatten())
        if len(in_colors) != 1 or len(out_colors) != 1:
            return None
        c_in = in_colors.pop()
        c_out = out_colors.pop()
        if c_in in color_map and color_map[c_in] != c_out:
            return None
        color_map[c_in] = c_out
    return {'type': 'invert_recolor', 'bg': bg, 'color_map': color_map}


def apply_symmetrize_4fold(inp, bg):
    """Complete 4-fold symmetry around center of non-bg region."""
    a = np.array(inp)
    h, w = a.shape
    nonbg = np.where(a != bg)
    if len(nonbg[0]) == 0:
        return inp
    r_min, r_max = int(nonbg[0].min()), int(nonbg[0].max())
    c_min, c_max = int(nonbg[1].min()), int(nonbg[1].max())
    cr = (r_min + r_max) / 2
    cc = (c_min + c_max) / 2
    result = a.copy()
    for r in range(h):
        for c in range(w):
            if a[r, c] != bg:
                dr, dc = r - cr, c - cc
                for mr, mc in [
                    (int(round(cr - dr)), int(round(cc + dc))),   # mirror V
                    (int(round(cr + dr)), int(round(cc - dc))),   # mirror H
                    (int(round(cr - dr)), int(round(cc - dc))),   # mirror VH
                    (int(round(cr + dc)), int(round(cc + dr))),   # rot 90
                    (int(round(cr - dc)), int(round(cc - dr))),   # rot 270
                    (int(round(cr + dc)), int(round(cc - dr))),   # rot90+mirrorH
                    (int(round(cr - dc)), int(round(cc + dr))),   # rot270+mirrorH
                ]:
                    if 0 <= mr < h and 0 <= mc < w and result[mr, mc] == bg:
                        result[mr, mc] = int(a[r, c])
    return result.tolist()


def _try_symmetrize_4fold(train_pairs, bg):
    for inp, out in train_pairs:
        if apply_symmetrize_4fold(inp, bg) != out:
            return False
    # Must change at least one pair
    return any(apply_symmetrize_4fold(inp, bg) != inp for inp, _ in train_pairs)


def apply_invert_recolor(inp, bg, color_map):
    a = np.array(inp)
    in_colors = set(int(v) for v in a[a != bg].flatten())
    if len(in_colors) != 1:
        return None
    c_in = in_colors.pop()
    if c_in not in color_map:
        return None
    c_out = color_map[c_in]
    result = np.where(a == bg, c_out, bg)
    return result.tolist()


def _try_odd_one_out(train_pairs, bg):
    """Extract the object whose (bbox_h, bbox_w, area) is unique among all objects."""
    from scipy import ndimage
    
    for inp, out in train_pairs:
        a = np.array(inp)
        o = np.array(out)
        labeled, n = ndimage.label(a != bg)
        if n < 2:
            return False
        
        objs = []
        for lbl in range(1, n + 1):
            rows, cols = np.where(labeled == lbl)
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
            area = int((labeled == lbl).sum())
            objs.append({
                'shape_key': (r1 - r0 + 1, c1 - c0 + 1, area),
                'subgrid': a[r0:r1+1, c0:c1+1].tolist(),
            })
        
        key_counts = Counter(obj['shape_key'] for obj in objs)
        unique = [obj for obj in objs if key_counts[obj['shape_key']] == 1]
        if len(unique) != 1 or unique[0]['subgrid'] != o.tolist():
            return False
    
    return True


def apply_odd_one_out(inp, bg):
    from scipy import ndimage
    a = np.array(inp)
    labeled, n = ndimage.label(a != bg)
    if n < 2:
        return None
    
    objs = []
    for lbl in range(1, n + 1):
        rows, cols = np.where(labeled == lbl)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        area = int((labeled == lbl).sum())
        objs.append((r1 - r0 + 1, c1 - c0 + 1, area, r0, c0))
    
    key_counts = Counter((h, w, ar) for h, w, ar, _, _ in objs)
    unique = [(h, w, ar, r0, c0) for h, w, ar, r0, c0 in objs if key_counts[(h, w, ar)] == 1]
    if len(unique) != 1:
        return None
    
    h, w, _, r0, c0 = unique[0]
    return a[r0:r0+h, c0:c0+w].tolist()


def _try_midpoint_cross(train_pairs, bg):
    """2 dots of same color → cross at midpoint with learned color."""
    from collections import defaultdict
    
    in0, out0 = train_pairs[0]
    a_in = np.array(in0)
    a_out = np.array(out0)
    if a_in.shape != a_out.shape:
        return None
    h, w = a_in.shape
    
    cg = defaultdict(list)
    for r in range(h):
        for c in range(w):
            if a_in[r, c] != bg:
                cg[int(a_in[r, c])].append((r, c))
    
    dot_colors = [c for c, pos in cg.items() if len(pos) == 2]
    if len(dot_colors) != 1:
        return None
    
    dot_color = dot_colors[0]
    (r1, c1), (r2, c2) = cg[dot_color]
    mr, mc = (r1 + r2) / 2, (c1 + c2) / 2
    if mr != int(mr) or mc != int(mc):
        return None
    mr, mc = int(mr), int(mc)
    
    new_color = int(a_out[mr, mc])
    if new_color == bg:
        return None
    
    # Verify cross at midpoint
    expected = a_in.copy()
    for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = mr + dr, mc + dc
        if 0 <= nr < h and 0 <= nc < w and expected[nr, nc] == bg:
            expected[nr, nc] = new_color
    
    if not np.array_equal(expected, a_out):
        return None
    
    # Verify all pairs
    for inp, out in train_pairs[1:]:
        a_i = np.array(inp)
        a_o = np.array(out)
        if a_i.shape != a_o.shape:
            return None
        ph, pw = a_i.shape
        
        cg2 = defaultdict(list)
        for r in range(ph):
            for c in range(pw):
                if a_i[r, c] != bg:
                    cg2[int(a_i[r, c])].append((r, c))
        
        dc2 = [c for c, pos in cg2.items() if len(pos) == 2]
        if len(dc2) != 1:
            return None
        
        (r1, c1), (r2, c2) = cg2[dc2[0]]
        mr2, mc2 = (r1 + r2) / 2, (c1 + c2) / 2
        if mr2 != int(mr2) or mc2 != int(mc2):
            return None
        mr2, mc2 = int(mr2), int(mc2)
        
        exp = a_i.copy()
        for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = mr2 + dr, mc2 + dc
            if 0 <= nr < ph and 0 <= nc < pw and exp[nr, nc] == bg:
                exp[nr, nc] = new_color
        
        if not np.array_equal(exp, a_o):
            return None
    
    return {'dot_color': dot_color, 'new_color': new_color}


def apply_midpoint_cross(inp, bg, new_color):
    from collections import defaultdict
    a = np.array(inp)
    h, w = a.shape
    
    cg = defaultdict(list)
    for r in range(h):
        for c in range(w):
            if a[r, c] != bg:
                cg[int(a[r, c])].append((r, c))
    
    dc = [c for c, pos in cg.items() if len(pos) == 2]
    if len(dc) != 1:
        return None
    
    (r1, c1), (r2, c2) = cg[dc[0]]
    mr, mc = (r1 + r2) / 2, (c1 + c2) / 2
    if mr != int(mr) or mc != int(mc):
        return None
    mr, mc = int(mr), int(mc)
    
    result = a.copy()
    for dr, dcc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = mr + dr, mc + dcc
        if 0 <= nr < h and 0 <= nc < w and result[nr, nc] == bg:
            result[nr, nc] = new_color
    return result.tolist()


def _try_1x1_feature_rule(train_pairs, bg):
    """Try feature→output mapping for 1x1 output tasks."""
    from scipy import ndimage
    
    def n_distinct(inp, bg):
        return len(set(int(v) for v in np.array(inp).flatten()) - {bg})
    
    def majority_nonbg(inp, bg):
        a = np.array(inp)
        nonbg = [int(v) for v in a.flatten() if v != bg]
        if not nonbg: return -1
        return Counter(nonbg).most_common(1)[0][0]
    
    def minority_nonbg(inp, bg):
        a = np.array(inp)
        nonbg = [int(v) for v in a.flatten() if v != bg]
        if not nonbg: return -1
        return Counter(nonbg).most_common()[-1][0]
    
    def parity_bg(inp, bg):
        return int((np.array(inp) == bg).sum()) % 2
    
    def n_objects(inp, bg):
        a = np.array(inp)
        labeled, n = ndimage.label(a != bg)
        return n
    
    def parity_nonbg(inp, bg):
        return int((np.array(inp) != bg).sum()) % 2
    
    # Binary shape key (all inputs must be same size)
    in_shapes = set(np.array(inp).shape for inp, _ in train_pairs)
    if len(in_shapes) == 1:
        def binary_shape(inp, bg):
            return tuple((np.array(inp) != 0).flatten().tolist())  # use 0 as bg for shape
        features_extra = [('binary_shape', binary_shape)]
    else:
        features_extra = []
    
    features = [
        ('parity_bg', parity_bg),
        ('parity_nonbg', parity_nonbg),
        ('n_distinct', n_distinct),
        ('n_objects', n_objects),
        ('majority', majority_nonbg),
        ('minority', minority_nonbg),
    ] + features_extra
    
    candidates = []
    for feat_name, feat_fn in features:
        mapping = {}
        ok = True
        for inp, out in train_pairs:
            feat = feat_fn(inp, bg)
            out_val = int(np.array(out)[0, 0])
            if feat in mapping:
                if mapping[feat] != out_val:
                    ok = False
                    break
            else:
                mapping[feat] = out_val
        
        if ok and len(mapping) >= 2:
            # Require at least one mapping value to appear 2+ times (not all unique)
            from collections import Counter as _Cnt
            feat_counts = _Cnt(feat_fn(inp, bg) for inp, _ in train_pairs)
            has_repeated = any(v >= 2 for v in feat_counts.values())
            if has_repeated or len(train_pairs) <= 3:
                candidates.append({'name': feat_name, 'feat_fn': feat_fn, 'mapping': mapping})
    
    # Return first candidate (caller will try all via pieces)
    # But generate pieces for ALL candidates so at least one can predict test
    return candidates if candidates else None


def generate_3d_cross_pieces(train_pairs: List[Tuple[Grid, Grid]]):
    """立体cross構造由来のCrossPieceを生成する。"""
    from arc.cross_engine import CrossPiece
    
    pieces = []
    
    bg = _detect_bg(train_pairs[0][0])
    
    # Standard 3D cross rules
    rule = learn_3d_cross_rule(train_pairs)
    if rule is not None:
        def _apply(inp, _rule=rule):
            return apply_3d_cross_rule(inp, _rule)
        pieces.append(CrossPiece(
            f'cross3d:{rule["type"]}',
            _apply
        ))
    
    # Panel compact
    rule_pc = _try_panel_compact(train_pairs, bg)
    if rule_pc is not None:
        def _apply_pc(inp, _bg=bg):
            return apply_panel_compact(inp, _bg)
        pieces.append(CrossPiece('cross3d:panel_compact', _apply_pc))
    
    # Odd one out extraction
    if _try_odd_one_out(train_pairs, bg):
        def _apply_ooo(inp, _bg=bg):
            return apply_odd_one_out(inp, _bg)
        pieces.append(CrossPiece('cross3d:odd_one_out', _apply_ooo))
    
    # Midpoint cross
    rule_mc = _try_midpoint_cross(train_pairs, bg)
    if rule_mc is not None:
        _nc = rule_mc['new_color']
        def _apply_mc(inp, _bg=bg, _nc=_nc):
            return apply_midpoint_cross(inp, _bg, _nc)
        pieces.append(CrossPiece('cross3d:midpoint_cross', _apply_mc))
    
    # 4-fold symmetry completion
    if _try_symmetrize_4fold(train_pairs, bg):
        def _apply_sym4(inp, _bg=bg):
            return apply_symmetrize_4fold(inp, _bg)
        pieces.append(CrossPiece('cross3d:symmetrize_4fold', _apply_sym4))
    
    # Invert + recolor
    rule_ir = _try_invert_recolor(train_pairs, bg)
    if rule_ir is not None:
        _cmap = rule_ir['color_map']
        def _apply_ir(inp, _bg=bg, _cm=_cmap):
            return apply_invert_recolor(inp, _bg, _cm)
        pieces.append(CrossPiece('cross3d:invert_recolor', _apply_ir))
    
    # 1x1 output feature rules
    if all(np.array(out).shape == (1, 1) for _, out in train_pairs):
        rules_1x1 = _try_1x1_feature_rule(train_pairs, bg)
        if rules_1x1 is not None:
            for _rule in rules_1x1:
                _feat_fn = _rule['feat_fn']
                _mapping = _rule['mapping']
                _name = _rule['name']
                def _apply_1x1(inp, _fn=_feat_fn, _m=_mapping, _bg=bg):
                    feat = _fn(inp, _bg)
                    if feat in _m:
                        return [[_m[feat]]]
                    return None
                pieces.append(CrossPiece(f'cross3d:1x1_{_name}', _apply_1x1))
    
    return pieces
