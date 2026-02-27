"""
arc/cross_multiscale.py — Multi-Scale Cross Structure

kofdai設計思想:
- 15x15グリッドを3x3/5x5のcrossブロックにカット
- crossブロック間の空間関係をcross構造で把握
- 大きいcrossが小さいcrossを包含 = 穴にオブジェクトがはまる
- 次元(サイズ)が数値で確定 → 配置がずれない
- ノイズはcross構造に属さない要素として除去

4層構成:
  Layer 1: マクロ構造認識（グリッド→crossブロック分割）
  Layer 2: オブジェクト検出（cross同士の相対大きさ・位置）
  Layer 3: ノイズ除去（cross構造に属さない＝ノイズ）
  Layer 4: 出力再構成（cross構造→出力次元マッピング）
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from scipy import ndimage
from arc.grid import Grid, grid_shape, grid_eq, most_common_color, grid_colors
from arc.cross_engine import CrossPiece


# ---- Cross Block ----

class CrossBlock:
    """グリッド内の矩形領域をcross構造の1ブロックとして表現"""
    def __init__(self, grid: np.ndarray, top: int, left: int, height: int, width: int, 
                 block_id: int = 0, color: int = -1):
        self.grid = grid  # full grid reference
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.block_id = block_id
        self.color = color  # dominant non-bg color
        self.area = height * width
        
        # Extract subgrid
        self.subgrid = grid[top:top+height, left:left+width].copy()
    
    @property
    def bottom(self) -> int:
        return self.top + self.height
    
    @property
    def right(self) -> int:
        return self.left + self.width
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.top + self.height / 2, self.left + self.width / 2)
    
    def contains(self, other: 'CrossBlock') -> bool:
        """このブロックが他のブロックを完全に包含するか"""
        return (self.top <= other.top and self.left <= other.left and
                self.bottom >= other.bottom and self.right >= other.right)
    
    def overlaps(self, other: 'CrossBlock') -> bool:
        return not (self.bottom <= other.top or other.bottom <= self.top or
                    self.right <= other.left or other.right <= self.left)
    
    def hole_positions(self, bg: int = 0) -> List[Tuple[int, int, int, int]]:
        """ブロック内の穴（bg色の矩形領域）を検出"""
        holes = []
        mask = (self.subgrid == bg)
        labeled, n = ndimage.label(mask)
        for lbl in range(1, n + 1):
            positions = np.argwhere(labeled == lbl)
            if len(positions) == 0:
                continue
            r_min, c_min = positions.min(axis=0)
            r_max, c_max = positions.max(axis=0)
            h = r_max - r_min + 1
            w = c_max - c_min + 1
            # Check if it's a rectangular hole (not edge-touching)
            if r_min > 0 and c_min > 0 and r_max < self.height - 1 and c_max < self.width - 1:
                holes.append((r_min, c_min, h, w))
        return holes
    
    def probe_holes(self, bg: int = 0) -> List[Tuple[int, int, int, int, Optional[List]]]:
        """
        プローブ方式による穴検出 — cross構造交差認識
        
        1. 内部bg領域を検出（ボーダー接続を除外）
        2. 十字型/L字型の穴は中心点から4方向にプローブ
        3. 各腕の長さを測定 → 腕ごとに分離した穴として返す
        
        Returns: [(r, c, h, w, arms), ...]
          arms = None (矩形穴) or [(dir, length, width), ...] (十字型)
        """
        if self.height < 3 or self.width < 3:
            return []
        
        mask = (self.subgrid == bg)
        border_mask = np.zeros_like(mask)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        
        from scipy.ndimage import label as nd_label
        bg_labeled, n_bg = nd_label(mask)
        border_labels = set()
        for lbl in range(1, n_bg + 1):
            if np.any(bg_labeled[border_mask] == lbl):
                border_labels.add(lbl)
        
        interior_hole_mask = np.zeros_like(mask)
        for lbl in range(1, n_bg + 1):
            if lbl not in border_labels:
                interior_hole_mask |= (bg_labeled == lbl)
        
        hole_labeled, n_holes = nd_label(interior_hole_mask)
        results = []
        
        for lbl in range(1, n_holes + 1):
            positions = np.argwhere(hole_labeled == lbl)
            if len(positions) == 0:
                continue
            r_min, c_min = positions.min(axis=0)
            r_max, c_max = positions.max(axis=0)
            region_h = r_max - r_min + 1
            region_w = c_max - c_min + 1
            n_pixels = len(positions)
            
            # Check if it's a simple rectangle
            if n_pixels == region_h * region_w:
                results.append((int(r_min), int(c_min), int(region_h), int(region_w), None))
                continue
            
            # Non-rectangular: cross/L/T shape
            # Decompose into arms by probing from each bg pixel in 4 directions
            region_mask = np.zeros((region_h, region_w), dtype=bool)
            for pos in positions:
                region_mask[pos[0] - r_min, pos[1] - c_min] = True
            
            # Find arms: maximal horizontal and vertical runs
            arms = self._decompose_cross_arms(region_mask, r_min, c_min)
            
            if arms:
                for arm_r, arm_c, arm_h, arm_w in arms:
                    results.append((int(arm_r), int(arm_c), int(arm_h), int(arm_w), None))
            else:
                # Fallback: use bounding box
                results.append((int(r_min), int(c_min), int(region_h), int(region_w), None))
        
        return results
    
    def _decompose_cross_arms(self, mask: np.ndarray, 
                               offset_r: int, offset_c: int) -> List[Tuple[int,int,int,int]]:
        """
        十字型穴を腕に分解
        
        中心点を検出し、4方向にプローブして各腕を独立した矩形として返す
        """
        h, w = mask.shape
        
        # Find all maximal horizontal runs
        h_runs = []
        for r in range(h):
            c_start = None
            for c in range(w + 1):
                if c < w and mask[r, c]:
                    if c_start is None:
                        c_start = c
                else:
                    if c_start is not None:
                        h_runs.append((r, c_start, 1, c - c_start))
                        c_start = None
        
        # Find all maximal vertical runs
        v_runs = []
        for c in range(w):
            r_start = None
            for r in range(h + 1):
                if r < h and mask[r, c]:
                    if r_start is None:
                        r_start = r
                else:
                    if r_start is not None:
                        v_runs.append((r_start, c, r - r_start, 1))
                        r_start = None
        
        # Merge adjacent same-width horizontal runs into taller rectangles
        # Group h_runs by (c_start, width)
        from collections import defaultdict
        h_groups = defaultdict(list)
        for r, c, rh, rw in h_runs:
            h_groups[(c, rw)].append(r)
        
        merged_h = []
        for (c_start, rw), rows in h_groups.items():
            rows.sort()
            # Merge consecutive rows
            i = 0
            while i < len(rows):
                start = rows[i]
                end = start
                while i + 1 < len(rows) and rows[i + 1] == end + 1:
                    end = rows[i + 1]
                    i += 1
                merged_h.append((start + offset_r, c_start + offset_c, end - start + 1, rw))
                i += 1
        
        # Same for vertical
        v_groups = defaultdict(list)
        for r, c, rh, rw in v_runs:
            v_groups[(r, rh)].append(c)
        
        merged_v = []
        for (r_start, rh), cols in v_groups.items():
            cols.sort()
            i = 0
            while i < len(cols):
                start = cols[i]
                end = start
                while i + 1 < len(cols) and cols[i + 1] == end + 1:
                    end = cols[i + 1]
                    i += 1
                merged_v.append((r_start + offset_r, start + offset_c, rh, end - start + 1))
                i += 1
        
        # Combine: pick non-overlapping rectangles that cover the most pixels
        all_rects = merged_h + merged_v
        if not all_rects:
            return []
        
        # Sort by area descending — greedy cover
        all_rects.sort(key=lambda x: -(x[2] * x[3]))
        
        covered = set()
        selected = []
        for r, c, rh, rw in all_rects:
            pixels = set()
            for dr in range(rh):
                for dc in range(rw):
                    pr, pc = r - offset_r, c - offset_c
                    if 0 <= pr + dr < mask.shape[0] and 0 <= pc + dc < mask.shape[1]:
                        if mask[pr + dr, pc + dc]:
                            pixels.add((pr + dr, pc + dc))
            
            new_pixels = pixels - covered
            if len(new_pixels) > 0:
                selected.append((r, c, rh, rw))
                covered |= pixels
        
        return selected if selected else []

    def cross_descriptor(self, bg: int = 0) -> Dict:
        """
        6軸Cross記述子: 図形をcross構造として正確にマッピング
        
        Axis 1: Position (center_r, center_c) — 空間座標
        Axis 2: Scale (height, width, pixel_count) — サイズ次元
        Axis 3: Color distribution — 各色の出現比率
        Axis 4: Shape signature — 塗りつぶし率、アスペクト比、穴の有無
        Axis 5: Orientation — 主軸方向（横長/縦長/正方形）
        Axis 6: Adjacency — 隣接する他オブジェクトとの関係
        """
        pixel_count = getattr(self, '_pixel_count', self.area)
        fill_ratio = getattr(self, '_fill_ratio', 1.0)
        
        # Axis 3: Color distribution
        color_dist = {}
        for r in range(self.height):
            for c in range(self.width):
                v = int(self.subgrid[r, c])
                if v != bg:
                    color_dist[v] = color_dist.get(v, 0) + 1
        total_px = sum(color_dist.values()) or 1
        color_ratios = {k: v / total_px for k, v in color_dist.items()}
        dominant_color = max(color_dist, key=color_dist.get) if color_dist else bg
        
        # Axis 4: Shape
        aspect = self.width / self.height if self.height > 0 else 1.0
        holes = self.hole_positions(bg)
        
        # Axis 5: Orientation
        if self.height > self.width:
            orientation = 'vertical'
        elif self.width > self.height:
            orientation = 'horizontal'
        else:
            orientation = 'square'
        
        return {
            'position': self.center,
            'scale': (self.height, self.width, pixel_count),
            'color_dist': color_ratios,
            'dominant_color': dominant_color,
            'shape': (fill_ratio, aspect, len(holes)),
            'orientation': orientation,
            'n_holes': len(holes),
            'holes': holes,
        }
    
    def __repr__(self):
        return f"CrossBlock(id={self.block_id}, {self.height}x{self.width} @({self.top},{self.left}), color={self.color})"


# ---- Multi-Scale Cross Universe ----

class MultiScaleCross:
    """
    多スケールCross構造
    
    グリッドを複数スケールで分解:
    - Macro: 大きな矩形オブジェクト（フレーム/コンテナ）
    - Meso: 中サイズオブジェクト（穴に入る候補）
    - Micro: 小さいオブジェクト（ドット/マーカー）
    - Noise: cross構造に属さない孤立セル
    """
    
    def __init__(self, grid: Grid, bg: int = 0):
        self.raw = np.array(grid, dtype=np.int8)
        self.bg = bg
        self.h, self.w = self.raw.shape
        
        # Layer 1: Object detection at all scales
        self.all_objects: List[CrossBlock] = []
        self.macro_blocks: List[CrossBlock] = []  # large containers
        self.meso_blocks: List[CrossBlock] = []   # medium objects
        self.micro_blocks: List[CrossBlock] = []  # small markers
        self.noise_mask: np.ndarray = np.zeros_like(self.raw, dtype=bool)
        
        self._detect_all_objects()
        self._classify_by_scale()
        self._detect_noise()
    
    def _detect_all_objects(self):
        """全オブジェクトを検出（connected component per color）"""
        bid = 0
        for color in range(1, 10):
            mask = (self.raw == color)
            if not mask.any():
                continue
            labeled, n = ndimage.label(mask)
            for lbl in range(1, n + 1):
                positions = np.argwhere(labeled == lbl)
                if len(positions) == 0:
                    continue
                r_min, c_min = positions.min(axis=0)
                r_max, c_max = positions.max(axis=0)
                h = r_max - r_min + 1
                w = c_max - c_min + 1
                
                block = CrossBlock(self.raw, r_min, c_min, h, w, bid, color)
                block._pixel_count = len(positions)
                block._fill_ratio = len(positions) / (h * w) if h * w > 0 else 0
                self.all_objects.append(block)
                bid += 1
    
    def _classify_by_scale(self):
        """オブジェクトをスケールで分類"""
        if not self.all_objects:
            return
        
        areas = [obj.area for obj in self.all_objects]
        max_area = max(areas) if areas else 1
        grid_area = self.h * self.w
        
        for obj in self.all_objects:
            pixel_count = getattr(obj, '_pixel_count', obj.area)
            fill_ratio = getattr(obj, '_fill_ratio', 1.0)
            
            # Macro: large objects with holes (containers/frames)
            if obj.area >= grid_area * 0.1 and fill_ratio > 0.3:
                holes = obj.hole_positions(self.bg)
                if holes or fill_ratio > 0.5:
                    self.macro_blocks.append(obj)
                    continue
            
            # Micro: very small objects (1-4 pixels)
            if pixel_count <= 4:
                self.micro_blocks.append(obj)
                continue
            
            # Meso: everything else
            self.meso_blocks.append(obj)
    
    def _detect_noise(self):
        """
        ノイズ除去層: cross構造に属さない孤立セルを検出
        
        ノイズの定義:
        - 他のオブジェクトと同色でない孤立ピクセル
        - macro/mesoブロックの外にある小さいオブジェクト
        - 繰り返しパターンに属さないランダム配置
        """
        # Simple heuristic: isolated single pixels not part of any macro/meso
        macro_meso_mask = np.zeros_like(self.raw, dtype=bool)
        for obj in self.macro_blocks + self.meso_blocks:
            macro_meso_mask[obj.top:obj.bottom, obj.left:obj.right] = True
        
        # Single pixels outside macro/meso regions
        for obj in self.micro_blocks:
            pixel_count = getattr(obj, '_pixel_count', 1)
            if pixel_count == 1:
                # Check if this single pixel is isolated
                r, c = obj.top, obj.left
                if not macro_meso_mask[r, c]:
                    # Check if there are other same-color objects nearby
                    nearby = False
                    for other in self.all_objects:
                        if other.block_id == obj.block_id:
                            continue
                        if other.color == obj.color and abs(other.center[0] - r) + abs(other.center[1] - c) < 3:
                            nearby = True
                            break
                    if not nearby:
                        self.noise_mask[r, c] = True
    
    def denoise(self) -> np.ndarray:
        """ノイズ除去したグリッドを返す"""
        result = self.raw.copy()
        result[self.noise_mask] = self.bg
        return result
    
    def find_hole_fillers(self) -> List[Tuple[CrossBlock, Tuple[int,int,int,int], 'CrossBlock', np.ndarray]]:
        """
        6軸Cross記述子ベースの穴-filler マッチング
        
        各穴と各filler候補の6軸記述子を比較:
        - Scale軸: fillerサイズ ≈ 穴サイズ（回転/tile考慮）
        - Color軸: fillerの色が穴の期待色と一致
        - Position軸: fillerと穴の空間的近接性
        - Shape軸: fillerの形状が穴の形状と整合
        """
        # Filter: noise should not be fillers
        candidates = [obj for obj in self.meso_blocks + self.micro_blocks
                      if getattr(obj, '_pixel_count', 1) >= 2]
        
        # Build 6-axis descriptors for all candidates
        cand_descriptors = [(c, c.cross_descriptor(self.bg)) for c in candidates]
        
        # Collect all (hole, container) pairs with descriptors
        # Use probe_holes for consistent detection across training pairs
        hole_list = []
        for macro in self.macro_blocks:
            macro_desc = macro.cross_descriptor(self.bg)
            # Try probe_holes first (more robust), fall back to hole_positions
            probe_results = macro.probe_holes(self.bg)
            if probe_results:
                holes = [(r, c, h, w) for r, c, h, w, _ in probe_results]
            else:
                holes = macro.hole_positions(self.bg)
            for hole_r, hole_c, hole_h, hole_w in holes:
                hole_center = (macro.top + hole_r + hole_h/2, 
                             macro.left + hole_c + hole_w/2)
                hole_list.append((macro, (hole_r, hole_c, hole_h, hole_w), 
                                hole_center, macro_desc))
        
        # Score all (hole, candidate) pairs using 6-axis matching
        scored_matches = []
        
        for macro, (hr, hc, hh, hw), hole_center, macro_desc in hole_list:
            for cand, cand_desc in cand_descriptors:
                if macro.contains(cand):
                    continue
                
                filler_grid = cand.subgrid.copy()
                
                # Try orientations (B方式)
                orientations = self._generate_orientations(filler_grid)
                
                for oriented in orientations:
                    oh, ow = oriented.shape
                    
                    # === 6軸スコアリング ===
                    
                    # Axis 2 (Scale): size compatibility
                    if oh == hh and ow == hw:
                        scale_score = 1.0
                        final_grid = oriented
                    elif oh <= hh and ow <= hw:
                        # 立体推定: tile to fit
                        tiled = self._tile_to_fit(oriented, hh, hw)
                        if tiled is not None:
                            scale_score = 0.8
                            final_grid = tiled
                        else:
                            continue
                    else:
                        continue
                    
                    # Axis 1 (Position): spatial proximity
                    filler_center = cand_desc['position']
                    dist = abs(hole_center[0] - filler_center[0]) + abs(hole_center[1] - filler_center[1])
                    max_dist = self.h + self.w
                    position_score = 1.0 - dist / max_dist
                    
                    # Axis 3 (Color): color compatibility
                    # filler's color should NOT be the container's color
                    color_score = 1.0 if cand_desc['dominant_color'] != macro_desc['dominant_color'] else 0.2
                    
                    # Axis 4 (Shape): dimension alignment
                    # Prefer fillers where at least one dimension matches exactly
                    if oh == hh or ow == hw:
                        shape_score = 1.0  # one dimension matches → natural tile
                    elif oh == hw or ow == hh:
                        shape_score = 0.7  # transposed match
                    else:
                        filler_aspect = ow / oh if oh > 0 else 1.0
                        hole_aspect = hw / hh if hh > 0 else 1.0
                        shape_score = max(0, 1.0 - abs(filler_aspect - hole_aspect))
                    
                    # Axis 5 (Orientation): orientation match
                    orient_score = 1.0  # already tried rotations
                    
                    # Axis 6 (Adjacency): uniqueness — prefer fillers that appear 
                    # as isolated objects (not part of a larger pattern)
                    adj_score = 1.0 if getattr(cand, '_pixel_count', 0) <= hh * hw * 2 else 0.5
                    
                    # Weighted combination
                    total_score = (scale_score * 0.30 + 
                                 position_score * 0.15 +
                                 color_score * 0.25 +
                                 shape_score * 0.15 +
                                 orient_score * 0.05 +
                                 adj_score * 0.10)
                    
                    scored_matches.append((total_score, macro, (hr, hc, hh, hw), cand, final_grid))
        
        # Greedy assignment: best score first, no reuse of holes or fillers
        scored_matches.sort(key=lambda x: -x[0])
        used_holes = set()
        used_fillers = set()
        final = []
        
        for score, container, (hr, hc, hh, hw), filler, oriented in scored_matches:
            hole_key = (container.block_id, hr, hc)
            if hole_key in used_holes or filler.block_id in used_fillers:
                continue
            if score < 0.3:
                continue
            used_holes.add(hole_key)
            used_fillers.add(filler.block_id)
            final.append((container, (hr, hc, hh, hw), filler, oriented))
        
        return final
    
    def _generate_orientations(self, grid: np.ndarray) -> List[np.ndarray]:
        """回転/転置の全バリエーションを生成"""
        results = [grid]
        for k in [1, 2, 3]:
            r = np.rot90(grid, k)
            if not any(np.array_equal(r, existing) for existing in results):
                results.append(r)
        for flip_fn in [np.fliplr, np.flipud]:
            f = flip_fn(grid)
            if not any(np.array_equal(f, existing) for existing in results):
                results.append(f)
        if grid.shape[0] != grid.shape[1]:
            t = grid.T
            if not any(np.array_equal(t, existing) for existing in results):
                results.append(t)
        return results
    
    def _tile_to_fit(self, filler: np.ndarray, target_h: int, target_w: int) -> Optional[np.ndarray]:
        """
        立体サイズ推定: fillerを穴サイズに拡張
        
        - 片方の次元が一致 → もう片方を繰り返し(tile)で埋める
        - 例: 3x1 filler → 3x2 hole → 各行を2回繰り返し
        - 例: 1x5 filler → 2x5 hole → 各列を2回繰り返し
        """
        fh, fw = filler.shape
        
        # Exact fit
        if fh == target_h and fw == target_w:
            return filler.copy()
        
        # One dimension matches → tile the other
        if fh == target_h and target_w % fw == 0:
            reps = target_w // fw
            return np.tile(filler, (1, reps))
        
        if fw == target_w and target_h % fh == 0:
            reps = target_h // fh
            return np.tile(filler, (reps, 1))
        
        # Fill with the filler's dominant non-bg color
        # 3x1 → 3x2: fill entire hole with filler's color
        non_bg = filler[filler != self.bg]
        if len(non_bg) > 0 and len(set(non_bg.tolist())) == 1:
            color = int(non_bg[0])
            # If filler is single-color, fill entire hole
            result = np.full((target_h, target_w), color, dtype=filler.dtype)
            return result
        
        return None
    
    def _match_score(self, match) -> float:
        """Score a hole-filler match by size fit + spatial proximity."""
        container, (hr, hc, hh, hw), filler, oriented = match
        oh, ow = oriented.shape
        
        # Size fit: exact match = 1.0
        size_score = 1.0 if (oh == hh and ow == hw) else 0.5
        
        # Spatial proximity: distance between filler and hole (in grid coords)
        hole_center_r = container.top + hr + hh / 2
        hole_center_c = container.left + hc + hw / 2
        filler_center_r, filler_center_c = filler.center
        dist = abs(hole_center_r - filler_center_r) + abs(hole_center_c - filler_center_c)
        max_dist = self.h + self.w
        proximity_score = 1.0 - dist / max_dist
        
        return size_score * 0.6 + proximity_score * 0.4
    
    def find_dot_to_cross_mapping(self) -> Optional[Dict]:
        """
        ドット（micro）の位置 → cross構造の空間配置を学習
        
        例: ドットの行位置に基づいて横線を描く
        """
        if not self.micro_blocks:
            return None
        
        # Check if micro blocks define spatial regions
        # Sort by position
        sorted_micros = sorted(self.micro_blocks, key=lambda o: o.top)
        
        mapping = {
            'dots': [(m.top, m.left, m.color) for m in sorted_micros],
            'grid_h': self.h,
            'grid_w': self.w,
        }
        
        return mapping


# ---- Transformation Generators ----

def _try_hole_fill_transform(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    穴埋め変換: macroブロックの穴にmesoブロックをはめ込む
    """
    pieces = []
    
    # Analyze first training pair
    inp0, out0 = train_pairs[0]
    ms_in = MultiScaleCross(inp0, bg)
    ms_out = MultiScaleCross(out0, bg)
    
    fillers = ms_in.find_hole_fillers()
    if not fillers:
        return pieces
    
    # Learn the fill pattern from training pairs
    # First verify on train[0] to understand the mapping
    
    def _apply_hole_fill(inp, _bg=bg):
        ms = MultiScaleCross(inp, _bg)
        result = ms.denoise()  # Layer 3: denoise first
        
        fillers = ms.find_hole_fillers()
        used_fillers = set()
        
        for container, (hole_r, hole_c, hole_h, hole_w), filler, oriented in fillers:
            if filler.block_id in used_fillers:
                continue
            
            # Place oriented filler into container's hole
            abs_r = container.top + hole_r
            abs_c = container.left + hole_c
            
            oh, ow = oriented.shape
            for r in range(min(oh, hole_h)):
                for c in range(min(ow, hole_w)):
                    val = int(oriented[r, c])
                    if val != _bg:
                        result[abs_r + r, abs_c + c] = val
            
            used_fillers.add(filler.block_id)
        
        # Remove ALL external objects that are not macro blocks
        # (fillers placed into holes + noise + unused external objects)
        macro_mask = np.zeros_like(result, dtype=bool)
        for m_block in ms.macro_blocks:
            macro_mask[m_block.top:m_block.bottom, m_block.left:m_block.right] = True
        
        # Clear everything outside macro blocks
        for r in range(result.shape[0]):
            for c in range(result.shape[1]):
                if not macro_mask[r, c]:
                    result[r, c] = _bg
        
        return result.tolist()
    
    pieces.append(CrossPiece('multiscale:hole_fill', _apply_hole_fill))
    return pieces


def _try_dot_to_cross_lines(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    ドット→cross線変換: 各ドットの位置にcross構造（十字線/枠線）を描く
    """
    pieces = []
    
    inp0, out0 = train_pairs[0]
    out0_np = np.array(out0)
    ms_in = MultiScaleCross(inp0, bg)
    
    if not ms_in.micro_blocks:
        return pieces
    
    # Check if dots define horizontal bands
    dots = sorted(ms_in.micro_blocks, key=lambda o: o.top)
    
    # Learn: does each dot create a horizontal line at its row?
    h, w = grid_shape(inp0)
    
    # Check pattern: dot at row r, color c → entire row r filled with color c
    pattern_h_line = True
    pattern_cross = True
    
    for dot in dots:
        r, c_pos, color = dot.top, dot.left, dot.color
        # Check if row r in output is filled with this color
        if not all(out0_np[r, :] == color):
            pattern_h_line = False
        # Check if both row and col are filled
        if not (all(out0_np[r, :] == color) or all(out0_np[:, c_pos] == color)):
            pattern_cross = False
    
    if pattern_h_line:
        def _apply_h_lines(inp, _bg=bg):
            ms = MultiScaleCross(inp, _bg)
            result = np.full_like(ms.raw, _bg)
            dots = sorted(ms.micro_blocks, key=lambda o: o.top)
            
            # Determine regions between dots
            h, w = ms.h, ms.w
            boundaries = [0] + [d.top for d in dots] + [h]
            
            for i, dot in enumerate(dots):
                r = dot.top
                color = dot.color
                # Fill horizontal line
                result[r, :] = color
                # Fill borders in the region around this dot
                region_top = boundaries[i]
                region_bot = boundaries[i + 2] if i + 2 < len(boundaries) else h
                for row in range(region_top, region_bot):
                    result[row, 0] = color
                    result[row, w - 1] = color
            
            return result.tolist()
        
        pieces.append(CrossPiece('multiscale:dot_to_hlines', _apply_h_lines))
    
    if pattern_cross:
        def _apply_cross_lines(inp, _bg=bg):
            ms = MultiScaleCross(inp, _bg)
            result = np.full_like(ms.raw, _bg)
            
            for dot in ms.micro_blocks:
                r, c, color = dot.top, dot.left, dot.color
                result[r, :] = color  # horizontal
                result[:, c] = color  # vertical
            
            return result.tolist()
        
        pieces.append(CrossPiece('multiscale:dot_to_cross', _apply_cross_lines))
    
    return pieces


def _try_denoise_and_keep(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    ノイズ除去変換: cross構造に属さない孤立セルを除去
    """
    pieces = []
    
    inp0, out0 = train_pairs[0]
    ms = MultiScaleCross(inp0, bg)
    
    if not ms.noise_mask.any():
        return pieces
    
    denoised = ms.denoise()
    if grid_eq(denoised.tolist(), out0):
        def _apply_denoise(inp, _bg=bg):
            ms = MultiScaleCross(inp, _bg)
            return ms.denoise().tolist()
        pieces.append(CrossPiece('multiscale:denoise', _apply_denoise))
    
    return pieces


def _try_macro_extract_recolor(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    macroブロック抽出 + 内部パターンの色変換
    """
    pieces = []
    
    inp0, out0 = train_pairs[0]
    out0_np = np.array(out0)
    ms_in = MultiScaleCross(inp0, bg)
    
    if not ms_in.macro_blocks:
        return pieces
    
    # Check if output is just the macro blocks with modifications
    # (noise removed, holes filled, colors changed)
    for macro in ms_in.macro_blocks:
        # Check if output has this macro block in same position
        out_region = out0_np[macro.top:macro.bottom, macro.left:macro.right]
        in_region = ms_in.raw[macro.top:macro.bottom, macro.left:macro.right]
        
        # Check if non-bg cells in macro are preserved
        preserved = True
        for r in range(macro.height):
            for c in range(macro.width):
                if in_region[r, c] == macro.color and out_region[r, c] != macro.color:
                    preserved = False
                    break
            if not preserved:
                break
    
    return pieces


# ---- Main Entry Point ----

def generate_multiscale_cross_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """
    多スケールCross構造に基づく変換候補を生成
    """
    pieces = []
    
    # Determine background color
    inp0, out0 = train_pairs[0]
    bg = most_common_color(inp0)
    
    # Try each transformation type
    pieces.extend(_try_hole_fill_transform(train_pairs, bg))
    pieces.extend(_try_dot_to_cross_lines(train_pairs, bg))
    pieces.extend(_try_denoise_and_keep(train_pairs, bg))
    pieces.extend(_try_macro_extract_recolor(train_pairs, bg))
    
    return pieces
