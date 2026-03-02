"""
arc/cross3d.py — Cross3D: ARCグリッドのCross構造表現

全ての問題をまずCross3D空間に変換してから解く。

表現:
  - grid (H x W, 色0-9) → CrossVolume (10 x H x W, bool/tag)
  - 各層 = 1色に対応するCross構造
  - 背景色の層も明示的にCrossで埋める
  - 各Crossにはサイズ・位置・タグ情報がある

Cross構造:
  - CrossCell: 1セル単位のCross要素
  - CrossRegion: 連結した同色セルの集合 = 1つのCross塊
  - CrossVolume: 全層の集合 = 問題の完全なCross表現
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from collections import Counter
from scipy.ndimage import label as scipy_label


@dataclass
class CrossRegion:
    """連結した同色セルの集合 = 1つのCross塊"""
    color: int
    cells: Set[Tuple[int, int]]  # (row, col) の集合
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (r_min, c_min, r_max, c_max)
    is_background: bool = False
    
    def compute_bbox(self):
        if not self.cells:
            return
        rows = [r for r, c in self.cells]
        cols = [c for r, c in self.cells]
        self.bbox = (min(rows), min(cols), max(rows), max(cols))
    
    @property
    def size(self) -> int:
        return len(self.cells)
    
    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1
    
    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1
    
    @property
    def center(self) -> Tuple[float, float]:
        if not self.cells:
            return (0, 0)
        rows = [r for r, c in self.cells]
        cols = [c for r, c in self.cells]
        return (sum(rows) / len(rows), sum(cols) / len(cols))
    
    def distance_to(self, other: CrossRegion) -> float:
        """別のCrossRegionとの最近接距離"""
        min_dist = float('inf')
        for r1, c1 in self.cells:
            for r2, c2 in other.cells:
                d = abs(r1 - r2) + abs(c1 - c2)  # Manhattan
                if d < min_dist:
                    min_dist = d
        return min_dist
    
    def center_distance_to(self, other: CrossRegion) -> float:
        """中心間距離"""
        c1 = self.center
        c2 = other.center
        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) ** 0.5
    
    def translate(self, dr: int, dc: int) -> CrossRegion:
        """移動"""
        new_cells = {(r + dr, c + dc) for r, c in self.cells}
        region = CrossRegion(color=self.color, cells=new_cells, is_background=self.is_background)
        region.compute_bbox()
        return region
    
    def recolor(self, new_color: int) -> CrossRegion:
        """色変更"""
        region = CrossRegion(color=new_color, cells=set(self.cells), is_background=self.is_background)
        region.bbox = self.bbox
        return region
    
    def to_mask(self, h: int, w: int) -> np.ndarray:
        """H x W のboolマスク"""
        mask = np.zeros((h, w), dtype=bool)
        for r, c in self.cells:
            if 0 <= r < h and 0 <= c < w:
                mask[r, c] = True
        return mask
    
    def overlap(self, other: CrossRegion) -> int:
        """重なるセル数"""
        return len(self.cells & other.cells)
    
    def contains(self, other: CrossRegion) -> bool:
        """otherが完全に内包されているか"""
        return other.cells.issubset(self.cells)
    
    def shape_signature(self) -> Tuple:
        """形状のシグネチャ（平行移動不変）"""
        if not self.cells:
            return ()
        r_min = min(r for r, c in self.cells)
        c_min = min(c for r, c in self.cells)
        normalized = tuple(sorted((r - r_min, c - c_min) for r, c in self.cells))
        return normalized
    
    def __repr__(self):
        tag = "BG" if self.is_background else f"C{self.color}"
        return f"CrossRegion({tag}, size={self.size}, bbox={self.bbox})"


@dataclass
class CrossLayer:
    """1色分のCross層"""
    color: int
    regions: List[CrossRegion]
    is_background: bool = False
    
    @property
    def total_cells(self) -> int:
        return sum(r.size for r in self.regions)
    
    @property
    def n_regions(self) -> int:
        return len(self.regions)


class CrossVolume:
    """全層の集合 = 問題の完全なCross3D表現"""
    
    def __init__(self, grid: List[List[int]], is_output: bool = False):
        self.grid = grid
        self.h = len(grid)
        self.w = len(grid[0]) if grid else 0
        self.is_output = is_output
        
        g = np.array(grid)
        
        # 背景色検出
        color_counts = Counter(g.flatten())
        self.bg_color = color_counts.most_common(1)[0][0]
        
        # 全色を検出
        self.colors = sorted(set(g.flatten()))
        
        # 各色のCross層を構築
        self.layers: Dict[int, CrossLayer] = {}
        for color in self.colors:
            mask = (g == color)
            regions = self._extract_regions(mask, color)
            is_bg = (color == self.bg_color)
            
            # 背景タグ付け
            for region in regions:
                region.is_background = is_bg
            
            self.layers[color] = CrossLayer(
                color=color, 
                regions=regions, 
                is_background=is_bg
            )
        
        # 全リージョンのフラットリスト（非背景のみ）
        self.objects: List[CrossRegion] = []
        for color, layer in self.layers.items():
            if not layer.is_background:
                self.objects.extend(layer.regions)
    
    def _extract_regions(self, mask: np.ndarray, color: int) -> List[CrossRegion]:
        """マスクから連結領域を抽出"""
        if not mask.any():
            return []
        
        labeled, n = scipy_label(mask)
        regions = []
        for i in range(1, n + 1):
            cells = set(zip(*np.where(labeled == i)))
            region = CrossRegion(color=color, cells=cells)
            region.compute_bbox()
            regions.append(region)
        
        # サイズ降順
        regions.sort(key=lambda r: -r.size)
        return regions
    
    def to_grid(self) -> List[List[int]]:
        """Cross3D表現からgridに逆変換"""
        grid = [[self.bg_color] * self.w for _ in range(self.h)]
        
        # 非背景レイヤーを塗る（小さいリージョンを後に = 上書き優先）
        all_regions = []
        for layer in self.layers.values():
            if not layer.is_background:
                all_regions.extend(layer.regions)
        
        # サイズ大→小の順で塗る
        all_regions.sort(key=lambda r: -r.size)
        for region in all_regions:
            for r, c in region.cells:
                if 0 <= r < self.h and 0 <= c < self.w:
                    grid[r][c] = region.color
        
        return grid
    
    # ─── Cross操作 ───
    
    def find_matching_regions(self, source: CrossVolume) -> List[Tuple[CrossRegion, CrossRegion]]:
        """入力と出力のCrossVolume間で形状が一致するリージョンを見つける"""
        matches = []
        for obj_s in source.objects:
            sig_s = obj_s.shape_signature()
            for obj_t in self.objects:
                sig_t = obj_t.shape_signature()
                if sig_s == sig_t:
                    matches.append((obj_s, obj_t))
        return matches
    
    def detect_translations(self, other: CrossVolume) -> List[Dict]:
        """self(入力) → other(出力) の移動を検出"""
        translations = []
        
        for color in self.colors:
            if color == self.bg_color:
                continue
            
            if color not in self.layers or color not in other.layers:
                continue
            
            src_regions = self.layers[color].regions
            dst_regions = other.layers[color].regions
            
            for sr in src_regions:
                for dr in dst_regions:
                    if sr.shape_signature() == dr.shape_signature():
                        # Same shape — detect translation
                        sr_min_r = min(r for r, c in sr.cells)
                        sr_min_c = min(c for r, c in sr.cells)
                        dr_min_r = min(r for r, c in dr.cells)
                        dr_min_c = min(c for r, c in dr.cells)
                        
                        delta_r = dr_min_r - sr_min_r
                        delta_c = dr_min_c - sr_min_c
                        
                        if delta_r != 0 or delta_c != 0:
                            translations.append({
                                'color': color,
                                'delta': (delta_r, delta_c),
                                'src_bbox': sr.bbox,
                                'dst_bbox': dr.bbox,
                                'size': sr.size,
                            })
        
        return translations
    
    def detect_recoloring(self, other: CrossVolume) -> List[Dict]:
        """self(入力) → other(出力) の色変更を検出"""
        recolorings = []
        
        for sr in self.objects:
            sig = sr.shape_signature()
            for dr in other.objects:
                if dr.shape_signature() == sig and sr.color != dr.color:
                    # Same position check
                    if sr.cells == dr.cells:
                        recolorings.append({
                            'from_color': sr.color,
                            'to_color': dr.color,
                            'cells': sr.cells,
                            'size': sr.size,
                        })
        
        return recolorings
    
    def detect_cuts(self, other: CrossVolume) -> List[Dict]:
        """self(入力) → other(出力) のカット（削除）を検出"""
        cuts = []
        
        for sr in self.objects:
            # Check if this region exists in output
            found = False
            for dr in other.objects:
                if sr.cells == dr.cells:
                    found = True
                    break
                if sr.overlap(dr) > 0:
                    found = True
                    break
            
            if not found:
                cuts.append({
                    'color': sr.color,
                    'cells': sr.cells,
                    'bbox': sr.bbox,
                    'size': sr.size,
                })
        
        return cuts
    
    def detect_additions(self, other: CrossVolume) -> List[Dict]:
        """self(入力) → other(出力) の追加セルを検出"""
        additions = []
        
        for dr in other.objects:
            found = False
            for sr in self.objects:
                if sr.cells == dr.cells:
                    found = True
                    break
                if sr.overlap(dr) > 0:
                    found = True
                    break
            
            if not found:
                additions.append({
                    'color': dr.color,
                    'cells': dr.cells,
                    'bbox': dr.bbox,
                    'size': dr.size,
                })
        
        return additions
    
    def measure_distances(self) -> List[Dict]:
        """全オブジェクト間の距離を計測"""
        distances = []
        for i, a in enumerate(self.objects):
            for j, b in enumerate(self.objects):
                if i >= j:
                    continue
                d = a.distance_to(b)
                cd = a.center_distance_to(b)
                distances.append({
                    'a': (a.color, a.bbox),
                    'b': (b.color, b.bbox),
                    'manhattan': d,
                    'center_dist': cd,
                })
        return distances
    
    def summary(self) -> str:
        """Cross3D表現のサマリー"""
        lines = [f"CrossVolume {self.h}x{self.w}, bg={self.bg_color}, colors={self.colors}"]
        for color, layer in sorted(self.layers.items()):
            tag = " [BG]" if layer.is_background else ""
            lines.append(f"  Layer {color}{tag}: {layer.n_regions} regions, {layer.total_cells} cells")
            for r in layer.regions[:5]:
                lines.append(f"    {r}")
        return "\n".join(lines)


# ─── Cross3D ベースのソルバー ───

def cross3d_analyze(train_pairs: List[Tuple[List[List[int]], List[List[int]]]]) -> Dict:
    """train全例をCross3Dに変換して共通パターンを分析"""
    
    analyses = []
    for inp, out in train_pairs:
        cv_in = CrossVolume(inp)
        cv_out = CrossVolume(out, is_output=True)
        
        analysis = {
            'translations': cv_in.detect_translations(cv_out),
            'recolorings': cv_in.detect_recoloring(cv_out),
            'cuts': cv_in.detect_cuts(cv_out),
            'additions': cv_in.detect_additions(cv_out),
            'in_summary': cv_in.summary(),
            'out_summary': cv_out.summary(),
            'in_objects': len(cv_in.objects),
            'out_objects': len(cv_out.objects),
            'size_change': (cv_out.h / cv_in.h, cv_out.w / cv_in.w),
        }
        analyses.append(analysis)
    
    # Find consistent patterns across all examples
    consistent = {
        'all_translations': all(a['translations'] for a in analyses),
        'all_recolorings': all(a['recolorings'] for a in analyses),
        'all_cuts': all(a['cuts'] for a in analyses),
        'all_additions': all(a['additions'] for a in analyses),
        'same_size': all(a['size_change'] == (1.0, 1.0) for a in analyses),
        'n_examples': len(analyses),
    }
    
    return {
        'per_example': analyses,
        'consistent': consistent,
    }


def cross3d_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """Cross3Dベースでタスクを解く
    
    1. 全例をCross3Dに変換
    2. 共通する変換パターンを検出
    3. テスト入力に適用
    """
    analysis = cross3d_analyze(train_pairs)
    consistent = analysis['consistent']
    
    cv_test = CrossVolume(test_input)
    
    # Strategy 1: Translation（移動）
    if consistent['all_translations'] and consistent['same_size']:
        # 全例で一貫した移動パターンがあるか？
        all_deltas = []
        for a in analysis['per_example']:
            for t in a['translations']:
                all_deltas.append(t['delta'])
        
        if all_deltas:
            # 最頻出の移動量
            delta_counts = Counter(all_deltas)
            best_delta = delta_counts.most_common(1)[0][0]
            
            # Apply translation to test
            result_grid = [row[:] for row in test_input]
            h, w = cv_test.h, cv_test.w
            
            # Move all non-bg objects
            # First clear original positions
            new_grid = [[cv_test.bg_color] * w for _ in range(h)]
            for obj in cv_test.objects:
                moved = obj.translate(best_delta[0], best_delta[1])
                for r, c in moved.cells:
                    if 0 <= r < h and 0 <= c < w:
                        new_grid[r][c] = obj.color
            
            # Verify on train
            ok = True
            for inp, out in train_pairs:
                cv_i = CrossVolume(inp)
                test_grid = [[cv_i.bg_color] * cv_i.w for _ in range(cv_i.h)]
                for obj in cv_i.objects:
                    moved = obj.translate(best_delta[0], best_delta[1])
                    for r, c in moved.cells:
                        if 0 <= r < cv_i.h and 0 <= c < cv_i.w:
                            test_grid[r][c] = obj.color
                if test_grid != out:
                    ok = False
                    break
            
            if ok:
                return new_grid
    
    # Strategy 2: Recoloring（色変更）
    if consistent['all_recolorings'] and consistent['same_size']:
        all_recolors = []
        for a in analysis['per_example']:
            for rc in a['recolorings']:
                all_recolors.append((rc['from_color'], rc['to_color']))
        
        if all_recolors:
            color_map = {}
            for f, t in all_recolors:
                color_map[f] = t
            
            # Apply to test
            result = [[color_map.get(cell, cell) for cell in row] for row in test_input]
            
            # Verify
            ok = True
            for inp, out in train_pairs:
                pred = [[color_map.get(cell, cell) for cell in row] for row in inp]
                if pred != out:
                    ok = False
                    break
            
            if ok:
                return result
    
    # Strategy 3: Cross shape matching + transform
    # Find objects in input that match objects in output by shape
    # Then determine the transformation rule
    
    return None


# ─── CLI ───

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python3 -m arc.cross3d <task.json>")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        task = json.load(f)
    
    train_pairs = [(ex['input'], ex['output']) for ex in task['train']]
    
    # Analyze
    for i, (inp, out) in enumerate(train_pairs):
        print(f"\n=== Example {i} ===")
        cv_in = CrossVolume(inp)
        cv_out = CrossVolume(out, is_output=True)
        print("INPUT:")
        print(cv_in.summary())
        print("\nOUTPUT:")
        print(cv_out.summary())
        
        print("\nTranslations:", cv_in.detect_translations(cv_out))
        print("Recolorings:", cv_in.detect_recoloring(cv_out))
        print("Cuts:", cv_in.detect_cuts(cv_out))
        print("Additions:", cv_in.detect_additions(cv_out))
    
    # Try to solve
    test_input = task['test'][0]['input']
    result = cross3d_solve(train_pairs, test_input)
    if result:
        print("\n=== SOLUTION ===")
        for row in result:
            print(' '.join(str(c) for c in row))
    else:
        print("\n=== No solution found ===")
