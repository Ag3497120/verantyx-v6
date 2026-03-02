"""
arc/cross6axis.py — 6軸Cross全セル埋め込み

kofdai設計:
全セル（背景含む）に6軸Cross記述子を付与。
色ごとにレイヤーを作り、各レイヤーでCross構造で埋める。
この表現空間で入力と出力の対応を取り、変換を学習。

6軸:
  1. 上下方向の同色連続長 (vertical run)
  2. 左右方向の同色連続長 (horizontal run)
  3. 左上-右下対角の同色連続長 (diagonal ↘)
  4. 右上-左下対角の同色連続長 (diagonal ↙)
  5. 色ID (color layer)
  6. 背景/前景タグ (is_bg)

各セルの6軸Cross = そのセルから6方向に伸ばした十字の長さ + メタ情報
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color


def compute_cross6(grid: List[List[int]]) -> np.ndarray:
    """全セルの6軸Cross記述子を計算
    
    Returns:
        (H, W, 6) array:
          [0] = 上方向の同色連続長
          [1] = 下方向の同色連続長
          [2] = 左方向の同色連続長
          [3] = 右方向の同色連続長
          [4] = 左上対角の同色連続長
          [5] = 右下対角の同色連続長
    """
    g = np.array(grid)
    h, w = g.shape
    cross = np.zeros((h, w, 6), dtype=np.int32)
    
    # Axis 0: Up (同色が上にどこまで続くか)
    for r in range(h):
        for c in range(w):
            color = g[r, c]
            length = 0
            for r2 in range(r - 1, -1, -1):
                if g[r2, c] == color:
                    length += 1
                else:
                    break
            cross[r, c, 0] = length
    
    # Axis 1: Down
    for r in range(h - 1, -1, -1):
        for c in range(w):
            color = g[r, c]
            length = 0
            for r2 in range(r + 1, h):
                if g[r2, c] == color:
                    length += 1
                else:
                    break
            cross[r, c, 1] = length
    
    # Axis 2: Left
    for r in range(h):
        for c in range(w):
            color = g[r, c]
            length = 0
            for c2 in range(c - 1, -1, -1):
                if g[r, c2] == color:
                    length += 1
                else:
                    break
            cross[r, c, 2] = length
    
    # Axis 3: Right
    for r in range(h):
        for c in range(w - 1, -1, -1):
            color = g[r, c]
            length = 0
            for c2 in range(c + 1, w):
                if g[r, c2] == color:
                    length += 1
                else:
                    break
            cross[r, c, 3] = length
    
    # Axis 4: Diagonal ↖ (left-up)
    for r in range(h):
        for c in range(w):
            color = g[r, c]
            length = 0
            r2, c2 = r - 1, c - 1
            while r2 >= 0 and c2 >= 0:
                if g[r2, c2] == color:
                    length += 1
                else:
                    break
                r2 -= 1
                c2 -= 1
            cross[r, c, 4] = length
    
    # Axis 5: Diagonal ↘ (right-down)
    for r in range(h):
        for c in range(w):
            color = g[r, c]
            length = 0
            r2, c2 = r + 1, c + 1
            while r2 < h and c2 < w:
                if g[r2, c2] == color:
                    length += 1
                else:
                    break
                r2 += 1
                c2 += 1
            cross[r, c, 5] = length
    
    return cross


class Cross6Volume:
    """6軸Cross埋め込み済みのグリッド表現"""
    
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.g = np.array(grid)
        self.h, self.w = self.g.shape
        self.bg = int(Counter(self.g.flatten()).most_common(1)[0][0])
        self.colors = sorted(set(self.g.flatten()))
        
        # 6軸Cross記述子
        self.cross = compute_cross6(grid)
        
        # 色ごとのレイヤー (color -> bool mask)
        self.color_layers: Dict[int, np.ndarray] = {}
        for color in self.colors:
            self.color_layers[color] = (self.g == color)
        
        # 色ごとの6軸Cross (各色のセルだけ抽出)
        self.color_crosses: Dict[int, np.ndarray] = {}
        for color in self.colors:
            mask = self.color_layers[color]
            cc = np.zeros_like(self.cross)
            cc[mask] = self.cross[mask]
            self.color_crosses[color] = cc
    
    def cell_descriptor(self, r: int, c: int) -> Dict:
        """1セルの完全記述子"""
        return {
            'color': int(self.g[r, c]),
            'is_bg': int(self.g[r, c]) == self.bg,
            'cross': tuple(self.cross[r, c].tolist()),
            'pos': (r, c),
            'total_cross_length': int(self.cross[r, c].sum()),
        }
    
    def find_cross_pattern(self, pattern: Tuple[int, ...]) -> List[Tuple[int, int]]:
        """特定のCrossパターン(6軸)を持つセルを検索"""
        matches = []
        pat = np.array(pattern)
        for r in range(self.h):
            for c in range(self.w):
                if np.array_equal(self.cross[r, c], pat):
                    matches.append((r, c))
        return matches
    
    def cross_signature(self) -> Tuple:
        """グリッド全体のCrossシグネチャ（比較用）"""
        # 各色レイヤーのCross長さ統計
        sigs = []
        for color in self.colors:
            mask = self.color_layers[color]
            if mask.any():
                cross_lengths = self.cross[mask].sum(axis=1)
                sigs.append((color, int(mask.sum()), 
                           float(cross_lengths.mean()), float(cross_lengths.max())))
        return tuple(sigs)
    
    def diff(self, other: 'Cross6Volume') -> Dict:
        """2つのCross6Volume間の差分を分析"""
        if self.h != other.h or self.w != other.w:
            return {'same_size': False, 'size_ratio': (other.h / self.h, other.w / self.w)}
        
        # セルごとの色変化
        color_changes = []
        cross_changes = []
        
        for r in range(self.h):
            for c in range(self.w):
                c_in = int(self.g[r, c])
                c_out = int(other.g[r, c])
                
                if c_in != c_out:
                    cross_in = tuple(self.cross[r, c].tolist())
                    cross_out = tuple(other.cross[r, c].tolist())
                    
                    color_changes.append({
                        'pos': (r, c),
                        'from_color': c_in,
                        'to_color': c_out,
                        'from_cross': cross_in,
                        'to_cross': cross_out,
                        'from_is_bg': c_in == self.bg,
                    })
        
        # パターン分析: 変化のグループ化
        change_groups = Counter()
        for ch in color_changes:
            # Abstract key: (from_cross, from_color==bg) -> to_color_role
            to_role = 'bg' if ch['to_color'] == other.bg else 'other'
            from_role = 'bg' if ch['from_is_bg'] else 'fg'
            change_groups[(from_role, to_role)] += 1
        
        return {
            'same_size': True,
            'n_changed': len(color_changes),
            'total_cells': self.h * self.w,
            'change_pct': len(color_changes) / (self.h * self.w) * 100,
            'changes': color_changes,
            'change_groups': dict(change_groups),
        }


def cross6_learn_rule(train_pairs: List[Tuple[List[List[int]], List[List[int]]]]) -> Optional[Dict]:
    """
    6軸Cross空間でtrain全例の変換ルールを学習
    
    各セルの入力Cross記述子 → 出力色 のマッピングを作る
    """
    bg = most_common_color(train_pairs[0][0])
    
    # Same size check
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    # Build mapping: (input_color, input_cross_pattern) -> output_color
    cross_rule = {}
    
    for inp, out in train_pairs:
        cv_in = Cross6Volume(inp)
        h, w = cv_in.h, cv_in.w
        
        for r in range(h):
            for c in range(w):
                in_color = int(cv_in.g[r, c])
                in_cross = tuple(cv_in.cross[r, c].tolist())
                out_color = int(np.array(out)[r, c])
                
                # Key: (色, 6軸Crossパターン)
                key = (in_color, in_cross)
                
                if key in cross_rule:
                    if cross_rule[key] != out_color:
                        # Conflict — try abstract key
                        # Use total cross length instead of exact pattern
                        total = sum(in_cross)
                        abs_key = (in_color, total)
                        if abs_key in cross_rule:
                            if cross_rule[abs_key] != out_color:
                                cross_rule[key] = None  # irreconcilable
                        else:
                            cross_rule[abs_key] = out_color
                else:
                    cross_rule[key] = out_color
    
    # Remove conflicts
    cross_rule = {k: v for k, v in cross_rule.items() if v is not None}
    
    if not cross_rule:
        return None
    
    # Verify on train
    for inp, out in train_pairs:
        cv_in = Cross6Volume(inp)
        h, w = cv_in.h, cv_in.w
        
        for r in range(h):
            for c in range(w):
                in_color = int(cv_in.g[r, c])
                in_cross = tuple(cv_in.cross[r, c].tolist())
                out_color = int(np.array(out)[r, c])
                
                key = (in_color, in_cross)
                if key in cross_rule:
                    if cross_rule[key] != out_color:
                        return None  # verification failed
    
    return {
        'cross_rule': cross_rule,
        'bg': bg,
    }


def cross6_apply(inp: List[List[int]], rule: Dict) -> List[List[int]]:
    """6軸Crossルールを適用"""
    cross_rule = rule['cross_rule']
    cv = Cross6Volume(inp)
    
    result = [row[:] for row in inp]
    
    for r in range(cv.h):
        for c in range(cv.w):
            in_color = int(cv.g[r, c])
            in_cross = tuple(cv.cross[r, c].tolist())
            
            key = (in_color, in_cross)
            if key in cross_rule:
                result[r][c] = cross_rule[key]
            else:
                # Fallback: try abstract key (total cross length)
                total = sum(in_cross)
                abs_key = (in_color, total)
                if abs_key in cross_rule:
                    result[r][c] = cross_rule[abs_key]
                # else: keep original
    
    return result


def cross6_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """6軸Cross空間でタスクを解く"""
    rule = cross6_learn_rule(train_pairs)
    if rule is None:
        return None
    
    result = cross6_apply(test_input, rule)
    
    # Verify on train
    for inp, out in train_pairs:
        pred = cross6_apply(inp, rule)
        if not grid_eq(pred, out):
            return None
    
    return result


# ─── CLI ───

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python3 -m arc.cross6axis <task.json>")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        task = json.load(f)
    
    train_pairs = [(ex['input'], ex['output']) for ex in task['train']]
    test_input = task['test'][0]['input']
    test_output = task['test'][0].get('output')
    
    # Analyze
    for i, (inp, out) in enumerate(train_pairs[:2]):
        cv_in = Cross6Volume(inp)
        cv_out = Cross6Volume(out)
        diff = cv_in.diff(cv_out)
        print(f"Example {i}: {diff['n_changed']}/{diff['total_cells']} changed ({diff['change_pct']:.1f}%)")
        print(f"  Groups: {diff['change_groups']}")
        
        # Show some changed cells
        for ch in diff['changes'][:5]:
            print(f"  ({ch['pos'][0]:2d},{ch['pos'][1]:2d}): color {ch['from_color']}->{ch['to_color']}  "
                  f"cross {ch['from_cross']}")
    
    # Solve
    result = cross6_solve(train_pairs, test_input)
    if result:
        if test_output and grid_eq(result, test_output):
            print("\n✅ SOLVED!")
        else:
            print("\n⚠️ Solution generated (not verified)")
    else:
        print("\n✗ No solution")
