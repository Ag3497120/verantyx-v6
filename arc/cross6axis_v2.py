"""
arc/cross6axis_v2.py — 6軸Cross v2: 抽象化 + 多スケール

v1の問題: exact match → カバレッジ不足
v2の改善:
  1. 抽象化レベル: exact → binned → categorical
  2. 多スケール: 1x1, 2x2, 3x3 ブロック単位のCross
  3. 相対Cross: 隣接色の情報も含める
  4. 背景Cross: 背景の連続長 = 位置情報として活用
  5. フォールバック階層: exact → binned → categorical → keep
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color


# ─── 6軸 Run Length 計算 ───

def _run_lengths_all(g: np.ndarray) -> np.ndarray:
    """全セルの8方向run length (H, W, 8)
    
    方向: 0=↑ 1=↓ 2=← 3=→ 4=↖ 5=↘ 6=↗ 7=↙
    """
    h, w = g.shape
    runs = np.zeros((h, w, 8), dtype=np.int16)
    
    DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    
    for d, (dr, dc) in enumerate(DIRS):
        for r in range(h):
            for c in range(w):
                color = g[r, c]
                length = 0
                r2, c2 = r + dr, c + dc
                while 0 <= r2 < h and 0 <= c2 < w and g[r2, c2] == color:
                    length += 1
                    r2 += dr
                    c2 += dc
                runs[r, c, d] = length
    
    return runs


def _run_to_boundary(g: np.ndarray, bg: int) -> np.ndarray:
    """全セルの4方向 境界までの距離 (H, W, 4)
    
    「背景のCrossで位置認識」= 背景セルから見てどの方向にどれだけ
    非背景(or 境界)があるか
    """
    h, w = g.shape
    boundary = np.zeros((h, w, 4), dtype=np.int16)
    
    for r in range(h):
        for c in range(w):
            # ↑ 上端までの距離
            boundary[r, c, 0] = r
            # ↓ 下端までの距離
            boundary[r, c, 1] = h - 1 - r
            # ← 左端までの距離
            boundary[r, c, 2] = c
            # → 右端までの距離
            boundary[r, c, 3] = w - 1 - c
    
    return boundary


# ─── 抽象化関数 ───

def _bin_run(length: int) -> int:
    """run lengthをビニング (0,1,2,3-4,5+)"""
    if length == 0: return 0
    if length == 1: return 1
    if length == 2: return 2
    if length <= 4: return 3
    return 4

def _cat_run(length: int) -> str:
    """run lengthをカテゴリ化"""
    if length == 0: return 'Z'  # zero
    if length <= 2: return 'S'  # short
    if length <= 5: return 'M'  # medium
    return 'L'                  # long

def _abstract_cross(runs: np.ndarray, level: str = 'exact') -> tuple:
    """8方向runをキーに変換
    
    level:
      'exact'  — 生の値 (0,3,0,5,0,0,2,1)
      'binned' — ビニング (0,2,0,4,0,0,1,1)  
      'cat'    — カテゴリ (Z,M,Z,L,Z,Z,S,S)
      'cross4' — 4方向のみ (上下左右)
      'total'  — 合計値のみ
    """
    if level == 'exact':
        return tuple(int(x) for x in runs)
    elif level == 'binned':
        return tuple(_bin_run(int(x)) for x in runs)
    elif level == 'cat':
        return tuple(_cat_run(int(x)) for x in runs)
    elif level == 'cross4':
        return tuple(int(x) for x in runs[:4])
    elif level == 'total':
        return (int(runs.sum()),)
    else:
        return tuple(int(x) for x in runs)


# ─── 多スケールCross ───

def _downscale(g: np.ndarray, block_size: int) -> np.ndarray:
    """grid を block_size x block_size ブロックに縮小
    各ブロック = 最頻色
    """
    h, w = g.shape
    nh = h // block_size
    nw = w // block_size
    if nh == 0 or nw == 0:
        return None
    
    result = np.zeros((nh, nw), dtype=g.dtype)
    for r in range(nh):
        for c in range(nw):
            block = g[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size]
            result[r, c] = Counter(block.flatten()).most_common(1)[0][0]
    
    return result


# ─── Cross6 v2 メインクラス ───

class Cross6v2:
    """多スケール・抽象化6軸Cross"""
    
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.g = np.array(grid)
        self.h, self.w = self.g.shape
        self.bg = int(Counter(self.g.flatten()).most_common(1)[0][0])
        self.colors = sorted(set(int(x) for x in self.g.flatten()))
        
        # Scale 1: 原寸
        self.runs = _run_lengths_all(self.g)
        self.boundary = _run_to_boundary(self.g, self.bg)
        
        # Scale 2, 3: 縮小版 (2x2, 3x3)
        self.scales = {1: (self.g, self.runs)}
        for bs in [2, 3]:
            ds = _downscale(self.g, bs)
            if ds is not None and ds.shape[0] >= 3 and ds.shape[1] >= 3:
                self.scales[bs] = (ds, _run_lengths_all(ds))
    
    def cell_key(self, r: int, c: int, level: str = 'exact', 
                 include_boundary: bool = True,
                 include_color: bool = True) -> tuple:
        """1セルのCrossキーを生成"""
        parts = []
        
        if include_color:
            color = int(self.g[r, c])
            is_bg = color == self.bg
            parts.append(('bg' if is_bg else color,))
        
        # 8方向run length (抽象化レベル適用)
        runs = self.runs[r, c]
        parts.append(_abstract_cross(runs, level))
        
        # 境界距離 (位置情報)
        if include_boundary:
            bd = self.boundary[r, c]
            if level == 'exact':
                parts.append(tuple(int(x) for x in bd))
            elif level in ('binned', 'cat'):
                parts.append(tuple(_bin_run(int(x)) for x in bd))
            else:
                parts.append(tuple(int(x) for x in bd))
        
        return sum(parts, ())


def cross6v2_learn(train_pairs: List[Tuple[List[List[int]], List[List[int]]]],
                   ) -> Optional[Dict]:
    """
    6軸Cross v2 でルールを学習
    
    階層的フォールバック:
    1. exact + boundary + color → 最も具体的
    2. exact + color (boundary なし)
    3. binned + color
    4. cat + color
    5. total + color
    """
    bg = most_common_color(train_pairs[0][0])
    
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    # 各レベルでルールを構築
    levels = [
        ('exact', True, True),     # exact + boundary + color
        ('exact', False, True),    # exact + color
        ('cross4', False, True),   # 4方向 + color
        ('binned', False, True),   # binned + color
        ('cat', False, True),      # categorical + color
        ('total', False, True),    # total + color
    ]
    
    rules_by_level = {}
    
    for level, inc_bd, inc_color in levels:
        rule = {}
        consistent = True
        
        for inp, out in train_pairs:
            cv = Cross6v2(inp)
            go = np.array(out)
            h, w = cv.h, cv.w
            
            for r in range(h):
                for c in range(w):
                    key = cv.cell_key(r, c, level=level, 
                                      include_boundary=inc_bd,
                                      include_color=inc_color)
                    out_color = int(go[r, c])
                    
                    if key in rule:
                        if rule[key] != out_color:
                            consistent = False
                            rule[key] = None  # conflict
                    else:
                        rule[key] = out_color
        
        # Remove conflicts
        rule = {k: v for k, v in rule.items() if v is not None}
        rules_by_level[(level, inc_bd, inc_color)] = rule
    
    # Verify: find the most specific level that passes train
    best_config = None
    
    for level, inc_bd, inc_color in levels:
        rule = rules_by_level[(level, inc_bd, inc_color)]
        if not rule:
            continue
        
        passes = True
        for inp, out in train_pairs:
            cv = Cross6v2(inp)
            go = np.array(out)
            
            for r in range(cv.h):
                for c in range(cv.w):
                    key = cv.cell_key(r, c, level=level,
                                      include_boundary=inc_bd,
                                      include_color=inc_color)
                    if key in rule:
                        if rule[key] != int(go[r, c]):
                            passes = False
                            break
                    else:
                        # Missing key — check if output == input (keep is ok)
                        if int(cv.g[r, c]) != int(go[r, c]):
                            passes = False
                            break
                if not passes:
                    break
            if not passes:
                break
        
        if passes:
            best_config = (level, inc_bd, inc_color)
            break
    
    if best_config is None:
        return None
    
    return {
        'rules_by_level': rules_by_level,
        'best_config': best_config,
        'bg': bg,
        'levels': levels,
    }


def cross6v2_apply(inp: List[List[int]], learned: Dict) -> List[List[int]]:
    """6軸Cross v2 でルールを適用（階層フォールバック付き）"""
    cv = Cross6v2(inp)
    result = [row[:] for row in inp]
    levels = learned['levels']
    rules_by_level = learned['rules_by_level']
    best_config = learned['best_config']
    
    # Best config でまず適用
    best_rule = rules_by_level[best_config]
    level, inc_bd, inc_color = best_config
    
    for r in range(cv.h):
        for c in range(cv.w):
            key = cv.cell_key(r, c, level=level,
                              include_boundary=inc_bd,
                              include_color=inc_color)
            if key in best_rule:
                result[r][c] = best_rule[key]
            else:
                # Fallback: try less specific levels
                for fb_level, fb_bd, fb_color in levels:
                    if (fb_level, fb_bd, fb_color) == best_config:
                        continue
                    fb_rule = rules_by_level.get((fb_level, fb_bd, fb_color), {})
                    fb_key = cv.cell_key(r, c, level=fb_level,
                                         include_boundary=fb_bd,
                                         include_color=fb_color)
                    if fb_key in fb_rule:
                        result[r][c] = fb_rule[fb_key]
                        break
                # If no fallback hit, keep original (already in result)
    
    return result


def cross6v2_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """6軸Cross v2 でタスクを解く"""
    learned = cross6v2_learn(train_pairs)
    if learned is None:
        return None
    
    # Verify on train
    for inp, out in train_pairs:
        pred = cross6v2_apply(inp, learned)
        if not grid_eq(pred, out):
            return None
    
    return cross6v2_apply(test_input, learned)


# ─── 多スケール統合 ───

def cross6v2_multiscale_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """多スケールで試行 — 原寸で解けなければ縮小版で"""
    # Scale 1: 原寸
    result = cross6v2_solve(train_pairs, test_input)
    if result:
        return result
    
    # TODO: Scale 2, 3 での変換は入出力サイズ変換が必要
    # 今はスキップ
    
    return None


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python3 -m arc.cross6axis_v2 <task.json>")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        task = json.load(f)
    
    train_pairs = [(ex['input'], ex['output']) for ex in task['train']]
    test_input = task['test'][0]['input']
    test_output = task['test'][0].get('output')
    
    result = cross6v2_solve(train_pairs, test_input)
    if result:
        if test_output and grid_eq(result, test_output):
            print("✅ SOLVED!")
        else:
            print("⚠️ Solution (unverified)")
    else:
        print("✗ No solution")
