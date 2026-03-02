"""
arc/output_repair.py — NB rule出力の自己整合性修復

Cross Engine NBルールの出力が90-99%正解だが数セル間違える問題を修復。
trainで学習した「出力パターン→出力値」マッピングを使い、
テスト出力の各セルの近傍を見て、矛盾するセルを修正する。

原理:
  1. trainの出力ペアから「出力近傍→出力値」ルールを学習
  2. テスト出力の各セルについて、その出力近傍からの予測と比較
  3. 矛盾するセルを多数決で修正
  4. 収束するまで繰り返す（最大5回）
"""

from typing import List, Tuple, Optional, Dict
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color


def learn_output_nb_rule(train_pairs: List[Tuple[Grid, Grid]], 
                         radius: int = 1) -> Optional[Dict]:
    """trainの出力ペアから出力近傍→出力値のルールを学習"""
    mapping = {}  # output_nb_pattern -> output_value
    
    for inp, out in train_pairs:
        h, w = grid_shape(out)
        for r in range(h):
            for c in range(w):
                nb = []
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if dr == 0 and dc == 0:
                            continue  # skip center
                        nr, nc = r + dr, c + dc
                        nb.append(out[nr][nc] if 0 <= nr < h and 0 <= nc < w else -1)
                
                key = tuple(nb)
                val = out[r][c]
                
                if key in mapping:
                    if mapping[key] != val:
                        mapping[key] = None  # ambiguous
                else:
                    mapping[key] = val
    
    # Remove ambiguous entries
    mapping = {k: v for k, v in mapping.items() if v is not None}
    
    if not mapping:
        return None
    
    return {"mapping": mapping, "radius": radius}


def repair_output(output: Grid, rule: Dict, max_iters: int = 5) -> Grid:
    """出力の自己整合性を修復"""
    mapping = rule["mapping"]
    radius = rule["radius"]
    
    # Copy output
    grid = [row[:] for row in output]
    h, w = grid_shape(grid)
    
    for iteration in range(max_iters):
        changes = 0
        new_grid = [row[:] for row in grid]
        
        for r in range(h):
            for c in range(w):
                nb = []
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        nb.append(grid[nr][nc] if 0 <= nr < h and 0 <= nc < w else -1)
                
                key = tuple(nb)
                if key in mapping and mapping[key] != grid[r][c]:
                    new_grid[r][c] = mapping[key]
                    changes += 1
        
        grid = new_grid
        if changes == 0:
            break
    
    return grid


def repair_with_voting(output: Grid, train_pairs: List[Tuple[Grid, Grid]],
                       radius: int = 1, max_iters: int = 3) -> Grid:
    """多数決ベースの出力修復（より安全）
    
    各セルについて:
    1. 出力近傍パターンからの予測
    2. 入力近傍パターンからの予測（元のNBルール相当）
    3. 現在の値
    を多数決で決める
    """
    # Learn output-NB mapping from train outputs
    out_mapping = {}  # output_nb -> Counter of output values
    
    for inp, out in train_pairs:
        h, w = grid_shape(out)
        for r in range(h):
            for c in range(w):
                nb = []
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        nb.append(out[nr][nc] if 0 <= nr < h and 0 <= nc < w else -1)
                key = tuple(nb)
                if key not in out_mapping:
                    out_mapping[key] = Counter()
                out_mapping[key][out[r][c]] += 1
    
    # For each key, keep only if there's a clear majority (>80%)
    confident_mapping = {}
    for key, counter in out_mapping.items():
        total = sum(counter.values())
        most_common_val, most_common_count = counter.most_common(1)[0]
        if most_common_count / total >= 0.8:
            confident_mapping[key] = most_common_val
    
    grid = [row[:] for row in output]
    h, w = grid_shape(grid)
    
    for iteration in range(max_iters):
        changes = 0
        new_grid = [row[:] for row in grid]
        
        for r in range(h):
            for c in range(w):
                nb = []
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        nb.append(grid[nr][nc] if 0 <= nr < h and 0 <= nc < w else -1)
                key = tuple(nb)
                
                if key in confident_mapping:
                    predicted = confident_mapping[key]
                    if predicted != grid[r][c]:
                        new_grid[r][c] = predicted
                        changes += 1
        
        grid = new_grid
        if changes == 0:
            break
    
    return grid
