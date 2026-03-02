"""
arc/boundary_repair.py — NB rule出力の境界交差点修復

NBルールが90-99%正解する問題の残りエラーは、
ほぼ全て「2色の領域の境界交差点」で発生する。

修復方法:
1. trainのinput→output差分から「境界修復ルール」を学習
2. テスト出力（NB rule適用済み）の各セルで、
   入力の4近傍の色分布を見て境界セルを検出
3. trainで学習した修復ルールを適用

Key insight: 
  trainで学習する修復ルールは「4近傍の色分布→正しい出力色」
  これは5x5近傍よりはるかに少ないパターン空間なので汎化しやすい
"""

from typing import List, Tuple, Optional, Dict
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color
import numpy as np


def learn_boundary_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """trainのinput→output差分から境界修復ルールを学習
    
    Returns:
        {
            "repair_map": {(center, sorted_nb_colors_tuple): output_color, ...},
            "bg": background_color,
        }
    """
    bg = most_common_color(train_pairs[0][0])
    
    # Only applicable for same-size transformations
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    # Collect boundary repair patterns from train
    # For each cell that CHANGES, record (center_color, 4nb_color_signature) -> output_color
    repair_map = {}
    
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        for r in range(h):
            for c in range(w):
                if inp[r][c] == out[r][c]:
                    continue  # only learn from changed cells
                
                center = inp[r][c]
                
                # 4-neighbor signature (abstracted)
                nbs = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w:
                        v = inp[nr][nc]
                        if v == bg:
                            nbs.append('bg')
                        elif v == center:
                            nbs.append('self')
                        else:
                            nbs.append('other')
                    else:
                        nbs.append('oob')
                
                # Sort to make rotation-invariant
                nb_sig = tuple(sorted(nbs))
                
                # Output role
                out_val = out[r][c]
                if out_val == bg:
                    role = 'bg'
                elif out_val == center:
                    role = 'keep'  # shouldn't happen since we filter unchanged
                else:
                    role = 'other'
                
                key = (center == bg, nb_sig, role)
                
                # More specific: what IS the other color?
                # Use "which neighbor color becomes the output"
                other_colors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w:
                        v = inp[nr][nc]
                        if v == out_val:
                            other_colors.append(True)
                        else:
                            other_colors.append(False)
                
                # Pattern: (is_center_bg, nb_abstract_signature) -> (role, copy_from_direction)
                # Simpler: just record the concrete pattern
                concrete_key = _make_concrete_key(inp, r, c, h, w, bg)
                repair_map[concrete_key] = out_val
    
    if not repair_map:
        return None
    
    return {"repair_map": repair_map, "bg": bg}


def _make_concrete_key(grid, r, c, h, w, bg):
    """Create a repair key for a cell"""
    center = grid[r][c]
    
    # 4-neighbor: abstract to (bg/self/other1/other2/oob)
    nb_abstract = []
    nb_colors = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < h and 0 <= nc < w:
            v = grid[nr][nc]
            nb_colors.append(v)
            if v == bg:
                nb_abstract.append(0)
            elif v == center:
                nb_abstract.append(1)
            else:
                nb_abstract.append(2)
        else:
            nb_abstract.append(-1)
            nb_colors.append(-1)
    
    # Count-based key: (center_is_bg, n_self, n_other, n_bg, n_oob)
    center_is_bg = center == bg
    n_self = nb_abstract.count(1)
    n_other = nb_abstract.count(2)
    n_bg = nb_abstract.count(0)
    n_oob = nb_abstract.count(-1)
    
    return (center_is_bg, n_self, n_other, n_bg, n_oob)


def apply_boundary_repair(nb_output: Grid, inp: Grid, 
                          train_pairs: List[Tuple[Grid, Grid]],
                          max_iters: int = 3) -> Grid:
    """NB rule出力を境界修復
    
    Strategy:
    1. trainで「変更されたセルの入力近傍パターン」を学習
    2. テスト出力の各セルが入力と異なるべきかを判定
    3. 異なるべきなら、近傍の色から正しい値を推定
    """
    bg = most_common_color(train_pairs[0][0])
    h, w = grid_shape(inp)
    
    # Phase 1: Learn what "should change" patterns look like
    # and what they should change TO
    change_patterns = {}  # (count_key) -> Counter of output_role
    keep_patterns = {}    # (count_key) -> True
    
    for train_inp, train_out in train_pairs:
        th, tw = grid_shape(train_inp)
        for r in range(th):
            for c in range(tw):
                key = _make_concrete_key(train_inp, r, c, th, tw, bg)
                
                if train_inp[r][c] != train_out[r][c]:
                    # This cell changes
                    out_val = train_out[r][c]
                    center = train_inp[r][c]
                    
                    # What is the output relative to neighbors?
                    # Find which neighbor color the output matches
                    output_source = 'unknown'
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < th and 0 <= nc < tw:
                            if train_inp[nr][nc] == out_val:
                                output_source = 'copy_neighbor'
                                break
                    if out_val == bg:
                        output_source = 'to_bg'
                    
                    if key not in change_patterns:
                        change_patterns[key] = Counter()
                    change_patterns[key][output_source] += 1
                else:
                    keep_patterns[key] = True
    
    # Phase 2: For each cell in nb_output that equals inp (i.e., fallback happened),
    # check if it should have changed based on train patterns
    grid = [row[:] for row in nb_output]
    
    for r in range(h):
        for c in range(w):
            # Only fix cells where NB rule fell back to center (output == input)
            if grid[r][c] != inp[r][c]:
                continue  # NB rule already changed it
            
            key = _make_concrete_key(inp, r, c, h, w, bg)
            
            if key in change_patterns and key not in keep_patterns:
                # This pattern should change but didn't!
                source = change_patterns[key].most_common(1)[0][0]
                center = inp[r][c]
                
                if source == 'copy_neighbor':
                    # Copy the most common non-center, non-bg neighbor
                    nb_colors = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w:
                            v = inp[nr][nc]
                            if v != center and v != bg:
                                nb_colors.append(v)
                    if nb_colors:
                        grid[r][c] = Counter(nb_colors).most_common(1)[0][0]
                elif source == 'to_bg':
                    grid[r][c] = bg
    
    return grid
