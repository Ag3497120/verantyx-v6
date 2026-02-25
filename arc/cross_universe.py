"""
CrossUniverse — kofdaiの思想実装

ピクセルレベルでcross構造を割り当て、フロー制御で多様なパターンを処理。
cross構造の上にcross構造の宇宙を重ね、再帰的自己相似で多重観察を実現。

核心原理:
1. 各セルはcross node（入出力4方向 + 自己状態）
2. cross node間のフロー（伝播/遮断）で変換を記述
3. グリッド全体がcross構造、ブロックもcross構造、その上もcross構造
4. 囲むものと囲まれるものの相互監視 → 内外の境界消失
5. 収束はレイヤー間伝播の自然な帰結
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter

Grid = List[List[int]]


class CrossNode:
    """1セルのcross構造ノード。4方向のフロー状態を持つ。"""
    __slots__ = ['color', 'flows', 'blocked']
    
    def __init__(self, color: int):
        self.color = color
        # 4方向のフロー: up, right, down, left
        # 各方向: (active: bool, color: int)
        self.flows = [None, None, None, None]  # U, R, D, L
        # 遮断マスク: Trueの方向はフローを遮断
        self.blocked = [False, False, False, False]


class CrossLayer:
    """1レベルのcross構造レイヤー。グリッド上の全セルにcross nodeを配置。"""
    
    def __init__(self, grid: Grid, bg: int = 0):
        self.arr = np.array(grid)
        self.h, self.w = self.arr.shape
        self.bg = bg
        self.nodes = [[CrossNode(int(self.arr[r, c])) 
                       for c in range(self.w)] for r in range(self.h)]
    
    def propagate_color(self, max_steps: int = 50) -> np.ndarray:
        """色の伝播: 非bgセルから4方向にフローを送り、bgセルを埋める。
        遮断(blocked)がある方向にはフローが流れない。
        blocked_maskがNoneの場合は遮断なし（高速パス）。"""
        result = self.arr.copy()
        
        # Build blocked mask: (h, w, 4)
        has_blocks = any(
            any(self.nodes[r][c].blocked[d] for d in range(4))
            for r in range(self.h) for c in range(self.w)
        )
        
        dirs = [(-1, 0, 0), (0, 1, 1), (1, 0, 2), (0, -1, 3)]
        
        if has_blocks:
            blocked = np.array([
                [self.nodes[r][c].blocked for c in range(self.w)]
                for r in range(self.h)
            ])  # (h, w, 4)
        
        for step in range(max_steps):
            changed = False
            new_result = result.copy()
            
            for r in range(self.h):
                for c in range(self.w):
                    if result[r, c] == self.bg:
                        continue
                    for dr, dc, di in dirs:
                        if has_blocks and blocked[r, c, di]:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.h and 0 <= nc < self.w:
                            if has_blocks and blocked[nr, nc, (di + 2) % 4]:
                                continue
                            if new_result[nr, nc] == self.bg:
                                new_result[nr, nc] = result[r, c]
                                changed = True
            
            result = new_result
            if not changed:
                break
        
        return result
    
    def propagate_nearest(self, max_steps: int = 50) -> np.ndarray:
        """最近接伝播: 各bgセルに最も近い非bgセルの色を割り当てる。
        遮断を考慮したBFS。"""
        result = self.arr.copy()
        dist = np.full((self.h, self.w), float('inf'))
        
        # 初期化: 非bgセルは距離0
        from collections import deque
        queue = deque()
        for r in range(self.h):
            for c in range(self.w):
                if self.arr[r, c] != self.bg:
                    dist[r, c] = 0
                    queue.append((r, c, 0))
        
        dirs = [(-1, 0, 0), (0, 1, 1), (1, 0, 2), (0, -1, 3)]
        
        while queue:
            r, c, d = queue.popleft()
            if d > dist[r, c]:
                continue
            
            for dr, dc, di in dirs:
                if self.nodes[r][c].blocked[di]:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.h and 0 <= nc < self.w:
                    opp_di = (di + 2) % 4
                    if self.nodes[nr][nc].blocked[opp_di]:
                        continue
                    new_d = d + 1
                    if new_d < dist[nr, nc]:
                        dist[nr, nc] = new_d
                        result[nr, nc] = result[r, c]
                        queue.append((nr, nc, new_d))
        
        return result


class CrossUniverse:
    """多重cross構造の宇宙。各レベルが異なるスケールのcross構造。
    
    Level 0: ピクセルレベル
    Level 1: ブロックレベル（separator検出による自動分割）
    Level 2: ブロックのブロック（再帰的）
    
    各レベルのcross構造が相互監視し、情報が上下に流れる。
    """
    
    def __init__(self, grid: Grid, bg: int = 0):
        self.grid = grid
        self.bg = bg
        self.layers: List[CrossLayer] = []
        self.block_params: List[Dict] = []
        self._build_hierarchy()
    
    def _build_hierarchy(self):
        """レベル0（ピクセルレベル）のみ構築。上位レベルは必要時に追加。"""
        layer = CrossLayer(self.grid, self.bg)
        self.layers.append(layer)
    
    @property
    def depth(self) -> int:
        return len(self.layers)
    
    def observe(self, level: int = 0) -> np.ndarray:
        """指定レベルのcross構造を観察（現在の状態を返す）。"""
        if level < 0 or level >= self.depth:
            return None
        return self.layers[level].arr.copy()
    
    def propagate(self, level: int = 0, mode: str = 'color') -> np.ndarray:
        """指定レベルでフロー伝播を実行。"""
        if level < 0 or level >= self.depth:
            return None
        
        layer = self.layers[level]
        if mode == 'color':
            return layer.propagate_color()
        elif mode == 'nearest':
            return layer.propagate_nearest()
        return layer.arr.copy()
    
    def set_barrier(self, level: int, barriers: List[Tuple[int, int, int]]):
        """遮断を設定。barriers = [(row, col, direction), ...]"""
        if level < 0 or level >= self.depth:
            return
        layer = self.layers[level]
        for r, c, d in barriers:
            if 0 <= r < layer.h and 0 <= c < layer.w and 0 <= d < 4:
                layer.nodes[r][c].blocked[d] = True
    
    def reconstruct(self, level: int, result: np.ndarray) -> Grid:
        """上位レベルの結果をピクセルレベルに戻す。"""
        from arc.block_ir import block_colors_to_grid
        
        current = result
        for lv in range(level, 0, -1):
            params = self.block_params[lv - 1]
            current = block_colors_to_grid(
                np.array(current) if not isinstance(current, np.ndarray) else current,
                params['block_h'], params['block_w'],
                params['sep_w'], params['sep_color']
            )
            if isinstance(current, list):
                current = np.array(current)
        
        return current.tolist() if isinstance(current, np.ndarray) else current


# ============================================================
# フロー制御ルール学習
# ============================================================

def _fast_sep_propagate(arr: np.ndarray, bg: int, sep_color: int) -> np.ndarray:
    """Fast separator-based flood fill without CrossUniverse overhead."""
    from scipy import ndimage
    
    h, w = arr.shape
    result = arr.copy()
    
    # Mask of non-separator, non-bg cells (seed cells)
    # Mask of bg cells (to fill)
    sep_mask = (arr == sep_color)
    bg_mask = (arr == bg)
    fillable = bg_mask  # cells that can be filled
    
    # Connected components of non-separator cells
    passable = ~sep_mask
    labeled, n = ndimage.label(passable)
    
    # For each component, find seed colors and fill bg cells
    for comp_id in range(1, n + 1):
        comp_mask = (labeled == comp_id)
        # Seeds: non-bg, non-sep cells in this component
        seed_mask = comp_mask & ~bg_mask & ~sep_mask
        seed_colors = set(int(v) for v in arr[seed_mask]) if seed_mask.any() else set()
        seed_colors.discard(bg)
        seed_colors.discard(sep_color)
        
        if len(seed_colors) != 1:
            continue  # Multiple seed colors or none → skip
        
        fill_color = seed_colors.pop()
        # Fill all bg cells in this component
        fill_mask = comp_mask & bg_mask
        result[fill_mask] = fill_color
    
    return result


def learn_flow_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """訓練ペアからフロー伝播ルールを学習する。"""
    from arc.grid import grid_eq, most_common_color, grid_shape
    
    if not train_pairs:
        return None
    
    # Same-shape only
    for inp, out in train_pairs:
        if grid_shape(inp) != grid_shape(out):
            return None
    
    # Skip large grids for performance
    h, w = grid_shape(train_pairs[0][0])
    if h * w > 400:
        return None
    
    bg = most_common_color(train_pairs[0][0])
    
    # Quick check: need >= 3 distinct colors
    arr0 = np.array(train_pairs[0][0])
    colors = set(int(v) for v in arr0.flatten())
    if len(colors) < 3:
        return None
    
    cnt = Counter(int(v) for v in arr0.flatten())
    candidates = [c for c, _ in cnt.most_common() if c != bg][:2]
    
    for sep_color in candidates:
        sep_ok = True
        for inp, out in train_pairs:
            result = _fast_sep_propagate(np.array(inp), bg, sep_color)
            if not grid_eq(result.tolist(), out):
                sep_ok = False
                break
        if sep_ok:
            return {'type': 'separator_propagate', 'bg': bg,
                    'sep_color': sep_color}
    
    return None


def apply_flow_rule(inp: Grid, rule: Dict) -> Optional[Grid]:
    """学習したフロールールを適用する。"""
    from arc.grid import most_common_color
    
    bg = rule.get('bg', most_common_color(inp))
    universe = CrossUniverse(inp, bg)
    
    if rule['type'] == 'propagate_color':
        return universe.propagate(0, 'color').tolist()
    
    elif rule['type'] == 'propagate_nearest':
        return universe.propagate(0, 'nearest').tolist()
    
    elif rule['type'] == 'directional_propagate':
        h, w = universe.layers[0].h, universe.layers[0].w
        barriers = [(r, c, d) 
                     for r in range(h) for c in range(w) 
                     for d in rule['blocked']]
        universe.set_barrier(0, barriers)
        return universe.propagate(0, 'color').tolist()
    
    elif rule['type'] == 'separator_propagate':
        return _fast_sep_propagate(np.array(inp), bg, rule['sep_color']).tolist()
    
    return None


# ============================================================
# CrossPiece生成（cross_engineへの統合用）
# ============================================================

def generate_cross_universe_pieces(train_pairs: List[Tuple[Grid, Grid]]):
    """CrossUniverse由来のCrossPieceを生成する。"""
    from arc.cross_engine import CrossPiece
    
    pieces = []
    
    rule = learn_flow_rule(train_pairs)
    if rule is not None:
        def _apply(inp, _rule=rule):
            return apply_flow_rule(inp, _rule)
        pieces.append(CrossPiece(
            f'cross_universe:{rule["type"]}',
            _apply
        ))
    
    return pieces
