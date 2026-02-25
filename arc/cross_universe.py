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
        遮断(blocked)がある方向にはフローが流れない。"""
        result = self.arr.copy()
        
        for step in range(max_steps):
            changed = False
            new_result = result.copy()
            
            # 4方向: (dr, dc, direction_index)
            dirs = [(-1, 0, 0), (0, 1, 1), (1, 0, 2), (0, -1, 3)]
            
            for r in range(self.h):
                for c in range(self.w):
                    if result[r, c] != self.bg:
                        # 非bgセルから4方向に伝播
                        for dr, dc, di in dirs:
                            if self.nodes[r][c].blocked[di]:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.h and 0 <= nc < self.w:
                                # 反対方向の遮断もチェック
                                opp_di = (di + 2) % 4
                                if self.nodes[nr][nc].blocked[opp_di]:
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
        """再帰的にcross構造の階層を構築。"""
        from arc.block_ir import detect_block_grid
        
        current_grid = self.grid
        current_bg = self.bg
        
        for level in range(5):  # 最大5レベル
            layer = CrossLayer(current_grid, current_bg)
            self.layers.append(layer)
            
            # ブロック構造を検出して次レベルへ
            ir = detect_block_grid(current_grid, current_bg)
            if ir is None:
                break
            
            self.block_params.append({
                'block_h': ir['block_h'],
                'block_w': ir['block_w'],
                'sep_w': ir['sep_w'],
                'sep_color': ir['sep_color'],
                'n_rows': ir['n_rows'],
                'n_cols': ir['n_cols'],
            })
            
            # 次レベル = ブロック色グリッド
            current_grid = ir['block_colors'].tolist()
            current_bg = int(Counter(
                int(v) for row in current_grid for v in row
            ).most_common(1)[0][0])
    
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

def learn_flow_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """訓練ペアからフロー伝播ルールを学習する。
    
    試行するルール:
    1. 単純伝播（全方向開放）
    2. 色境界遮断（異なる色間はフロー遮断）
    3. separator遮断（特定色の行/列で遮断）
    4. 方向限定伝播（上のみ、右のみ、etc.）
    """
    from arc.grid import grid_eq, most_common_color
    
    if not train_pairs:
        return None
    
    bg = most_common_color(train_pairs[0][0])
    
    # Rule 1: 全方向開放伝播
    rule1_ok = True
    for inp, out in train_pairs:
        universe = CrossUniverse(inp, bg)
        result = universe.propagate(0, 'color')
        if not grid_eq(result.tolist(), out):
            rule1_ok = False
            break
    if rule1_ok:
        return {'type': 'propagate_color', 'bg': bg}
    
    # Rule 2: 最近接伝播
    rule2_ok = True
    for inp, out in train_pairs:
        universe = CrossUniverse(inp, bg)
        result = universe.propagate(0, 'nearest')
        if not grid_eq(result.tolist(), out):
            rule2_ok = False
            break
    if rule2_ok:
        return {'type': 'propagate_nearest', 'bg': bg}
    
    # Rule 3: 方向限定伝播（4方向 + 4対角 = 8パターン）
    for blocked_dirs in [[0], [1], [2], [3],  # 上/右/下/左を遮断
                          [0, 2], [1, 3],  # 縦/横を遮断
                          [0, 1], [0, 3], [2, 1], [2, 3]]:  # 2方向遮断
        dir_ok = True
        for inp, out in train_pairs:
            universe = CrossUniverse(inp, bg)
            # 全セルの指定方向を遮断
            barriers = []
            for r in range(universe.layers[0].h):
                for c in range(universe.layers[0].w):
                    for d in blocked_dirs:
                        barriers.append((r, c, d))
            universe.set_barrier(0, barriers)
            result = universe.propagate(0, 'color')
            if not grid_eq(result.tolist(), out):
                dir_ok = False
                break
        if dir_ok:
            return {'type': 'directional_propagate', 'bg': bg, 
                    'blocked': blocked_dirs}
    
    # Rule 4: separator色で遮断
    arr0 = np.array(train_pairs[0][0])
    colors = set(int(v) for v in arr0.flatten())
    for sep_color in colors:
        if sep_color == bg:
            continue
        sep_ok = True
        for inp, out in train_pairs:
            universe = CrossUniverse(inp, bg)
            arr = np.array(inp)
            h, w = arr.shape
            barriers = []
            for r in range(h):
                for c in range(w):
                    if arr[r, c] == sep_color:
                        # separatorセルは全方向遮断
                        for d in range(4):
                            barriers.append((r, c, d))
            universe.set_barrier(0, barriers)
            result = universe.propagate(0, 'color')
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
        arr = np.array(inp)
        h, w = arr.shape
        barriers = [(r, c, d) 
                     for r in range(h) for c in range(w)
                     if arr[r, c] == rule['sep_color']
                     for d in range(4)]
        universe.set_barrier(0, barriers)
        return universe.propagate(0, 'color').tolist()
    
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
