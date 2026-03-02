"""
arc/game_rules.py — 世界のゲームルールDB

立体十字の4面情報からゲームルールをマッチングし、
ARC問題の操作を特定する。

=== ゲームルール = グリッド操作パターン ===

各ゲームルールは:
1. 局所パターン（4面の条件）
2. 操作（何をするか）
3. 適用条件（いつ発火するか）

をエンコードする。
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
from scipy.ndimage import label as scipy_label


@dataclass
class GameRule:
    """1つのゲームルール"""
    name: str          # ルール名
    game: str          # 出典ゲーム
    description: str   # 説明
    
    def match(self, grid, r, c, bg) -> bool:
        """このルールが(r,c)に適用可能か"""
        raise NotImplementedError
    
    def apply(self, grid, bg) -> Optional[np.ndarray]:
        """グリッド全体にルールを適用"""
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════
# 立体十字 4面記述子
# ══════════════════════════════════════════════════════════════

def cross_descriptor(g, r, c, bg):
    """セル(r,c)の4面(上下左右)の情報を抽出"""
    h, w = g.shape
    center = int(g[r, c])
    
    desc = {
        'center': center,
        'is_bg': center == bg,
        'up': [],    # 上方向の色列
        'down': [],  # 下方向
        'left': [],  # 左方向
        'right': [], # 右方向
    }
    
    # 4方向のray
    for dr, dc, key in [(-1,0,'up'), (1,0,'down'), (0,-1,'left'), (0,1,'right')]:
        ray = []
        nr, nc = r+dr, c+dc
        while 0 <= nr < h and 0 <= nc < w:
            ray.append(int(g[nr, nc]))
            nr += dr; nc += dc
        desc[key] = ray
    
    # 各方向の要約
    for key in ['up', 'down', 'left', 'right']:
        ray = desc[key]
        desc[f'{key}_first_fg'] = next((v for v in ray if v != bg), None)
        desc[f'{key}_dist_fg'] = next((i for i, v in enumerate(ray) if v != bg), -1)
        desc[f'{key}_count_fg'] = sum(1 for v in ray if v != bg)
    
    # 8近傍
    nb8 = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0: continue
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                nb8.append(int(g[nr, nc]))
            else:
                nb8.append(-1)  # OOB
    desc['nb8'] = nb8
    desc['nb8_fg_count'] = sum(1 for v in nb8 if v != bg and v != -1)
    desc['nb8_colors'] = set(v for v in nb8 if v != bg and v != -1)
    
    return desc


def full_cross_map(g, bg):
    """グリッド全体の十字記述子マップ"""
    h, w = g.shape
    return [[cross_descriptor(g, r, c, bg) for c in range(w)] for r in range(h)]


# ══════════════════════════════════════════════════════════════
# ゲームルール実装
# ══════════════════════════════════════════════════════════════

# --- 1. リバーシ（オセロ）: 同色で挟まれた異色/BGを塗る ---

class ReversiRule(GameRule):
    def __init__(self):
        super().__init__(
            'reversi_sandwich', 'Othello/Reversi',
            '同色2点の間にあるBG/異色セルを塗りつぶす'
        )
    
    def apply(self, grid, bg):
        g = grid.copy()
        h, w = g.shape
        changed = False
        
        # 8方向で同色挟み
        for r in range(h):
            for c in range(w):
                if g[r, c] != bg:
                    continue
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                    # この方向に非BGがあるか
                    nr, nc = r+dr, c+dc
                    if not (0 <= nr < h and 0 <= nc < w): continue
                    if g[nr, nc] == bg: continue
                    color = int(g[nr, nc])
                    # 反対方向にも同色があるか
                    nr2, nc2 = r-dr, c-dc
                    if not (0 <= nr2 < h and 0 <= nc2 < w): continue
                    if int(g[nr2, nc2]) == color:
                        g[r, c] = color
                        changed = True
                        break
        
        return g if changed else None


# --- 2. 囲碁: 囲まれた領域を塗る ---

class GoRule(GameRule):
    def __init__(self):
        super().__init__(
            'go_territory', 'Go/囲碁',
            '完全に囲まれたBG領域を囲んだ色で塗る'
        )
    
    def apply(self, grid, bg):
        g = grid.copy()
        h, w = g.shape
        
        # 各BG連結成分を特定
        bg_mask = (g == bg).astype(int)
        labeled, n = scipy_label(bg_mask)
        
        changed = False
        for i in range(1, n+1):
            cells = list(zip(*np.where(labeled == i)))
            
            # エッジに接しているか
            touches_edge = any(r == 0 or r == h-1 or c == 0 or c == w-1 for r, c in cells)
            if touches_edge:
                continue
            
            # 周囲の色を収集
            border_colors = set()
            for r, c in cells:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != bg:
                        border_colors.add(int(g[nr, nc]))
            
            # 1色で囲まれている場合、その色で塗る
            if len(border_colors) == 1:
                fill_color = border_colors.pop()
                for r, c in cells:
                    g[r, c] = fill_color
                    changed = True
        
        return g if changed else None


# --- 3. マインスイーパー: 近傍の数を数える ---

class MinesweeperRule(GameRule):
    def __init__(self):
        super().__init__(
            'minesweeper_count', 'Minesweeper',
            'BGセルの8近傍にある非BGセル数を色として書く'
        )
    
    def apply(self, grid, bg):
        g = grid.copy()
        h, w = g.shape
        changed = False
        
        for r in range(h):
            for c in range(w):
                if g[r, c] != bg:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != bg:
                            count += 1
                if 0 < count <= 9:
                    g[r, c] = count
                    changed = True
        
        return g if changed else None


# --- 4. テトリス: 重力で落下 ---

class TetrisRule(GameRule):
    def __init__(self):
        super().__init__(
            'tetris_gravity', 'Tetris',
            '非BGセルが重力で下に落ちる'
        )
    
    def apply(self, grid, bg):
        g = grid.copy()
        h, w = g.shape
        result = np.full_like(g, bg)
        
        for c in range(w):
            colors = [int(g[r, c]) for r in range(h) if g[r, c] != bg]
            for i, color in enumerate(reversed(colors)):
                result[h-1-i, c] = color
        
        if np.array_equal(result, g):
            return None
        return result


# --- 5. チェス・ナイト: L字移動 ---

class ChessKnightRule(GameRule):
    def __init__(self):
        super().__init__(
            'chess_knight_move', 'Chess',
            'オブジェクトがナイトのL字(2+1)で移動'
        )
    
    def apply(self, grid, bg):
        # ナイト移動は特殊すぎるので、オブジェクト移動の一般形で
        return None


# --- 6. ドミノ: 隣接ペアのマッチング ---

class DominoRule(GameRule):
    def __init__(self):
        super().__init__(
            'domino_pair', 'Domino',
            '隣接する2セルのペアを形成し、ペア内で操作'
        )
    
    def apply(self, grid, bg):
        return None


# --- 7. ライフゲーム: セルオートマトン ---

class GameOfLifeRule(GameRule):
    def __init__(self):
        super().__init__(
            'game_of_life', "Conway's Game of Life",
            '生存: 2-3近傍で生存、誕生: 3近傍で誕生'
        )
    
    def apply(self, grid, bg):
        g = grid.copy()
        h, w = g.shape
        result = np.full_like(g, bg)
        fg_color = None
        
        for r in range(h):
            for c in range(w):
                if g[r, c] != bg:
                    fg_color = int(g[r, c])
                    break
            if fg_color: break
        
        if not fg_color:
            return None
        
        for r in range(h):
            for c in range(w):
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != bg:
                            count += 1
                
                if g[r, c] != bg:
                    if count in (2, 3):
                        result[r, c] = g[r, c]
                else:
                    if count == 3:
                        result[r, c] = fg_color
        
        if np.array_equal(result, g):
            return None
        return result


# --- 8. 数独: 行列ブロック内のユニーク制約 ---

class SudokuRule(GameRule):
    def __init__(self):
        super().__init__(
            'sudoku_unique', 'Sudoku',
            '行/列/ブロック内で色がユニークになるよう欠損を補完'
        )
    
    def apply(self, grid, bg):
        g = grid.copy()
        h, w = g.shape
        changed = True
        any_changed = False
        
        while changed:
            changed = False
            for r in range(h):
                for c in range(w):
                    if g[r, c] != bg:
                        continue
                    
                    # 行にある色
                    row_colors = set(int(v) for v in g[r] if v != bg)
                    # 列にある色
                    col_colors = set(int(g[rr, c]) for rr in range(h) if g[rr, c] != bg)
                    
                    # 全色候補
                    all_colors = set()
                    for rr in range(h):
                        for cc in range(w):
                            if g[rr, cc] != bg:
                                all_colors.add(int(g[rr, cc]))
                    
                    possible = all_colors - row_colors - col_colors
                    if len(possible) == 1:
                        g[r, c] = possible.pop()
                        changed = True
                        any_changed = True
        
        return g if any_changed else None


# --- 9. 花札/トランプ: スート(色)×ランク(位置)のマッチ ---

class CardMatchRule(GameRule):
    def __init__(self):
        super().__init__(
            'card_suit_match', 'Card Games',
            '同色(スート)または同位置(ランク)のペアをマッチング'
        )
    
    def apply(self, grid, bg):
        return None


# --- 10. パズルボビー/ぷよぷよ: 同色連結4+を消す ---

class PuyoRule(GameRule):
    def __init__(self):
        super().__init__(
            'puyo_chain', 'Puyo Puyo',
            '同色4連結以上を消去'
        )
    
    def apply(self, grid, bg):
        g = grid.copy()
        h, w = g.shape
        
        # 各色の連結成分
        to_remove = set()
        for color in set(int(v) for v in g.flatten()) - {bg}:
            mask = (g == color).astype(int)
            labeled, n = scipy_label(mask)
            for i in range(1, n+1):
                cells = list(zip(*np.where(labeled == i)))
                if len(cells) >= 4:
                    for r, c in cells:
                        to_remove.add((r, c))
        
        if not to_remove:
            return None
        
        for r, c in to_remove:
            g[r, c] = bg
        return g


# --- 11. 迷路: 始点→終点の最短経路 ---

class MazeRule(GameRule):
    def __init__(self):
        super().__init__(
            'maze_path', 'Maze',
            'BG上を始点から終点まで最短経路で塗る'
        )
    
    def apply(self, grid, bg):
        g = grid.copy()
        h, w = g.shape
        
        # 特定色の点を始点/終点として検出
        special_colors = []
        for color in set(int(v) for v in g.flatten()) - {bg}:
            cells = list(zip(*np.where(g == color)))
            if len(cells) == 1:
                special_colors.append((color, cells[0]))
        
        if len(special_colors) < 2:
            return None
        
        start = special_colors[0][1]
        end = special_colors[1][1]
        path_color = special_colors[0][0]
        
        # BFS
        from collections import deque
        visited = set()
        parent = {}
        q = deque([start])
        visited.add(start)
        
        while q:
            r, c = q.popleft()
            if (r, c) == end:
                # パスを塗る
                cur = end
                changed = False
                while cur in parent:
                    pr, pc = cur
                    if g[pr, pc] == bg:
                        g[pr, pc] = path_color
                        changed = True
                    cur = parent[cur]
                return g if changed else None
            
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    if g[nr, nc] == bg or (nr, nc) == end:
                        visited.add((nr, nc))
                        parent[(nr, nc)] = (r, c)
                        q.append((nr, nc))
        
        return None


# --- 12. ソリティア: 並べ替え/整列 ---

class SortRule(GameRule):
    def __init__(self):
        super().__init__(
            'solitaire_sort', 'Solitaire',
            'オブジェクトをサイズ/色/位置でソートして整列'
        )
    
    def apply(self, grid, bg):
        return None


# --- 13. 将棋: 取った駒を打つ(stampの源) ---

class ShogiDropRule(GameRule):
    def __init__(self):
        super().__init__(
            'shogi_drop', 'Shogi/将棋',
            'あるオブジェクトを取り、別の位置にスタンプ（将棋の駒打ち）'
        )
    
    def apply(self, grid, bg):
        return None


# --- 14. ジグソーパズル: パーツの組み合わせ ---

class JigsawRule(GameRule):
    def __init__(self):
        super().__init__(
            'jigsaw_combine', 'Jigsaw Puzzle',
            '複数オブジェクトを組み合わせて1つの形を作る'
        )
    
    def apply(self, grid, bg):
        return None


# --- 15. ブロック崩し: 反射 ---

class BreakoutRule(GameRule):
    def __init__(self):
        super().__init__(
            'breakout_reflect', 'Breakout',
            '壁/障害物に当たって反射する線を描く'
        )
    
    def apply(self, grid, bg):
        g = grid.copy()
        h, w = g.shape
        
        # 単独点を探す
        special = []
        for color in set(int(v) for v in g.flatten()) - {bg}:
            cells = list(zip(*np.where(g == color)))
            if len(cells) == 1:
                special.append((color, cells[0]))
        
        if not special:
            return None
        
        # 各点から4方向に反射線を引く(簡易版: 直線のみ)
        changed = False
        for color, (sr, sc) in special:
            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                r, c = sr+dr, sc+dc
                while 0 <= r < h and 0 <= c < w:
                    if g[r, c] != bg:
                        break
                    g[r, c] = color
                    changed = True
                    r += dr; c += dc
        
        return g if changed else None


# ══════════════════════════════════════════════════════════════
# ルールDB & マッチエンジン
# ══════════════════════════════════════════════════════════════

ALL_GAME_RULES = [
    ReversiRule(),
    GoRule(),
    MinesweeperRule(),
    TetrisRule(),
    GameOfLifeRule(),
    SudokuRule(),
    PuyoRule(),
    MazeRule(),
    BreakoutRule(),
]

def game_rule_solve(train_pairs, test_input):
    """全ゲームルールを試行"""
    from arc.grid import grid_eq
    
    gi = np.array(test_input)
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    for rule in ALL_GAME_RULES:
        try:
            ok = True
            for inp, out in train_pairs:
                ga = np.array(inp)
                bg_t = int(Counter(ga.flatten()).most_common(1)[0][0])
                result = rule.apply(ga, bg_t)
                if result is None or not grid_eq(result.tolist(), out):
                    ok = False; break
            
            if ok:
                result = rule.apply(gi, bg)
                if result is not None:
                    return result.tolist(), rule.name
        except Exception:
            continue
    
    # 複数ステップ（ゲームルールの連続適用）
    for rule in ALL_GAME_RULES:
        try:
            ok = True
            for inp, out in train_pairs:
                ga = np.array(inp)
                bg_t = int(Counter(ga.flatten()).most_common(1)[0][0])
                # 最大10ステップ
                cur = ga.copy()
                for _ in range(10):
                    r = rule.apply(cur, bg_t)
                    if r is None: break
                    if grid_eq(r.tolist(), out):
                        cur = r; break
                    cur = r
                if not grid_eq(cur.tolist(), out):
                    ok = False; break
            
            if ok:
                cur = gi.copy()
                for _ in range(10):
                    r = rule.apply(cur, bg)
                    if r is None: break
                    cur = r
                return cur.tolist(), rule.name + '_multi'
        except Exception:
            continue
    
    return None, None


if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    from arc.grid import grid_eq
    
    split = 'evaluation' if '--eval' in sys.argv else 'training'
    data_dir = Path(f'/tmp/arc-agi-2/data/{split}')
    
    solved = []
    for tf in sorted(data_dir.glob('*.json')):
        tid = tf.stem
        with open(tf) as f: task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti, to = task['test'][0]['input'], task['test'][0].get('output')
        
        result, rule_name = game_rule_solve(tp, ti)
        if result and to and grid_eq(result, to):
            solved.append((tid, rule_name))
            print(f'  ✓ {tid} [{rule_name}]')
    
    total = len(list(data_dir.glob('*.json')))
    print(f'\n{split}: {len(solved)}/{total}')
    rule_counts = Counter(r for _, r in solved)
    for r, c in rule_counts.most_common():
        print(f'  {c:3d} — {r}')
