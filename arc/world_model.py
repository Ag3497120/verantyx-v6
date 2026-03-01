#!/usr/bin/env python3
"""
Verantyx World Model — Cross Simulator as World Physics Engine
==============================================================
世界のルール（物理法則）を「実行可能なシミュレーションコマンド」として定義。
各コマンドはグリッド → グリッドの変換。
コマンドの組み合わせ（プログラム）を探索し、trainで検証。

設計思想:
  - 各コマンドは「意味」ではなく「操作」
  - コマンドは合成可能（A→B→Cのパイプライン）
  - パラメータ付きコマンドはtrainから自動推定
  - 世界モデル = コマンド群 + 合成ルール + シミュレータ

使い方:
  from arc.world_model import WorldModel
  wm = WorldModel()
  solutions = wm.solve(train_pairs, test_inputs)
"""

from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Callable, Dict, Any
from arc.grid import Grid, grid_shape, grid_eq, most_common_color

# ═══════════════════════════════════════════════════════
# コマンド基盤
# ═══════════════════════════════════════════════════════

class SimCommand:
    """1つのシミュレーションコマンド（世界の法則1つ）"""
    __slots__ = ['name', 'fn', 'category']
    
    def __init__(self, name: str, fn: Callable[[Grid], Optional[Grid]], category: str = ""):
        self.name = name
        self.fn = fn
        self.category = category
    
    def __call__(self, grid: Grid) -> Optional[Grid]:
        try:
            return self.fn(grid)
        except:
            return None

class SimProgram:
    """コマンドの列（パイプライン）"""
    __slots__ = ['commands', 'name']
    
    def __init__(self, commands: List[SimCommand]):
        self.commands = commands
        self.name = " → ".join(c.name for c in commands)
    
    def apply(self, grid: Grid) -> Optional[Grid]:
        g = grid
        for cmd in self.commands:
            g = cmd(g)
            if g is None:
                return None
        return g

# ═══════════════════════════════════════════════════════
# ユーティリティ（コマンド内部で使用）
# ═══════════════════════════════════════════════════════

def _bg(grid):
    c = Counter()
    for row in grid: c.update(row)
    return c.most_common(1)[0][0]

def _color_counts(grid):
    c = Counter()
    for row in grid: c.update(row)
    return c

def _objects_4(grid, bg):
    """4-connected non-bg objects."""
    h, w = len(grid), len(grid[0])
    visited = [[False]*w for _ in range(h)]
    objs = []
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] != bg:
                obj = []
                stack = [(r,c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    obj.append((cr, cc, grid[cr][cc]))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            stack.append((nr,nc))
                objs.append(obj)
    return objs

def _objects_8(grid, bg):
    """8-connected non-bg objects."""
    h, w = len(grid), len(grid[0])
    visited = [[False]*w for _ in range(h)]
    objs = []
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] != bg:
                obj = []
                stack = [(r,c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    obj.append((cr, cc, grid[cr][cc]))
                    for dr in [-1,0,1]:
                        for dc in [-1,0,1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc] != bg:
                                visited[nr][nc] = True
                                stack.append((nr,nc))
                objs.append(obj)
    return objs

def _bbox(cells):
    rs = [r for r,c,_ in cells]
    cs = [c for r,c,_ in cells]
    return min(rs), min(cs), max(rs), max(cs)

def _crop_cells(grid, cells, bg):
    r0, c0, r1, c1 = _bbox(cells)
    h, w = r1-r0+1, c1-c0+1
    result = [[bg]*w for _ in range(h)]
    for r, c, v in cells:
        result[r-r0][c-c0] = v
    return result

def _grid_copy(g):
    return [row[:] for row in g]

def _is_exterior_bg(grid, bg):
    """Returns visited[][] marking exterior bg cells (reachable from border)."""
    h, w = len(grid), len(grid[0])
    visited = [[False]*w for _ in range(h)]
    queue = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h-1 or c == 0 or c == w-1) and grid[r][c] == bg:
                if not visited[r][c]:
                    visited[r][c] = True
                    queue.append((r,c))
    while queue:
        cr, cc = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc] == bg:
                visited[nr][nc] = True
                queue.append((nr,nc))
    return visited

# ═══════════════════════════════════════════════════════
# 物理法則コマンド群
# ═══════════════════════════════════════════════════════

def build_physics_commands() -> List[SimCommand]:
    """世界の物理法則をシミュレーションコマンドとして定義。"""
    cmds = []
    
    # ──── 幾何変換 ────
    cmds.append(SimCommand("rot90",
        lambda g: [list(r) for r in zip(*g[::-1])], "geometry"))
    cmds.append(SimCommand("rot180",
        lambda g: [row[::-1] for row in g[::-1]], "geometry"))
    cmds.append(SimCommand("rot270",
        lambda g: [list(r) for r in zip(*[row[::-1] for row in g])], "geometry"))
    cmds.append(SimCommand("flip_h",
        lambda g: [row[::-1] for row in g], "geometry"))
    cmds.append(SimCommand("flip_v",
        lambda g: g[::-1], "geometry"))
    cmds.append(SimCommand("transpose",
        lambda g: [list(r) for r in zip(*g)], "geometry"))
    
    # ──── 重力（物が落ちる）────
    for direction in ["down", "up", "left", "right"]:
        def make_gravity(d):
            def fn(g):
                bg = _bg(g)
                h, w = len(g), len(g[0])
                result = [[bg]*w for _ in range(h)]
                if d == "down":
                    for c in range(w):
                        non = [g[r][c] for r in range(h) if g[r][c] != bg]
                        for i, v in enumerate(non):
                            result[h-len(non)+i][c] = v
                elif d == "up":
                    for c in range(w):
                        non = [g[r][c] for r in range(h) if g[r][c] != bg]
                        for i, v in enumerate(non):
                            result[i][c] = v
                elif d == "left":
                    for r in range(h):
                        non = [g[r][c] for c in range(w) if g[r][c] != bg]
                        for i, v in enumerate(non):
                            result[r][i] = v
                elif d == "right":
                    for r in range(h):
                        non = [g[r][c] for c in range(w) if g[r][c] != bg]
                        for i, v in enumerate(non):
                            result[r][w-len(non)+i] = v
                return result
            return fn
        cmds.append(SimCommand(f"gravity_{direction}", make_gravity(direction), "physics"))
    
    # ──── 色の重力（特定色だけ落ちる）────
    for color in range(10):
        for d in ["down", "up"]:
            def make_color_gravity(col, direction):
                def fn(g):
                    bg = _bg(g)
                    h, w = len(g), len(g[0])
                    result = _grid_copy(g)
                    for c in range(w):
                        positions = [r for r in range(h) if g[r][c] == col]
                        empties = [r for r in range(h) if g[r][c] == bg]
                        if not positions: continue
                        # Remove color cells
                        for r in positions:
                            result[r][c] = bg
                        # Drop to bottom/top
                        if direction == "down":
                            available = sorted([r for r in range(h) if result[r][c] == bg], reverse=True)
                        else:
                            available = sorted([r for r in range(h) if result[r][c] == bg])
                        for i, r in enumerate(available[:len(positions)]):
                            result[r][c] = col
                    return result
                return fn
            cmds.append(SimCommand(f"gravity_{d}_c{color}", make_color_gravity(color, d), "physics"))
    
    # ──── 塗りつぶし（液体の充填）────
    def flood_fill_enclosed(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        exterior = _is_exterior_bg(g, bg)
        result = _grid_copy(g)
        for r in range(h):
            for c in range(w):
                if g[r][c] == bg and not exterior[r][c]:
                    # Find adjacent non-bg color
                    neighbors = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and g[nr][nc] != bg:
                            neighbors.append(g[nr][nc])
                    if neighbors:
                        result[r][c] = Counter(neighbors).most_common(1)[0][0]
        return result
    cmds.append(SimCommand("fill_enclosed", flood_fill_enclosed, "physics"))
    
    # 色指定のフラッドフィル
    for color in range(10):
        def make_fill_color(col):
            def fn(g):
                bg = _bg(g)
                h, w = len(g), len(g[0])
                exterior = _is_exterior_bg(g, bg)
                result = _grid_copy(g)
                for r in range(h):
                    for c in range(w):
                        if g[r][c] == bg and not exterior[r][c]:
                            result[r][c] = col
                return result
            return fn
        cmds.append(SimCommand(f"fill_enclosed_c{color}", make_fill_color(color), "physics"))
    
    # ──── 対称性（鏡の法則）────
    def sym_h(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = _grid_copy(g)
        for r in range(h):
            for c in range(w//2):
                mc = w-1-c
                if result[r][c] == bg and result[r][mc] != bg:
                    result[r][c] = result[r][mc]
                elif result[r][mc] == bg and result[r][c] != bg:
                    result[r][mc] = result[r][c]
        return result
    cmds.append(SimCommand("symmetry_h", sym_h, "symmetry"))
    
    def sym_v(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = _grid_copy(g)
        for r in range(h//2):
            mr = h-1-r
            for c in range(w):
                if result[r][c] == bg and result[mr][c] != bg:
                    result[r][c] = result[mr][c]
                elif result[mr][c] == bg and result[r][c] != bg:
                    result[mr][c] = result[r][c]
        return result
    cmds.append(SimCommand("symmetry_v", sym_v, "symmetry"))
    
    def sym_4fold(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = _grid_copy(g)
        for r in range((h+1)//2):
            for c in range((w+1)//2):
                positions = [(r,c),(r,w-1-c),(h-1-r,c),(h-1-r,w-1-c)]
                vals = [result[pr][pc] for pr, pc in positions
                        if 0<=pr<h and 0<=pc<w and result[pr][pc] != bg]
                if vals:
                    fill = Counter(vals).most_common(1)[0][0]
                    for pr, pc in positions:
                        if 0<=pr<h and 0<=pc<w and result[pr][pc] == bg:
                            result[pr][pc] = fill
        return result
    cmds.append(SimCommand("symmetry_4fold", sym_4fold, "symmetry"))
    
    def sym_diag(g):
        """対角対称 (r,c) ↔ (c,r)"""
        bg = _bg(g)
        h, w = len(g), len(g[0])
        if h != w: return g
        result = _grid_copy(g)
        for r in range(h):
            for c in range(r+1, w):
                if result[r][c] == bg and result[c][r] != bg:
                    result[r][c] = result[c][r]
                elif result[c][r] == bg and result[r][c] != bg:
                    result[c][r] = result[r][c]
        return result
    cmds.append(SimCommand("symmetry_diag", sym_diag, "symmetry"))
    
    # ──── クロップ（視野の切り出し）────
    def crop_content(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        rows = [r for r in range(h) if any(g[r][c] != bg for c in range(w))]
        cols = [c for c in range(w) if any(g[r][c] != bg for r in range(h))]
        if not rows or not cols: return g
        return [g[r][min(cols):max(cols)+1] for r in range(min(rows), max(rows)+1)]
    cmds.append(SimCommand("crop", crop_content, "view"))
    
    # ──── 色の消去（特定色を背景にする）────
    for color in range(10):
        def make_erase(col):
            def fn(g):
                bg = _bg(g)
                if col == bg: return g
                return [[bg if c == col else c for c in row] for row in g]
            return fn
        cmds.append(SimCommand(f"erase_c{color}", make_erase(color), "color"))
    
    # ──── 色の置換（A→B）────
    # これはパラメトリックなのでbuild_parametric_commandsで生成
    
    # ──── 線の延長（点から端まで伸ばす）────
    def extend_h(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = _grid_copy(g)
        for r in range(h):
            colors = set(g[r][c] for c in range(w) if g[r][c] != bg)
            if len(colors) == 1:
                col = colors.pop()
                positions = [c for c in range(w) if g[r][c] == col]
                if len(positions) >= 2:
                    for c in range(min(positions), max(positions)+1):
                        result[r][c] = col
        return result
    cmds.append(SimCommand("extend_h", extend_h, "line"))
    
    def extend_v(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = _grid_copy(g)
        for c in range(w):
            colors = set(g[r][c] for r in range(h) if g[r][c] != bg)
            if len(colors) == 1:
                col = colors.pop()
                positions = [r for r in range(h) if g[r][c] == col]
                if len(positions) >= 2:
                    for r in range(min(positions), max(positions)+1):
                        result[r][c] = col
        return result
    cmds.append(SimCommand("extend_v", extend_v, "line"))
    
    def extend_full_h(g):
        """行の非bg色が1種なら行全体をその色にする"""
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = _grid_copy(g)
        for r in range(h):
            colors = set(g[r][c] for c in range(w) if g[r][c] != bg)
            if len(colors) == 1 and sum(1 for c in range(w) if g[r][c] != bg) >= 1:
                col = colors.pop()
                for c in range(w):
                    result[r][c] = col
        return result
    cmds.append(SimCommand("extend_full_h", extend_full_h, "line"))
    
    def extend_full_v(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = _grid_copy(g)
        for c in range(w):
            colors = set(g[r][c] for r in range(h) if g[r][c] != bg)
            if len(colors) == 1 and sum(1 for r in range(h) if g[r][c] != bg) >= 1:
                col = colors.pop()
                for r in range(h):
                    result[r][c] = col
        return result
    cmds.append(SimCommand("extend_full_v", extend_full_v, "line"))
    
    # ──── オブジェクト操作 ────
    def remove_largest(g):
        bg = _bg(g)
        objs = _objects_4(g, bg)
        if not objs: return g
        largest = max(objs, key=len)
        result = _grid_copy(g)
        for r, c, v in largest:
            result[r][c] = bg
        return result
    cmds.append(SimCommand("remove_largest", remove_largest, "object"))
    
    def remove_smallest(g):
        bg = _bg(g)
        objs = _objects_4(g, bg)
        if not objs: return g
        smallest = min(objs, key=len)
        result = _grid_copy(g)
        for r, c, v in smallest:
            result[r][c] = bg
        return result
    cmds.append(SimCommand("remove_smallest", remove_smallest, "object"))
    
    def keep_largest(g):
        bg = _bg(g)
        objs = _objects_4(g, bg)
        if not objs: return g
        largest = max(objs, key=len)
        h, w = len(g), len(g[0])
        result = [[bg]*w for _ in range(h)]
        for r, c, v in largest:
            result[r][c] = v
        return result
    cmds.append(SimCommand("keep_largest", keep_largest, "object"))
    
    def keep_smallest(g):
        bg = _bg(g)
        objs = _objects_4(g, bg)
        if not objs: return g
        smallest = min(objs, key=len)
        h, w = len(g), len(g[0])
        result = [[bg]*w for _ in range(h)]
        for r, c, v in smallest:
            result[r][c] = v
        return result
    cmds.append(SimCommand("keep_smallest", keep_smallest, "object"))
    
    def fill_obj_bbox(g):
        """各オブジェクトのbounding boxを塗りつぶす"""
        bg = _bg(g)
        objs = _objects_4(g, bg)
        result = _grid_copy(g)
        for obj in objs:
            r0, c0, r1, c1 = _bbox(obj)
            col = Counter(v for _,_,v in obj).most_common(1)[0][0]
            for r in range(r0, r1+1):
                for c in range(c0, c1+1):
                    result[r][c] = col
        return result
    cmds.append(SimCommand("fill_bbox", fill_obj_bbox, "object"))
    
    def outline_objects(g):
        """各オブジェクトの輪郭（bgに隣接するセル）だけ残す"""
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = [[bg]*w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if g[r][c] != bg:
                    is_border = False
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if not (0<=nr<h and 0<=nc<w) or g[nr][nc] == bg:
                            is_border = True; break
                    if is_border:
                        result[r][c] = g[r][c]
        return result
    cmds.append(SimCommand("outline", outline_objects, "object"))
    
    def interior_only(g):
        """オブジェクトの内部（輪郭以外）だけ残す"""
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = [[bg]*w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if g[r][c] != bg:
                    is_border = False
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if not (0<=nr<h and 0<=nc<w) or g[nr][nc] == bg:
                            is_border = True; break
                    if not is_border:
                        result[r][c] = g[r][c]
        return result
    cmds.append(SimCommand("interior", interior_only, "object"))
    
    # ──── セルオートマトン（1ステップ）────
    def ca_life_step(g):
        """Conway's Game of Life 1step (non-bg = alive)"""
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = [[bg]*w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                alive_neighbors = 0
                dominant_color = Counter()
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and g[nr][nc] != bg:
                            alive_neighbors += 1
                            dominant_color[g[nr][nc]] += 1
                
                is_alive = g[r][c] != bg
                if is_alive:
                    if alive_neighbors in [2, 3]:
                        result[r][c] = g[r][c]
                else:
                    if alive_neighbors == 3 and dominant_color:
                        result[r][c] = dominant_color.most_common(1)[0][0]
        return result
    cmds.append(SimCommand("ca_life", ca_life_step, "automaton"))
    
    def ca_grow(g):
        """各非bg色セルが4方向に1セル成長"""
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = _grid_copy(g)
        for r in range(h):
            for c in range(w):
                if g[r][c] != bg:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and result[nr][nc] == bg:
                            result[nr][nc] = g[r][c]
        return result
    cmds.append(SimCommand("grow_1", ca_grow, "automaton"))
    
    def ca_shrink(g):
        """bgに隣接する非bgセルを削除（侵食）"""
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = _grid_copy(g)
        for r in range(h):
            for c in range(w):
                if g[r][c] != bg:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if not (0<=nr<h and 0<=nc<w) or g[nr][nc] == bg:
                            result[r][c] = bg
                            break
        return result
    cmds.append(SimCommand("shrink_1", ca_shrink, "automaton"))
    
    # ──── マジョリティ/マイノリティ ────
    def replace_minority(g):
        bg = _bg(g)
        cc = Counter()
        for row in g:
            for v in row:
                if v != bg: cc[v] += 1
        if len(cc) < 2: return g
        majority = cc.most_common(1)[0][0]
        minority = cc.most_common()[-1][0]
        return [[majority if v == minority else v for v in row] for row in g]
    cmds.append(SimCommand("replace_minority", replace_minority, "color"))
    
    # ──── 行/列の除去 ────
    def remove_bg_rows(g):
        bg = _bg(g)
        return [row for row in g if any(c != bg for c in row)] or g
    cmds.append(SimCommand("remove_bg_rows", remove_bg_rows, "view"))
    
    def remove_bg_cols(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        keep = [c for c in range(w) if any(g[r][c] != bg for r in range(h))]
        if not keep: return g
        return [[g[r][c] for c in keep] for r in range(h)]
    cmds.append(SimCommand("remove_bg_cols", remove_bg_cols, "view"))
    
    # ──── 並べ替え ────
    def sort_rows_by_count(g):
        bg = _bg(g)
        rows = [(sum(1 for c in row if c != bg), i, row) for i, row in enumerate(g)]
        rows.sort(key=lambda x: x[0])
        return [row for _, _, row in rows]
    cmds.append(SimCommand("sort_rows", sort_rows_by_count, "order"))
    
    def sort_cols_by_count(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        cols = []
        for c in range(w):
            col = [g[r][c] for r in range(h)]
            cnt = sum(1 for v in col if v != bg)
            cols.append((cnt, c, col))
        cols.sort(key=lambda x: x[0])
        return [[cols[c][2][r] for c in range(w)] for r in range(h)]
    cmds.append(SimCommand("sort_cols", sort_cols_by_count, "order"))
    
    def reverse_rows(g):
        return g[::-1]
    cmds.append(SimCommand("reverse_rows", reverse_rows, "order"))
    
    def reverse_cols(g):
        return [row[::-1] for row in g]
    cmds.append(SimCommand("reverse_cols", reverse_cols, "order"))
    
    # ──── スケーリング ────
    for s in [2, 3]:
        def make_up(sc):
            def fn(g):
                h, w = len(g), len(g[0])
                return [[g[r//sc][c//sc] for c in range(w*sc)] for r in range(h*sc)]
            return fn
        cmds.append(SimCommand(f"upscale_{s}x", make_up(s), "scale"))
        
        def make_down(sc):
            def fn(g):
                h, w = len(g), len(g[0])
                nh, nw = h//sc, w//sc
                if nh == 0 or nw == 0: return None
                return [[Counter(g[r*sc+dr][c*sc+dc] for dr in range(sc) for dc in range(sc)
                                 if r*sc+dr < h and c*sc+dc < w).most_common(1)[0][0]
                         for c in range(nw)] for r in range(nh)]
            return fn
        cmds.append(SimCommand(f"downscale_{s}x", make_down(s), "scale"))
    
    # ──── タイリング ────
    for rh in [2, 3]:
        for rw in [2, 3]:
            def make_tile(th, tw):
                def fn(g):
                    h, w = len(g), len(g[0])
                    return [[g[r%h][c%w] for c in range(w*tw)] for r in range(h*th)]
                return fn
            cmds.append(SimCommand(f"tile_{rh}x{rw}", make_tile(rh, rw), "scale"))
    
    # ──── 収束（同じコマンドを変化がなくなるまで繰り返す）────
    # これはsearchレベルで実現（後述）
    
    return cmds


def build_parametric_commands(train_pairs) -> List[SimCommand]:
    """trainから学習するパラメトリックコマンド。"""
    cmds = []
    
    inp0, out0 = train_pairs[0]
    ih, iw = len(inp0), len(inp0[0])
    oh, ow = len(out0), len(out0[0])
    
    # ──── 色マッピング（trainから学習）────
    if (ih, iw) == (oh, ow):
        cmap = {}
        ok = True
        for inp, out in train_pairs:
            h, w = len(inp), len(inp[0])
            if len(out) != h or len(out[0]) != w:
                ok = False; break
            for r in range(h):
                for c in range(w):
                    ic, oc = inp[r][c], out[r][c]
                    if ic in cmap:
                        if cmap[ic] != oc: ok = False; break
                    else:
                        cmap[ic] = oc
                if not ok: break
            if not ok: break
        
        if ok and cmap and any(k != v for k, v in cmap.items()):
            cm = dict(cmap)
            cmds.append(SimCommand("color_map_learned",
                lambda g, m=cm: [[m.get(c, c) for c in row] for row in g], "parametric"))
    
    # ──── サイズ固定の出力 ────
    out_sizes = set((len(o), len(o[0])) for _, o in train_pairs)
    if len(out_sizes) == 1:
        foh, fow = out_sizes.pop()
        if (foh, fow) != (ih, iw):
            # Crop to fixed size from top-left
            cmds.append(SimCommand(f"crop_to_{foh}x{fow}",
                lambda g, h=foh, w=fow: [row[:w] for row in g[:h]], "parametric"))
    
    # ──── パネル抽出（仕切り線検出）────
    bg0 = _bg(inp0)
    # 行全体が同じ非bg色 → セパレータ
    for sep_color in range(10):
        if sep_color == bg0: continue
        sep_rows = [r for r in range(ih) if all(inp0[r][c] == sep_color for c in range(iw))]
        sep_cols = [c for c in range(iw) if all(inp0[r][c] == sep_color for r in range(ih))]
        
        if len(sep_rows) >= 1:
            # Extract panels between separators
            boundaries = [-1] + sep_rows + [ih]
            panels = []
            for i in range(len(boundaries)-1):
                r_start = boundaries[i] + 1
                r_end = boundaries[i+1]
                if r_start < r_end:
                    panels.append([inp0[r][:] for r in range(r_start, r_end)])
            
            if len(panels) >= 2:
                for pi in range(len(panels)):
                    def make_extract_panel(idx, srows, sc):
                        def fn(g):
                            h, w = len(g), len(g[0])
                            sr = [r for r in range(h) if all(g[r][c] == sc for c in range(w))]
                            bounds = [-1] + sr + [h]
                            ps = []
                            for i in range(len(bounds)-1):
                                rs, re = bounds[i]+1, bounds[i+1]
                                if rs < re:
                                    ps.append([g[r][:] for r in range(rs, re)])
                            if idx < len(ps):
                                return ps[idx]
                            return None
                        return fn
                    cmds.append(SimCommand(f"panel_{pi}_sep_c{sep_color}",
                        make_extract_panel(pi, sep_rows, sep_color), "parametric"))
    
    return cmds


# ═══════════════════════════════════════════════════════
# 世界モデル（コマンド探索エンジン）
# ═══════════════════════════════════════════════════════

class WorldModel:
    """
    世界モデル = 物理法則コマンド群 + プログラム合成 + シミュレーション検証
    
    探索戦略:
      1. 単一コマンド（depth=1）
      2. 2コマンド合成（depth=2）
      3. 収束ループ（コマンドを変化なしまで繰り返す）
      4. 3コマンド合成（depth=3, 限定的）
    """
    
    def __init__(self):
        self.physics = build_physics_commands()
    
    def solve(self, train_pairs, test_inputs, max_depth=3, timeout_s=60):
        """
        trainで検証される全プログラムを探索し、testに適用。
        Returns: List of (program_name, test_predictions, confidence)
        """
        import time
        t0 = time.time()
        
        # パラメトリックコマンドを追加
        parametric = build_parametric_commands(train_pairs)
        all_cmds = self.physics + parametric
        
        solutions = []
        
        # ── Depth 1: 単一コマンド ──
        for cmd in all_cmds:
            if time.time() - t0 > timeout_s: break
            prog = SimProgram([cmd])
            if self._verify(prog, train_pairs):
                preds = self._predict(prog, test_inputs)
                if preds:
                    solutions.append((prog.name, preds, 1.0))
        
        # ── 収束ループ: 単一コマンドを繰り返す ──
        for cmd in all_cmds:
            if time.time() - t0 > timeout_s: break
            conv_prog = self._make_converge_program(cmd)
            if conv_prog and self._verify(conv_prog, train_pairs):
                preds = self._predict(conv_prog, test_inputs)
                if preds:
                    solutions.append((f"converge({cmd.name})", preds, 0.9))
        
        # ── Depth 2: 2コマンド合成 ──
        # Pre-filter: run each command on train[0] input, keep those that produce valid grids
        inp0, out0 = train_pairs[0]
        active = []
        mid_results = {}  # cmd_idx -> intermediate result for train[0]
        for i, cmd in enumerate(all_cmds):
            if time.time() - t0 > timeout_s: break
            try:
                r = cmd(inp0)
                if r is not None and isinstance(r, list) and len(r) > 0:
                    active.append((i, cmd))
                    mid_results[i] = r
            except:
                pass
        
        active = active[:25]  # limit
        
        # Second step: for each mid-result, find a cmd that maps it to out0
        for i, c1 in active:
            if time.time() - t0 > timeout_s: break
            mid = mid_results[i]
            for c2 in all_cmds:
                if time.time() - t0 > timeout_s: break
                try:
                    r2 = c2(mid)
                    if r2 is None or not grid_eq(r2, out0):
                        continue
                except:
                    continue
                # Passes train[0], verify all
                prog = SimProgram([c1, c2])
                if self._verify(prog, train_pairs):
                    preds = self._predict(prog, test_inputs)
                    if preds:
                        solutions.append((prog.name, preds, 0.8))
        
        # ── Depth 3: 限定的 ──
        if max_depth >= 3 and len(solutions) == 0:
            active3 = active[:8]
            for i1, c1 in active3:
                if time.time() - t0 > timeout_s: break
                mid1 = mid_results.get(i1)
                if mid1 is None: continue
                for i2, c2 in active3:
                    if time.time() - t0 > timeout_s: break
                    try:
                        mid2 = c2(mid1)
                        if mid2 is None: continue
                    except: continue
                    for c3 in all_cmds:
                        if time.time() - t0 > timeout_s: break
                        try:
                            r3 = c3(mid2)
                            if r3 is None or not grid_eq(r3, out0): continue
                        except: continue
                        prog = SimProgram([c1, c2, c3])
                        if self._verify(prog, train_pairs):
                            preds = self._predict(prog, test_inputs)
                            if preds:
                                solutions.append((prog.name, preds, 0.6))
        
        return solutions
    
    def _verify(self, prog: SimProgram, train_pairs) -> bool:
        """Train全例でexact matchするか検証。"""
        for inp, expected in train_pairs:
            result = prog.apply(inp)
            if result is None or not grid_eq(result, expected):
                return False
        return True
    
    def _predict(self, prog: SimProgram, test_inputs) -> Optional[List[Grid]]:
        """Test入力に対する予測を生成。"""
        preds = []
        for ti in test_inputs:
            p = prog.apply(ti)
            if p is None:
                return None
            preds.append(p)
        return preds
    
    def _make_converge_program(self, cmd: SimCommand, max_iter=20):
        """コマンドを収束まで繰り返すプログラム。"""
        class ConvergeProgram:
            def __init__(self, c, mi):
                self.name = f"converge({c.name})"
                self._cmd = c
                self._max = mi
            def apply(self, grid):
                g = grid
                for _ in range(self._max):
                    g2 = self._cmd(g)
                    if g2 is None: return None
                    if grid_eq(g, g2): return g
                    g = g2
                return g
        
        return ConvergeProgram(cmd, max_iter)
