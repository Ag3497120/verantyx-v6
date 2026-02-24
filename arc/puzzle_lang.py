"""
puzzle_lang.py — パズル推論DSL (Puzzle Reasoning Language)

自然言語的な語彙を使い、ARC-AGI2タスクの変換ルールを記述する。
各語彙は「意味」ではなく「指示」として機械的に解釈される。

=== 語彙体系 ===

【量化子 (Quantifiers)】
  ALL(selector)     — 条件に合う全セル/オブジェクト
  EACH(selector)    — 各オブジェクトに個別適用
  BETWEEN(a, b)     — 2つの要素の間の領域
  PAIR(selector)    — 2つずつ組にする

【選択子 (Selectors)】
  cells(color=C)         — 色Cのセル群
  objects()              — 連結成分オブジェクト群
  rows() / cols()        — 行/列
  regions(separator=C)   — セパレータ色で区切られた領域
  largest / smallest     — サイズで選択
  nearest(ref)           — 最近傍

【空間子 (Spatial)】
  adjacent(n=4/8)        — 隣接セル
  enclosed()             — 囲まれた領域
  through(point)         — 点を通る直線
  from_to(a, b)          — aからbまで
  bbox()                 — バウンディングボックス
  row_of / col_of        — 同じ行/列

【操作子 (Operations)】
  fill(color)            — 塗りつぶし
  draw_line(dir, color)  — 線を引く
  extend(dir)            — 既存パターンを延長
  reflect(axis)          — 反転
  count()                — 数える
  connect(color)         — 2点間を接続
  recolor(from, to)      — 色を変更
  move(dir, dist)        — 移動
  copy_to(pos)           — コピー
  crop()                 — 切り出し
  summarize(fn)          — 集約

【条件子 (Conditions)】
  where(pred)            — 条件フィルタ
  having(prop, val)      — 属性フィルタ
  if_then(cond, action)  — 条件分岐
"""

from typing import List, Tuple, Optional, Dict, Any, Callable, Set
from dataclasses import dataclass, field
from collections import Counter, deque
from arc.grid import Grid, grid_shape, most_common_color, grid_eq
import itertools


# ============================================================
# Core Types
# ============================================================

@dataclass
class CellSet:
    """セル座標の集合 + メタデータ"""
    cells: Set[Tuple[int, int]]
    color: Optional[int] = None
    label: str = ""


@dataclass  
class Object:
    """連結成分オブジェクト"""
    cells: Set[Tuple[int, int]]
    color: int
    bbox: Tuple[int, int, int, int]  # r1, c1, r2, c2
    
    @property
    def size(self):
        return len(self.cells)
    
    @property
    def center(self):
        r1, c1, r2, c2 = self.bbox
        return ((r1 + r2) / 2, (c1 + c2) / 2)
    
    @property
    def width(self):
        return self.bbox[3] - self.bbox[1] + 1
    
    @property
    def height(self):
        return self.bbox[2] - self.bbox[0] + 1


@dataclass
class Region:
    """セパレータで区切られた矩形領域"""
    r1: int
    c1: int
    r2: int
    c2: int
    cells: List[Tuple[int, int]] = field(default_factory=list)


# ============================================================
# Selectors — グリッドから要素を選択
# ============================================================

def select_objects(grid: Grid, bg: int, connectivity: int = 4) -> List[Object]:
    """連結成分を抽出"""
    h, w = grid_shape(grid)
    visited = set()
    objects = []
    
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg or (r, c) in visited:
                continue
            color = grid[r][c]
            component = set()
            queue = deque([(r, c)])
            visited.add((r, c))
            while queue:
                cr, cc = queue.popleft()
                component.add((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < h and 0 <= nc < w and 
                        (nr, nc) not in visited and grid[nr][nc] == color):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            
            rs = [r for r, c in component]
            cs = [c for r, c in component]
            bbox = (min(rs), min(cs), max(rs), max(cs))
            objects.append(Object(cells=component, color=color, bbox=bbox))
    
    return objects


def select_objects_multicolor(grid: Grid, bg: int, connectivity: int = 4) -> List[Object]:
    """多色オブジェクト: bg以外の隣接セルをまとめる"""
    h, w = grid_shape(grid)
    visited = set()
    objects = []
    
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg or (r, c) in visited:
                continue
            component = set()
            queue = deque([(r, c)])
            visited.add((r, c))
            while queue:
                cr, cc = queue.popleft()
                component.add((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < h and 0 <= nc < w and 
                        (nr, nc) not in visited and grid[nr][nc] != bg):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            
            rs = [r for r, c in component]
            cs = [c for r, c in component]
            bbox = (min(rs), min(cs), max(rs), max(cs))
            # Use most common color
            colors = Counter(grid[r][c] for r, c in component)
            mc = colors.most_common(1)[0][0]
            objects.append(Object(cells=component, color=mc, bbox=bbox))
    
    return objects


def select_cells(grid: Grid, color: int) -> CellSet:
    """特定色のセルを全て選択"""
    h, w = grid_shape(grid)
    cells = {(r, c) for r in range(h) for c in range(w) if grid[r][c] == color}
    return CellSet(cells=cells, color=color)


def select_regions(grid: Grid, bg: int) -> Tuple[Optional[int], List[int], List[int], List[Region]]:
    """セパレータ色で区切られた領域を返す"""
    h, w = grid_shape(grid)
    
    for color in range(10):
        if color == bg:
            continue
        full_rows = [r for r in range(h) if all(grid[r][c] == color for c in range(w))]
        full_cols = [c for c in range(w) if all(grid[r][c] == color for r in range(h))]
        if full_rows or full_cols:
            row_bounds = sorted(set([-1] + full_rows + [h]))
            col_bounds = sorted(set([-1] + full_cols + [w]))
            regions = []
            for i in range(len(row_bounds) - 1):
                r1, r2 = row_bounds[i] + 1, row_bounds[i + 1] - 1
                if r1 > r2:
                    continue
                for j in range(len(col_bounds) - 1):
                    c1, c2 = col_bounds[j] + 1, col_bounds[j + 1] - 1
                    if c1 > c2:
                        continue
                    cells = [(r, c) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)]
                    regions.append(Region(r1=r1, c1=c1, r2=r2, c2=c2, cells=cells))
            return color, full_rows, full_cols, regions
    
    return None, [], [], []


# ============================================================
# Operations — グリッドを変換
# ============================================================

def op_fill(grid: Grid, cells: Set[Tuple[int, int]], color: int) -> Grid:
    """セル群を指定色で塗りつぶす"""
    result = [row[:] for row in grid]
    for r, c in cells:
        if 0 <= r < len(result) and 0 <= c < len(result[0]):
            result[r][c] = color
    return result


def op_draw_line(grid: Grid, start: Tuple[int, int], direction: Tuple[int, int], 
                 color: int, length: int = -1) -> Grid:
    """始点から方向に線を引く (length=-1で端まで)"""
    result = [row[:] for row in grid]
    h, w = grid_shape(grid)
    r, c = start
    dr, dc = direction
    steps = 0
    while 0 <= r < h and 0 <= c < w:
        result[r][c] = color
        r += dr
        c += dc
        steps += 1
        if length > 0 and steps >= length:
            break
    return result


def op_connect(grid: Grid, p1: Tuple[int, int], p2: Tuple[int, int], 
               color: int) -> Grid:
    """2点間を直線で接続 (水平/垂直/対角)"""
    result = [row[:] for row in grid]
    r1, c1 = p1
    r2, c2 = p2
    
    dr = 0 if r2 == r1 else (1 if r2 > r1 else -1)
    dc = 0 if c2 == c1 else (1 if c2 > c1 else -1)
    
    r, c = r1, c1
    while True:
        result[r][c] = color
        if r == r2 and c == c2:
            break
        r += dr
        c += dc
    return result


def op_reflect_4fold(grid: Grid, obj: Object, anchor: Tuple[int, int], 
                     bg: int) -> Grid:
    """オブジェクトをアンカー点を中心に4回対称反射"""
    result = [row[:] for row in grid]
    h, w = grid_shape(grid)
    ar, ac = anchor
    
    for r, c in obj.cells:
        # Original position relative to anchor
        dr, dc = r - ar, c - ac
        # 4 reflections
        for nr, nc in [(ar + dr, ac + dc), (ar + dr, ac - dc),
                       (ar - dr, ac + dc), (ar - dr, ac - dc)]:
            if 0 <= nr < h and 0 <= nc < w:
                result[nr][nc] = grid[r][c]
    return result


def op_extend_line(grid: Grid, cells: List[Tuple[int, int]], 
                   color: int, bg: int) -> Grid:
    """水平ラインを端まで延長"""
    result = [row[:] for row in grid]
    h, w = grid_shape(grid)
    
    if not cells:
        return result
    
    # Determine direction
    rows = {r for r, c in cells}
    cols = {c for r, c in cells}
    
    if len(rows) == 1:
        # Horizontal line
        r = list(rows)[0]
        c_min, c_max = min(cols), max(cols)
        for c in range(c_min, c_max + 1):
            result[r][c] = color
    elif len(cols) == 1:
        # Vertical line
        c = list(cols)[0]
        r_min, r_max = min(rows), max(rows)
        for r in range(r_min, r_max + 1):
            result[r][c] = color
    
    return result


def op_flood_fill(grid: Grid, seeds: Set[Tuple[int, int]], 
                  color: int, bg: int, barrier: Optional[int] = None) -> Grid:
    """シードセルからフラッドフィル"""
    result = [row[:] for row in grid]
    h, w = grid_shape(grid)
    visited = set(seeds)
    queue = deque(seeds)
    
    while queue:
        r, c = queue.popleft()
        result[r][c] = color
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited):
                v = grid[nr][nc]
                if v == bg and (barrier is None or v != barrier):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    
    return result


def op_crop_bbox(grid: Grid, bg: int) -> Grid:
    """非bgセルのバウンディングボックスで切り出し"""
    h, w = grid_shape(grid)
    non_bg = [(r, c) for r in range(h) for c in range(w) if grid[r][c] != bg]
    if not non_bg:
        return grid
    rmin = min(r for r, c in non_bg)
    rmax = max(r for r, c in non_bg)
    cmin = min(c for r, c in non_bg)
    cmax = max(c for r, c in non_bg)
    return [[grid[r][c] for c in range(cmin, cmax + 1)] for r in range(rmin, rmax + 1)]


def op_extract_region(grid: Grid, region: Region) -> Grid:
    """領域を切り出す"""
    return [[grid[r][c] for c in range(region.c1, region.c2 + 1)] 
            for r in range(region.r1, region.r2 + 1)]


# ============================================================
# Compound Operations — 複合操作 (よくあるパターン)
# ============================================================

def pattern_cross_through_points(grid: Grid, bg: int) -> Optional[Grid]:
    """2点を通る十字線を引き、囲まれた領域を塗りつぶす"""
    objects = select_objects(grid, bg)
    if len(objects) != 2:
        return None
    
    # Each object should be a single cell
    points = []
    for obj in objects:
        if obj.size != 1:
            return None
        points.append(list(obj.cells)[0])
    
    color = objects[0].color
    h, w = grid_shape(grid)
    result = [row[:] for row in grid]
    
    # Draw vertical and horizontal lines through each point
    for r, c in points:
        for ri in range(h):
            result[ri][c] = color
        for ci in range(w):
            result[r][ci] = color
    
    return result


def pattern_reflect_around_anchor(grid: Grid, bg: int) -> Optional[Grid]:
    """オブジェクトをアンカー(異色)を中心に4回反射"""
    objects = select_objects(grid, bg)
    if len(objects) < 2:
        return None
    
    # Find anchor (different color from main object)
    colors = Counter(o.color for o in objects)
    if len(colors) < 2:
        return None
    
    # Anchor = smallest/unique color object
    anchor_color = min(colors, key=lambda c: colors[c])
    anchor_objs = [o for o in objects if o.color == anchor_color]
    main_objs = [o for o in objects if o.color != anchor_color]
    
    if len(anchor_objs) != 1:
        return None
    
    anchor = anchor_objs[0]
    anchor_center = (int(anchor.center[0]), int(anchor.center[1]))
    
    result = [row[:] for row in grid]
    for obj in main_objs:
        result = op_reflect_4fold(result, obj, anchor_center, bg)
    
    return result


def pattern_fill_between_objects(grid: Grid, bg: int) -> Optional[Grid]:
    """オブジェクト間の空間を塗りつぶす"""
    h, w = grid_shape(grid)
    result = [row[:] for row in grid]
    
    # Find non-bg cells in each row, fill between them
    for r in range(h):
        non_bg_cols = [c for c in range(w) if grid[r][c] != bg]
        if len(non_bg_cols) >= 2:
            c_min, c_max = min(non_bg_cols), max(non_bg_cols)
            for c in range(c_min, c_max + 1):
                if result[r][c] == bg:
                    result[r][c] = grid[r][non_bg_cols[0]]  # Use nearest color
    
    # Same for columns
    for c in range(w):
        non_bg_rows = [r for r in range(h) if grid[r][c] != bg]
        if len(non_bg_rows) >= 2:
            r_min, r_max = min(non_bg_rows), max(non_bg_rows)
            for r in range(r_min, r_max + 1):
                if result[r][c] == bg:
                    result[r][c] = grid[non_bg_rows[0]][c]
    
    return result


def pattern_count_per_region(grid: Grid, bg: int, out_shape: Tuple[int, int]) -> Optional[Grid]:
    """各領域の非bgセル数をカウントし、小さいグリッドに要約"""
    h, w = grid_shape(grid)
    oh, ow = out_shape
    
    if h % oh != 0 or w % ow != 0:
        return None
    
    rh, rw = h // oh, w // ow
    result = [[bg] * ow for _ in range(oh)]
    
    for ri in range(oh):
        for ci in range(ow):
            count = 0
            for r in range(ri * rh, (ri + 1) * rh):
                for c in range(ci * rw, (ci + 1) * rw):
                    if grid[r][c] != bg:
                        count += 1
            result[ri][ci] = count
    
    return result


def pattern_draw_cross_and_fill(grid: Grid, bg: int) -> Optional[Grid]:
    """2点を通る十字線 + 囲まれた矩形をfill"""
    objects = select_objects(grid, bg)
    if len(objects) != 2:
        return None
    
    points = []
    for obj in objects:
        if obj.size != 1:
            return None
        points.append(list(obj.cells)[0])
    
    color = objects[0].color
    h, w = grid_shape(grid)
    result = [row[:] for row in grid]
    
    r1, c1 = points[0]
    r2, c2 = points[1]
    
    # Draw full cross lines through each point
    for r, c in points:
        for ri in range(h):
            result[ri][c] = color
        for ci in range(w):
            result[r][ci] = color
    
    # Fill the enclosed rectangle with a new color
    min_r, max_r = min(r1, r2), max(r1, r2)
    min_c, max_c = min(c1, c2), max(c1, c2)
    
    new_color = None
    for nc in range(1, 10):
        if nc != bg and nc != color:
            new_color = nc
            break
    
    if new_color:
        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                if result[r][c] == bg:
                    result[r][c] = new_color
    
    return result


# ============================================================
# Program Synthesis — 語彙の組み合わせでプログラムを合成
# ============================================================

@dataclass
class PuzzleProgram:
    """パズル推論プログラム"""
    name: str
    apply_fn: Callable[[Grid], Optional[Grid]]
    description: str = ""


def synthesize_programs(train_pairs: List[Tuple[Grid, Grid]]) -> List[PuzzleProgram]:
    """訓練ペアからパズルプログラムを合成する"""
    programs = []
    
    if not train_pairs:
        return programs
    
    inp0, out0 = train_pairs[0]
    ih, iw = grid_shape(inp0)
    oh, ow = grid_shape(out0)
    bg = most_common_color(inp0)
    
    # === Pattern 1: Cross through points + fill ===
    programs.append(PuzzleProgram(
        name="cross_and_fill",
        apply_fn=lambda g: pattern_draw_cross_and_fill(g, most_common_color(g)),
        description="FIND 2 points → DRAW cross THROUGH each → FILL enclosed WITH new_color"
    ))
    
    # === Pattern 2: Reflect around anchor ===
    programs.append(PuzzleProgram(
        name="reflect_around_anchor",
        apply_fn=lambda g: pattern_reflect_around_anchor(g, most_common_color(g)),
        description="FIND object+anchor → REFLECT object AROUND anchor 4-fold"
    ))
    
    # === Pattern 3: Fill between objects (h+v) ===
    programs.append(PuzzleProgram(
        name="fill_between",
        apply_fn=lambda g: pattern_fill_between_objects(g, most_common_color(g)),
        description="EACH row/col → FILL BETWEEN non-bg cells"
    ))
    
    # === Pattern 4: Count per region → summary grid ===
    if (oh, ow) != (ih, iw) and oh < ih and ow < iw:
        programs.append(PuzzleProgram(
            name="count_per_region",
            apply_fn=lambda g, os=(oh,ow): pattern_count_per_region(g, most_common_color(g), os),
            description=f"DIVIDE into {oh}x{ow} regions → COUNT non-bg in each"
        ))
    
    # === Pattern 5: Separator-based operations ===
    sep_color, full_rows, full_cols, regions = select_regions(inp0, bg)
    if sep_color is not None and len(regions) >= 2:
        # Try: extract region with unique content
        def try_extract_unique(g):
            bg2 = most_common_color(g)
            sc, fr, fc, regs = select_regions(g, bg2)
            if sc is None:
                return None
            for reg in regs:
                non_bg = [(r, c) for r, c in reg.cells if g[r][c] != bg2 and g[r][c] != sc]
                if len(non_bg) == 1:
                    return op_extract_region(g, reg)
            return None
        
        programs.append(PuzzleProgram(
            name="extract_unique_panel",
            apply_fn=try_extract_unique,
            description="FIND separator → EXTRACT panel WHERE unique_cell"
        ))
        
        # Try: fill regions based on content
        # Try: compare regions (XOR/OR/AND)
    
    # === Pattern 6: Line extension (horizontal lines → mark length) ===
    def try_line_length_marker(g):
        bg2 = most_common_color(g)
        h2, w2 = grid_shape(g)
        result = [row[:] for row in g]
        objects = select_objects(g, bg2)
        
        # Find marker (single cell, specific color)
        markers = [o for o in objects if o.size == 1]
        lines = [o for o in objects if o.size > 1]
        
        if not markers or not lines:
            return None
        
        marker = markers[0]
        mr, mc = list(marker.cells)[0]
        
        # Each line: compute length, write to marker column
        for line in lines:
            line_cells = sorted(line.cells)
            rows = {r for r, c in line_cells}
            if len(rows) == 1:  # horizontal line
                r = list(rows)[0]
                length = len(line_cells)
                # Map length to color
                result[r][mc] = length
        
        return result
    
    programs.append(PuzzleProgram(
        name="line_length_marker",
        apply_fn=try_line_length_marker,
        description="EACH horizontal_line → WRITE LENGTH AT marker_column"
    ))
    
    # === Pattern 7: Global color map ===
    cmap = {}
    ok = True
    for inp, out in train_pairs:
        if grid_shape(inp) != grid_shape(out):
            ok = False
            break
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                iv, ov = inp[r][c], out[r][c]
                if iv in cmap:
                    if cmap[iv] != ov:
                        ok = False
                        break
                else:
                    cmap[iv] = ov
            if not ok:
                break
        if not ok:
            break
    
    if ok and any(k != v for k, v in cmap.items()):
        frozen_cmap = dict(cmap)
        programs.append(PuzzleProgram(
            name="global_color_map",
            apply_fn=lambda g, cm=frozen_cmap: [
                [cm.get(g[r][c], g[r][c]) for c in range(len(g[0]))] 
                for r in range(len(g))
            ],
            description=f"EACH cell → RECOLOR by map {cmap}"
        ))
    
    # === Pattern 8: Split + combine (XOR/OR/AND) ===
    for split in ['h_equal', 'v_equal']:
        if split == 'h_equal' and ih % 2 != 0:
            continue
        if split == 'v_equal' and iw % 2 != 0:
            continue
        
        half = ih // 2 if split == 'h_equal' else iw // 2
        exp_out = (half, iw) if split == 'h_equal' else (ih, half)
        
        if (oh, ow) != exp_out:
            continue
        
        for op_name in ['xor', 'or', 'and']:
            out_color = None
            valid = True
            
            for inp, out in train_pairs:
                if grid_shape(out) != exp_out:
                    valid = False
                    break
                
                bg2 = most_common_color(inp)
                h2, w2 = grid_shape(inp)
                
                if split == 'h_equal':
                    A = [[inp[r][c] for c in range(w2)] for r in range(half)]
                    B = [[inp[r][c] for c in range(w2)] for r in range(half, h2)]
                else:
                    A = [[inp[r][c] for c in range(half)] for r in range(h2)]
                    B = [[inp[r][c] for c in range(half, w2)] for r in range(h2)]
                
                ah, aw = grid_shape(A)
                for r in range(ah):
                    for c in range(aw):
                        a_nb = A[r][c] != bg2
                        b_nb = B[r][c] != bg2
                        if op_name == 'xor':
                            mark = a_nb != b_nb
                        elif op_name == 'or':
                            mark = a_nb or b_nb
                        else:
                            mark = a_nb and b_nb
                        
                        if mark:
                            if out_color is None:
                                out_color = out[r][c]
                            elif out_color != out[r][c]:
                                valid = False
                                break
                        else:
                            if out[r][c] != bg2:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid and out_color is not None:
                frozen = {'split': split, 'op': op_name, 'color': out_color}
                
                def make_split_fn(params):
                    def fn(g):
                        bg3 = most_common_color(g)
                        h3, w3 = grid_shape(g)
                        sp = params['split']
                        half3 = h3 // 2 if sp == 'h_equal' else w3 // 2
                        
                        if sp == 'h_equal':
                            A = [[g[r][c] for c in range(w3)] for r in range(half3)]
                            B = [[g[r][c] for c in range(w3)] for r in range(half3, h3)]
                            res = [[bg3] * w3 for _ in range(half3)]
                        else:
                            A = [[g[r][c] for c in range(half3)] for r in range(h3)]
                            B = [[g[r][c] for c in range(half3, w3)] for r in range(h3)]
                            res = [[bg3] * half3 for _ in range(h3)]
                        
                        ah, aw = grid_shape(A)
                        for r in range(ah):
                            for c in range(aw):
                                a_nb = A[r][c] != bg3
                                b_nb = B[r][c] != bg3
                                if params['op'] == 'xor':
                                    mark = a_nb != b_nb
                                elif params['op'] == 'or':
                                    mark = a_nb or b_nb
                                else:
                                    mark = a_nb and b_nb
                                if mark:
                                    res[r][c] = params['color']
                        return res
                    return fn
                
                programs.append(PuzzleProgram(
                    name=f"split_{split}_{op_name}",
                    apply_fn=make_split_fn(frozen),
                    description=f"SPLIT {split} → {op_name.upper()} → FILL WITH {out_color}"
                ))
    
    return programs


def solve_with_puzzle_lang(train_pairs: List[Tuple[Grid, Grid]], 
                           test_input: Grid) -> Optional[Grid]:
    """パズル推論言語でタスクを解く"""
    programs = synthesize_programs(train_pairs)
    
    for prog in programs:
        try:
            # Verify on all training pairs
            valid = True
            for inp, out in train_pairs:
                result = prog.apply_fn(inp)
                if result is None or not grid_eq(result, out):
                    valid = False
                    break
            
            if valid:
                result = prog.apply_fn(test_input)
                if result is not None:
                    return result
        except Exception:
            continue
    
    return None
