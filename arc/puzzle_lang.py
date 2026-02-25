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


def _nonzero_count_row(grid: Grid) -> Optional[Grid]:
    """Output a 1-row grid: [fg_color] × count_of_fg_cells.
    Always treats 0 as background (non-zero = fg)."""
    h, w = grid_shape(grid)
    nz_vals = [grid[r][c] for r in range(h) for c in range(w) if grid[r][c] != 0]
    if not nz_vals:
        return None
    nz_colors = set(nz_vals)
    if len(nz_colors) != 1:
        return None
    fc = list(nz_colors)[0]
    return [[fc] * len(nz_vals)]


def _u_drop_ball(grid: Grid) -> Optional[Grid]:
    """Find U-shaped frames (open at bottom) and drop color-4 from opening to bottom."""
    bg = most_common_color(grid)
    h, w = grid_shape(grid)
    result = [row[:] for row in grid]
    openings = []
    
    for r in range(h - 1):
        row = grid[r]
        c = 0
        while c < w:
            if row[c] == bg:
                c += 1
                continue
            start = c
            color = row[c]
            while c < w and row[c] == color:
                c += 1
            end = c - 1
            if end - start < 2:
                continue
            if r + 1 >= h:
                continue
            next_row = grid[r + 1]
            # Check side walls (same color at start and end of next row)
            if next_row[start] == color and next_row[end] == color:
                # Find empty (bg) positions in middle of next row
                empty_mid = [cc for cc in range(start + 1, end) if next_row[cc] == bg]
                if empty_mid:
                    center_col = (start + end) // 2
                    openings.append((r + 1, center_col))
    
    if not openings:
        return None
    
    for (open_r, open_c) in openings:
        # Drop from open_r to the last bg cell in that column
        for r in range(h - 1, open_r - 1, -1):
            if result[r][open_c] == bg:
                result[r][open_c] = 4
                break
    
    return result


def _connect_same_color(grid: Grid) -> Optional[Grid]:
    """EACH color → draw straight lines connecting all cells of that color"""
    bg = most_common_color(grid)
    h, w = grid_shape(grid)
    result = [row[:] for row in grid]
    colors = set(grid[r][c] for r in range(h) for c in range(w)) - {bg}
    for color in colors:
        pts = sorted([(r, c) for r in range(h) for c in range(w) if grid[r][c] == color])
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            r1, c1 = pts[i]
            r2, c2 = pts[i + 1]
            dr = 0 if r2 == r1 else (1 if r2 > r1 else -1)
            dc = 0 if c2 == c1 else (1 if c2 > c1 else -1)
            r, c = r1, c1
            steps = 0
            while steps < h * w:
                if 0 <= r < h and 0 <= c < w:
                    result[r][c] = color
                if r == r2 and c == c2:
                    break
                r += dr
                c += dc
                steps += 1
    return result


def _fill_intersection_gap(grid: Grid) -> Optional[Grid]:
    """WHERE full-row meets partial-col (or full-col meets partial-row):
    the intersection cell gets the color of the partial line."""
    bg = most_common_color(grid)
    h, w = grid_shape(grid)
    result = [row[:] for row in grid]
    
    # Find full rows (uniform non-bg color)
    full_rows = {}
    for r in range(h):
        vals = set(grid[r][c] for c in range(w))
        if len(vals) == 1 and grid[r][0] != bg:
            full_rows[r] = grid[r][0]
    
    # For each column, find dominant color; fill cells where full_row color "interrupted" the col
    for c in range(w):
        col_colors = Counter(
            grid[r][c] for r in range(h)
            if grid[r][c] != bg and grid[r][c] not in full_rows.values()
        )
        if not col_colors:
            continue
        dom = col_colors.most_common(1)[0][0]
        for r in full_rows:
            if grid[r][c] == full_rows[r] and full_rows[r] != dom:
                if sum(1 for rr in range(h) if rr != r and grid[rr][c] == dom) >= 1:
                    result[r][c] = dom
    
    # Find full cols
    full_cols = {}
    for c in range(w):
        vals = set(grid[r][c] for r in range(h))
        if len(vals) == 1 and grid[0][c] != bg:
            full_cols[c] = grid[0][c]
    
    # For each row, fill intersections where full_col interrupted the row
    for r in range(h):
        row_colors = Counter(
            grid[r][c] for c in range(w)
            if grid[r][c] != bg and grid[r][c] not in full_cols.values()
        )
        if not row_colors:
            continue
        dom = row_colors.most_common(1)[0][0]
        for c in full_cols:
            if grid[r][c] == full_cols[c] and full_cols[c] != dom:
                if sum(1 for cc in range(w) if cc != c and grid[r][cc] == dom) >= 1:
                    result[r][c] = dom
    
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
    
    # === Pattern 9_rot: Whole-grid transformations (fallback for missing transforms) ===
    # Rotate 180° — whole grid AND content-only reflected to opposite corner
    programs.append(PuzzleProgram(
        name="rotate_180",
        apply_fn=lambda g: [row[::-1] for row in reversed(g)],
        description="ROTATE grid 180°"
    ))
    # Rotate content 180° around grid center (content-bbox rot + reflection)
    def rot180_content_around_center(g):
        h, w = grid_shape(g)
        bg = most_common_color(g)
        # Find bounding box of non-bg content
        rows_nz = [r for r in range(h) if any(g[r][c] != bg for c in range(w))]
        cols_nz = [c for c in range(w) if any(g[r][c] != bg for r in range(h))]
        if not rows_nz or not cols_nz:
            return None
        r1, r2 = min(rows_nz), max(rows_nz)
        c1, c2 = min(cols_nz), max(cols_nz)
        # Extract and rotate bbox 180°
        bbox = [[g[r][c] for c in range(c1, c2+1)] for r in range(r1, r2+1)]
        rotated = [row[::-1] for row in reversed(bbox)]
        bh, bw = len(bbox), len(bbox[0])
        # Reflect bbox position around grid center
        # Grid center: ((h-1)/2, (w-1)/2)
        # Content center: ((r1+r2)/2, (c1+c2)/2)
        # Reflected center: (h-1 - (r1+r2)/2, w-1 - (c1+c2)/2)
        new_r1 = (h - 1) - r2  # = h-1-r2
        new_c1 = (w - 1) - c2  # = w-1-c2
        # Place rotated content at new position
        result = [[bg] * w for _ in range(h)]
        for dr in range(bh):
            for dc in range(bw):
                nr = new_r1 + dr
                nc = new_c1 + dc
                if 0 <= nr < h and 0 <= nc < w:
                    result[nr][nc] = rotated[dr][dc]
        return result
    programs.append(PuzzleProgram(
        name="rotate_180_content",
        apply_fn=rot180_content_around_center,
        description="ROTATE content bbox 180° and reflect to opposite grid corner"
    ))
    # Complement tile 2x2 (for 2-color grids)
    if len(set(inp0[r][c] for r in range(ih) for c in range(iw))) == 2:
        colors_2 = set(inp0[r][c] for r in range(ih) for c in range(iw))
        c1_tile, c2_tile = list(colors_2)
        def make_comp_tile_2x2(ca, cb):
            def fn(g):
                h, w = grid_shape(g)
                cs = set(g[r][c] for r in range(h) for c in range(w))
                if len(cs) != 2: return None
                ca2, cb2 = sorted(cs)
                # comp: swap the two colors
                comp = [[cb2 if g[r][c] == ca2 else ca2 for c in range(w)] for r in range(h)]
                # Tile 2x2
                result = []
                for r in range(h * 2):
                    row = []
                    for c in range(w * 2):
                        row.append(comp[r % h][c % w])
                    result.append(row)
                return result
            return fn
        programs.append(PuzzleProgram(
            name="comp_tile_2x2",
            apply_fn=make_comp_tile_2x2(c1_tile, c2_tile),
            description="COMPLEMENT the 2-color grid then TILE 2x2"
        ))
    # Staircase grow: 1-row input → triangle pattern
    # The fg is the color that STARTS from left; bg fills the right tail
    if ih == 1:
        def staircase_grow(g):
            h, w = grid_shape(g)
            if h != 1: return None
            # The "fg" is the left-aligned color, "bg" is the right-side filler
            fg_color = g[0][0]  # leftmost color
            bg_color = g[0][-1]  # rightmost color (the filler)
            if fg_color == bg_color: return None
            # Count how many fg cells are in input (contiguous from left)
            n_fg = 0
            for c in range(w):
                if g[0][c] == fg_color: n_fg += 1
                else: break
            n_rows = w // 2
            if n_rows < 1: return None
            result = []
            for row_idx in range(n_rows):
                n_this_row = min(n_fg + row_idx, w)
                row = [fg_color] * n_this_row + [bg_color] * (w - n_this_row)
                result.append(row)
            return result
        programs.append(PuzzleProgram(
            name="staircase_grow",
            apply_fn=staircase_grow,
            description="1-row input → grow fg count by 1 each row (staircase)"
        ))
    
    # === Pattern 9_interleave: 2-row interleave (checkerboard by row) ===
    if ih == 2 and oh == 2 and iw == ow:
        def two_row_interleave(g):
            h, w = grid_shape(g)
            if h != 2: return None
            # Each row must be uniform
            c0 = set(g[0])
            c1 = set(g[1])
            if len(c0) != 1 or len(c1) != 1: return None
            col0 = g[0][0]
            col1 = g[1][0]
            result = []
            for r in range(h):
                row = []
                for c in range(w):
                    # c even → own row's color; c odd → other row's color
                    if c % 2 == 0:
                        row.append(col0 if r == 0 else col1)
                    else:
                        row.append(col1 if r == 0 else col0)
                result.append(row)
            return result
        # Verify on all pairs
        ok_il = all(grid_eq(two_row_interleave(inp), out) for inp, out in train_pairs)
        if ok_il:
            programs.append(PuzzleProgram(
                name="two_row_interleave",
                apply_fn=two_row_interleave,
                description="INTERLEAVE two uniform rows in checkerboard pattern"
            ))
    
    # === Pattern 9_nor: Stack-NOR — two halves stacked, mark where both are bg ===
    if ih % 2 == 0 and oh == ih // 2 and ow == iw:
        # Try: top half (A) and bottom half (B) stacked
        # Output[r][c] = mark_color where A[r][c]==bg AND B[r][c]==bg
        half_h = ih // 2
        # Detect mark color from first pair
        bg0 = most_common_color(inp0)
        top0 = inp0[:half_h]
        bot0 = inp0[half_h:]
        # Find mark color = the non-bg color in out0 where both top0/bot0 are bg
        mark_color_nor = None
        for r in range(half_h):
            for c in range(iw):
                if top0[r][c] == bg0 and bot0[r][c] == bg0 and out0[r][c] != bg0:
                    mark_color_nor = out0[r][c]
                    break
            if mark_color_nor: break
        if mark_color_nor is not None:
            # bg is detected from out0 (simpler grid, clearer background)
            fixed_bg_nor = most_common_color(out0)
            def make_nor(mark, fixed_bg):
                def fn(g):
                    h, w = grid_shape(g)
                    if h % 2 != 0: return None
                    hh = h // 2
                    top = g[:hh]
                    bot = g[hh:]
                    result = []
                    for r in range(hh):
                        row = []
                        for c in range(w):
                            if top[r][c] == fixed_bg and bot[r][c] == fixed_bg:
                                row.append(mark)
                            else:
                                row.append(fixed_bg)
                        result.append(row)
                    return result
                return fn
            nor_fn = make_nor(mark_color_nor, fixed_bg_nor)
            ok_nor = all(grid_eq(nor_fn(inp), out) for inp, out in train_pairs)
            if ok_nor:
                programs.append(PuzzleProgram(
                    name="stack_nor",
                    apply_fn=nor_fn,
                    description=f"STACK-NOR: mark {mark_color_nor} where both halves are bg"
                ))
    
    # === Pattern 9_frame: Add repeat-border around input (corners=bg) ===
    if oh == ih + 2 and ow == iw + 2:
        # Try with corner_color = 0 (global bg) and with detected bg
        for corner_c in [0, most_common_color(inp0)]:
            def make_frame(cc):
                def frame_fn(g):
                    h, w = grid_shape(g)
                    result = []
                    result.append([cc] + list(g[0]) + [cc])
                    for r in range(h):
                        result.append([g[r][0]] + list(g[r]) + [g[r][w-1]])
                    result.append([cc] + list(g[-1]) + [cc])
                    return result
                return frame_fn
            frame_fn = make_frame(corner_c)
            ok_fr = all(grid_eq(frame_fn(inp), out) for inp, out in train_pairs)
            if ok_fr:
                programs.append(PuzzleProgram(
                    name=f"frame_repeat_border_c{corner_c}",
                    apply_fn=frame_fn,
                    description=f"FRAME input with repeat-border (corners={corner_c})"
                ))
                break
    
    # === Pattern 9_uniform_row: Each row → uniform mark if all same, else bg ===
    if (oh, ow) == (ih, iw):
        out_colors_ur = set(out0[r][c] for r in range(oh) for c in range(ow))
        if len(out_colors_ur) == 2:
            for mark_ur, bg_ur in [
                (min(out_colors_ur), max(out_colors_ur)),
                (max(out_colors_ur), min(out_colors_ur)),
            ]:
                def make_ur(mk, bk):
                    def fn(g):
                        h, w = grid_shape(g)
                        result = []
                        for r in range(h):
                            result.append([mk]*w if len(set(g[r]))==1 else [bk]*w)
                        return result
                    return fn
                ur_fn = make_ur(mark_ur, bg_ur)
                if all(grid_eq(ur_fn(inp), out) for inp, out in train_pairs):
                    programs.append(PuzzleProgram(
                        name=f"uniform_row_detect_{mark_ur}",
                        apply_fn=ur_fn,
                        description=f"MARK uniform rows with {mark_ur}, non-uniform with {bg_ur}"
                    ))
                    break
        # Also try: uniform COLUMNS
        out_colors_uc = set(out0[r][c] for r in range(oh) for c in range(ow))
        if len(out_colors_uc) == 2:
            for mark_uc, bg_uc in [
                (min(out_colors_uc), max(out_colors_uc)),
                (max(out_colors_uc), min(out_colors_uc)),
            ]:
                def make_uc(mk, bk):
                    def fn(g):
                        h, w = grid_shape(g)
                        result = []
                        for r in range(h):
                            row = []
                            for c in range(w):
                                col_vals = [g[rr][c] for rr in range(h)]
                                row.append(mk if len(set(col_vals))==1 else bk)
                            result.append(row)
                        return result
                    return fn
                uc_fn = make_uc(mark_uc, bg_uc)
                if all(grid_eq(uc_fn(inp), out) for inp, out in train_pairs):
                    programs.append(PuzzleProgram(
                        name=f"uniform_col_detect_{mark_uc}",
                        apply_fn=uc_fn,
                        description=f"MARK uniform cols with {mark_uc}, else {bg_uc}"
                    ))
                    break

    # === Pattern 9_grid_pattern: All-bg input → grid/checkerboard/lattice ===
    if all(inp0[r][c] == inp0[0][0] for r in range(ih) for c in range(iw)):
        out_vals = set(out0[r][c] for r in range(oh) for c in range(ow))
        if len(out_vals) == 2:
            ov = list(out_vals)
            # Try multiple parity patterns
            for mark_gp, bg_gp in [(ov[0], ov[1]), (ov[1], ov[0])]:
                for pat_name, pat_fn in [
                    ('checkerboard', lambda r,c: (r+c)%2==0),
                    ('grid_lattice', lambda r,c: not (r%2==1 and c%2==1)),
                    ('grid_inv', lambda r,c: (r%2==1 and c%2==1)),
                    ('even_rows', lambda r,c: r%2==0),
                    ('odd_rows', lambda r,c: r%2==1),
                    ('even_cols', lambda r,c: c%2==0),
                    ('odd_cols', lambda r,c: c%2==1),
                ]:
                    is_match = True
                    for r in range(oh):
                        for c in range(ow):
                            expected = mark_gp if pat_fn(r,c) else bg_gp
                            if out0[r][c] != expected:
                                is_match = False; break
                        if not is_match: break
                    if is_match:
                        def make_grid_pat(mk, bk, pf):
                            def fn(g):
                                h, w = grid_shape(g)
                                return [[mk if pf(r,c) else bk for c in range(w)] for r in range(h)]
                            return fn
                        gp_fn = make_grid_pat(mark_gp, bg_gp, pat_fn)
                        if all(grid_eq(gp_fn(inp), out) for inp, out in train_pairs):
                            programs.append(PuzzleProgram(
                                name=f"grid_pattern_{pat_name}_{mark_gp}",
                                apply_fn=gp_fn,
                                description=f"FILL {pat_name}: {mark_gp}/{bg_gp}"
                            ))
                        break
                if any(p.name.startswith('grid_pattern') for p in programs):
                    break

    # === Pattern 9_antidiag: Left column + anti-diagonal line + bottom row fill ===
    if (oh,ow) == (ih,iw) and ih == iw:
        bg_ad = most_common_color(inp0)
        # Check: left column is non-bg, rest is bg
        left_col = set(inp0[r][0] for r in range(ih))
        rest_bg = all(inp0[r][c] == bg_ad for r in range(ih) for c in range(1, iw))
        if len(left_col) == 1 and left_col.pop() != bg_ad and rest_bg:
            # Check output: anti-diagonal (color 2) + bottom row (color 4)
            left_c = inp0[0][0]
            # Detect anti-diag color and bottom row color
            anti_c = out0[0][iw-1] if out0[0][iw-1] != bg_ad and out0[0][iw-1] != left_c else None
            bot_c = out0[ih-1][1] if ih>1 and out0[ih-1][1] != bg_ad and out0[ih-1][1] != left_c else None
            if anti_c and bot_c:
                def make_antidiag(lc, ac, bc, bg_v):
                    def fn(g):
                        h, w = grid_shape(g)
                        if h != w: return None
                        bg2 = most_common_color(g)
                        lcolor = g[0][0]
                        result = [[bg2]*w for _ in range(h)]
                        # Left column
                        for r in range(h):
                            result[r][0] = lcolor
                        # Anti-diagonal (from top-right to bottom-left, excluding col 0)
                        for r in range(h-1):
                            c = w - 1 - r
                            if c > 0:
                                result[r][c] = ac
                        # Bottom row (non-col-0)
                        for c in range(1, w):
                            result[h-1][c] = bc
                        return result
                    return fn
                ad_fn = make_antidiag(left_c, anti_c, bot_c, bg_ad)
                if all(grid_eq(ad_fn(inp), out) for inp, out in train_pairs):
                    programs.append(PuzzleProgram(
                        name="antidiag_fill",
                        apply_fn=ad_fn,
                        description=f"DRAW anti-diagonal ({anti_c}) + fill bottom row ({bot_c})"
                    ))

    # === Pattern 9_latin_square: Complete a Latin square (each row/col has all colors) ===
    if (oh,ow) == (ih,iw) and ih == iw:
        # Check if output is a Latin square (each row/col contains 1..N exactly once)
        N = ih
        is_latin = True
        out_vals_set = set()
        for r in range(N):
            row_vals = set(out0[r])
            out_vals_set |= row_vals
            if len(row_vals) != N:
                is_latin = False; break
        if is_latin:
            for c in range(N):
                col_vals = set(out0[r][c] for r in range(N))
                if len(col_vals) != N:
                    is_latin = False; break
        if is_latin and len(out_vals_set) == N:
            # It IS a Latin square! Try to solve by constraint propagation
            def solve_latin(g):
                h, w = grid_shape(g)
                if h != w: return None
                N2 = h
                bg2 = most_common_color(g)
                # Find the set of values used
                vals = set(g[r][c] for r in range(N2) for c in range(N2) if g[r][c] != bg2)
                if len(vals) < N2 - 1: return None  # not enough clues
                all_vals = vals | {bg2}  # bg might be a real value
                # Actually need to figure out the full value set from output
                # Use the values that appear in out0
                target_vals = out_vals_set
                
                result = [row[:] for row in g]
                changed = True
                while changed:
                    changed = False
                    for r in range(N2):
                        for c in range(N2):
                            if result[r][c] == bg2:
                                row_vals = set(result[r][cc] for cc in range(N2) if result[r][cc] != bg2)
                                col_vals = set(result[rr][c] for rr in range(N2) if result[rr][c] != bg2)
                                possible = target_vals - row_vals - col_vals
                                if len(possible) == 1:
                                    result[r][c] = possible.pop()
                                    changed = True
                                elif len(possible) == 0:
                                    return None
                # Check if complete
                if any(result[r][c] == bg2 for r in range(N2) for c in range(N2)):
                    return None
                return result
            if all(grid_eq(solve_latin(inp), out) for inp, out in train_pairs):
                programs.append(PuzzleProgram(
                    name="latin_square",
                    apply_fn=solve_latin,
                    description="COMPLETE Latin square (each row/col has all values)"
                ))

    # === Pattern 9_col_color_map: Each row has 1 fg cell; map its column to output color ===
    if (oh, ow) == (ih, iw):
        ccmap_cc = {}
        valid_ccmap = True
        for inp_cc, out_cc in train_pairs:
            h2, w2 = grid_shape(inp_cc)
            bg2 = most_common_color(inp_cc)
            for r in range(h2):
                fg_c = [c for c in range(w2) if inp_cc[r][c] != bg2]
                out_vals = set(out_cc[r])
                if len(fg_c) == 1 and len(out_vals) == 1:
                    col = fg_c[0]
                    oc2 = out_cc[r][0]
                    if col in ccmap_cc and ccmap_cc[col] != oc2:
                        valid_ccmap = False; break
                    ccmap_cc[col] = oc2
                elif fg_c:  # more than 1 fg cell or no fg → not this pattern
                    valid_ccmap = False; break
            if not valid_ccmap: break
        if valid_ccmap and ccmap_cc:
            def make_ccmap_fn(cm):
                def fn(g):
                    h, w = grid_shape(g)
                    bg2 = most_common_color(g)
                    result = []
                    for r in range(h):
                        fg_c = [c for c in range(w) if g[r][c] != bg2]
                        if len(fg_c) == 1 and fg_c[0] in cm:
                            result.append([cm[fg_c[0]]] * w)
                        else:
                            return None
                    return result
                return fn
            programs.append(PuzzleProgram(
                name=f"col_color_map",
                apply_fn=make_ccmap_fn(dict(ccmap_cc)),
                description=f"MAP column position → output row color {ccmap_cc}"
            ))

    # === Pattern 9_sep_and: Vertical separator splits L/R; AND fg → mark ===
    if oh == ih:
        bg_out_sep = most_common_color(out0)
        out_colors_sep = set(out0[r][c] for r in range(oh) for c in range(ow)) - {bg_out_sep}
        if out_colors_sep:
            mark_sep = next(iter(out_colors_sep))
            # Find separator column
            for col_sep in range(iw):
                sep_c_set = set(inp0[r][col_sep] for r in range(ih))
                if len(sep_c_set) != 1: continue
                sc = sep_c_set.pop()
                if sc == bg_out_sep: continue  # separator must be different from output bg
                left_w = col_sep
                right_w = iw - col_sep - 1
                if left_w != right_w or left_w != ow: continue
                def make_sep_and(mk, bk):
                    def fn(g):
                        h, w = grid_shape(g)
                        # Find separator (uniform non-bg column in center)
                        sep_col = -1
                        for c2 in range(1, w-1):
                            cs = set(g[r][c2] for r in range(h))
                            if len(cs)==1 and cs.pop()!=bk:
                                lw2 = c2; rw2 = w-c2-1
                                if lw2 == rw2:
                                    sep_col = c2; break
                        if sep_col < 0: return None
                        result = []
                        for r in range(h):
                            row = []
                            for c in range(sep_col):
                                lv = g[r][c]
                                rv = g[r][sep_col+1+c]
                                row.append(mk if lv!=bk and rv!=bk else bk)
                            result.append(row)
                        return result
                    return fn
                sep_fn = make_sep_and(mark_sep, bg_out_sep)
                if all(grid_eq(sep_fn(inp), out) for inp, out in train_pairs):
                    programs.append(PuzzleProgram(
                        name="split_vsep_and",
                        apply_fn=sep_fn,
                        description=f"SPLIT at vertical separator, AND fg → mark {mark_sep}"
                    ))
                    break

    # === Pattern 9_shift: Shift content by 1 step and recolor ===
    if (oh, ow) == (ih, iw):
        # Detect fg and output color from first pair
        bg_sh = most_common_color(inp0)
        bg_sh_out = most_common_color(out0)
        fg_cells_sh = [(r,c,inp0[r][c]) for r in range(ih) for c in range(iw) if inp0[r][c] != bg_sh]
        out_cells_sh = [(r,c,out0[r][c]) for r in range(oh) for c in range(ow) if out0[r][c] != bg_sh_out]
        if fg_cells_sh and out_cells_sh:
            fg_color_sh = Counter(x[2] for x in fg_cells_sh).most_common(1)[0][0]
            out_color_sh = Counter(x[2] for x in out_cells_sh).most_common(1)[0][0]
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                def make_shift(fg, oc, bg, drow, dcol):
                    def fn(g):
                        h, w = grid_shape(g)
                        result = [[bg]*w for _ in range(h)]
                        for r in range(h):
                            for c in range(w):
                                if g[r][c] == fg:
                                    nr, nc = r+drow, c+dcol
                                    if 0<=nr<h and 0<=nc<w:
                                        result[nr][nc] = oc
                        return result
                    return fn
                sh_fn = make_shift(fg_color_sh, out_color_sh, bg_sh_out, dr, dc)
                if all(grid_eq(sh_fn(inp), out) for inp, out in train_pairs):
                    programs.append(PuzzleProgram(
                        name=f"shift_recolor_d{dr}{dc}",
                        apply_fn=sh_fn,
                        description=f"SHIFT {fg_color_sh}→{out_color_sh} by ({dr},{dc})"
                    ))
                    break

    # === Pattern 9_dedup: Extract unique tile from a repeated grid ===
    # Handles: top_half, bottom_half, left_half, right_half on a per-pair basis
    def extract_tile(g):
        """Auto-detect if grid is tiled and extract unique tile."""
        h, w = grid_shape(g)
        # Try left half (tiled 1×2 horizontally)
        if w % 2 == 0:
            half_w = w // 2
            left = [row[:half_w] for row in g]
            right = [row[half_w:] for row in g]
            if grid_eq(left, right):
                return left
        # Try top half (tiled 2×1 vertically)
        if h % 2 == 0:
            half_h = h // 2
            top = g[:half_h]
            bot = g[half_h:]
            if grid_eq(top, bot):
                return top
        # Try top third, quarter, etc.
        for divisor in [3, 4]:
            if h % divisor == 0:
                tile_h = h // divisor
                tile = g[:tile_h]
                if all(grid_eq(tile, g[k*tile_h:(k+1)*tile_h]) for k in range(1, divisor)):
                    return tile
            if w % divisor == 0:
                tile_w = w // divisor
                tile = [row[:tile_w] for row in g]
                tiles = [[row[k*tile_w:(k+1)*tile_w] for row in g] for k in range(1, divisor)]
                if all(grid_eq(tile, t) for t in tiles):
                    return tile
        return None
    
    # Verify on all pairs
    ok_et = True
    for inp_et, out_et in train_pairs:
        pred_et = extract_tile(inp_et)
        if pred_et is None or not grid_eq(pred_et, out_et):
            ok_et = False
            break
    if ok_et:
        programs.append(PuzzleProgram(
            name="extract_tile",
            apply_fn=extract_tile,
            description="DETECT and EXTRACT unique repeated tile"
        ))
    
    # === Pattern 9_sd: Scale-down (NxN blocks → single cell, all-same aggregation) ===
    for N_sd in [2, 3, 4, 5]:
        if ih % N_sd != 0 or iw % N_sd != 0:
            continue
        if (oh, ow) != (ih // N_sd, iw // N_sd):
            continue
        # Try: each NxN block → block[0] if all same, else 0
        def make_sd(n):
            def fn(g):
                h, w = grid_shape(g)
                if h % n != 0 or w % n != 0:
                    return None
                result = []
                for r in range(0, h, n):
                    row = []
                    for c in range(0, w, n):
                        block = [g[r+dr][c+dc] for dr in range(n) for dc in range(n)]
                        row.append(block[0] if len(set(block)) == 1 else 0)
                    result.append(row)
                return result
            return fn
        
        sd_fn = make_sd(N_sd)
        # Quick verify on first pair
        r0 = sd_fn(inp0)
        if r0 is not None and grid_eq(r0, out0):
            programs.append(PuzzleProgram(
                name=f"scale_down_{N_sd}_uniform",
                apply_fn=sd_fn,
                description=f"COMPRESS {N_sd}x{N_sd} blocks → single cell (uniform→color, else 0)"
            ))
    
    # === Pattern 9_vec: Non-zero color × its count as row vector ===
    programs.append(PuzzleProgram(
        name="nonzero_count_row",
        apply_fn=lambda g: _nonzero_count_row(g),
        description="FIND non-zero fg color → OUTPUT [fg_color] × count_fg as 1-row grid"
    ))
    
    # === Pattern 9a: U-shaped frame → drop ball ===
    # "FIND U-shaped frame (open at bottom) → DROP color-4 from opening to bottom"
    programs.append(PuzzleProgram(
        name="u_drop_ball",
        apply_fn=lambda g: _u_drop_ball(g),
        description="FIND U-frame opening → DROP color-4 to bottom of grid"
    ))
    
    # === Pattern 9b: Connect same-color cells with lines ===
    programs.append(PuzzleProgram(
        name="connect_same_color",
        apply_fn=lambda g: _connect_same_color(g),
        description="EACH color → CONNECT all cells of that color with straight lines"
    ))
    
    # === Pattern 9c: Fill intersection gap ===
    # Where a full row and a partial col (or full col + partial row) intersect,
    # the cell at intersection gets filled with the "expected" color of the partial line.
    programs.append(PuzzleProgram(
        name="fill_intersection_gap",
        apply_fn=lambda g: _fill_intersection_gap(g),
        description="FIND full-row/full-col + partial line → FILL gap at intersection"
    ))
    
    # === Pattern 9: Most frequent color → center of empty region ===
    # "ABOVE separator → COUNT colors → MOST_FREQUENT → WRITE AT center of below"
    for inp_t, out_t in train_pairs[:1]:
        ih_t, iw_t = grid_shape(inp_t)
        bg_t = most_common_color(inp_t)
        sep_r = None
        for r in range(ih_t):
            vals = set(inp_t[r][c] for c in range(iw_t))
            if len(vals) == 1 and inp_t[r][0] != bg_t:
                sep_r = r
                break
        if sep_r is not None and sep_r > 0:
            def make_mfc():
                def fn(g):
                    h2, w2 = grid_shape(g)
                    bg2 = most_common_color(g)
                    sr = None
                    for r in range(h2):
                        vals = set(g[r][c2] for c2 in range(w2))
                        if len(vals) == 1 and g[r][0] != bg2:
                            sr = r
                            break
                    if sr is None or sr == 0:
                        return None
                    colors = Counter()
                    for r in range(sr):
                        for c in range(w2):
                            v = g[r][c]
                            if v != bg2 and v != g[sr][0]:
                                colors[v] += 1
                    if not colors:
                        return None
                    mc = colors.most_common(1)[0][0]
                    result = [row[:] for row in g]
                    result[h2 - 1][w2 // 2] = mc
                    return result
                return fn
            programs.append(PuzzleProgram(
                name="most_freq_center",
                apply_fn=make_mfc(),
                description="ABOVE separator → MOST_FREQUENT color → WRITE AT bottom center"
            ))
    
    # === Pattern 10: Diagonal extend ===
    # "FIND collinear points → EXTEND pattern beyond last point WITH new_color"
    if (ih, iw) == (oh, ow):
        objs = sorted([(r, c) for r in range(ih) for c in range(iw) if inp0[r][c] != bg])
        if 2 <= len(objs) <= 30:
            dr_d = objs[1][0] - objs[0][0]
            dc_d = objs[1][1] - objs[0][1]
            if (dr_d != 0 or dc_d != 0):
                consistent = all(
                    objs[j][0] - objs[j-1][0] == dr_d and objs[j][1] - objs[j-1][1] == dc_d
                    for j in range(2, len(objs))
                )
                if consistent:
                    # Learn new_color from first training pair diff
                    new_cs = set(
                        out0[r][c] for r in range(ih) for c in range(iw)
                        if inp0[r][c] != out0[r][c] and inp0[r][c] == bg
                    )
                    if len(new_cs) == 1:
                        nc_d = list(new_cs)[0]
                        frozen_d = (dr_d, dc_d, nc_d)
                        
                        def make_diag_ext(params):
                            _, _, nc2 = params  # dr/dc learned dynamically per input
                            def fn(g):
                                bg3 = most_common_color(g)
                                h2, w2 = grid_shape(g)
                                pts = sorted([(r, c) for r in range(h2) for c in range(w2)
                                              if g[r][c] != bg3])
                                if len(pts) < 2:
                                    return None
                                # Dynamic dr/dc from this input's points
                                d_r = pts[1][0] - pts[0][0]
                                d_c = pts[1][1] - pts[0][1]
                                if d_r == 0 and d_c == 0:
                                    return None
                                # Verify all points are collinear
                                for j in range(2, len(pts)):
                                    if pts[j][0] - pts[j-1][0] != d_r or pts[j][1] - pts[j-1][1] != d_c:
                                        return None
                                result = [row[:] for row in g]
                                r, c = pts[-1]
                                while True:
                                    r += d_r
                                    c += d_c
                                    if not (0 <= r < h2 and 0 <= c < w2):
                                        break
                                    result[r][c] = nc2
                                return result
                            return fn
                        
                        programs.append(PuzzleProgram(
                            name="diagonal_extend",
                            apply_fn=make_diag_ext(frozen_d),
                            description=f"FIND collinear points (dr={dr_d},dc={dc_d}) → EXTEND WITH color {nc_d}"
                        ))
    
    # === Pattern 11: Separator split + compare (non-equal halves) ===
    # For single horizontal or vertical separator, output = shrunk operation result
    for sep_c_try in range(10):
        if sep_c_try == bg:
            continue
        h_seps = [r for r in range(ih) if all(inp0[r][c] == sep_c_try for c in range(iw))]
        v_seps = [c for c in range(iw) if all(inp0[r][c] == sep_c_try for r in range(ih))]
        
        if len(h_seps) == 1 and not v_seps:
            sr = h_seps[0]
            top_h = sr
            bot_h = ih - sr - 1
            if top_h == bot_h and (oh, ow) == (top_h, iw):
                for op_s in ['xor', 'or', 'and']:
                    for fill_c_s in range(10):
                        frozen_s = {'sep_c': sep_c_try, 'op': op_s, 'fill': fill_c_s, 'bg': bg}
                        
                        def make_sep_h(params):
                            def fn(g):
                                h2, w2 = grid_shape(g)
                                bg2 = params['bg']
                                sc = params['sep_c']
                                sr2 = None
                                for r in range(h2):
                                    if all(g[r][c2] == sc for c2 in range(w2)):
                                        sr2 = r
                                        break
                                if sr2 is None:
                                    return None
                                th = sr2
                                result = [[bg2] * w2 for _ in range(th)]
                                for r in range(th):
                                    for c in range(w2):
                                        a = g[r][c] != bg2
                                        b = g[sr2 + 1 + r][c] != bg2 if sr2 + 1 + r < h2 else False
                                        if params['op'] == 'xor':
                                            m = a != b
                                        elif params['op'] == 'or':
                                            m = a or b
                                        else:
                                            m = a and b
                                        if m:
                                            result[r][c] = params['fill']
                                return result
                            return fn
                        
                        # Quick verify on first pair only
                        test_fn = make_sep_h(frozen_s)
                        r0 = test_fn(inp0)
                        if r0 is not None and grid_eq(r0, out0):
                            programs.append(PuzzleProgram(
                                name=f"sep_h_{op_s}_{fill_c_s}",
                                apply_fn=test_fn,
                                description=f"H-separator → {op_s.upper()} halves → FILL {fill_c_s}"
                            ))
                            break
                    else:
                        continue
                    break
    
    # === Pattern 12: Separator XOR/diff with new mark color ===
    # Handles: sep splits grid into two halves, compare non-bg cells, mark differences
    # Supports both h-sep and v-sep, output may be smaller (half size)
    # Output mark color may be NEW (not in input)
    for sep_c_try in range(10):
        if sep_c_try == bg:
            continue
        # Horizontal separator
        h_seps = [r for r in range(ih) if all(inp0[r][c] == sep_c_try for c in range(iw))]
        if len(h_seps) == 1:
            sr = h_seps[0]
            top_h, bot_h = sr, ih - sr - 1
            if top_h > 0 and bot_h > 0:
                # Determine which comparison produces the output
                for out_bg_try in set(out0[r][c] for r in range(oh) for c in range(ow)):
                    mark_cs = set(out0[r][c] for r in range(oh) for c in range(ow)) - {out_bg_try}
                    if len(mark_cs) != 1:
                        continue
                    mark_c = mark_cs.pop()
                    # Try: output = top_h x iw or bot_h x iw
                    for out_h, use_top_for_A in [(top_h, True), (bot_h, False)]:
                        if (oh, ow) != (out_h, iw):
                            continue
                        for op_name in ['xor', 'diff_ab', 'diff_ba', 'nor', 'and', 'or']:
                            frozen_sep = {'sep_c': sep_c_try, 'mark': mark_c, 'out_bg': out_bg_try,
                                          'op': op_name, 'use_top_A': use_top_for_A}
                            def make_sep_xor_h(p):
                                def fn(g):
                                    h2, w2 = grid_shape(g)
                                    sr2 = None
                                    for r in range(h2):
                                        if all(g[r][c2] == p['sep_c'] for c2 in range(w2)):
                                            sr2 = r; break
                                    if sr2 is None: return None
                                    th, bh = sr2, h2 - sr2 - 1
                                    if p['use_top_A']:
                                        A = g[:th]; B = g[sr2+1:sr2+1+th]
                                        out_h2 = th
                                    else:
                                        A = g[sr2+1:sr2+1+bh]; B = g[:bh]
                                        out_h2 = bh
                                    if len(A) != out_h2 or len(B) != out_h2: return None
                                    bg2 = p['out_bg']
                                    result = []
                                    for r in range(out_h2):
                                        row = []
                                        for c in range(w2):
                                            a_fg = A[r][c] != bg2 and A[r][c] != p['sep_c']
                                            b_fg = B[r][c] != bg2 and B[r][c] != p['sep_c']
                                            if p['op'] == 'xor': m = a_fg != b_fg
                                            elif p['op'] == 'diff_ab': m = a_fg and not b_fg
                                            elif p['op'] == 'diff_ba': m = b_fg and not a_fg
                                            elif p['op'] == 'nor': m = not a_fg and not b_fg
                                            elif p['op'] == 'and': m = a_fg and b_fg
                                            else: m = a_fg or b_fg  # or
                                            row.append(p['mark'] if m else bg2)
                                        result.append(row)
                                    return result
                                return fn
                            sx_fn = make_sep_xor_h(frozen_sep)
                            if all(grid_eq(sx_fn(inp_t), out_t) for inp_t, out_t in train_pairs):
                                programs.append(PuzzleProgram(
                                    name=f"sep_h_{op_name}_mark{mark_c}",
                                    apply_fn=sx_fn,
                                    description=f"H-sep → {op_name} → mark {mark_c} (bg={out_bg_try})"
                                ))
                                break
                        else:
                            continue
                        break

        # Vertical separator
        v_seps = [c for c in range(iw) if all(inp0[r][c] == sep_c_try for r in range(ih))]
        if len(v_seps) == 1:
            sc = v_seps[0]
            left_w, right_w = sc, iw - sc - 1
            if left_w > 0 and right_w > 0:
                for out_bg_try in set(out0[r][c] for r in range(oh) for c in range(ow)):
                    mark_cs = set(out0[r][c] for r in range(oh) for c in range(ow)) - {out_bg_try}
                    if len(mark_cs) != 1:
                        continue
                    mark_c = mark_cs.pop()
                    for out_w, use_left_A in [(left_w, True), (right_w, False)]:
                        if (oh, ow) != (ih, out_w):
                            continue
                        for op_name in ['xor', 'diff_ab', 'diff_ba', 'nor', 'and', 'or']:
                            frozen_sep_v = {'sep_c': sep_c_try, 'mark': mark_c, 'out_bg': out_bg_try,
                                            'op': op_name, 'use_left_A': use_left_A}
                            def make_sep_xor_v(p):
                                def fn(g):
                                    h2, w2 = grid_shape(g)
                                    sc2 = None
                                    for c in range(w2):
                                        if all(g[r2][c] == p['sep_c'] for r2 in range(h2)):
                                            sc2 = c; break
                                    if sc2 is None: return None
                                    lw, rw = sc2, w2 - sc2 - 1
                                    if p['use_left_A']:
                                        A = [row[:lw] for row in g]; B = [row[sc2+1:sc2+1+lw] for row in g]
                                        out_w2 = lw
                                    else:
                                        A = [row[sc2+1:sc2+1+rw] for row in g]; B = [row[:rw] for row in g]
                                        out_w2 = rw
                                    bg2 = p['out_bg']
                                    result = []
                                    for r in range(h2):
                                        row = []
                                        for c in range(out_w2):
                                            a_fg = A[r][c] != bg2 and A[r][c] != p['sep_c']
                                            b_fg = B[r][c] != bg2 and B[r][c] != p['sep_c']
                                            if p['op'] == 'xor': m = a_fg != b_fg
                                            elif p['op'] == 'diff_ab': m = a_fg and not b_fg
                                            elif p['op'] == 'diff_ba': m = b_fg and not a_fg
                                            elif p['op'] == 'nor': m = not a_fg and not b_fg
                                            elif p['op'] == 'and': m = a_fg and b_fg
                                            else: m = a_fg or b_fg
                                            row.append(p['mark'] if m else bg2)
                                        result.append(row)
                                    return result
                                return fn
                            sx_fn_v = make_sep_xor_v(frozen_sep_v)
                            if all(grid_eq(sx_fn_v(inp_t), out_t) for inp_t, out_t in train_pairs):
                                programs.append(PuzzleProgram(
                                    name=f"sep_v_{op_name}_mark{mark_c}",
                                    apply_fn=sx_fn_v,
                                    description=f"V-sep → {op_name} → mark {mark_c} (bg={out_bg_try})"
                                ))
                                break
                        else:
                            continue
                        break

    # === Pattern 13: Object move by own width/height ===
    if (oh, ow) == (ih, iw):
        objs_in = select_objects(inp0, bg)
        objs_out = select_objects(out0, bg)
        if len(objs_in) >= 1 and len(objs_in) == len(objs_out):
            # Sort both by size desc then position
            objs_in_s = sorted(objs_in, key=lambda o: (-o.size, o.center))
            objs_out_s = sorted(objs_out, key=lambda o: (-o.size, o.center))
            # Check: each object moves by its own width (right) or height (down)
            for move_type in ['width_right', 'width_left', 'height_down', 'height_up']:
                match = True
                for oi, oo in zip(objs_in_s, objs_out_s):
                    if oi.size != oo.size:
                        match = False; break
                    dc = oo.center[1] - oi.center[1]
                    dr = oo.center[0] - oi.center[0]
                    if move_type == 'width_right' and (dc != oi.width or dr != 0):
                        match = False; break
                    elif move_type == 'width_left' and (dc != -oi.width or dr != 0):
                        match = False; break
                    elif move_type == 'height_down' and (dr != oi.height or dc != 0):
                        match = False; break
                    elif move_type == 'height_up' and (dr != -oi.height or dc != 0):
                        match = False; break
                if match:
                    def make_obj_self_move(mt):
                        def fn(g):
                            bg2 = most_common_color(g)
                            h2, w2 = grid_shape(g)
                            objs2 = select_objects(g, bg2)
                            if not objs2: return None
                            result = [[bg2]*w2 for _ in range(h2)]
                            for o in objs2:
                                if mt == 'width_right': dr2, dc2 = 0, o.width
                                elif mt == 'width_left': dr2, dc2 = 0, -o.width
                                elif mt == 'height_down': dr2, dc2 = o.height, 0
                                else: dr2, dc2 = -o.height, 0
                                for r, c in o.cells:
                                    nr, nc = r + dr2, c + dc2
                                    if 0 <= nr < h2 and 0 <= nc < w2:
                                        result[nr][nc] = g[r][c]
                                    else:
                                        return None
                            return result
                        return fn
                    mv_fn = make_obj_self_move(move_type)
                    if all(grid_eq(mv_fn(inp_t), out_t) for inp_t, out_t in train_pairs):
                        programs.append(PuzzleProgram(
                            name=f"move_obj_by_{move_type}",
                            apply_fn=mv_fn,
                            description=f"EACH object → MOVE by own {move_type}"
                        ))
                    break

    # === Pattern: cell_to_inverted_block (0692e18c type) ===
    # Input NxN → Output (N*N)x(N*N): each non-zero cell → place inverted copy of input
    if oh == ih * ih and ow == iw * iw:
        def try_cell_invert_block(g, ratio=ih):
            h2, w2 = grid_shape(g)
            r = h2  # ratio = input size
            oh2, ow2 = h2 * r, w2 * r
            result = [[0] * ow2 for _ in range(oh2)]
            for row in range(h2):
                for col in range(w2):
                    if g[row][col] != 0:
                        color = g[row][col]
                        inv = [[color if g[rr][cc] == 0 else 0
                                for cc in range(w2)] for rr in range(h2)]
                        for rr in range(h2):
                            for cc in range(w2):
                                result[row * h2 + rr][col * w2 + cc] = inv[rr][cc]
            return result

        programs.append(PuzzleProgram(
            name="cell_to_inverted_block",
            apply_fn=try_cell_invert_block,
            description="EACH non-zero cell → PLACE inverted input copy AT cell position"
        ))

    # === Pattern: cell_to_solid_block ===
    # Input NxN → Output (N*N)x(N*N): each non-zero cell → fill NxN block with cell's color
    if oh == ih * ih and ow == iw * iw:
        def try_cell_solid_block(g, ratio=ih):
            h2, w2 = grid_shape(g)
            r = h2
            oh2, ow2 = h2 * r, w2 * r
            result = [[0] * ow2 for _ in range(oh2)]
            for row in range(h2):
                for col in range(w2):
                    if g[row][col] != 0:
                        color = g[row][col]
                        for rr in range(h2):
                            for cc in range(w2):
                                result[row * h2 + rr][col * w2 + cc] = color
            return result

        programs.append(PuzzleProgram(
            name="cell_to_solid_block",
            apply_fn=try_cell_solid_block,
            description="EACH non-zero cell → FILL NxN block WITH cell color"
        ))

    # === Pattern: cell_to_copy_block ===
    # Input NxN → Output (N*N)x(N*N): each non-zero cell → place copy of input
    if oh == ih * ih and ow == iw * iw:
        def try_cell_copy_block(g, ratio=ih):
            h2, w2 = grid_shape(g)
            r = h2
            oh2, ow2 = h2 * r, w2 * r
            result = [[0] * ow2 for _ in range(oh2)]
            for row in range(h2):
                for col in range(w2):
                    if g[row][col] != 0:
                        for rr in range(h2):
                            for cc in range(w2):
                                result[row * h2 + rr][col * w2 + cc] = g[rr][cc]
            return result

        programs.append(PuzzleProgram(
            name="cell_to_copy_block",
            apply_fn=try_cell_copy_block,
            description="EACH non-zero cell → COPY input AT cell position"
        ))

    # === Pattern: simple_tile_NxN ===
    # Input → tile N times in each direction to fill output
    # Dynamic: detect ratio from output shape per-pair (handled by cross_engine)
    if oh > ih and ow > iw and oh % ih == 0 and ow % iw == 0:
        # Check if all train pairs have integer ratio
        all_int_ratio = True
        for inp_t, out_t in train_pairs:
            h_t, w_t = grid_shape(inp_t)
            oh_t, ow_t = grid_shape(out_t)
            if oh_t % h_t != 0 or ow_t % w_t != 0:
                all_int_ratio = False
                break

        if all_int_ratio:
            # Fixed ratio version
            frozen_ratio_h = oh // ih
            frozen_ratio_w = ow // iw
            def try_simple_tile(g, rh=frozen_ratio_h, rw=frozen_ratio_w):
                h2, w2 = grid_shape(g)
                return [[g[r % h2][c % w2] for c in range(w2 * rw)] for r in range(h2 * rh)]

            programs.append(PuzzleProgram(
                name="simple_tile",
                apply_fn=try_simple_tile,
                description=f"TILE input {oh//ih}x{ow//iw} times"
            ))

    # === Pattern: diagonal_slide_tile (d13f3404 type) ===
    # Input pattern slides diagonally in output: each row shifts right by 1 column
    if oh == ih * 2 and ow == iw * 2:
        for shift_r, shift_c in [(0, 1), (1, 0), (1, 1)]:
            def make_diag_slide(sr, sc):
                def fn(g):
                    h2, w2 = grid_shape(g)
                    oh2, ow2 = h2 * 2, w2 * 2
                    result = [[0] * ow2 for _ in range(oh2)]
                    # Find non-zero cells and their diagonal pattern
                    nz = [(r, c, g[r][c]) for r in range(h2) for c in range(w2) if g[r][c] != 0]
                    if not nz:
                        return None
                    # Place copies diagonally
                    for step in range(max(oh2, ow2)):
                        for r0, c0, color in nz:
                            nr = r0 + step * sr
                            nc = c0 + step * sc
                            if 0 <= nr < oh2 and 0 <= nc < ow2:
                                result[nr][nc] = color
                    return result
                return fn
            programs.append(PuzzleProgram(
                name=f"diagonal_slide_{shift_r}_{shift_c}",
                apply_fn=make_diag_slide(shift_r, shift_c),
                description=f"TILE input SLIDING diagonally ({shift_r},{shift_c})"
            ))

    # === Pattern: fill_dot_to_corner ===
    # Single non-bg dot on uniform bg → fill rectangle from dot to nearest corner
    def try_fill_dot_to_corner(g):
        bg2 = most_common_color(g)
        h2, w2 = grid_shape(g)
        non_bg = [(r, c, g[r][c]) for r in range(h2) for c in range(w2) if g[r][c] != bg2]
        if len(non_bg) != 1:
            return None
        r0, c0, color = non_bg[0]
        # nearest corner
        corners = [(0, 0), (0, w2-1), (h2-1, 0), (h2-1, w2-1)]
        best_corner = min(corners, key=lambda cr: abs(cr[0]-r0) + abs(cr[1]-c0))
        cr, cc = best_corner
        rmin, rmax = min(r0, cr), max(r0, cr)
        cmin, cmax = min(c0, cc), max(c0, cc)
        result = [row[:] for row in g]
        for r in range(rmin, rmax+1):
            for c in range(cmin, cmax+1):
                result[r][c] = color
        return result

    programs.append(PuzzleProgram(
        name="fill_dot_to_corner",
        apply_fn=try_fill_dot_to_corner,
        description="FIND single dot → FILL rectangle TO nearest corner"
    ))

    # === Pattern: majority_vote_fix ===
    # Find cells where all 4-neighbors of same non-bg color disagree → fix to majority
    def try_majority_vote_fix(g):
        bg2 = most_common_color(g)
        h2, w2 = grid_shape(g)
        result = [row[:] for row in g]
        changed = False
        for r in range(h2):
            for c in range(w2):
                nbrs = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h2 and 0 <= nc < w2:
                        nbrs.append(g[nr][nc])
                if not nbrs:
                    continue
                from collections import Counter
                cnt = Counter(nbrs)
                # If cell differs from all neighbors and neighbors mostly agree
                majority_color, majority_count = cnt.most_common(1)[0]
                if majority_count >= 3 and g[r][c] != majority_color and g[r][c] != bg2:
                    result[r][c] = majority_color
                    changed = True
        return result if changed else None

    programs.append(PuzzleProgram(
        name="majority_vote_fix",
        apply_fn=try_majority_vote_fix,
        description="EACH cell WHERE neighbors disagree → REPLACE WITH majority neighbor color"
    ))

    # === Pattern: bar_rotate_at_pivot ===
    # Horizontal bar + single pivot cell of different color → rotate bar 90° at pivot
    def try_bar_rotate_pivot(g):
        bg2 = most_common_color(g)
        h2, w2 = grid_shape(g)
        objs = select_objects(g, bg2)
        if len(objs) != 1:
            return None
        obj = objs[0]
        cells_sorted = sorted(obj.cells)
        rows_set = {r for r, c in cells_sorted}
        cols_set = {c for r, c in cells_sorted}
        if len(rows_set) != 1:
            return None
        row0 = list(rows_set)[0]
        # Find colors in the bar
        colors = {}
        for r, c in cells_sorted:
            colors.setdefault(g[r][c], []).append((r, c))
        if len(colors) != 2:
            return None
        # Pivot = color with fewer cells (ideally 1)
        sorted_colors = sorted(colors.items(), key=lambda x: len(x[1]))
        pivot_color, pivot_cells = sorted_colors[0]
        bar_color, bar_cells = sorted_colors[1]
        if len(pivot_cells) != 1:
            return None
        pr, pc = pivot_cells[0]
        # Bar is horizontal, rotate to vertical at pivot
        result = [row[:] for row in g]
        # Remove original bar (except pivot)
        for r, c in bar_cells:
            result[r][c] = bg2
        # Count cells on each side of pivot
        left_count = sum(1 for r, c in bar_cells if c < pc)
        right_count = sum(1 for r, c in bar_cells if c > pc)
        # Rotate: left→up, right→down (or vice versa based on which side has more)
        for i in range(1, left_count + 1):
            nr = pr - i
            if 0 <= nr < h2:
                result[nr][pc] = bar_color
        for i in range(1, right_count + 1):
            nr = pr + i
            if 0 <= nr < h2:
                result[nr][pc] = bar_color
        return result

    programs.append(PuzzleProgram(
        name="bar_rotate_at_pivot",
        apply_fn=try_bar_rotate_pivot,
        description="FIND horizontal bar WITH pivot → ROTATE 90° at pivot"
    ))

    # === Pattern: dot_L_to_corner ===
    # Each dot draws L-shaped path to nearest corner (right+down or similar)
    def try_dot_L_to_corner(g):
        bg2 = most_common_color(g)
        h2, w2 = grid_shape(g)
        non_bg = [(r, c, g[r][c]) for r in range(h2) for c in range(w2) if g[r][c] != bg2]
        if len(non_bg) < 1 or len(non_bg) > 10:
            return None
        result = [row[:] for row in g]
        for r0, c0, color in non_bg:
            # nearest corner
            corners = [(0, 0), (0, w2-1), (h2-1, 0), (h2-1, w2-1)]
            best_corner = min(corners, key=lambda cr: abs(cr[0]-r0) + abs(cr[1]-c0))
            cr, cc = best_corner
            # Draw L: first horizontal to corner col, then vertical to corner row
            dc = 1 if cc > c0 else -1 if cc < c0 else 0
            dr = 1 if cr > r0 else -1 if cr < r0 else 0
            # Horizontal leg
            c = c0
            while c != cc + dc:
                result[r0][c] = color
                c += dc
            # Vertical leg
            r = r0
            while r != cr + dr:
                result[r][cc] = color
                r += dr
        return result

    programs.append(PuzzleProgram(
        name="dot_L_to_corner",
        apply_fn=try_dot_L_to_corner,
        description="EACH dot → DRAW L-path TO nearest corner"
    ))

    # === Pattern: diagonal_object_continue ===
    # Two same-colored objects placed diagonally → predict continuation position
    def try_diagonal_continue(g):
        bg2 = most_common_color(g)
        h2, w2 = grid_shape(g)
        objs = select_objects_multicolor(g, bg2)
        if len(objs) < 2 or len(objs) > 4:
            return None
        # Check if objects have same shape
        shapes = []
        for o in objs:
            min_r = min(r for r, c in o.cells)
            min_c = min(c for r, c in o.cells)
            norm = frozenset((r - min_r, c - min_c) for r, c in o.cells)
            shapes.append((norm, min_r, min_c, o))
        if len(set(s[0] for s in shapes)) != 1:
            return None
        # Sort by position
        shapes.sort(key=lambda s: (s[1], s[2]))
        if len(shapes) == 2:
            dr = shapes[1][1] - shapes[0][1]
            dc = shapes[1][2] - shapes[0][2]
            # Predict next position
            new_r = shapes[1][1] + dr
            new_c = shapes[1][2] + dc
            # Also predict before first
            pre_r = shapes[0][1] - dr
            pre_c = shapes[0][2] - dc
            result = [row[:] for row in g]
            new_color = 8  # Will be learned from training
            candidates = []
            # Try placing at continuation
            for nr, nc in [(new_r, new_c), (pre_r, pre_c)]:
                for r_off, c_off in shapes[0][0]:
                    rr, cc = nr + r_off, nc + c_off
                    if 0 <= rr < h2 and 0 <= cc < w2:
                        candidates.append((rr, cc))
            return None  # Too complex for generic, skip
        return None

    # === Pattern: reflect_object_4fold ===
    # One shape object + one center object → reflect shape around center in 4 directions
    def try_reflect_4fold(g):
        bg2 = most_common_color(g)
        h2, w2 = grid_shape(g)
        objs = select_objects(g, bg2)
        if len(objs) != 2:
            return None
        # Identify center object (more compact/square) and shape object
        o1, o2 = objs
        # Center = the one that's more centered in the grid
        def center_dist(o):
            cr = sum(r for r, c in o.cells) / len(o.cells)
            cc = sum(c for r, c in o.cells) / len(o.cells)
            return abs(cr - h2/2) + abs(cc - w2/2)
        if center_dist(o1) < center_dist(o2):
            center_obj, shape_obj = o1, o2
        else:
            center_obj, shape_obj = o2, o1
        # Center point of center object
        cr = sum(r for r, c in center_obj.cells) / len(center_obj.cells)
        cc = sum(c for r, c in center_obj.cells) / len(center_obj.cells)
        result = [row[:] for row in g]
        shape_color = shape_obj.color
        # Reflect shape across center point
        for r, c in shape_obj.cells:
            # 180° rotation
            nr, nc = int(2*cr - r), int(2*cc - c)
            if 0 <= nr < h2 and 0 <= nc < w2 and result[nr][nc] == bg2:
                result[nr][nc] = g[r][c]
            # Horizontal reflection
            nr2, nc2 = r, int(2*cc - c)
            if 0 <= nr2 < h2 and 0 <= nc2 < w2 and result[nr2][nc2] == bg2:
                result[nr2][nc2] = g[r][c]
            # Vertical reflection
            nr3, nc3 = int(2*cr - r), c
            if 0 <= nr3 < h2 and 0 <= nc3 < w2 and result[nr3][nc3] == bg2:
                result[nr3][nc3] = g[r][c]
        return result

    programs.append(PuzzleProgram(
        name="reflect_4fold_center",
        apply_fn=try_reflect_4fold,
        description="FIND shape+center → REFLECT shape 4-fold AROUND center"
    ))

    # === Pattern: connect_dots_line ===
    # Dots of same color → connect with straight lines (horizontal/vertical)
    def try_connect_same_color_lines(g):
        bg2 = most_common_color(g)
        h2, w2 = grid_shape(g)
        # Group cells by color
        color_cells = {}
        for r in range(h2):
            for c in range(w2):
                if g[r][c] != bg2:
                    color_cells.setdefault(g[r][c], []).append((r, c))
        if not color_cells:
            return None
        result = [row[:] for row in g]
        changed = False
        for color, cells in color_cells.items():
            if len(cells) < 2 or len(cells) > 20:
                continue
            # Connect pairs that share a row or column
            for i, (r1, c1) in enumerate(cells):
                for r2, c2 in cells[i+1:]:
                    if r1 == r2:
                        cmin, cmax = min(c1, c2), max(c1, c2)
                        for c in range(cmin, cmax+1):
                            if result[r1][c] == bg2:
                                result[r1][c] = color
                                changed = True
                    elif c1 == c2:
                        rmin, rmax = min(r1, r2), max(r1, r2)
                        for r in range(rmin, rmax+1):
                            if result[r][c1] == bg2:
                                result[r][c1] = color
                                changed = True
        return result if changed else None

    programs.append(PuzzleProgram(
        name="connect_same_color_lines",
        apply_fn=try_connect_same_color_lines,
        description="EACH color → CONNECT dots WITH straight lines"
    ))

    # === Pattern: template_row_project ===
    # Separator row splits grid; template row below separator has colors;
    # marker row above has non-zero cells; project template colors upward
    def try_template_project(g):
        bg2 = most_common_color(g)
        h2, w2 = grid_shape(g)
        # Find full separator row
        sep_rows = []
        for r in range(h2):
            vals = set(g[r])
            if len(vals) == 1 and list(vals)[0] != bg2:
                sep_rows.append(r)
        if len(sep_rows) != 1:
            return None
        sep_r = sep_rows[0]
        sep_c = g[sep_r][0]
        # Template row: first non-bg row below separator (skip empty rows)
        template_r = None
        for r in range(sep_r+1, h2):
            if any(g[r][c] != bg2 for c in range(w2)):
                template_r = r
                break
        if template_r is None:
            return None
        # Marker: non-bg cells above separator (e.g., row 0 has 8s)
        marker_cells = []
        for r in range(sep_r):
            for c in range(w2):
                if g[r][c] != bg2 and g[r][c] != sep_c:
                    marker_cells.append((r, c, g[r][c]))
        if not marker_cells:
            return None
        # Count marker cells per column span
        marker_count = len(marker_cells)
        # For each column: template color + fill upward from separator by marker_count rows
        result = [row[:] for row in g]
        for c in range(w2):
            tmpl_color = g[template_r][c]
            if tmpl_color == bg2 or tmpl_color == sep_c:
                continue
            # Fill upward from sep_r-1 for marker_count rows
            for i in range(1, marker_count + 1):
                nr = sep_r - i
                if 0 <= nr < h2 and result[nr][c] == bg2:
                    result[nr][c] = tmpl_color
        return result

    programs.append(PuzzleProgram(
        name="template_row_project",
        apply_fn=try_template_project,
        description="FIND separator → TEMPLATE row below → PROJECT colors upward BY marker count"
    ))

    # === Pattern: gravity_fill (dot falls to nearest edge in each quadrant) ===
    # Single dot + direction = flood toward edge
    # Actually more general: each non-bg cell expands in all 4 directions until blocked

    # === Pattern: concentric_frames ===
    # Single dot → expand outward with concentric rectangles of alternating colors
    def try_concentric_frames(g):
        bg2 = most_common_color(g)
        h2, w2 = grid_shape(g)
        non_bg = [(r, c, g[r][c]) for r in range(h2) for c in range(w2) if g[r][c] != bg2]
        if len(non_bg) < 2 or len(non_bg) > 10:
            return None
        # Check if they form concentric pattern already
        # Sort by distance from center of all points
        cr = sum(r for r, c, _ in non_bg) / len(non_bg)
        cc = sum(c for r, c, _ in non_bg) / len(non_bg)
        # Group by Chebyshev distance from center
        dist_colors = {}
        for r, c, color in non_bg:
            d = max(abs(r - cr), abs(c - cc))
            d_int = round(d)
            dist_colors.setdefault(d_int, set()).add(color)
        if len(dist_colors) < 2:
            return None
        # Each distance should have exactly one color
        if not all(len(cs) == 1 for cs in dist_colors.values()):
            return None
        # Get color sequence from innermost
        max_dist = max(dist_colors.keys())
        color_seq = []
        for d in range(max_dist + 1):
            if d in dist_colors:
                color_seq.append(list(dist_colors[d])[0])
            else:
                color_seq.append(bg2)
        # Expand: fill remaining cells by distance
        result = [row[:] for row in g]
        center_r, center_c = round(cr), round(cc)
        for r in range(h2):
            for c in range(w2):
                d = max(abs(r - center_r), abs(c - center_c))
                if d < len(color_seq) and result[r][c] == bg2:
                    result[r][c] = color_seq[d]
        return result

    programs.append(PuzzleProgram(
        name="concentric_frames",
        apply_fn=try_concentric_frames,
        description="FIND dots → GROUP BY distance → FILL concentric frames"
    ))

    # === Pattern: ray_extend_to_edge ===
    # Each non-bg cell extends rays in learned directions until hitting edge or obstacle
    # Learn direction set from training pairs
    if ih == oh and iw == ow:
        DIRS_4 = [(0,1),(0,-1),(1,0),(-1,0)]
        DIRS_DIAG = [(1,1),(1,-1),(-1,1),(-1,-1)]
        DIRS_8 = DIRS_4 + DIRS_DIAG

        for dir_set_name, dir_set in [('4dir', DIRS_4), ('8dir', DIRS_8), ('diag', DIRS_DIAG)]:
            # Learn: which subset of directions is needed?
            # For each direction, extend every non-bg cell until edge/obstacle
            # Check if this exactly reproduces training outputs

            def make_ray_fn(dirs):
                def fn(g):
                    bg2 = most_common_color(g)
                    h2, w2 = grid_shape(g)
                    result = [row[:] for row in g]
                    for r in range(h2):
                        for c in range(w2):
                            if g[r][c] != bg2:
                                color = g[r][c]
                                for dr, dc in dirs:
                                    nr, nc = r + dr, c + dc
                                    while 0 <= nr < h2 and 0 <= nc < w2:
                                        if g[nr][nc] != bg2:
                                            break
                                        result[nr][nc] = color
                                        nr += dr
                                        nc += dc
                    return result
                return fn

            ray_fn = make_ray_fn(dir_set)
            if all(grid_eq(ray_fn(inp_t), out_t) for inp_t, out_t in train_pairs):
                programs.append(PuzzleProgram(
                    name=f"ray_extend_{dir_set_name}",
                    apply_fn=ray_fn,
                    description=f"EACH non-bg cell → EXTEND rays {dir_set_name} TO edge/obstacle"
                ))
                break  # found the right direction set

        # Also try per-direction variants (only H, only V, only certain dirs)
        for single_dir_name, single_dirs in [
            ('right', [(0,1)]), ('left', [(0,-1)]),
            ('down', [(1,0)]), ('up', [(-1,0)]),
            ('h', [(0,1),(0,-1)]), ('v', [(1,0),(-1,0)]),
        ]:
            sfn = make_ray_fn(single_dirs)
            if all(grid_eq(sfn(inp_t), out_t) for inp_t, out_t in train_pairs):
                programs.append(PuzzleProgram(
                    name=f"ray_extend_{single_dir_name}",
                    apply_fn=sfn,
                    description=f"EACH non-bg cell → EXTEND ray {single_dir_name} TO edge/obstacle"
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
