#!/usr/bin/env python3
"""
World Priors — ARC問題で起こりうる全変換の事前知識
=================================================
各関数は (train_pairs) → List[CrossPiece] を返す。

設計思想:
- 具体的な色の値に依存しない（抽象化）
- 入出力のサイズ関係から変換の種類を推定
- オブジェクト検出は背景色（最頻色）ベース
- 各変換は自己完結（他の変換との依存なし）
"""

from collections import Counter
from typing import List, Tuple, Optional, Set, Dict
from arc.grid import Grid, grid_shape, grid_eq, most_common_color

class WorldPiece:
    """A candidate transformation with a name and apply function."""
    def __init__(self, name, apply_fn):
        self.name = name
        self._apply = apply_fn
    
    def apply(self, grid):
        return self._apply(grid)

# ═══════════════════════════════════════
# ユーティリティ
# ═══════════════════════════════════════

def _bg(grid):
    """Background color (most frequent)."""
    c = Counter()
    for row in grid: c.update(row)
    return c.most_common(1)[0][0]

def _colors(grid):
    """All colors in grid."""
    s = set()
    for row in grid: s.update(row)
    return s

def _objects(grid, bg, connectivity=4):
    """Extract connected components of non-bg cells."""
    h, w = len(grid), len(grid[0])
    visited = [[False]*w for _ in range(h)]
    objects = []
    
    deltas = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8:
        deltas += [(-1,-1),(-1,1),(1,-1),(1,1)]
    
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] != bg:
                # BFS
                obj = []
                queue = [(r,c)]
                visited[r][c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    obj.append((cr, cc, grid[cr][cc]))
                    for dr, dc in deltas:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                objects.append(obj)
    return objects

def _bbox(cells):
    """Bounding box of cells: (min_r, min_c, max_r, max_c)."""
    rs = [r for r,c,_ in cells]
    cs = [c for r,c,_ in cells]
    return min(rs), min(cs), max(rs), max(cs)

def _crop_obj(grid, cells, bg):
    """Crop object to its bounding box."""
    r0, c0, r1, c1 = _bbox(cells)
    h, w = r1-r0+1, c1-c0+1
    result = [[bg]*w for _ in range(h)]
    for r, c, v in cells:
        result[r-r0][c-c0] = v
    return result

def _paste(grid, patch, r0, c0, bg):
    """Paste patch onto grid at position (r0, c0), skipping bg."""
    result = [row[:] for row in grid]
    ph, pw = len(patch), len(patch[0])
    h, w = len(grid), len(grid[0])
    for r in range(ph):
        for c in range(pw):
            nr, nc = r0+r, c0+c
            if 0<=nr<h and 0<=nc<w and patch[r][c] != bg:
                result[nr][nc] = patch[r][c]
    return result

# ═══════════════════════════════════════
# カテゴリ1: サイズ不変変換 (input.shape == output.shape)
# ═══════════════════════════════════════

def _gen_same_size(train_pairs):
    pieces = []
    
    # --- 1a: 色の置換マッピング ---
    # 入力の各色を出力の対応する色にマップ
    def learn_color_map(pairs):
        """Learn a consistent color → color mapping."""
        cmap = {}
        for inp, out in pairs:
            h, w = len(inp), len(inp[0])
            if len(out) != h or len(out[0]) != w:
                return None
            for r in range(h):
                for c in range(w):
                    ic, oc = inp[r][c], out[r][c]
                    if ic in cmap:
                        if cmap[ic] != oc:
                            return None
                    else:
                        cmap[ic] = oc
        return cmap
    
    cmap = learn_color_map(train_pairs)
    if cmap:
        cm = dict(cmap)
        pieces.append(WorldPiece("wp:color_map",
            lambda g, m=cm: [[m.get(c,c) for c in row] for row in g]))
    
    # --- 1b: 位置ベースの色マッピング（色が変わるが位置が同じ）---
    # 入力のbg→出力のbg、入力のfg→出力のfg (色の値は変わる)
    def learn_role_map(pairs):
        """bg→bg', fg_i→fg_i' by frequency rank."""
        role_maps = []
        for inp, out in pairs:
            h, w = len(inp), len(inp[0])
            if len(out) != h or len(out[0]) != w: return None
            
            ibg = _bg(inp)
            obg = _bg(out)
            
            ic = Counter()
            for row in inp: ic.update(row)
            oc = Counter()
            for row in out: oc.update(row)
            
            # Rank non-bg colors by frequency
            i_ranked = sorted(set(ic.keys()) - {ibg}, key=lambda x: (-ic[x], x))
            o_ranked = sorted(set(oc.keys()) - {obg}, key=lambda x: (-oc[x], x))
            
            if len(i_ranked) != len(o_ranked): return None
            
            rm = {ibg: obg}
            for a, b in zip(i_ranked, o_ranked):
                rm[a] = b
            
            # Verify this mapping works
            ok = True
            for r in range(h):
                for c in range(w):
                    if rm.get(inp[r][c], -99) != out[r][c]:
                        ok = False; break
                if not ok: break
            if not ok: return None
            role_maps.append(rm)
        
        return role_maps
    
    role_maps = learn_role_map(train_pairs)
    if role_maps:
        # Apply by learning role mapping for new input
        def apply_role_map(g, ref_pairs=train_pairs):
            bg_g = _bg(g)
            gc = Counter()
            for row in g: gc.update(row)
            g_ranked = sorted(set(gc.keys()) - {bg_g}, key=lambda x: (-gc[x], x))
            
            # For identity transform (same colors), no need
            # We need to figure out what the OUTPUT colors should be
            # Use training examples to learn the abstract rule
            # (e.g., bg stays bg, color i stays color i)
            # The output uses the SAME colors as the input (identity role map)
            # ... unless there's a consistent re-coloring pattern
            
            # Check if training shows identity (output colors = input colors for each role)
            is_identity = True
            for rm in role_maps:
                if any(k != v for k, v in rm.items()):
                    is_identity = False; break
            
            if is_identity:
                return g  # identity, no color change
            
            # Non-trivial: check if bg always maps to bg (yes) and fg roles shift
            # For now, just apply identity role
            return [row[:] for row in g]
        
        # Only add if it's actually doing something (not identity)
        if role_maps and any(any(k != v for k, v in rm.items()) for rm in role_maps):
            pieces.append(WorldPiece("wp:role_color_map", apply_role_map))
    
    # --- 1c: フラッドフィル（囲まれた領域を塗る）---
    def flood_fill_enclosed(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = [row[:] for row in g]
        
        # Find exterior bg via BFS from edges
        visited = [[False]*w for _ in range(h)]
        queue = []
        for r in range(h):
            for c in [0, w-1]:
                if g[r][c] == bg and not visited[r][c]:
                    visited[r][c] = True
                    queue.append((r,c))
        for c in range(w):
            for r in [0, h-1]:
                if g[r][c] == bg and not visited[r][c]:
                    visited[r][c] = True
                    queue.append((r,c))
        while queue:
            cr, cc = queue.pop(0)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and g[nr][nc] == bg:
                    visited[nr][nc] = True
                    queue.append((nr,nc))
        
        # Fill interior bg with most common adjacent non-bg
        for r in range(h):
            for c in range(w):
                if g[r][c] == bg and not visited[r][c]:
                    neighbors = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and g[nr][nc] != bg:
                            neighbors.append(g[nr][nc])
                    if neighbors:
                        result[r][c] = Counter(neighbors).most_common(1)[0][0]
        return result
    pieces.append(WorldPiece("wp:flood_fill_enclosed", flood_fill_enclosed))
    
    # --- 1d: 重力（各列の非bg色を下に落とす）---
    for direction, dname in [("down","gravity_down"), ("up","gravity_up"), 
                             ("left","gravity_left"), ("right","gravity_right")]:
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
        pieces.append(WorldPiece(f"wp:{dname}", make_gravity(direction)))
    
    # --- 1e: 対称性修復 ---
    for sym_type in ["h", "v", "4fold"]:
        def make_sym(st):
            def fn(g):
                bg = _bg(g)
                h, w = len(g), len(g[0])
                result = [row[:] for row in g]
                if st == "h":
                    for r in range(h):
                        for c in range(w//2):
                            mc = w-1-c
                            if result[r][c] == bg and result[r][mc] != bg:
                                result[r][c] = result[r][mc]
                            elif result[r][mc] == bg and result[r][c] != bg:
                                result[r][mc] = result[r][c]
                elif st == "v":
                    for r in range(h//2):
                        mr = h-1-r
                        for c in range(w):
                            if result[r][c] == bg and result[mr][c] != bg:
                                result[r][c] = result[mr][c]
                            elif result[mr][c] == bg and result[r][c] != bg:
                                result[mr][c] = result[r][c]
                elif st == "4fold":
                    for r in range(h//2+1):
                        for c in range(w//2+1):
                            vals = []
                            positions = [(r,c),(r,w-1-c),(h-1-r,c),(h-1-r,w-1-c)]
                            for pr, pc in positions:
                                if 0<=pr<h and 0<=pc<w and result[pr][pc] != bg:
                                    vals.append(result[pr][pc])
                            if vals:
                                fill = Counter(vals).most_common(1)[0][0]
                                for pr, pc in positions:
                                    if 0<=pr<h and 0<=pc<w:
                                        result[pr][pc] = fill
                return result
            return fn
        pieces.append(WorldPiece(f"wp:symmetry_{sym_type}", make_sym(sym_type)))
    
    # --- 1f: 行/列でソート ---
    def sort_rows(g):
        bg = _bg(g)
        rows_with_count = []
        for r, row in enumerate(g):
            cnt = sum(1 for c in row if c != bg)
            rows_with_count.append((cnt, r, row))
        rows_with_count.sort(key=lambda x: x[0])
        return [row for _, _, row in rows_with_count]
    pieces.append(WorldPiece("wp:sort_rows_by_count", sort_rows))
    
    def sort_cols(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        cols = []
        for c in range(w):
            col = [g[r][c] for r in range(h)]
            cnt = sum(1 for v in col if v != bg)
            cols.append((cnt, c, col))
        cols.sort(key=lambda x: x[0])
        return [[cols[c][2][r] for c in range(w)] for r in range(h)]
    pieces.append(WorldPiece("wp:sort_cols_by_count", sort_cols))
    
    # --- 1g: マジョリティ色で少数色を置換 ---
    def replace_minority_color(g):
        bg = _bg(g)
        c = Counter()
        for row in g:
            for v in row:
                if v != bg: c[v] += 1
        if len(c) < 2: return g
        majority = c.most_common(1)[0][0]
        minority = c.most_common()[-1][0]
        return [[majority if v == minority else v for v in row] for row in g]
    pieces.append(WorldPiece("wp:replace_minority", replace_minority_color))
    
    # --- 1h: 行/列の線を延長（非bg色を行/列の端まで伸ばす）---
    def extend_lines_h(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = [row[:] for row in g]
        for r in range(h):
            colors = [g[r][c] for c in range(w) if g[r][c] != bg]
            if len(set(colors)) == 1 and len(colors) >= 2:
                for c in range(w):
                    result[r][c] = colors[0]
        return result
    pieces.append(WorldPiece("wp:extend_lines_h", extend_lines_h))
    
    def extend_lines_v(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        result = [row[:] for row in g]
        for c in range(w):
            colors = [g[r][c] for r in range(h) if g[r][c] != bg]
            if len(set(colors)) == 1 and len(colors) >= 2:
                for r in range(h):
                    result[r][c] = colors[0]
        return result
    pieces.append(WorldPiece("wp:extend_lines_v", extend_lines_v))
    
    # --- 1i: オブジェクトの最大を残す / 最小を残す ---
    def keep_largest_obj(g):
        bg = _bg(g)
        objs = _objects(g, bg)
        if not objs: return g
        largest = max(objs, key=len)
        h, w = len(g), len(g[0])
        result = [[bg]*w for _ in range(h)]
        for r, c, v in largest:
            result[r][c] = v
        return result
    pieces.append(WorldPiece("wp:keep_largest", keep_largest_obj))
    
    def keep_smallest_obj(g):
        bg = _bg(g)
        objs = _objects(g, bg)
        if not objs: return g
        smallest = min(objs, key=len)
        h, w = len(g), len(g[0])
        result = [[bg]*w for _ in range(h)]
        for r, c, v in smallest:
            result[r][c] = v
        return result
    pieces.append(WorldPiece("wp:keep_smallest", keep_smallest_obj))
    
    # --- 1j: オブジェクトを色でソートして並べ替え ---
    
    # --- 1k: 各オブジェクトをbboxで囲む（rectangle化）---
    def fill_object_bboxes(g):
        bg = _bg(g)
        objs = _objects(g, bg)
        result = [row[:] for row in g]
        for obj in objs:
            r0, c0, r1, c1 = _bbox(obj)
            # Most common color in object
            oc = Counter(v for _,_,v in obj).most_common(1)[0][0]
            for r in range(r0, r1+1):
                for c in range(c0, c1+1):
                    if result[r][c] == bg:
                        result[r][c] = oc
        return result
    pieces.append(WorldPiece("wp:fill_bboxes", fill_object_bboxes))
    
    # --- 1l: 各オブジェクト間を線でつなぐ ---
    def connect_objects(g):
        bg = _bg(g)
        objs = _objects(g, bg)
        if len(objs) < 2: return g
        result = [row[:] for row in g]
        # Connect centroids of same-color objects
        color_groups = {}
        for obj in objs:
            colors = Counter(v for _,_,v in obj)
            main_color = colors.most_common(1)[0][0]
            cr = sum(r for r,c,v in obj) // len(obj)
            cc = sum(c for r,c,v in obj) // len(obj)
            color_groups.setdefault(main_color, []).append((cr, cc))
        
        h, w = len(g), len(g[0])
        for color, centroids in color_groups.items():
            for i in range(len(centroids)-1):
                r1, c1 = centroids[i]
                r2, c2 = centroids[i+1]
                # Draw line (horizontal then vertical)
                for c in range(min(c1,c2), max(c1,c2)+1):
                    if 0<=r1<h and 0<=c<w:
                        result[r1][c] = color
                for r in range(min(r1,r2), max(r1,r2)+1):
                    if 0<=r<h and 0<=c2<w:
                        result[r][c2] = color
        return result
    pieces.append(WorldPiece("wp:connect_objects", connect_objects))
    
    return pieces

# ═══════════════════════════════════════
# カテゴリ2: サイズ変化変換
# ═══════════════════════════════════════

def _gen_size_change(train_pairs):
    pieces = []
    inp0, out0 = train_pairs[0]
    ih, iw = len(inp0), len(inp0[0])
    oh, ow = len(out0), len(out0[0])
    
    # --- 2a: クロップ（非bgのbboxに切り出す）---
    def crop_to_content(g):
        bg = _bg(g)
        h, w = len(g), len(g[0])
        rows = [r for r in range(h) if any(g[r][c] != bg for c in range(w))]
        cols = [c for c in range(w) if any(g[r][c] != bg for r in range(h))]
        if not rows or not cols: return g
        return [g[r][min(cols):max(cols)+1] for r in range(min(rows), max(rows)+1)]
    pieces.append(WorldPiece("wp:crop", crop_to_content))
    
    # --- 2b: スケーリング ---
    for s in [2, 3, 4, 5]:
        if oh == ih * s and ow == iw * s:
            def make_up(sc):
                def fn(g):
                    return [[g[r//sc][c//sc] for c in range(len(g[0])*sc)] for r in range(len(g)*sc)]
                return fn
            pieces.append(WorldPiece(f"wp:upscale_{s}x", make_up(s)))
        
        if ih == oh * s and iw == ow * s:
            def make_down(sc):
                def fn(g):
                    h, w = len(g), len(g[0])
                    return [[Counter(g[r*sc+dr][c*sc+dc] for dr in range(sc) for dc in range(sc)).most_common(1)[0][0]
                             for c in range(w//sc)] for r in range(h//sc)]
                return fn
            pieces.append(WorldPiece(f"wp:downscale_{s}x", make_down(s)))
    
    # --- 2c: パネル分割（仕切り線で分割）---
    # 入力にセパレータ色の行/列がある → パネルに分割
    
    # --- 2d: 最大オブジェクトをクロップ ---
    def crop_largest(g):
        bg = _bg(g)
        objs = _objects(g, bg)
        if not objs: return g
        largest = max(objs, key=len)
        return _crop_obj(g, largest, bg)
    pieces.append(WorldPiece("wp:crop_largest", crop_largest))
    
    # --- 2e: 最小オブジェクトをクロップ ---
    def crop_smallest(g):
        bg = _bg(g)
        objs = _objects(g, bg)
        if not objs: return g
        smallest = min(objs, key=len)
        return _crop_obj(g, smallest, bg)
    pieces.append(WorldPiece("wp:crop_smallest", crop_smallest))
    
    # --- 2f: 特定色をクロップ ---
    for color in range(10):
        def make_crop_color(col):
            def fn(g):
                h, w = len(g), len(g[0])
                cells = [(r,c,g[r][c]) for r in range(h) for c in range(w) if g[r][c] == col]
                if not cells: return None
                bg = _bg(g)
                return _crop_obj(g, cells, bg)
            return fn
        pieces.append(WorldPiece(f"wp:crop_color_{color}", make_crop_color(color)))
    
    # --- 2g: タイル化（小グリッドを繰り返す）---
    if oh > ih and ow > iw and oh % ih == 0 and ow % iw == 0:
        rh, rw = oh // ih, ow // iw
        def make_tile(th, tw):
            def fn(g):
                h, w = len(g), len(g[0])
                return [[g[r % h][c % w] for c in range(w*tw)] for r in range(h*th)]
            return fn
        pieces.append(WorldPiece(f"wp:tile_{rh}x{rw}", make_tile(rh, rw)))
    
    return pieces

# ═══════════════════════════════════════
# カテゴリ3: オブジェクト操作
# ═══════════════════════════════════════

def _gen_object_transforms(train_pairs):
    pieces = []
    
    # --- 3a: 各オブジェクトを90度回転 ---
    def rotate_each_obj(g):
        bg = _bg(g)
        objs = _objects(g, bg)
        h, w = len(g), len(g[0])
        result = [[bg]*w for _ in range(h)]
        for obj in objs:
            r0, c0, r1, c1 = _bbox(obj)
            oh, ow = r1-r0+1, c1-c0+1
            patch = [[bg]*ow for _ in range(oh)]
            for r, c, v in obj:
                patch[r-r0][c-c0] = v
            # Rotate 90 CW
            rotated = [list(r) for r in zip(*patch[::-1])]
            rh, rw = len(rotated), len(rotated[0])
            # Center rotation at original center
            cr, cc = (r0+r1)//2, (c0+c1)//2
            nr0 = cr - rh//2
            nc0 = cc - rw//2
            for r in range(rh):
                for c in range(rw):
                    nr, nc = nr0+r, nc0+c
                    if 0<=nr<h and 0<=nc<w and rotated[r][c] != bg:
                        result[nr][nc] = rotated[r][c]
        return result
    pieces.append(WorldPiece("wp:rotate_each_obj", rotate_each_obj))
    
    # --- 3b: 各オブジェクトを水平/垂直反転 ---
    def flip_each_obj_h(g):
        bg = _bg(g)
        objs = _objects(g, bg)
        h, w = len(g), len(g[0])
        result = [row[:] for row in g]
        for obj in objs:
            r0, c0, r1, c1 = _bbox(obj)
            for r, c, v in obj:
                result[r][c] = bg
            for r, c, v in obj:
                nc = c1 - (c - c0)
                result[r][nc] = v
        return result
    pieces.append(WorldPiece("wp:flip_each_obj_h", flip_each_obj_h))
    
    def flip_each_obj_v(g):
        bg = _bg(g)
        objs = _objects(g, bg)
        h, w = len(g), len(g[0])
        result = [row[:] for row in g]
        for obj in objs:
            r0, c0, r1, c1 = _bbox(obj)
            for r, c, v in obj:
                result[r][c] = bg
            for r, c, v in obj:
                nr = r1 - (r - r0)
                result[nr][c] = v
        return result
    pieces.append(WorldPiece("wp:flip_each_obj_v", flip_each_obj_v))
    
    # --- 3c: オブジェクトをスタンプ（テンプレートをマーカー位置にコピー）---
    def stamp_template(g):
        """Find template (largest obj) and markers (smallest objs), stamp template at markers."""
        bg = _bg(g)
        objs = _objects(g, bg)
        if len(objs) < 2: return g
        
        # Sort by size
        objs_sorted = sorted(objs, key=len, reverse=True)
        template = objs_sorted[0]
        markers = objs_sorted[1:]
        
        t_patch = _crop_obj(g, template, bg)
        th, tw = len(t_patch), len(t_patch[0])
        
        h, w = len(g), len(g[0])
        result = [row[:] for row in g]
        
        for marker in markers:
            mr = sum(r for r,c,v in marker) // len(marker)
            mc = sum(c for r,c,v in marker) // len(marker)
            # Center template at marker
            r0 = mr - th//2
            c0 = mc - tw//2
            result = _paste(result, t_patch, r0, c0, bg)
        
        return result
    pieces.append(WorldPiece("wp:stamp_template", stamp_template))
    
    # --- 3d: 穴埋め（テンプレートから欠けたセルを補完）---
    def fill_holes_from_template(g):
        """Find repeated pattern, use complete instances to fill incomplete ones."""
        bg = _bg(g)
        objs = _objects(g, bg, connectivity=8)
        if len(objs) < 2: return g
        
        # Group by shape signature (bbox dimensions)
        from collections import defaultdict
        shape_groups = defaultdict(list)
        for obj in objs:
            r0, c0, r1, c1 = _bbox(obj)
            h, w = r1-r0+1, c1-c0+1
            shape_groups[(h,w)].append(obj)
        
        result = [row[:] for row in g]
        
        for (sh, sw), group in shape_groups.items():
            if len(group) < 2: continue
            
            # Find the "most complete" as template
            template = max(group, key=len)
            t_patch = _crop_obj(g, template, bg)
            
            # Fill others from template
            for obj in group:
                if obj is template: continue
                r0, c0, _, _ = _bbox(obj)
                for r in range(sh):
                    for c in range(sw):
                        nr, nc = r0+r, c0+c
                        if 0<=nr<len(g) and 0<=nc<len(g[0]):
                            if result[nr][nc] == bg and t_patch[r][c] != bg:
                                result[nr][nc] = t_patch[r][c]
        return result
    pieces.append(WorldPiece("wp:fill_holes", fill_holes_from_template))
    
    # --- 3e: Boolean AND/OR/XOR of objects ---
    def obj_boolean_and(g):
        bg = _bg(g)
        objs = _objects(g, bg)
        if len(objs) != 2: return None
        h, w = len(g), len(g[0])
        cells1 = set((r,c) for r,c,v in objs[0])
        cells2 = set((r,c) for r,c,v in objs[1])
        # Align by centroid
        c1r = sum(r for r,c in cells1)//len(cells1)
        c1c = sum(c for r,c in cells1)//len(cells1)
        c2r = sum(r for r,c in cells2)//len(cells2)
        c2c = sum(c for r,c in cells2)//len(cells2)
        dr, dc = c2r-c1r, c2c-c1c
        shifted2 = set((r-dr, c-dc) for r,c in cells2)
        
        # AND: cells in both
        both = cells1 & shifted2
        if not both: return None
        result = [[bg]*w for _ in range(h)]
        color = objs[0][0][2]
        for r, c in both:
            if 0<=r<h and 0<=c<w:
                result[r][c] = color
        return result
    pieces.append(WorldPiece("wp:obj_and", obj_boolean_and))
    
    def obj_boolean_xor(g):
        bg = _bg(g)
        objs = _objects(g, bg)
        if len(objs) != 2: return None
        h, w = len(g), len(g[0])
        cells1 = set((r,c) for r,c,v in objs[0])
        cells2 = set((r,c) for r,c,v in objs[1])
        c1r = sum(r for r,c in cells1)//len(cells1)
        c1c = sum(c for r,c in cells1)//len(cells1)
        c2r = sum(r for r,c in cells2)//len(cells2)
        c2c = sum(c for r,c in cells2)//len(cells2)
        dr, dc = c2r-c1r, c2c-c1c
        shifted2 = set((r-dr, c-dc) for r,c in cells2)
        xor = (cells1 | shifted2) - (cells1 & shifted2)
        if not xor: return None
        result = [[bg]*w for _ in range(h)]
        color = objs[0][0][2]
        for r, c in xor:
            if 0<=r<h and 0<=c<w:
                result[r][c] = color
        return result
    pieces.append(WorldPiece("wp:obj_xor", obj_boolean_xor))
    
    return pieces

# ═══════════════════════════════════════
# カテゴリ4: 基本幾何変換
# ═══════════════════════════════════════

def _gen_basic_transforms(train_pairs):
    pieces = []
    pieces.append(WorldPiece("wp:rot90", lambda g: [list(r) for r in zip(*g[::-1])]))
    pieces.append(WorldPiece("wp:rot180", lambda g: [row[::-1] for row in g[::-1]]))
    pieces.append(WorldPiece("wp:rot270", lambda g: [list(r) for r in zip(*[row[::-1] for row in g])]))
    pieces.append(WorldPiece("wp:flip_h", lambda g: [row[::-1] for row in g]))
    pieces.append(WorldPiece("wp:flip_v", lambda g: g[::-1]))
    pieces.append(WorldPiece("wp:transpose", lambda g: [list(r) for r in zip(*g)]))
    return pieces

# ═══════════════════════════════════════
# メイン: 全事前知識を統合
# ═══════════════════════════════════════

def generate_world_prior_pieces(train_pairs):
    """Generate all world-prior candidates for a task."""
    all_pieces = []
    
    all_pieces.extend(_gen_basic_transforms(train_pairs))
    all_pieces.extend(_gen_same_size(train_pairs))
    all_pieces.extend(_gen_size_change(train_pairs))
    all_pieces.extend(_gen_object_transforms(train_pairs))
    
    return all_pieces
