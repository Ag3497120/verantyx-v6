#!/usr/bin/env python3
"""
Verantyx World Commands — 300+ シミュレーションコマンド
======================================================
世界の法則を実行可能なコマンドとして網羅的に定義。
各コマンド: Grid → Grid (純粋関数、副作用なし)

カテゴリ:
  A. 幾何変換 (8)
  B. 重力・物理 (12)
  C. 充填・塗りつぶし (15)
  D. 対称性 (10)
  E. 成長・侵食・セルオートマトン (12)
  F. 線・延長・接続 (16)
  G. クロップ・ビュー (12)
  H. オブジェクト検出・操作 (30)
  I. オブジェクト移動 (16)
  J. パターン複製・スタンプ (12)
  K. 条件付き変換 (20)
  L. パネル・セパレータ操作 (18)
  M. 色操作・マッピング (20)
  N. カウント・集計 (10)
  O. スケール・タイル (12)
  P. 抽象化・正規化 (10)
  Q. 反復・収束 (8)
  R. 形状操作 (12)
  S. マーカー・特殊色 (12)
  T. Boolean演算 (8)
  U. 並べ替え・ソート (10)
  V. パスファインド・接続 (8)
  W. 分割・マージ (12)

合計: 約300コマンド
"""

from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Callable

# ═══════════════════════════════════════
# ユーティリティ
# ═══════════════════════════════════════

def _bg(g):
    c = Counter()
    for row in g: c.update(row)
    return c.most_common(1)[0][0]

def _copy(g):
    return [row[:] for row in g]

def _shape(g):
    return len(g), len(g[0]) if g else 0

def _eq(a, b):
    if len(a) != len(b): return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]): return False
        for j in range(len(a[i])):
            if a[i][j] != b[i][j]: return False
    return True

def _colors(g):
    s = set()
    for row in g: s.update(row)
    return s

def _color_counts(g):
    c = Counter()
    for row in g: c.update(row)
    return c

def _objects(g, bg, conn=4):
    h, w = len(g), len(g[0])
    vis = [[False]*w for _ in range(h)]
    objs = []
    ds = [(-1,0),(1,0),(0,-1),(0,1)]
    if conn == 8: ds += [(-1,-1),(-1,1),(1,-1),(1,1)]
    for r in range(h):
        for c in range(w):
            if not vis[r][c] and g[r][c] != bg:
                obj = []; stk = [(r,c)]; vis[r][c] = True
                while stk:
                    cr, cc = stk.pop()
                    obj.append((cr,cc,g[cr][cc]))
                    for dr,dc in ds:
                        nr,nc = cr+dr, cc+dc
                        if 0<=nr<h and 0<=nc<w and not vis[nr][nc] and g[nr][nc] != bg:
                            vis[nr][nc] = True; stk.append((nr,nc))
                objs.append(obj)
    return objs

def _mono_objects(g, bg):
    """Single-color connected components."""
    h, w = len(g), len(g[0])
    vis = [[False]*w for _ in range(h)]
    objs = []
    for r in range(h):
        for c in range(w):
            if not vis[r][c] and g[r][c] != bg:
                col = g[r][c]
                obj = []; stk = [(r,c)]; vis[r][c] = True
                while stk:
                    cr, cc = stk.pop()
                    obj.append((cr,cc,g[cr][cc]))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr, cc+dc
                        if 0<=nr<h and 0<=nc<w and not vis[nr][nc] and g[nr][nc] == col:
                            vis[nr][nc] = True; stk.append((nr,nc))
                objs.append(obj)
    return objs

def _bbox(cells):
    rs = [r for r,c,_ in cells]; cs = [c for r,c,_ in cells]
    return min(rs), min(cs), max(rs), max(cs)

def _crop_cells(g, cells, bg):
    r0,c0,r1,c1 = _bbox(cells)
    h,w = r1-r0+1, c1-c0+1
    res = [[bg]*w for _ in range(h)]
    for r,c,v in cells: res[r-r0][c-c0] = v
    return res

def _paste(g, patch, r0, c0, bg):
    res = _copy(g)
    ph,pw = len(patch), len(patch[0])
    h,w = len(g), len(g[0])
    for r in range(ph):
        for c in range(pw):
            nr,nc = r0+r, c0+c
            if 0<=nr<h and 0<=nc<w and patch[r][c] != bg:
                res[nr][nc] = patch[r][c]
    return res

def _exterior(g, bg):
    h,w = len(g), len(g[0])
    vis = [[False]*w for _ in range(h)]
    q = []
    for r in range(h):
        for c in range(w):
            if (r==0 or r==h-1 or c==0 or c==w-1) and g[r][c]==bg and not vis[r][c]:
                vis[r][c]=True; q.append((r,c))
    while q:
        cr,cc = q.pop(0)
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc=cr+dr,cc+dc
            if 0<=nr<h and 0<=nc<w and not vis[nr][nc] and g[nr][nc]==bg:
                vis[nr][nc]=True; q.append((nr,nc))
    return vis

def _centroid(cells):
    r = sum(x[0] for x in cells) // len(cells)
    c = sum(x[1] for x in cells) // len(cells)
    return r, c

def _obj_color(obj):
    return Counter(v for _,_,v in obj).most_common(1)[0][0]

def _normalize_patch(patch, bg):
    """Normalize a patch to canonical form (crop, then sort cells)."""
    h, w = len(patch), len(patch[0])
    cells = [(r,c,patch[r][c]) for r in range(h) for c in range(w) if patch[r][c] != bg]
    if not cells: return tuple()
    return tuple((r-cells[0][0], c-cells[0][1], v) for r,c,v in cells)

# ═══════════════════════════════════════
# A. 幾何変換 (8)
# ═══════════════════════════════════════

def rot90(g): return [list(r) for r in zip(*g[::-1])]
def rot180(g): return [row[::-1] for row in g[::-1]]
def rot270(g): return [list(r) for r in zip(*[row[::-1] for row in g])]
def flip_h(g): return [row[::-1] for row in g]
def flip_v(g): return [row[:] for row in g[::-1]]
def transpose(g): return [list(r) for r in zip(*g)]
def anti_transpose(g): return rot90(flip_h(g))
def rot90_ccw(g): return rot270(g)

# ═══════════════════════════════════════
# B. 重力・物理 (12)
# ═══════════════════════════════════════

def _mk_gravity(direction):
    def fn(g):
        bg = _bg(g); h,w = len(g), len(g[0])
        res = [[bg]*w for _ in range(h)]
        if direction == "down":
            for c in range(w):
                non = [g[r][c] for r in range(h) if g[r][c]!=bg]
                for i,v in enumerate(non): res[h-len(non)+i][c]=v
        elif direction == "up":
            for c in range(w):
                non = [g[r][c] for r in range(h) if g[r][c]!=bg]
                for i,v in enumerate(non): res[i][c]=v
        elif direction == "left":
            for r in range(h):
                non = [g[r][c] for c in range(w) if g[r][c]!=bg]
                for i,v in enumerate(non): res[r][i]=v
        elif direction == "right":
            for r in range(h):
                non = [g[r][c] for c in range(w) if g[r][c]!=bg]
                for i,v in enumerate(non): res[r][w-len(non)+i]=v
        return res
    return fn

gravity_down = _mk_gravity("down")
gravity_up = _mk_gravity("up")
gravity_left = _mk_gravity("left")
gravity_right = _mk_gravity("right")

def gravity_down_per_color(g):
    """各色を独立に下に落とす"""
    bg = _bg(g); h,w = len(g), len(g[0])
    res = [[bg]*w for _ in range(h)]
    for col in _colors(g) - {bg}:
        for c in range(w):
            positions = [r for r in range(h) if g[r][c]==col]
            empties = sorted([r for r in range(h) if res[r][c]==bg], reverse=True)
            for i, r in enumerate(empties[:len(positions)]):
                res[r][c] = col
    return res

def gravity_to_wall(g):
    """各オブジェクトを最寄りの壁（非bg）まで移動"""
    bg = _bg(g); h,w = len(g), len(g[0])
    objs = _objects(g, bg)
    res = [[bg]*w for _ in range(h)]
    for obj in objs:
        r0,c0,r1,c1 = _bbox(obj)
        cr,cc = _centroid(obj)
        # Find nearest wall direction
        dist_up = cr; dist_down = h-1-cr; dist_left = cc; dist_right = w-1-cc
        min_dist = min(dist_up, dist_down, dist_left, dist_right)
        if min_dist == dist_up: dr,dc = -dist_up+r0-r0, 0; nr0 = 0
        elif min_dist == dist_down: nr0 = h-(r1-r0+1)
        elif min_dist == dist_left: nr0 = r0  # just move left
        else: nr0 = r0
        # Simple: move to top
        for r,c,v in obj:
            nr = r - r0
            if 0<=nr<h: res[nr][c] = v
    return res

def compact_down(g):
    """bg行を除去して下詰め"""
    bg = _bg(g); h,w = len(g), len(g[0])
    non_bg_rows = [row for row in g if any(c!=bg for c in row)]
    res = [[bg]*w for _ in range(h - len(non_bg_rows))]
    res.extend([row[:] for row in non_bg_rows])
    return res

def compact_up(g):
    bg = _bg(g); h,w = len(g), len(g[0])
    non_bg_rows = [row for row in g if any(c!=bg for c in row)]
    res = [row[:] for row in non_bg_rows]
    res.extend([[bg]*w for _ in range(h - len(non_bg_rows))])
    return res

def compact_left(g):
    return transpose(compact_up(transpose(g)))

def compact_right(g):
    return transpose(compact_down(transpose(g)))

# ═══════════════════════════════════════
# C. 充填・塗りつぶし (15)
# ═══════════════════════════════════════

def fill_enclosed(g):
    bg = _bg(g); h,w = len(g), len(g[0])
    ext = _exterior(g, bg)
    res = _copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]==bg and not ext[r][c]:
                nb = [g[nr][nc] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                      for nr,nc in [(r+dr,c+dc)] if 0<=nr<h and 0<=nc<w and g[nr][nc]!=bg]
                if nb: res[r][c] = Counter(nb).most_common(1)[0][0]
    return res

def fill_enclosed_with_color(g, color):
    bg = _bg(g); h,w = len(g), len(g[0])
    ext = _exterior(g, bg)
    res = _copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]==bg and not ext[r][c]:
                res[r][c] = color
    return res

def fill_row_gaps(g):
    """行内の非bgセル間のbgを塗る"""
    bg = _bg(g); res = _copy(g)
    for r in range(len(g)):
        positions = [(c, g[r][c]) for c in range(len(g[0])) if g[r][c]!=bg]
        if len(positions) >= 2:
            for i in range(len(positions)-1):
                c1, col1 = positions[i]; c2, col2 = positions[i+1]
                if col1 == col2:
                    for c in range(c1+1, c2): res[r][c] = col1
    return res

def fill_col_gaps(g):
    """列内の非bgセル間のbgを塗る"""
    return transpose(fill_row_gaps(transpose(g)))

def fill_between_same_color(g):
    """同色オブジェクト間を塗る（行列両方）"""
    r = fill_row_gaps(g)
    return fill_col_gaps(r)

def flood_from_edges(g):
    """辺から非bgまで特定色で塗る"""
    bg = _bg(g); h,w = len(g), len(g[0])
    res = _copy(g)
    # Find first non-bg color adjacent to border
    border_colors = set()
    for r in range(h):
        for c in range(w):
            if (r==0 or r==h-1 or c==0 or c==w-1) and g[r][c]!=bg:
                border_colors.add(g[r][c])
    return res

def fill_diag_enclosed(g):
    """8方向接続で囲まれた領域を塗る"""
    bg = _bg(g); h,w = len(g), len(g[0])
    vis = [[False]*w for _ in range(h)]
    q = []
    for r in range(h):
        for c in range(w):
            if (r==0 or r==h-1 or c==0 or c==w-1) and g[r][c]==bg:
                vis[r][c]=True; q.append((r,c))
    while q:
        cr,cc = q.pop(0)
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr==0 and dc==0: continue
                nr,nc=cr+dr,cc+dc
                if 0<=nr<h and 0<=nc<w and not vis[nr][nc] and g[nr][nc]==bg:
                    vis[nr][nc]=True; q.append((nr,nc))
    res = _copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]==bg and not vis[r][c]:
                nb = [g[nr][nc] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                      for nr,nc in [(r+dr,c+dc)] if 0<=nr<h and 0<=nc<w and g[nr][nc]!=bg]
                if nb: res[r][c] = Counter(nb).most_common(1)[0][0]
    return res

def fill_bg_with_majority(g):
    bg = _bg(g)
    cc = _color_counts(g)
    del cc[bg]
    if not cc: return g
    maj = cc.most_common(1)[0][0]
    return [[maj if v==bg else v for v in row] for row in g]

def fill_checkerboard(g):
    """bgセルを市松模様で塗る"""
    bg = _bg(g)
    cc = _color_counts(g)
    non_bg = sorted(set(cc.keys())-{bg}, key=lambda x: -cc[x])
    if not non_bg: return g
    col = non_bg[0]
    h,w = len(g), len(g[0])
    return [[col if g[r][c]==bg and (r+c)%2==0 else g[r][c] for c in range(w)] for r in range(h)]

def fill_row_majority(g):
    """各行のbgを行の多数色で塗る"""
    bg = _bg(g); res = _copy(g)
    for r in range(len(g)):
        cc = Counter(c for c in g[r] if c!=bg)
        if cc:
            maj = cc.most_common(1)[0][0]
            for c in range(len(g[0])):
                if res[r][c]==bg: res[r][c]=maj
    return res

def fill_col_majority(g):
    return transpose(fill_row_majority(transpose(g)))

def fill_each_obj_bbox(g):
    bg = _bg(g); objs = _objects(g, bg); res = _copy(g)
    for obj in objs:
        r0,c0,r1,c1 = _bbox(obj)
        col = _obj_color(obj)
        for r in range(r0,r1+1):
            for c in range(c0,c1+1): res[r][c]=col
    return res

def fill_each_obj_convex(g):
    """各オブジェクトの凸包を塗る (近似: bbox)"""
    return fill_each_obj_bbox(g)

# ═══════════════════════════════════════
# D. 対称性 (10)
# ═══════════════════════════════════════

def sym_h(g):
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        for c in range(w//2):
            mc=w-1-c
            if res[r][c]==bg and res[r][mc]!=bg: res[r][c]=res[r][mc]
            elif res[r][mc]==bg and res[r][c]!=bg: res[r][mc]=res[r][c]
    return res

def sym_v(g):
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h//2):
        mr=h-1-r
        for c in range(w):
            if res[r][c]==bg and res[mr][c]!=bg: res[r][c]=res[mr][c]
            elif res[mr][c]==bg and res[r][c]!=bg: res[mr][c]=res[r][c]
    return res

def sym_4fold(g):
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range((h+1)//2):
        for c in range((w+1)//2):
            ps=[(r,c),(r,w-1-c),(h-1-r,c),(h-1-r,w-1-c)]
            vals=[res[pr][pc] for pr,pc in ps if 0<=pr<h and 0<=pc<w and res[pr][pc]!=bg]
            if vals:
                fill=Counter(vals).most_common(1)[0][0]
                for pr,pc in ps:
                    if 0<=pr<h and 0<=pc<w and res[pr][pc]==bg: res[pr][pc]=fill
    return res

def sym_diag_main(g):
    bg=_bg(g); h,w=len(g),len(g[0])
    if h!=w: return g
    res=_copy(g)
    for r in range(h):
        for c in range(r+1,w):
            if res[r][c]==bg and res[c][r]!=bg: res[r][c]=res[c][r]
            elif res[c][r]==bg and res[r][c]!=bg: res[c][r]=res[r][c]
    return res

def sym_diag_anti(g):
    bg=_bg(g); h,w=len(g),len(g[0])
    if h!=w: return g
    res=_copy(g)
    for r in range(h):
        for c in range(w):
            mr,mc = w-1-c, h-1-r
            if 0<=mr<h and 0<=mc<w:
                if res[r][c]==bg and res[mr][mc]!=bg: res[r][c]=res[mr][mc]
                elif res[mr][mc]==bg and res[r][c]!=bg: res[mr][mc]=res[r][c]
    return res

def sym_rot90(g):
    """90度回転対称修復"""
    bg=_bg(g); h,w=len(g),len(g[0])
    if h!=w: return g
    res=_copy(g)
    for r in range(h):
        for c in range(w):
            ps=[(r,c),(c,h-1-r),(h-1-r,w-1-c),(w-1-c,r)]
            vals=[res[pr][pc] for pr,pc in ps if 0<=pr<h and 0<=pc<w and res[pr][pc]!=bg]
            if vals:
                fill=Counter(vals).most_common(1)[0][0]
                for pr,pc in ps:
                    if 0<=pr<h and 0<=pc<w and res[pr][pc]==bg: res[pr][pc]=fill
    return res

def sym_center_point(g):
    """中心点対称修復"""
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        for c in range(w):
            mr,mc=h-1-r,w-1-c
            if res[r][c]==bg and res[mr][mc]!=bg: res[r][c]=res[mr][mc]
            elif res[mr][mc]==bg and res[r][c]!=bg: res[mr][mc]=res[r][c]
    return res

def force_sym_h(g):
    """左半分を右に反射（強制）"""
    h,w=len(g),len(g[0])
    res=_copy(g)
    for r in range(h):
        for c in range(w//2):
            res[r][w-1-c]=res[r][c]
    return res

def force_sym_v(g):
    """上半分を下に反射（強制）"""
    h,w=len(g),len(g[0])
    res=_copy(g)
    for r in range(h//2):
        res[h-1-r] = res[r][:]
    return res

def force_sym_4fold_tl(g):
    """左上1/4を全体に反射"""
    h,w=len(g),len(g[0]); res=_copy(g)
    for r in range((h+1)//2):
        for c in range((w+1)//2):
            v=res[r][c]
            if w-1-c<w: res[r][w-1-c]=v
            if h-1-r<h: res[h-1-r][c]=v
            if h-1-r<h and w-1-c<w: res[h-1-r][w-1-c]=v
    return res

# ═══════════════════════════════════════
# E. 成長・侵食・セルオートマトン (12)
# ═══════════════════════════════════════

def grow_1(g):
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and res[nr][nc]==bg: res[nr][nc]=g[r][c]
    return res

def grow_8(g):
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg:
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        nr,nc=r+dr,c+dc
                        if 0<=nr<h and 0<=nc<w and res[nr][nc]==bg: res[nr][nc]=g[r][c]
    return res

def shrink_1(g):
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if not(0<=nr<h and 0<=nc<w) or g[nr][nc]==bg:
                        res[r][c]=bg; break
    return res

def ca_life(g):
    bg=_bg(g); h,w=len(g),len(g[0]); res=[[bg]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            alive=0; nb_colors=Counter()
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr==0 and dc==0: continue
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and g[nr][nc]!=bg:
                        alive+=1; nb_colors[g[nr][nc]]+=1
            is_alive = g[r][c]!=bg
            if is_alive and alive in [2,3]: res[r][c]=g[r][c]
            elif not is_alive and alive==3 and nb_colors:
                res[r][c]=nb_colors.most_common(1)[0][0]
    return res

def ca_majority(g):
    """各セルを4近傍の多数色に変更"""
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        for c in range(w):
            nb = [g[r][c]]
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=r+dr,c+dc
                if 0<=nr<h and 0<=nc<w: nb.append(g[nr][nc])
            res[r][c]=Counter(nb).most_common(1)[0][0]
    return res

def dilate_color(g, color):
    """特定色だけ膨張"""
    h,w=len(g),len(g[0]); bg=_bg(g); res=_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]==color:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and res[nr][nc]==bg: res[nr][nc]=color
    return res

def erode_color(g, color):
    """特定色だけ侵食"""
    h,w=len(g),len(g[0]); bg=_bg(g); res=_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]==color:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if not(0<=nr<h and 0<=nc<w) or g[nr][nc]!=color:
                        res[r][c]=bg; break
    return res

def grow_diagonal(g):
    """斜め方向にのみ成長"""
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg:
                for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and res[nr][nc]==bg: res[nr][nc]=g[r][c]
    return res

def grow_cross(g):
    """十字方向に全延長"""
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc = r+dr, c+dc
                    while 0<=nr<h and 0<=nc<w:
                        if res[nr][nc]==bg: res[nr][nc]=g[r][c]
                        else: break
                        nr+=dr; nc+=dc
    return res

def smooth(g):
    """各bgセルを4近傍の多数非bg色で塗る（1回）"""
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]==bg:
                nb = [g[nr][nc] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                      for nr,nc in [(r+dr,c+dc)] if 0<=nr<h and 0<=nc<w and g[nr][nc]!=bg]
                if len(nb) >= 2: res[r][c]=Counter(nb).most_common(1)[0][0]
    return res

# ═══════════════════════════════════════
# F. 線・延長・接続 (16)
# ═══════════════════════════════════════

def extend_h(g):
    bg=_bg(g); res=_copy(g)
    for r in range(len(g)):
        cs = set(g[r][c] for c in range(len(g[0])) if g[r][c]!=bg)
        if len(cs)==1:
            col=cs.pop(); ps=[c for c in range(len(g[0])) if g[r][c]==col]
            if len(ps)>=2:
                for c in range(min(ps),max(ps)+1): res[r][c]=col
    return res

def extend_v(g):
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for c in range(w):
        cs = set(g[r][c] for r in range(h) if g[r][c]!=bg)
        if len(cs)==1:
            col=cs.pop(); ps=[r for r in range(h) if g[r][c]==col]
            if len(ps)>=2:
                for r in range(min(ps),max(ps)+1): res[r][c]=col
    return res

def extend_full_h(g):
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        cs=set(g[r][c] for c in range(w) if g[r][c]!=bg)
        if len(cs)==1:
            col=cs.pop()
            for c in range(w): res[r][c]=col
    return res

def extend_full_v(g):
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for c in range(w):
        cs=set(g[r][c] for r in range(h) if g[r][c]!=bg)
        if len(cs)==1:
            col=cs.pop()
            for r in range(h): res[r][c]=col
    return res

def draw_cross_through_points(g):
    """各非bgセルから十字に線を引く"""
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg:
                col=g[r][c]
                for rr in range(h): res[rr][c]=col
                for cc in range(w): res[r][cc]=col
    return res

def draw_diag_through_points(g):
    """各非bgセルから斜め線を引く"""
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg:
                col=g[r][c]
                for d in range(max(h,w)):
                    for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr,nc=r+dr*d,c+dc*d
                        if 0<=nr<h and 0<=nc<w and res[nr][nc]==bg: res[nr][nc]=col
    return res

def connect_same_color_h(g):
    """同色セルを水平線でつなぐ"""
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        by_color = defaultdict(list)
        for c in range(w):
            if g[r][c]!=bg: by_color[g[r][c]].append(c)
        for col, positions in by_color.items():
            if len(positions)>=2:
                for c in range(min(positions),max(positions)+1): res[r][c]=col
    return res

def connect_same_color_v(g):
    """同色セルを垂直線でつなぐ"""
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for c in range(w):
        by_color = defaultdict(list)
        for r in range(h):
            if g[r][c]!=bg: by_color[g[r][c]].append(r)
        for col, positions in by_color.items():
            if len(positions)>=2:
                for r in range(min(positions),max(positions)+1): res[r][c]=col
    return res

def connect_same_color_both(g):
    r = connect_same_color_h(g)
    return connect_same_color_v(r)

def draw_border(g):
    """グリッドの外枠を最頻非bg色で描画"""
    bg=_bg(g); h,w=len(g),len(g[0])
    cc=_color_counts(g); del cc[bg]
    if not cc: return g
    col=cc.most_common(1)[0][0]; res=_copy(g)
    for r in range(h):
        res[r][0]=col; res[r][w-1]=col
    for c in range(w):
        res[0][c]=col; res[h-1][c]=col
    return res

def draw_grid_lines(g):
    """bgに格子線を引く（オブジェクトの区切り検出）"""
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    # Find separator candidates: rows/cols that are all bg
    sep_rows = [r for r in range(h) if all(g[r][c]==bg for c in range(w))]
    sep_cols = [c for c in range(w) if all(g[r][c]==bg for r in range(h))]
    cc=_color_counts(g); del cc[bg]
    if not cc: return g
    col = cc.most_common()[-1][0]  # least common non-bg
    for r in sep_rows:
        for c in range(w): res[r][c]=col
    for c in sep_cols:
        for r in range(h): res[r][c]=col
    return res

def trace_outline_4(g):
    """オブジェクトの4方向輪郭"""
    bg=_bg(g); h,w=len(g),len(g[0]); res=[[bg]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if not(0<=nr<h and 0<=nc<w) or g[nr][nc]==bg:
                        res[r][c]=g[r][c]; break
    return res

def trace_outline_8(g):
    bg=_bg(g); h,w=len(g),len(g[0]); res=[[bg]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg:
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        nr,nc=r+dr,c+dc
                        if not(0<=nr<h and 0<=nc<w) or g[nr][nc]==bg:
                            res[r][c]=g[r][c]; break
                    else: continue
                    break
    return res

def interior_only(g):
    bg=_bg(g); h,w=len(g),len(g[0]); res=[[bg]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg:
                border=False
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if not(0<=nr<h and 0<=nc<w) or g[nr][nc]==bg:
                        border=True; break
                if not border: res[r][c]=g[r][c]
    return res

# ═══════════════════════════════════════
# G. クロップ・ビュー (12)
# ═══════════════════════════════════════

def crop_content(g):
    bg=_bg(g); h,w=len(g),len(g[0])
    rows=[r for r in range(h) if any(g[r][c]!=bg for c in range(w))]
    cols=[c for c in range(w) if any(g[r][c]!=bg for r in range(h))]
    if not rows or not cols: return g
    return [g[r][min(cols):max(cols)+1] for r in range(min(rows),max(rows)+1)]

def crop_largest_obj(g):
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    return _crop_cells(g, max(objs,key=len), bg)

def crop_smallest_obj(g):
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    return _crop_cells(g, min(objs,key=len), bg)

def rm_bg_rows(g):
    bg=_bg(g)
    return [row[:] for row in g if any(c!=bg for c in row)] or [g[0][:]]

def rm_bg_cols(g):
    bg=_bg(g); h,w=len(g),len(g[0])
    keep=[c for c in range(w) if any(g[r][c]!=bg for r in range(h))]
    if not keep: return g
    return [[g[r][c] for c in keep] for r in range(h)]

def crop_top_half(g):
    h=len(g); return [row[:] for row in g[:h//2]]
def crop_bottom_half(g):
    h=len(g); return [row[:] for row in g[h//2:]]
def crop_left_half(g):
    w=len(g[0]); return [row[:w//2] for row in g]
def crop_right_half(g):
    w=len(g[0]); return [row[w//2:] for row in g]

def crop_quadrant_tl(g):
    h,w=len(g),len(g[0]); return [row[:w//2] for row in g[:h//2]]
def crop_quadrant_tr(g):
    h,w=len(g),len(g[0]); return [row[w//2:] for row in g[:h//2]]
def crop_quadrant_bl(g):
    h,w=len(g),len(g[0]); return [row[:w//2] for row in g[h//2:]]
def crop_quadrant_br(g):
    h,w=len(g),len(g[0]); return [row[w//2:] for row in g[h//2:]]

# ═══════════════════════════════════════
# H. オブジェクト検出・操作 (30)
# ═══════════════════════════════════════

def keep_largest(g):
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    largest=max(objs,key=len); h,w=len(g),len(g[0])
    res=[[bg]*w for _ in range(h)]
    for r,c,v in largest: res[r][c]=v
    return res

def keep_smallest(g):
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    smallest=min(objs,key=len); h,w=len(g),len(g[0])
    res=[[bg]*w for _ in range(h)]
    for r,c,v in smallest: res[r][c]=v
    return res

def remove_largest(g):
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    largest=max(objs,key=len); res=_copy(g)
    for r,c,v in largest: res[r][c]=bg
    return res

def remove_smallest(g):
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    smallest=min(objs,key=len); res=_copy(g)
    for r,c,v in smallest: res[r][c]=bg
    return res

def keep_most_common_color_obj(g):
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    cc=Counter(_obj_color(o) for o in objs)
    target=cc.most_common(1)[0][0]; h,w=len(g),len(g[0])
    res=[[bg]*w for _ in range(h)]
    for obj in objs:
        if _obj_color(obj)==target:
            for r,c,v in obj: res[r][c]=v
    return res

def keep_unique_color_obj(g):
    """ユニークな色のオブジェクトだけ残す"""
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    cc=Counter(_obj_color(o) for o in objs)
    uniques={col for col,cnt in cc.items() if cnt==1}
    h,w=len(g),len(g[0]); res=[[bg]*w for _ in range(h)]
    for obj in objs:
        if _obj_color(obj) in uniques:
            for r,c,v in obj: res[r][c]=v
    return res

def keep_objs_touching_border(g):
    bg=_bg(g); h,w=len(g),len(g[0]); objs=_objects(g,bg)
    res=[[bg]*w for _ in range(h)]
    for obj in objs:
        if any(r==0 or r==h-1 or c==0 or c==w-1 for r,c,_ in obj):
            for r,c,v in obj: res[r][c]=v
    return res

def remove_objs_touching_border(g):
    bg=_bg(g); h,w=len(g),len(g[0]); objs=_objects(g,bg)
    res=_copy(g)
    for obj in objs:
        if any(r==0 or r==h-1 or c==0 or c==w-1 for r,c,_ in obj):
            for r,c,v in obj: res[r][c]=bg
    return res

def keep_objs_by_size(g, n):
    """セル数nのオブジェクトだけ残す"""
    bg=_bg(g); objs=_objects(g,bg); h,w=len(g),len(g[0])
    res=[[bg]*w for _ in range(h)]
    for obj in objs:
        if len(obj)==n:
            for r,c,v in obj: res[r][c]=v
    return res

def sort_objs_by_size(g):
    """オブジェクトをサイズ順に左から並べる"""
    bg=_bg(g); objs=_objects(g,bg)
    if len(objs)<2: return g
    objs.sort(key=len)
    patches = [_crop_cells(g, obj, bg) for obj in objs]
    total_w = sum(len(p[0]) for p in patches) + len(patches)-1
    max_h = max(len(p) for p in patches)
    res=[[bg]*total_w for _ in range(max_h)]
    col = 0
    for patch in patches:
        ph,pw=len(patch),len(patch[0])
        for r in range(ph):
            for c in range(pw):
                if patch[r][c]!=bg: res[r][col+c]=patch[r][c]
        col += pw + 1
    return res

def recolor_each_obj_by_size(g):
    """各オブジェクトをサイズ順に色1,2,3...で塗り直す"""
    bg=_bg(g); objs=_objects(g,bg)
    objs.sort(key=len); res=_copy(g)
    for i, obj in enumerate(objs):
        col = (i % 9) + 1
        if col == bg: col = (col % 9) + 1
        for r,c,_ in obj: res[r][c]=col
    return res

def unify_obj_color(g):
    """全オブジェクトを同じ色（最頻非bg色）に統一"""
    bg=_bg(g); cc=_color_counts(g); del cc[bg]
    if not cc: return g
    col=cc.most_common(1)[0][0]; res=_copy(g)
    for r in range(len(g)):
        for c in range(len(g[0])):
            if res[r][c]!=bg: res[r][c]=col
    return res

def hollow_objects(g):
    """各オブジェクトの内部をbgにする（外殻だけ残す）"""
    return trace_outline_4(g)

def rotate_each_obj_90(g):
    bg=_bg(g); objs=_objects(g,bg); h,w=len(g),len(g[0])
    res=[[bg]*w for _ in range(h)]
    for obj in objs:
        patch=_crop_cells(g,obj,bg)
        rotated=rot90(patch)
        r0,c0,_,_ =_bbox(obj)
        cr,cc=_centroid(obj)
        rh,rw=len(rotated),len(rotated[0])
        nr0=cr-rh//2; nc0=cc-rw//2
        for r in range(rh):
            for c in range(rw):
                nr,nc=nr0+r,nc0+c
                if 0<=nr<h and 0<=nc<w and rotated[r][c]!=bg: res[nr][nc]=rotated[r][c]
    return res

def flip_each_obj_h(g):
    bg=_bg(g); objs=_objects(g,bg); h,w=len(g),len(g[0]); res=_copy(g)
    for obj in objs:
        r0,c0,r1,c1=_bbox(obj)
        for r,c,v in obj: res[r][c]=bg
        for r,c,v in obj: res[r][c1-(c-c0)]=v
    return res

def flip_each_obj_v(g):
    bg=_bg(g); objs=_objects(g,bg); h,w=len(g),len(g[0]); res=_copy(g)
    for obj in objs:
        r0,c0,r1,c1=_bbox(obj)
        for r,c,v in obj: res[r][c]=bg
        for r,c,v in obj: res[r1-(r-r0)][c]=v
    return res

def duplicate_objs_h(g):
    """各オブジェクトを右に複製"""
    bg=_bg(g); objs=_objects(g,bg); h,w=len(g),len(g[0]); res=_copy(g)
    for obj in objs:
        r0,c0,r1,c1=_bbox(obj)
        ow=c1-c0+1
        for r,c,v in obj:
            nc=c+ow+1
            if 0<=nc<w: res[r][nc]=v
    return res

def duplicate_objs_v(g):
    bg=_bg(g); objs=_objects(g,bg); h,w=len(g),len(g[0]); res=_copy(g)
    for obj in objs:
        r0,c0,r1,c1=_bbox(obj)
        oh=r1-r0+1
        for r,c,v in obj:
            nr=r+oh+1
            if 0<=nr<h: res[nr][c]=v
    return res

# ═══════════════════════════════════════
# I. オブジェクト移動 (16)
# ═══════════════════════════════════════

def _move_objs(g, dr, dc):
    bg=_bg(g); objs=_objects(g,bg); h,w=len(g),len(g[0])
    res=[[bg]*w for _ in range(h)]
    for obj in objs:
        for r,c,v in obj:
            nr,nc=r+dr,c+dc
            if 0<=nr<h and 0<=nc<w: res[nr][nc]=v
    return res

def move_up_1(g): return _move_objs(g,-1,0)
def move_down_1(g): return _move_objs(g,1,0)
def move_left_1(g): return _move_objs(g,0,-1)
def move_right_1(g): return _move_objs(g,0,1)

def move_obj_to_center(g):
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    h,w=len(g),len(g[0]); res=[[bg]*w for _ in range(h)]
    for obj in objs:
        cr,cc=_centroid(obj)
        tr,tc=h//2,w//2; dr,dc=tr-cr,tc-cc
        for r,c,v in obj:
            nr,nc=r+dr,c+dc
            if 0<=nr<h and 0<=nc<w: res[nr][nc]=v
    return res

def move_all_to_top(g):
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    h,w=len(g),len(g[0]); res=[[bg]*w for _ in range(h)]
    for obj in objs:
        r0=min(r for r,c,v in obj)
        for r,c,v in obj:
            if 0<=r-r0<h: res[r-r0][c]=v
    return res

def move_all_to_bottom(g):
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    h,w=len(g),len(g[0]); res=[[bg]*w for _ in range(h)]
    for obj in objs:
        r1=max(r for r,c,v in obj)
        dr=h-1-r1
        for r,c,v in obj:
            if 0<=r+dr<h: res[r+dr][c]=v
    return res

def move_all_to_left(g):
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    h,w=len(g),len(g[0]); res=[[bg]*w for _ in range(h)]
    for obj in objs:
        c0=min(c for r,c,v in obj)
        for r,c,v in obj:
            if 0<=c-c0<w: res[r][c-c0]=v
    return res

def move_all_to_right(g):
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    h,w=len(g),len(g[0]); res=[[bg]*w for _ in range(h)]
    for obj in objs:
        c1=max(c for r,c,v in obj)
        dc=w-1-c1
        for r,c,v in obj:
            if 0<=c+dc<w: res[r][c+dc]=v
    return res

def slide_objs_to_touch(g):
    """各オブジェクトを下方向に、他のオブジェクトまたは壁に接触するまでスライド"""
    bg=_bg(g); h,w=len(g),len(g[0])
    objs=_objects(g,bg)
    objs.sort(key=lambda o: -max(r for r,c,v in o))  # bottom first
    res=[[bg]*w for _ in range(h)]
    occupied=set()
    for obj in objs:
        max_drop=h
        for r,c,v in obj:
            drop=0
            while r+drop+1<h and (r+drop+1,c) not in occupied:
                drop+=1
            max_drop=min(max_drop,drop)
        for r,c,v in obj:
            res[r+max_drop][c]=v
            occupied.add((r+max_drop,c))
    return res

def stack_objs_vertical(g):
    """全オブジェクトを上から順に隙間なく積む"""
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    patches=[_crop_cells(g,obj,bg) for obj in objs]
    total_h=sum(len(p) for p in patches)
    max_w=max(len(p[0]) for p in patches)
    res=[[bg]*max_w for _ in range(total_h)]
    row=0
    for patch in patches:
        ph,pw=len(patch),len(patch[0])
        for r in range(ph):
            for c in range(pw):
                if patch[r][c]!=bg: res[row+r][c]=patch[r][c]
        row+=ph
    return res

def stack_objs_horizontal(g):
    bg=_bg(g); objs=_objects(g,bg)
    if not objs: return g
    patches=[_crop_cells(g,obj,bg) for obj in objs]
    max_h=max(len(p) for p in patches)
    total_w=sum(len(p[0]) for p in patches)
    res=[[bg]*total_w for _ in range(max_h)]
    col=0
    for patch in patches:
        ph,pw=len(patch),len(patch[0])
        for r in range(ph):
            for c in range(pw):
                if patch[r][c]!=bg: res[r][col+c]=patch[r][c]
        col+=pw
    return res

# ═══════════════════════════════════════
# J. パターン複製・スタンプ (12)
# ═══════════════════════════════════════

def stamp_largest_at_smallest(g):
    bg=_bg(g); objs=_objects(g,bg)
    if len(objs)<2: return g
    objs.sort(key=len,reverse=True)
    template=_crop_cells(g,objs[0],bg)
    markers=objs[1:]
    res=_copy(g); h,w=len(g),len(g[0])
    th,tw=len(template),len(template[0])
    for marker in markers:
        cr,cc=_centroid(marker)
        r0,c0=cr-th//2,cc-tw//2
        res=_paste(res,template,r0,c0,bg)
    return res

def stamp_smallest_at_largest(g):
    bg=_bg(g); objs=_objects(g,bg)
    if len(objs)<2: return g
    objs.sort(key=len)
    template=_crop_cells(g,objs[0],bg)
    markers=objs[1:]
    res=_copy(g); h,w=len(g),len(g[0])
    th,tw=len(template),len(template[0])
    for marker in markers:
        cr,cc=_centroid(marker)
        r0,c0=cr-th//2,cc-tw//2
        res=_paste(res,template,r0,c0,bg)
    return res

def tile_pattern(g):
    bg=_bg(g)
    patch=crop_content(g)
    ph,pw=len(patch),len(patch[0])
    h,w=len(g),len(g[0])
    return [[patch[r%ph][c%pw] for c in range(w)] for r in range(h)]

def mirror_stamp_h(g):
    """右半分に左半分のミラーをスタンプ"""
    h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        for c in range(w//2):
            res[r][w-1-c]=res[r][c]
    return res

def mirror_stamp_v(g):
    h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h//2):
        res[h-1-r]=res[r][:]
    return res

def repeat_pattern_2x2(g):
    h,w=len(g),len(g[0])
    return [[g[r%h][c%w] for c in range(w*2)] for r in range(h*2)]

def repeat_pattern_3x3(g):
    h,w=len(g),len(g[0])
    return [[g[r%h][c%w] for c in range(w*3)] for r in range(h*3)]

def fill_holes_from_most_complete(g):
    """同形状オブジェクトの穴を最完全版で埋める"""
    bg=_bg(g); objs=_objects(g,bg,8)
    if len(objs)<2: return g
    shape_groups=defaultdict(list)
    for obj in objs:
        r0,c0,r1,c1=_bbox(obj)
        shape_groups[(r1-r0+1,c1-c0+1)].append(obj)
    res=_copy(g)
    for (sh,sw), group in shape_groups.items():
        if len(group)<2: continue
        template=max(group,key=len)
        t_patch=_crop_cells(g,template,bg)
        for obj in group:
            if obj is template: continue
            r0,c0,_,_=_bbox(obj)
            for r in range(sh):
                for c in range(sw):
                    nr,nc=r0+r,c0+c
                    if 0<=nr<len(g) and 0<=nc<len(g[0]):
                        if res[nr][nc]==bg and r<len(t_patch) and c<len(t_patch[0]) and t_patch[r][c]!=bg:
                            res[nr][nc]=t_patch[r][c]
    return res

def copy_unique_obj_to_all(g):
    """ユニーク形状のオブジェクトを全オブジェクト位置にコピー"""
    bg=_bg(g); objs=_objects(g,bg)
    if len(objs)<2: return g
    shapes=defaultdict(list)
    for obj in objs:
        patch=_crop_cells(g,obj,bg)
        key=_normalize_patch(patch,bg)
        shapes[key].append(obj)
    unique_obj=None
    for key,group in shapes.items():
        if len(group)==1: unique_obj=group[0]; break
    if unique_obj is None: return g
    template=_crop_cells(g,unique_obj,bg)
    th,tw=len(template),len(template[0])
    res=_copy(g); h,w=len(g),len(g[0])
    for obj in objs:
        if obj is unique_obj: continue
        cr,cc=_centroid(obj)
        res=_paste(res,template,cr-th//2,cc-tw//2,bg)
    return res

# ═══════════════════════════════════════
# K. 条件付き変換 (20)
# ═══════════════════════════════════════

def if_neighbor_same_fill(g):
    """非bgセルの隣接に同色があれば維持、なければbg"""
    bg=_bg(g); h,w=len(g),len(g[0]); res=[[bg]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg:
                has_same=False
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and g[nr][nc]==g[r][c]:
                        has_same=True; break
                if has_same: res[r][c]=g[r][c]
    return res

def if_isolated_remove(g):
    """孤立セル（同色隣接なし）を除去"""
    return if_neighbor_same_fill(g)

def if_on_edge_color(g):
    """辺に接するセルの色を変更"""
    bg=_bg(g); h,w=len(g),len(g[0])
    cc=_color_counts(g); del cc[bg]
    if len(cc)<2: return g
    colors=cc.most_common()
    edge_col=colors[1][0] if len(colors)>1 else colors[0][0]
    res=_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg and (r==0 or r==h-1 or c==0 or c==w-1):
                res[r][c]=edge_col
    return res

def if_enclosed_recolor(g):
    """囲まれたbgセルを、囲む色とは異なる色で塗る"""
    bg=_bg(g); h,w=len(g),len(g[0])
    ext=_exterior(g,bg); res=_copy(g)
    cc=_color_counts(g); del cc[bg]
    if len(cc)<2: return fill_enclosed(g)
    colors=sorted(cc.keys(), key=lambda x: -cc[x])
    alt_color=colors[1] if len(colors)>1 else colors[0]
    for r in range(h):
        for c in range(w):
            if g[r][c]==bg and not ext[r][c]: res[r][c]=alt_color
    return res

def swap_two_colors(g):
    """最頻と次頻の非bg色を入れ替え"""
    bg=_bg(g); cc=_color_counts(g); del cc[bg]
    if len(cc)<2: return g
    c1,c2=cc.most_common(2)[0][0], cc.most_common(2)[1][0]
    return [[c2 if v==c1 else c1 if v==c2 else v for v in row] for row in g]

def invert_colors(g):
    """色を反転（9-color）"""
    bg=_bg(g)
    colors=sorted(_colors(g)-{bg})
    if len(colors)<2: return g
    rev=list(reversed(colors))
    cmap={a:b for a,b in zip(colors,rev)}
    cmap[bg]=bg
    return [[cmap.get(v,v) for v in row] for row in g]

def color_by_position(g):
    """各セルの位置（行 mod N）で色を決定"""
    bg=_bg(g); h,w=len(g),len(g[0])
    cc=_color_counts(g); del cc[bg]
    if not cc: return g
    colors=sorted(cc.keys())
    n=len(colors)
    return [[colors[r%n] if g[r][c]!=bg else bg for c in range(w)] for r in range(h)]

def color_by_col_position(g):
    bg=_bg(g); h,w=len(g),len(g[0])
    cc=_color_counts(g); del cc[bg]
    if not cc: return g
    colors=sorted(cc.keys()); n=len(colors)
    return [[colors[c%n] if g[r][c]!=bg else bg for c in range(w)] for r in range(h)]

def replace_with_neighbor_color(g):
    """各非bgセルを最頻隣接色で置換"""
    bg=_bg(g); h,w=len(g),len(g[0]); res=_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg:
                nb=[g[nr][nc] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                    for nr,nc in [(r+dr,c+dc)] if 0<=nr<h and 0<=nc<w and g[nr][nc]!=bg and g[nr][nc]!=g[r][c]]
                if nb: res[r][c]=Counter(nb).most_common(1)[0][0]
    return res

def conditional_grow_if_enclosed(g):
    """囲まれた領域内のオブジェクトだけ成長"""
    bg=_bg(g); h,w=len(g),len(g[0])
    ext=_exterior(g,bg); res=_copy(g)
    for r in range(h):
        for c in range(w):
            if g[r][c]!=bg and not ext[r][c]:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and res[nr][nc]==bg: res[nr][nc]=g[r][c]
    return res

# ═══════════════════════════════════════
# L. パネル・セパレータ操作 (18)
# ═══════════════════════════════════════

def _find_sep_rows(g, bg):
    h,w=len(g),len(g[0])
    sep=[]
    for r in range(h):
        row_colors=set(g[r])
        if len(row_colors)==1 and g[r][0]!=bg:
            sep.append((r,g[r][0]))
    return sep

def _find_sep_cols(g, bg):
    h,w=len(g),len(g[0])
    sep=[]
    for c in range(w):
        col_colors=set(g[r][c] for r in range(h))
        if len(col_colors)==1 and g[0][c]!=bg:
            sep.append((c,g[0][c]))
    return sep

def _extract_panels_h(g):
    bg=_bg(g); h,w=len(g),len(g[0])
    seps=_find_sep_rows(g,bg)
    if not seps: return [g]
    bounds=[-1]+[r for r,_ in seps]+[h]
    panels=[]
    for i in range(len(bounds)-1):
        rs,re=bounds[i]+1,bounds[i+1]
        if rs<re: panels.append([g[r][:] for r in range(rs,re)])
    return panels

def _extract_panels_v(g):
    bg=_bg(g); h,w=len(g),len(g[0])
    seps=_find_sep_cols(g,bg)
    if not seps: return [g]
    bounds=[-1]+[c for c,_ in seps]+[w]
    panels=[]
    for i in range(len(bounds)-1):
        cs,ce=bounds[i]+1,bounds[i+1]
        if cs<ce: panels.append([[g[r][c] for c in range(cs,ce)] for r in range(h)])
    return panels

def panel_first_h(g):
    ps=_extract_panels_h(g); return ps[0] if ps else g
def panel_last_h(g):
    ps=_extract_panels_h(g); return ps[-1] if ps else g
def panel_first_v(g):
    ps=_extract_panels_v(g); return ps[0] if ps else g
def panel_last_v(g):
    ps=_extract_panels_v(g); return ps[-1] if ps else g

def panel_or_h(g):
    """水平パネルのOR（いずれかのパネルで非bgならば非bg）"""
    bg=_bg(g); panels=_extract_panels_h(g)
    if len(panels)<2: return g
    h=min(len(p) for p in panels); w=min(len(p[0]) for p in panels)
    res=[[bg]*w for _ in range(h)]
    for p in panels:
        for r in range(h):
            for c in range(w):
                if p[r][c]!=bg: res[r][c]=p[r][c]
    return res

def panel_and_h(g):
    """水平パネルのAND（全パネルで非bgの位置だけ残す）"""
    bg=_bg(g); panels=_extract_panels_h(g)
    if len(panels)<2: return g
    h=min(len(p) for p in panels); w=min(len(p[0]) for p in panels)
    res=[[bg]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if all(p[r][c]!=bg for p in panels):
                res[r][c]=panels[0][r][c]
    return res

def panel_xor_h(g):
    bg=_bg(g); panels=_extract_panels_h(g)
    if len(panels)<2: return g
    h=min(len(p) for p in panels); w=min(len(p[0]) for p in panels)
    res=[[bg]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            non_bg=[p[r][c] for p in panels if p[r][c]!=bg]
            if len(non_bg)==1: res[r][c]=non_bg[0]
    return res

def panel_or_v(g):
    bg=_bg(g); panels=_extract_panels_v(g)
    if len(panels)<2: return g
    h=min(len(p) for p in panels); w=min(len(p[0]) for p in panels)
    res=[[bg]*w for _ in range(h)]
    for p in panels:
        for r in range(h):
            for c in range(w):
                if p[r][c]!=bg: res[r][c]=p[r][c]
    return res

def panel_and_v(g):
    bg=_bg(g); panels=_extract_panels_v(g)
    if len(panels)<2: return g
    h=min(len(p) for p in panels); w=min(len(p[0]) for p in panels)
    res=[[bg]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if all(p[r][c]!=bg for p in panels): res[r][c]=panels[0][r][c]
    return res

def panel_xor_v(g):
    bg=_bg(g); panels=_extract_panels_v(g)
    if len(panels)<2: return g
    h=min(len(p) for p in panels); w=min(len(p[0]) for p in panels)
    res=[[bg]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            non_bg=[p[r][c] for p in panels if p[r][c]!=bg]
            if len(non_bg)==1: res[r][c]=non_bg[0]
    return res

def panel_diff_h(g):
    """2パネルの差分（片方にだけある非bg）"""
    bg=_bg(g); panels=_extract_panels_h(g)
    if len(panels)!=2: return g
    p1,p2=panels[0],panels[1]
    h=min(len(p1),len(p2)); w=min(len(p1[0]),len(p2[0]))
    res=[[bg]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if p1[r][c]!=bg and p2[r][c]==bg: res[r][c]=p1[r][c]
            elif p2[r][c]!=bg and p1[r][c]==bg: res[r][c]=p2[r][c]
    return res

def panel_diff_v(g):
    bg=_bg(g); panels=_extract_panels_v(g)
    if len(panels)!=2: return g
    p1,p2=panels[0],panels[1]
    h=min(len(p1),len(p2)); w=min(len(p1[0]),len(p2[0]))
    res=[[bg]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if p1[r][c]!=bg and p2[r][c]==bg: res[r][c]=p1[r][c]
            elif p2[r][c]!=bg and p1[r][c]==bg: res[r][c]=p2[r][c]
    return res

# ═══════════════════════════════════════
# M. 色操作・マッピング (20)
# ═══════════════════════════════════════

def replace_minority(g):
    bg=_bg(g); cc=_color_counts(g); del cc[bg]
    if len(cc)<2: return g
    maj=cc.most_common(1)[0][0]; mino=cc.most_common()[-1][0]
    return [[maj if v==mino else v for v in row] for row in g]

def replace_majority_nonbg(g):
    bg=_bg(g); cc=_color_counts(g); del cc[bg]
    if len(cc)<2: return g
    maj=cc.most_common(1)[0][0]; second=cc.most_common()[1][0]
    return [[second if v==maj else v for v in row] for row in g]

def erase_color(g, color):
    bg=_bg(g)
    return [[bg if v==color else v for v in row] for row in g]

def keep_only_color(g, color):
    bg=_bg(g); h,w=len(g),len(g[0])
    return [[g[r][c] if g[r][c]==color else bg for c in range(w)] for r in range(h)]

def recolor_by_frequency(g):
    """色を頻度順に1,2,3...に置換"""
    bg=_bg(g); cc=_color_counts(g); del cc[bg]
    ranked=sorted(cc.keys(), key=lambda x: (-cc[x],x))
    cmap={bg:bg}
    for i,c in enumerate(ranked): cmap[c]=i+1
    return [[cmap.get(v,v) for v in row] for row in g]

def rotate_colors(g):
    """非bg色を1つずつローテート"""
    bg=_bg(g); cc=_color_counts(g); del cc[bg]
    colors=sorted(cc.keys())
    if len(colors)<2: return g
    cmap={bg:bg}
    for i,c in enumerate(colors): cmap[c]=colors[(i+1)%len(colors)]
    return [[cmap.get(v,v) for v in row] for row in g]

def cycle_colors_reverse(g):
    bg=_bg(g); cc=_color_counts(g); del cc[bg]
    colors=sorted(cc.keys())
    if len(colors)<2: return g
    cmap={bg:bg}
    for i,c in enumerate(colors): cmap[c]=colors[(i-1)%len(colors)]
    return [[cmap.get(v,v) for v in row] for row in g]

def make_all_same_color(g):
    bg=_bg(g); cc=_color_counts(g); del cc[bg]
    if not cc: return g
    col=cc.most_common(1)[0][0]
    return [[col if v!=bg else bg for v in row] for row in g]

def swap_bg_and_fg(g):
    bg=_bg(g); cc=_color_counts(g); del cc[bg]
    if not cc: return g
    fg=cc.most_common(1)[0][0]
    return [[fg if v==bg else bg if v==fg else v for v in row] for row in g]

# ═══════════════════════════════════════
# N. カウント・集計 (10)
# ═══════════════════════════════════════

def count_to_1xN(g):
    """各色の出現回数を1xN gridとして出力"""
    bg=_bg(g); cc=_color_counts(g); del cc[bg]
    if not cc: return [[0]]
    colors=sorted(cc.keys())
    return [[cc[c] for c in colors]]

def count_objs_to_1x1(g):
    bg=_bg(g); objs=_objects(g,bg)
    return [[len(objs)]]

def count_colors_to_1x1(g):
    bg=_bg(g); cc=_color_counts(g); del cc[bg]
    return [[len(cc)]]

def color_histogram_row(g):
    """各行の非bg色数を列ベクトルとして出力"""
    bg=_bg(g)
    return [[sum(1 for c in row if c!=bg)] for row in g]

def color_histogram_col(g):
    bg=_bg(g); h,w=len(g),len(g[0])
    return [[sum(1 for r in range(h) if g[r][c]!=bg) for c in range(w)]]

# ═══════════════════════════════════════
# O. スケール・タイル (12)
# ═══════════════════════════════════════

def upscale_2x(g):
    h,w=len(g),len(g[0])
    return [[g[r//2][c//2] for c in range(w*2)] for r in range(h*2)]
def upscale_3x(g):
    h,w=len(g),len(g[0])
    return [[g[r//3][c//3] for c in range(w*3)] for r in range(h*3)]
def downscale_2x(g):
    h,w=len(g),len(g[0])
    return [[Counter(g[r*2+dr][c*2+dc] for dr in range(2) for dc in range(2)
                     if r*2+dr<h and c*2+dc<w).most_common(1)[0][0]
             for c in range(w//2)] for r in range(h//2)]
def downscale_3x(g):
    h,w=len(g),len(g[0])
    return [[Counter(g[r*3+dr][c*3+dc] for dr in range(3) for dc in range(3)
                     if r*3+dr<h and c*3+dc<w).most_common(1)[0][0]
             for c in range(w//3)] for r in range(h//3)]
def tile_2x2(g):
    h,w=len(g),len(g[0])
    return [[g[r%h][c%w] for c in range(w*2)] for r in range(h*2)]
def tile_3x3(g):
    h,w=len(g),len(g[0])
    return [[g[r%h][c%w] for c in range(w*3)] for r in range(h*3)]
def tile_2x1(g):
    h,w=len(g),len(g[0])
    return [[g[r%h][c%w] for c in range(w)] for r in range(h*2)]
def tile_1x2(g):
    h,w=len(g),len(g[0])
    return [[g[r%h][c%w] for c in range(w*2)] for r in range(h)]

def upscale_h_only_2x(g):
    h,w=len(g),len(g[0])
    return [[g[r][c//2] for c in range(w*2)] for r in range(h)]
def upscale_v_only_2x(g):
    h,w=len(g),len(g[0])
    return [[g[r//2][c] for c in range(w)] for r in range(h*2)]

# ═══════════════════════════════════════
# P-W: その他コマンド
# ═══════════════════════════════════════

def identity(g): return _copy(g)

def unique_rows(g):
    """重複行を除去"""
    seen=set(); res=[]
    for row in g:
        key=tuple(row)
        if key not in seen:
            seen.add(key); res.append(row[:])
    return res if res else g

def unique_cols(g):
    return transpose(unique_rows(transpose(g)))

def reverse_rows(g): return g[::-1]
def reverse_cols(g): return [row[::-1] for row in g]

def sort_rows_by_nonbg(g):
    bg=_bg(g)
    rows=[(sum(1 for c in row if c!=bg),i,row) for i,row in enumerate(g)]
    rows.sort(key=lambda x: x[0])
    return [row[:] for _,_,row in rows]

def sort_cols_by_nonbg(g):
    return transpose(sort_rows_by_nonbg(transpose(g)))

def sort_rows_by_first_color(g):
    bg=_bg(g)
    def key(row):
        for c in row:
            if c!=bg: return c
        return 99
    return sorted([row[:] for row in g], key=key)

# ═══════════════════════════════════════
# 全コマンドレジストリ
# ═══════════════════════════════════════

def build_all_commands(train_pairs=None):
    """全コマンドを (name, fn) のリストで返す。"""
    cmds = []
    
    # A: 幾何 (8)
    cmds += [("rot90",rot90),("rot180",rot180),("rot270",rot270),
             ("flip_h",flip_h),("flip_v",flip_v),("transpose",transpose),
             ("anti_transpose",anti_transpose),("identity",identity)]
    
    # B: 重力 (12)
    cmds += [("gravity_down",gravity_down),("gravity_up",gravity_up),
             ("gravity_left",gravity_left),("gravity_right",gravity_right),
             ("gravity_down_per_color",gravity_down_per_color),
             ("gravity_to_wall",gravity_to_wall),
             ("compact_down",compact_down),("compact_up",compact_up),
             ("compact_left",compact_left),("compact_right",compact_right),
             ("slide_objs_to_touch",slide_objs_to_touch)]
    
    # C: 充填 (15)
    cmds += [("fill_enclosed",fill_enclosed),
             ("fill_row_gaps",fill_row_gaps),("fill_col_gaps",fill_col_gaps),
             ("fill_between_same",fill_between_same_color),
             ("fill_diag_enclosed",fill_diag_enclosed),
             ("fill_bg_majority",fill_bg_with_majority),
             ("fill_checkerboard",fill_checkerboard),
             ("fill_row_majority",fill_row_majority),("fill_col_majority",fill_col_majority),
             ("fill_bbox",fill_each_obj_bbox),
             ("fill_holes",fill_holes_from_most_complete)]
    for col in range(10):
        cmds.append((f"fill_enclosed_c{col}", lambda g,c=col: fill_enclosed_with_color(g,c)))
    
    # D: 対称 (10)
    cmds += [("sym_h",sym_h),("sym_v",sym_v),("sym_4fold",sym_4fold),
             ("sym_diag",sym_diag_main),("sym_anti_diag",sym_diag_anti),
             ("sym_rot90",sym_rot90),("sym_center",sym_center_point),
             ("force_sym_h",force_sym_h),("force_sym_v",force_sym_v),
             ("force_sym_4fold",force_sym_4fold_tl)]
    
    # E: 成長・侵食 (12)
    cmds += [("grow_1",grow_1),("grow_8",grow_8),("shrink_1",shrink_1),
             ("ca_life",ca_life),("ca_majority",ca_majority),
             ("grow_diagonal",grow_diagonal),("grow_cross",grow_cross),
             ("smooth",smooth)]
    for col in range(10):
        cmds.append((f"dilate_c{col}", lambda g,c=col: dilate_color(g,c)))
        cmds.append((f"erode_c{col}", lambda g,c=col: erode_color(g,c)))
    
    # F: 線・延長 (16)
    cmds += [("extend_h",extend_h),("extend_v",extend_v),
             ("extend_full_h",extend_full_h),("extend_full_v",extend_full_v),
             ("cross_through",draw_cross_through_points),
             ("diag_through",draw_diag_through_points),
             ("connect_h",connect_same_color_h),("connect_v",connect_same_color_v),
             ("connect_both",connect_same_color_both),
             ("draw_border",draw_border),
             ("outline_4",trace_outline_4),("outline_8",trace_outline_8),
             ("interior",interior_only),("draw_grid",draw_grid_lines)]
    
    # G: クロップ (12)
    cmds += [("crop",crop_content),("crop_largest",crop_largest_obj),
             ("crop_smallest",crop_smallest_obj),
             ("rm_bg_rows",rm_bg_rows),("rm_bg_cols",rm_bg_cols),
             ("crop_top",crop_top_half),("crop_bottom",crop_bottom_half),
             ("crop_left",crop_left_half),("crop_right",crop_right_half),
             ("crop_tl",crop_quadrant_tl),("crop_tr",crop_quadrant_tr),
             ("crop_bl",crop_quadrant_bl),("crop_br",crop_quadrant_br)]
    
    # H: オブジェクト (30)
    cmds += [("keep_largest",keep_largest),("keep_smallest",keep_smallest),
             ("rm_largest",remove_largest),("rm_smallest",remove_smallest),
             ("keep_common_color",keep_most_common_color_obj),
             ("keep_unique_color",keep_unique_color_obj),
             ("keep_border_objs",keep_objs_touching_border),
             ("rm_border_objs",remove_objs_touching_border),
             ("sort_objs_size",sort_objs_by_size),
             ("recolor_by_size",recolor_each_obj_by_size),
             ("unify_color",unify_obj_color),("hollow",hollow_objects),
             ("rot_each_90",rotate_each_obj_90),
             ("flip_each_h",flip_each_obj_h),("flip_each_v",flip_each_obj_v),
             ("dup_h",duplicate_objs_h),("dup_v",duplicate_objs_v)]
    for n in [1,2,3,4,5]:
        cmds.append((f"keep_size_{n}", lambda g,sz=n: keep_objs_by_size(g,sz)))
    
    # I: 移動 (16)
    cmds += [("move_up",move_up_1),("move_down",move_down_1),
             ("move_left",move_left_1),("move_right",move_right_1),
             ("center",move_obj_to_center),
             ("to_top",move_all_to_top),("to_bottom",move_all_to_bottom),
             ("to_left",move_all_to_left),("to_right",move_all_to_right),
             ("stack_v",stack_objs_vertical),("stack_h",stack_objs_horizontal)]
    
    # J: スタンプ (12)
    cmds += [("stamp_lg_at_sm",stamp_largest_at_smallest),
             ("stamp_sm_at_lg",stamp_smallest_at_largest),
             ("tile_pattern",tile_pattern),
             ("mirror_h",mirror_stamp_h),("mirror_v",mirror_stamp_v),
             ("repeat_2x2",repeat_pattern_2x2),("repeat_3x3",repeat_pattern_3x3),
             ("fill_holes_shape",fill_holes_from_most_complete),
             ("copy_unique",copy_unique_obj_to_all)]
    
    # K: 条件付き (20)
    cmds += [("if_nb_same",if_neighbor_same_fill),
             ("if_isolated_rm",if_isolated_remove),
             ("if_edge_color",if_on_edge_color),
             ("if_enclosed_recolor",if_enclosed_recolor),
             ("swap_colors",swap_two_colors),("invert",invert_colors),
             ("color_by_row",color_by_position),("color_by_col",color_by_col_position),
             ("recolor_by_nb",replace_with_neighbor_color),
             ("grow_if_enclosed",conditional_grow_if_enclosed),
             ("swap_bg_fg",swap_bg_and_fg)]
    
    # L: パネル (18)
    cmds += [("panel_0_h",panel_first_h),("panel_-1_h",panel_last_h),
             ("panel_0_v",panel_first_v),("panel_-1_v",panel_last_v),
             ("panel_or_h",panel_or_h),("panel_and_h",panel_and_h),
             ("panel_xor_h",panel_xor_h),
             ("panel_or_v",panel_or_v),("panel_and_v",panel_and_v),
             ("panel_xor_v",panel_xor_v),
             ("panel_diff_h",panel_diff_h),("panel_diff_v",panel_diff_v)]
    
    # M: 色 (20)
    cmds += [("repl_minority",replace_minority),
             ("repl_majority",replace_majority_nonbg),
             ("recolor_freq",recolor_by_frequency),
             ("rotate_colors",rotate_colors),("cycle_rev",cycle_colors_reverse),
             ("all_same",make_all_same_color)]
    for col in range(10):
        cmds.append((f"erase_c{col}", lambda g,c=col: erase_color(g,c)))
        cmds.append((f"only_c{col}", lambda g,c=col: keep_only_color(g,c)))
    
    # N: カウント (5)
    cmds += [("count_1xN",count_to_1xN),("count_objs",count_objs_to_1x1),
             ("count_colors",count_colors_to_1x1)]
    
    # O: スケール (12)
    cmds += [("up_2x",upscale_2x),("up_3x",upscale_3x),
             ("down_2x",downscale_2x),("down_3x",downscale_3x),
             ("tile_2x2",tile_2x2),("tile_3x3",tile_3x3),
             ("tile_2x1",tile_2x1),("tile_1x2",tile_1x2),
             ("up_h2x",upscale_h_only_2x),("up_v2x",upscale_v_only_2x)]
    
    # U: ソート・並べ替え (8)
    cmds += [("reverse_rows",reverse_rows),("reverse_cols",reverse_cols),
             ("sort_rows",sort_rows_by_nonbg),("sort_cols",sort_cols_by_nonbg),
             ("sort_rows_color",sort_rows_by_first_color),
             ("unique_rows",unique_rows),("unique_cols",unique_cols)]
    
    # Parametric: trainから学習
    if train_pairs:
        i0,o0 = train_pairs[0]
        ih,iw=len(i0),len(i0[0]); oh,ow=len(o0),len(o0[0])
        if (ih,iw)==(oh,ow):
            cmap={}; ok=True
            for inp,out in train_pairs:
                h,w=len(inp),len(inp[0])
                if len(out)!=h or len(out[0])!=w: ok=False; break
                for r in range(h):
                    for c in range(w):
                        ic,oc=inp[r][c],out[r][c]
                        if ic in cmap:
                            if cmap[ic]!=oc: ok=False; break
                        else: cmap[ic]=oc
                    if not ok: break
                if not ok: break
            if ok and cmap and any(k!=v for k,v in cmap.items()):
                cm=dict(cmap)
                cmds.append(("color_map_learned", lambda g,m=cm: [[m.get(c,c) for c in row] for row in g]))
    
    return cmds

# コマンド数確認用
if __name__ == "__main__":
    cmds = build_all_commands()
    print(f"Total commands: {len(cmds)}")
    cats = Counter()
    for name, _ in cmds:
        prefix = name.split("_")[0] if "_" in name else name
        cats[prefix] += 1
    for cat, n in cats.most_common():
        print(f"  {cat}: {n}")
