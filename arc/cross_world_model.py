#!/usr/bin/env python3
"""
Verantyx Cross World Model
===========================
Cross Engineの全資産（90種WholeGridProgram + 94モジュール + パズル推論）を
世界モデルのシミュレーションコマンドとして統合。

Cross構造の分解 → 世界法則コマンドの適用 → Cross構造の再構成

設計:
  - Cross Engineそのものがシミュレータ
  - 各モジュールが生成するpieceをシミュレーション候補として扱う
  - 世界法則（重力、対称性等）も同じインターフェースで統合
  - depth 1-2の合成 + 収束ループ
  - train全例verify + 多数決
"""

import json, os, sys, time, gc
from collections import Counter
from typing import List, Tuple, Optional

from arc.grid import Grid, grid_shape, grid_eq, most_common_color

class SimResult:
    """シミュレーション結果"""
    __slots__ = ['name', 'preds', 'source', 'confidence']
    def __init__(self, name, preds, source="", confidence=1.0):
        self.name = name
        self.preds = preds
        self.source = source
        self.confidence = confidence

def _bg(grid):
    c = Counter()
    for row in grid: c.update(row)
    return c.most_common(1)[0][0]

def _grid_copy(g):
    return [row[:] for row in g]

def _is_exterior_bg(grid, bg):
    h, w = len(grid), len(grid[0])
    visited = [[False]*w for _ in range(h)]
    queue = []
    for r in range(h):
        for c in [0, w-1]:
            if grid[r][c] == bg and not visited[r][c]:
                visited[r][c] = True; queue.append((r,c))
    for c in range(w):
        for r in [0, h-1]:
            if grid[r][c] == bg and not visited[r][c]:
                visited[r][c] = True; queue.append((r,c))
    while queue:
        cr, cc = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc] == bg:
                visited[nr][nc] = True; queue.append((nr,nc))
    return visited


class CrossWorldModel:
    """
    Cross Engine全体を世界モデルとして使う。
    
    Phase A: Cross Engine全モジュール（そのまま使用）
    Phase B: 世界法則コマンド（重力、対称性、充填等）
    Phase C: 合成（B×B, A×B, B×A）
    Phase D: 収束ループ
    """
    
    def solve(self, train_pairs, test_inputs, timeout_s=90):
        t0 = time.time()
        solutions = []
        
        # ═══ Phase A: Cross Engine（全モジュール一括）═══
        try:
            from arc.cross_engine import solve_cross_engine
            _, verified = solve_cross_engine(train_pairs, test_inputs)
            for tag, piece in verified:
                name = getattr(piece, 'name', type(piece).__name__)
                try:
                    preds = [piece.apply(ti) for ti in test_inputs]
                    if all(p is not None for p in preds):
                        solutions.append(SimResult(
                            f"cross:{name}", preds, "cross_engine", 0.95))
                except:
                    pass
        except:
            pass
        
        if time.time() - t0 > timeout_s:
            return solutions
        
        # ═══ Phase B: 世界法則コマンド（軽量・高速）═══
        world_cmds = self._build_world_commands(train_pairs)
        
        inp0, out0 = train_pairs[0]
        
        # Depth 1: 単一コマンド
        for name, fn in world_cmds:
            if time.time() - t0 > timeout_s: break
            if self._verify_all(fn, train_pairs):
                preds = self._predict(fn, test_inputs)
                if preds:
                    solutions.append(SimResult(name, preds, "world_d1", 1.0))
        
        # 収束ループ
        for name, fn in world_cmds:
            if time.time() - t0 > timeout_s: break
            conv_fn = self._make_converge(fn)
            if self._verify_all(conv_fn, train_pairs):
                preds = self._predict(conv_fn, test_inputs)
                if preds:
                    solutions.append(SimResult(
                        f"converge({name})", preds, "world_conv", 0.9))
        
        # Depth 2: train[0]事前フィルタ付き
        active = []
        mid_cache = {}
        for i, (name, fn) in enumerate(world_cmds):
            if time.time() - t0 > timeout_s: break
            try:
                mid = fn(inp0)
                if mid is not None and isinstance(mid, list) and len(mid) > 0:
                    active.append((i, name, fn))
                    mid_cache[i] = mid
            except:
                pass
        
        active = active[:20]
        
        for i, n1, f1 in active:
            if time.time() - t0 > timeout_s: break
            mid = mid_cache[i]
            for n2, f2 in world_cmds:
                if time.time() - t0 > timeout_s: break
                try:
                    r2 = f2(mid)
                    if r2 is None or not grid_eq(r2, out0): continue
                except:
                    continue
                # train[0] passes, verify all
                def make_pipe(a, b):
                    def fn(g):
                        m = a(g)
                        if m is None: return None
                        return b(m)
                    return fn
                pipe = make_pipe(f1, f2)
                if self._verify_all(pipe, train_pairs):
                    preds = self._predict(pipe, test_inputs)
                    if preds:
                        solutions.append(SimResult(
                            f"{n1} → {n2}", preds, "world_d2", 0.8))
        
        return solutions
    
    def _verify_all(self, fn, train_pairs):
        for inp, expected in train_pairs:
            try:
                result = fn(inp)
                if result is None or not grid_eq(result, expected):
                    return False
            except:
                return False
        return True
    
    def _predict(self, fn, test_inputs):
        preds = []
        for ti in test_inputs:
            try:
                p = fn(ti)
                if p is None: return None
                preds.append(p)
            except:
                return None
        return preds
    
    def _make_converge(self, fn, max_iter=20):
        def conv(grid):
            g = grid
            for _ in range(max_iter):
                try:
                    g2 = fn(g)
                except:
                    return None
                if g2 is None: return None
                if grid_eq(g, g2): return g
                g = g2
            return g
        return conv
    
    def _build_world_commands(self, train_pairs):
        """世界法則コマンドをビルド。(name, fn)のリスト。"""
        cmds = []
        
        # ── 幾何 ──
        cmds.append(("rot90", lambda g: [list(r) for r in zip(*g[::-1])]))
        cmds.append(("rot180", lambda g: [row[::-1] for row in g[::-1]]))
        cmds.append(("rot270", lambda g: [list(r) for r in zip(*[row[::-1] for row in g])]))
        cmds.append(("flip_h", lambda g: [row[::-1] for row in g]))
        cmds.append(("flip_v", lambda g: g[::-1]))
        cmds.append(("transpose", lambda g: [list(r) for r in zip(*g)]))
        
        # ── 重力 ──
        for d in ["down", "up", "left", "right"]:
            def mk_grav(direction):
                def fn(g):
                    bg = _bg(g)
                    h, w = len(g), len(g[0])
                    result = [[bg]*w for _ in range(h)]
                    if direction == "down":
                        for c in range(w):
                            non = [g[r][c] for r in range(h) if g[r][c] != bg]
                            for i, v in enumerate(non): result[h-len(non)+i][c] = v
                    elif direction == "up":
                        for c in range(w):
                            non = [g[r][c] for r in range(h) if g[r][c] != bg]
                            for i, v in enumerate(non): result[i][c] = v
                    elif direction == "left":
                        for r in range(h):
                            non = [g[r][c] for c in range(w) if g[r][c] != bg]
                            for i, v in enumerate(non): result[r][i] = v
                    elif direction == "right":
                        for r in range(h):
                            non = [g[r][c] for c in range(w) if g[r][c] != bg]
                            for i, v in enumerate(non): result[r][w-len(non)+i] = v
                    return result
                return fn
            cmds.append((f"gravity_{d}", mk_grav(d)))
        
        # ── 充填 ──
        def fill_enclosed(g):
            bg = _bg(g)
            h, w = len(g), len(g[0])
            ext = _is_exterior_bg(g, bg)
            result = _grid_copy(g)
            for r in range(h):
                for c in range(w):
                    if g[r][c] == bg and not ext[r][c]:
                        nb = []
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0<=nr<h and 0<=nc<w and g[nr][nc] != bg:
                                nb.append(g[nr][nc])
                        if nb: result[r][c] = Counter(nb).most_common(1)[0][0]
            return result
        cmds.append(("fill_enclosed", fill_enclosed))
        
        # ── 対称性修復 ──
        for st in ["h", "v", "4fold"]:
            def mk_sym(s):
                def fn(g):
                    bg = _bg(g)
                    h, w = len(g), len(g[0])
                    result = _grid_copy(g)
                    if s == "h":
                        for r in range(h):
                            for c in range(w//2):
                                mc = w-1-c
                                if result[r][c] == bg and result[r][mc] != bg: result[r][c] = result[r][mc]
                                elif result[r][mc] == bg and result[r][c] != bg: result[r][mc] = result[r][c]
                    elif s == "v":
                        for r in range(h//2):
                            mr = h-1-r
                            for c in range(w):
                                if result[r][c] == bg and result[mr][c] != bg: result[r][c] = result[mr][c]
                                elif result[mr][c] == bg and result[r][c] != bg: result[mr][c] = result[r][c]
                    elif s == "4fold":
                        for r in range((h+1)//2):
                            for c in range((w+1)//2):
                                pos = [(r,c),(r,w-1-c),(h-1-r,c),(h-1-r,w-1-c)]
                                vals = [result[pr][pc] for pr,pc in pos if 0<=pr<h and 0<=pc<w and result[pr][pc]!=bg]
                                if vals:
                                    fill = Counter(vals).most_common(1)[0][0]
                                    for pr,pc in pos:
                                        if 0<=pr<h and 0<=pc<w and result[pr][pc]==bg: result[pr][pc]=fill
                    return result
                return fn
            cmds.append((f"sym_{st}", mk_sym(st)))
        
        # ── 成長・侵食（セルオートマトン）──
        def grow(g):
            bg = _bg(g)
            h, w = len(g), len(g[0])
            result = _grid_copy(g)
            for r in range(h):
                for c in range(w):
                    if g[r][c] != bg:
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0<=nr<h and 0<=nc<w and result[nr][nc]==bg: result[nr][nc]=g[r][c]
            return result
        cmds.append(("grow", grow))
        
        def shrink(g):
            bg = _bg(g)
            h, w = len(g), len(g[0])
            result = _grid_copy(g)
            for r in range(h):
                for c in range(w):
                    if g[r][c] != bg:
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if not (0<=nr<h and 0<=nc<w) or g[nr][nc]==bg:
                                result[r][c]=bg; break
            return result
        cmds.append(("shrink", shrink))
        
        # ── 線の延長 ──
        def extend_h(g):
            bg = _bg(g)
            h, w = len(g), len(g[0])
            result = _grid_copy(g)
            for r in range(h):
                colors = set(g[r][c] for c in range(w) if g[r][c] != bg)
                if len(colors) == 1:
                    col = colors.pop()
                    ps = [c for c in range(w) if g[r][c] == col]
                    if len(ps) >= 2:
                        for c in range(min(ps), max(ps)+1): result[r][c] = col
            return result
        cmds.append(("extend_h", extend_h))
        
        def extend_v(g):
            bg = _bg(g)
            h, w = len(g), len(g[0])
            result = _grid_copy(g)
            for c in range(w):
                colors = set(g[r][c] for r in range(h) if g[r][c] != bg)
                if len(colors) == 1:
                    col = colors.pop()
                    ps = [r for r in range(h) if g[r][c] == col]
                    if len(ps) >= 2:
                        for r in range(min(ps), max(ps)+1): result[r][c] = col
            return result
        cmds.append(("extend_v", extend_v))
        
        # ── クロップ ──
        def crop(g):
            bg = _bg(g)
            h, w = len(g), len(g[0])
            rows = [r for r in range(h) if any(g[r][c]!=bg for c in range(w))]
            cols = [c for c in range(w) if any(g[r][c]!=bg for r in range(h))]
            if not rows or not cols: return g
            return [g[r][min(cols):max(cols)+1] for r in range(min(rows), max(rows)+1)]
        cmds.append(("crop", crop))
        
        # ── bg行/列除去 ──
        def rm_bg_rows(g):
            bg = _bg(g)
            return [row for row in g if any(c!=bg for c in row)] or g
        cmds.append(("rm_bg_rows", rm_bg_rows))
        
        def rm_bg_cols(g):
            bg = _bg(g)
            h, w = len(g), len(g[0])
            keep = [c for c in range(w) if any(g[r][c]!=bg for r in range(h))]
            if not keep: return g
            return [[g[r][c] for c in keep] for r in range(h)]
        cmds.append(("rm_bg_cols", rm_bg_cols))
        
        # ── オブジェクト系 ──
        def keep_largest(g):
            bg = _bg(g)
            h, w = len(g), len(g[0])
            visited = [[False]*w for _ in range(h)]
            objs = []
            for r in range(h):
                for c in range(w):
                    if not visited[r][c] and g[r][c] != bg:
                        obj = []; stack = [(r,c)]; visited[r][c]=True
                        while stack:
                            cr,cc = stack.pop()
                            obj.append((cr,cc,g[cr][cc]))
                            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr,nc=cr+dr,cc+dc
                                if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and g[nr][nc]!=bg:
                                    visited[nr][nc]=True; stack.append((nr,nc))
                        objs.append(obj)
            if not objs: return g
            largest = max(objs, key=len)
            result = [[bg]*w for _ in range(h)]
            for r,c,v in largest: result[r][c]=v
            return result
        cmds.append(("keep_largest", keep_largest))
        
        def keep_smallest(g):
            bg = _bg(g)
            h, w = len(g), len(g[0])
            visited = [[False]*w for _ in range(h)]
            objs = []
            for r in range(h):
                for c in range(w):
                    if not visited[r][c] and g[r][c] != bg:
                        obj = []; stack = [(r,c)]; visited[r][c]=True
                        while stack:
                            cr,cc = stack.pop()
                            obj.append((cr,cc,g[cr][cc]))
                            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr,nc=cr+dr,cc+dc
                                if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and g[nr][nc]!=bg:
                                    visited[nr][nc]=True; stack.append((nr,nc))
                        objs.append(obj)
            if not objs: return g
            smallest = min(objs, key=len)
            result = [[bg]*w for _ in range(h)]
            for r,c,v in smallest: result[r][c]=v
            return result
        cmds.append(("keep_smallest", keep_smallest))
        
        def fill_bbox(g):
            bg = _bg(g)
            h, w = len(g), len(g[0])
            visited = [[False]*w for _ in range(h)]
            result = _grid_copy(g)
            for r in range(h):
                for c in range(w):
                    if not visited[r][c] and g[r][c] != bg:
                        obj = []; stack = [(r,c)]; visited[r][c]=True
                        while stack:
                            cr,cc = stack.pop()
                            obj.append((cr,cc,g[cr][cc]))
                            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr,nc=cr+dr,cc+dc
                                if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and g[nr][nc]!=bg:
                                    visited[nr][nc]=True; stack.append((nr,nc))
                        r0=min(x[0] for x in obj); c0=min(x[1] for x in obj)
                        r1=max(x[0] for x in obj); c1=max(x[1] for x in obj)
                        col = Counter(v for _,_,v in obj).most_common(1)[0][0]
                        for rr in range(r0,r1+1):
                            for cc in range(c0,c1+1):
                                result[rr][cc]=col
            return result
        cmds.append(("fill_bbox", fill_bbox))
        
        def outline(g):
            bg = _bg(g)
            h, w = len(g), len(g[0])
            result = [[bg]*w for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    if g[r][c] != bg:
                        border = False
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr,nc=r+dr,c+dc
                            if not (0<=nr<h and 0<=nc<w) or g[nr][nc]==bg:
                                border=True; break
                        if border: result[r][c]=g[r][c]
            return result
        cmds.append(("outline", outline))
        
        # ── マイノリティ色置換 ──
        def repl_minority(g):
            bg = _bg(g)
            cc = Counter()
            for row in g:
                for v in row:
                    if v != bg: cc[v] += 1
            if len(cc) < 2: return g
            maj = cc.most_common(1)[0][0]
            mino = cc.most_common()[-1][0]
            return [[maj if v==mino else v for v in row] for row in g]
        cmds.append(("repl_minority", repl_minority))
        
        # ── 色マッピング（trainから学習）──
        i0, o0 = train_pairs[0]
        if len(i0)==len(o0) and len(i0[0])==len(o0[0]):
            cmap = {}; ok = True
            for inp, out in train_pairs:
                h,w = len(inp), len(inp[0])
                if len(out)!=h or len(out[0])!=w: ok=False; break
                for r in range(h):
                    for c in range(w):
                        ic,oc = inp[r][c], out[r][c]
                        if ic in cmap:
                            if cmap[ic]!=oc: ok=False; break
                        else: cmap[ic]=oc
                    if not ok: break
                if not ok: break
            if ok and cmap and any(k!=v for k,v in cmap.items()):
                cm = dict(cmap)
                cmds.append(("color_map", lambda g,m=cm: [[m.get(c,c) for c in row] for row in g]))
        
        # ── スケーリング ──
        for s in [2,3]:
            def mk_up(sc):
                def fn(g):
                    h,w=len(g),len(g[0])
                    return [[g[r//sc][c//sc] for c in range(w*sc)] for r in range(h*sc)]
                return fn
            cmds.append((f"up_{s}x", mk_up(s)))
            def mk_down(sc):
                def fn(g):
                    h,w=len(g),len(g[0])
                    nh,nw=h//sc,w//sc
                    if nh==0 or nw==0: return None
                    return [[Counter(g[r*sc+dr][c*sc+dc] for dr in range(sc) for dc in range(sc)
                                     if r*sc+dr<h and c*sc+dc<w).most_common(1)[0][0]
                             for c in range(nw)] for r in range(nh)]
                return fn
            cmds.append((f"down_{s}x", mk_down(s)))
        
        # ── タイル ──
        for rh in [2,3]:
            for rw in [2,3]:
                def mk_tile(th,tw):
                    def fn(g):
                        h,w=len(g),len(g[0])
                        return [[g[r%h][c%w] for c in range(w*tw)] for r in range(h*th)]
                    return fn
                cmds.append((f"tile_{rh}x{rw}", mk_tile(rh,rw)))
        
        return cmds
