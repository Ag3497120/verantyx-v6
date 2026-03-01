#!/usr/bin/env python3
"""
cross2.py — Cross Structure v2 (再設計)
========================================
kofdaiの設計思想: すべてcross構造にする。
分解→道具→ルーティング→再構成が全部cross構造の中で完結。
"""

from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict, Any, Callable
import time

Grid = List[List[int]]

def grid_eq(a, b):
    if len(a) != len(b): return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]): return False
        for j in range(len(a[i])):
            if a[i][j] != b[i][j]: return False
    return True

def copy_grid(g): return [row[:] for row in g]

def bg_color(g):
    c = Counter(); 
    for row in g: c.update(row)
    return c.most_common(1)[0][0]

def grid_colors(g):
    s = set()
    for row in g: s.update(row)
    return s

def color_counts(g):
    c = Counter()
    for row in g: c.update(row)
    return c

# ═══════════════════════════════════════
# Decomposition — グリッドを構造に分解
# ═══════════════════════════════════════

class Decomposition:
    __slots__ = ['kind','grid','bg','objects','panels','colors','separators','regions','meta']
    def __init__(self, kind, grid, bg, **kw):
        self.kind = kind; self.grid = grid; self.bg = bg
        self.objects = kw.get('objects', [])
        self.panels = kw.get('panels', [])
        self.colors = kw.get('colors', {})
        self.separators = kw.get('separators', {})
        self.regions = kw.get('regions', [])
        self.meta = kw.get('meta', {})

class CrossDecomposer:
    @staticmethod
    def decompose_all(g):
        bg = bg_color(g); results = []
        objs4 = CrossDecomposer._find_objects(g, bg, 4)
        if objs4: results.append(Decomposition('obj4', g, bg, objects=objs4))
        objs8 = CrossDecomposer._find_objects(g, bg, 8)
        if objs8 and len(objs8) != len(objs4):
            results.append(Decomposition('obj8', g, bg, objects=objs8))
        mono = CrossDecomposer._find_mono(g, bg)
        if mono: results.append(Decomposition('mono', g, bg, objects=mono))
        ph, sh = CrossDecomposer._panels_h(g, bg)
        if ph and len(ph) >= 2: results.append(Decomposition('pan_h', g, bg, panels=ph, separators={'rows':sh}))
        pv, sv = CrossDecomposer._panels_v(g, bg)
        if pv and len(pv) >= 2: results.append(Decomposition('pan_v', g, bg, panels=pv, separators={'cols':sv}))
        cmasks = CrossDecomposer._color_masks(g, bg)
        if len(cmasks) >= 2: results.append(Decomposition('colors', g, bg, colors=cmasks))
        regs = CrossDecomposer._enclosed(g, bg)
        if regs: results.append(Decomposition('enclosed', g, bg, regions=regs))
        results.append(Decomposition('whole', g, bg))
        return results
    
    @staticmethod
    def _find_objects(g, bg, conn=4):
        h,w = len(g), len(g[0]); vis = [[False]*w for _ in range(h)]; objs = []
        ds = [(-1,0),(1,0),(0,-1),(0,1)]
        if conn == 8: ds += [(-1,-1),(-1,1),(1,-1),(1,1)]
        for r in range(h):
            for c in range(w):
                if not vis[r][c] and g[r][c] != bg:
                    obj=[]; stk=[(r,c)]; vis[r][c]=True
                    while stk:
                        cr,cc=stk.pop(); obj.append((cr,cc,g[cr][cc]))
                        for dr,dc in ds:
                            nr,nc=cr+dr,cc+dc
                            if 0<=nr<h and 0<=nc<w and not vis[nr][nc] and g[nr][nc]!=bg:
                                vis[nr][nc]=True; stk.append((nr,nc))
                    objs.append(obj)
        return objs
    
    @staticmethod
    def _find_mono(g, bg):
        h,w = len(g), len(g[0]); vis = [[False]*w for _ in range(h)]; objs = []
        for r in range(h):
            for c in range(w):
                if not vis[r][c] and g[r][c] != bg:
                    col=g[r][c]; obj=[]; stk=[(r,c)]; vis[r][c]=True
                    while stk:
                        cr,cc=stk.pop(); obj.append((cr,cc,g[cr][cc]))
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr,nc=cr+dr,cc+dc
                            if 0<=nr<h and 0<=nc<w and not vis[nr][nc] and g[nr][nc]==col:
                                vis[nr][nc]=True; stk.append((nr,nc))
                    objs.append(obj)
        return objs
    
    @staticmethod
    def _panels_h(g, bg):
        h,w=len(g),len(g[0]); seps=[]
        for r in range(h):
            vs=set(g[r])
            if len(vs)==1 and g[r][0]!=bg: seps.append((r,g[r][0]))
        if not seps: return [],[]
        bounds=[-1]+[r for r,_ in seps]+[h]; panels=[]
        for i in range(len(bounds)-1):
            rs,re=bounds[i]+1,bounds[i+1]
            if rs<re: panels.append([g[r][:] for r in range(rs,re)])
        return panels, seps
    
    @staticmethod
    def _panels_v(g, bg):
        h,w=len(g),len(g[0]); seps=[]
        for c in range(w):
            vs=set(g[r][c] for r in range(h))
            if len(vs)==1 and g[0][c]!=bg: seps.append((c,g[0][c]))
        if not seps: return [],[]
        bounds=[-1]+[c for c,_ in seps]+[w]; panels=[]
        for i in range(len(bounds)-1):
            cs,ce=bounds[i]+1,bounds[i+1]
            if cs<ce: panels.append([[g[r][c] for c in range(cs,ce)] for r in range(h)])
        return panels, seps
    
    @staticmethod
    def _color_masks(g, bg):
        h,w=len(g),len(g[0]); masks={}
        for r in range(h):
            for c in range(w):
                v=g[r][c]
                if v!=bg:
                    if v not in masks: masks[v]=[[0]*w for _ in range(h)]
                    masks[v][r][c]=1
        return masks
    
    @staticmethod
    def _enclosed(g, bg):
        h,w=len(g),len(g[0]); vis=[[False]*w for _ in range(h)]; q=[]
        for r in range(h):
            for c in range(w):
                if (r==0 or r==h-1 or c==0 or c==w-1) and g[r][c]==bg:
                    vis[r][c]=True; q.append((r,c))
        while q:
            cr,cc=q.pop(0)
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=cr+dr,cc+dc
                if 0<=nr<h and 0<=nc<w and not vis[nr][nc] and g[nr][nc]==bg:
                    vis[nr][nc]=True; q.append((nr,nc))
        interior=[[False]*w for _ in range(h)]; regions=[]
        for r in range(h):
            for c in range(w):
                if g[r][c]==bg and not vis[r][c] and not interior[r][c]:
                    reg=[]; stk=[(r,c)]; interior[r][c]=True
                    while stk:
                        cr,cc=stk.pop(); reg.append((cr,cc))
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr,nc=cr+dr,cc+dc
                            if 0<=nr<h and 0<=nc<w and g[nr][nc]==bg and not vis[nr][nc] and not interior[nr][nc]:
                                interior[nr][nc]=True; stk.append((nr,nc))
                    regions.append(reg)
        return regions

# ═══════════════════════════════════════
# CrossTool — 道具
# ═══════════════════════════════════════

def _obj_bbox(cells):
    rs=[r for r,c,_ in cells]; cs=[c for r,c,_ in cells]
    return min(rs),min(cs),max(rs),max(cs)

def _obj_patch(cells, bg):
    r0,c0,r1,c1=_obj_bbox(cells); h,w=r1-r0+1,c1-c0+1
    p=[[bg]*w for _ in range(h)]
    for r,c,v in cells: p[r-r0][c-c0]=v
    return p, r0, c0

def _obj_color(cells):
    return Counter(v for _,_,v in cells).most_common(1)[0][0]

def _paste(g, patch, r0, c0, bg):
    res=copy_grid(g); h,w=len(g),len(g[0])
    for r in range(len(patch)):
        for c in range(len(patch[0])):
            nr,nc=r0+r,c0+c
            if 0<=nr<h and 0<=nc<w and patch[r][c]!=bg: res[nr][nc]=patch[r][c]
    return res

class CrossTool:
    def __init__(self, name, fn):
        self.name = name; self.fn = fn
    def apply(self, g, d):
        try: return self.fn(g, d)
        except: return None

def _build_tools():
    tools = []
    
    # 幾何
    def rot90(g,d): return [list(r) for r in zip(*g[::-1])]
    def rot180(g,d): return [row[::-1] for row in g[::-1]]
    def rot270(g,d): return [list(r) for r in zip(*[row[::-1] for row in g])]
    def flip_h(g,d): return [row[::-1] for row in g]
    def flip_v(g,d): return [row[:] for row in g[::-1]]
    def transp(g,d): return [list(r) for r in zip(*g)]
    for n,f in [('rot90',rot90),('rot180',rot180),('rot270',rot270),
                ('flip_h',flip_h),('flip_v',flip_v),('transpose',transp)]:
        tools.append(CrossTool(n,f))
    
    # オブジェクト抽出
    def crop_largest(g,d):
        if not d.objects: return None
        o=max(d.objects,key=len); p,_,_=_obj_patch(o,d.bg); return p
    def crop_smallest(g,d):
        if not d.objects: return None
        o=min(d.objects,key=len); p,_,_=_obj_patch(o,d.bg); return p
    def rm_largest(g,d):
        if not d.objects: return None
        o=max(d.objects,key=len); res=copy_grid(g)
        for r,c,_ in o: res[r][c]=d.bg
        return res
    def rm_smallest(g,d):
        if not d.objects: return None
        o=min(d.objects,key=len); res=copy_grid(g)
        for r,c,_ in o: res[r][c]=d.bg
        return res
    for n,f in [('crop_lg',crop_largest),('crop_sm',crop_smallest),
                ('rm_lg',rm_largest),('rm_sm',rm_smallest)]:
        tools.append(CrossTool(n,f))
    
    # bbox充填
    def fill_bbox(g,d):
        if not d.objects: return None
        res=copy_grid(g)
        for obj in d.objects:
            r0,c0,r1,c1=_obj_bbox(obj); col=_obj_color(obj)
            for r in range(r0,r1+1):
                for c in range(c0,c1+1): res[r][c]=col
        return res
    tools.append(CrossTool('fill_bbox',fill_bbox))
    
    # 輪郭
    def outline(g,d):
        if not d.objects: return None
        h,w=len(g),len(g[0]); res=[[d.bg]*w for _ in range(h)]
        cells=set()
        for obj in d.objects:
            for r,c,v in obj: cells.add((r,c))
        for obj in d.objects:
            for r,c,v in obj:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if (nr,nc) not in cells or not(0<=nr<h and 0<=nc<w):
                        res[r][c]=v; break
        return res
    tools.append(CrossTool('outline',outline))
    
    # 重力
    for direction in ['down','up','left','right']:
        def mk(dr):
            def f(g,d):
                bg=d.bg; h,w=len(g),len(g[0]); res=[[bg]*w for _ in range(h)]
                if dr=='down':
                    for c in range(w):
                        non=[g[r][c] for r in range(h) if g[r][c]!=bg]
                        for i,v in enumerate(non): res[h-len(non)+i][c]=v
                elif dr=='up':
                    for c in range(w):
                        non=[g[r][c] for r in range(h) if g[r][c]!=bg]
                        for i,v in enumerate(non): res[i][c]=v
                elif dr=='left':
                    for r in range(h):
                        non=[g[r][c] for c in range(w) if g[r][c]!=bg]
                        for i,v in enumerate(non): res[r][i]=v
                elif dr=='right':
                    for r in range(h):
                        non=[g[r][c] for c in range(w) if g[r][c]!=bg]
                        for i,v in enumerate(non): res[r][w-len(non)+i]=v
                return res
            return f
        tools.append(CrossTool(f'grav_{direction}',mk(direction)))
    
    # 囲み充填
    def fill_enc(g,d):
        if not d.regions: return None
        res=copy_grid(g); h,w=len(g),len(g[0])
        for reg in d.regions:
            bc=Counter()
            for r,c in reg:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and g[nr][nc]!=d.bg: bc[g[nr][nc]]+=1
            if bc:
                col=bc.most_common(1)[0][0]
                for r,c in reg: res[r][c]=col
        return res
    tools.append(CrossTool('fill_enc',fill_enc))
    
    # 対称修復
    def sym_h(g,d):
        bg=d.bg; h,w=len(g),len(g[0]); res=copy_grid(g)
        for r in range(h):
            for c in range(w//2):
                mc=w-1-c
                if res[r][c]==bg and res[r][mc]!=bg: res[r][c]=res[r][mc]
                elif res[r][mc]==bg and res[r][c]!=bg: res[r][mc]=res[r][c]
        return res
    def sym_v(g,d):
        bg=d.bg; h,w=len(g),len(g[0]); res=copy_grid(g)
        for r in range(h//2):
            mr=h-1-r
            for c in range(w):
                if res[r][c]==bg and res[mr][c]!=bg: res[r][c]=res[mr][c]
                elif res[mr][c]==bg and res[r][c]!=bg: res[mr][c]=res[r][c]
        return res
    def sym_4(g,d):
        bg=d.bg; h,w=len(g),len(g[0]); res=copy_grid(g)
        for r in range((h+1)//2):
            for c in range((w+1)//2):
                ps=[(r,c),(r,w-1-c),(h-1-r,c),(h-1-r,w-1-c)]
                vals=[res[pr][pc] for pr,pc in ps if 0<=pr<h and 0<=pc<w and res[pr][pc]!=bg]
                if vals:
                    fill=Counter(vals).most_common(1)[0][0]
                    for pr,pc in ps:
                        if 0<=pr<h and 0<=pc<w and res[pr][pc]==bg: res[pr][pc]=fill
        return res
    for n,f in [('sym_h',sym_h),('sym_v',sym_v),('sym_4',sym_4)]:
        tools.append(CrossTool(n,f))
    
    # 成長/侵食
    def grow(g,d):
        bg=d.bg; h,w=len(g),len(g[0]); res=copy_grid(g)
        for r in range(h):
            for c in range(w):
                if g[r][c]!=bg:
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=r+dr,c+dc
                        if 0<=nr<h and 0<=nc<w and res[nr][nc]==bg: res[nr][nc]=g[r][c]
        return res
    def shrink(g,d):
        bg=d.bg; h,w=len(g),len(g[0]); res=copy_grid(g)
        for r in range(h):
            for c in range(w):
                if g[r][c]!=bg:
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=r+dr,c+dc
                        if not(0<=nr<h and 0<=nc<w) or g[nr][nc]==bg:
                            res[r][c]=bg; break
        return res
    tools.append(CrossTool('grow',grow))
    tools.append(CrossTool('shrink',shrink))
    
    # パネルBoolean
    def pan_or(g,d):
        if len(d.panels)<2: return None
        bg=d.bg; h=min(len(p) for p in d.panels); w=min(len(p[0]) for p in d.panels)
        res=[[bg]*w for _ in range(h)]
        for p in d.panels:
            for r in range(h):
                for c in range(w):
                    if p[r][c]!=bg: res[r][c]=p[r][c]
        return res
    def pan_and(g,d):
        if len(d.panels)<2: return None
        bg=d.bg; h=min(len(p) for p in d.panels); w=min(len(p[0]) for p in d.panels)
        res=[[bg]*w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if all(p[r][c]!=bg for p in d.panels): res[r][c]=d.panels[0][r][c]
        return res
    def pan_xor(g,d):
        if len(d.panels)<2: return None
        bg=d.bg; h=min(len(p) for p in d.panels); w=min(len(p[0]) for p in d.panels)
        res=[[bg]*w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                non=[p[r][c] for p in d.panels if p[r][c]!=bg]
                if len(non)==1: res[r][c]=non[0]
        return res
    def pan_diff(g,d):
        if len(d.panels)!=2: return None
        bg=d.bg; p1,p2=d.panels
        h=min(len(p1),len(p2)); w=min(len(p1[0]),len(p2[0]))
        res=[[bg]*w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if p1[r][c]!=bg and p2[r][c]==bg: res[r][c]=p1[r][c]
                elif p2[r][c]!=bg and p1[r][c]==bg: res[r][c]=p2[r][c]
        return res
    def pan_0(g,d): return [r[:] for r in d.panels[0]] if d.panels else None
    def pan_1(g,d): return [r[:] for r in d.panels[-1]] if d.panels else None
    for n,f in [('pan_or',pan_or),('pan_and',pan_and),('pan_xor',pan_xor),
                ('pan_diff',pan_diff),('pan_0',pan_0),('pan_-1',pan_1)]:
        tools.append(CrossTool(n,f))
    
    # 色操作
    def swap_2(g,d):
        bg=d.bg; cc=color_counts(g); del cc[bg]
        if len(cc)<2: return None
        c1,c2=cc.most_common(2)[0][0],cc.most_common(2)[1][0]
        return [[c2 if v==c1 else c1 if v==c2 else v for v in row] for row in g]
    def swap_bf(g,d):
        bg=d.bg; cc=color_counts(g); del cc[bg]
        if not cc: return None
        fg=cc.most_common(1)[0][0]
        return [[fg if v==bg else bg if v==fg else v for v in row] for row in g]
    tools.append(CrossTool('swap_2',swap_2))
    tools.append(CrossTool('swap_bf',swap_bf))
    
    for col in range(10):
        def mk_er(c):
            def f(g,d):
                bg=d.bg
                if c not in grid_colors(g) or c==bg: return None
                return [[bg if v==c else v for v in row] for row in g]
            return f
        tools.append(CrossTool(f'erase_{col}',mk_er(col)))
    
    # 接続
    def conn_h(g,d):
        bg=d.bg; h,w=len(g),len(g[0]); res=copy_grid(g)
        for r in range(h):
            by=defaultdict(list)
            for c in range(w):
                if g[r][c]!=bg: by[g[r][c]].append(c)
            for col,ps in by.items():
                if len(ps)>=2:
                    for c in range(min(ps),max(ps)+1): res[r][c]=col
        return res
    def conn_v(g,d):
        bg=d.bg; h,w=len(g),len(g[0]); res=copy_grid(g)
        for c in range(w):
            by=defaultdict(list)
            for r in range(h):
                if g[r][c]!=bg: by[g[r][c]].append(r)
            for col,ps in by.items():
                if len(ps)>=2:
                    for r in range(min(ps),max(ps)+1): res[r][c]=col
        return res
    tools.append(CrossTool('conn_h',conn_h))
    tools.append(CrossTool('conn_v',conn_v))
    
    # クロップ
    def crop(g,d):
        bg=d.bg; h,w=len(g),len(g[0])
        rows=[r for r in range(h) if any(g[r][c]!=bg for c in range(w))]
        cols=[c for c in range(w) if any(g[r][c]!=bg for r in range(h))]
        if not rows or not cols: return None
        return [g[r][min(cols):max(cols)+1] for r in range(min(rows),max(rows)+1)]
    tools.append(CrossTool('crop',crop))
    
    # スケール
    def up2(g,d):
        h,w=len(g),len(g[0])
        return [[g[r//2][c//2] for c in range(w*2)] for r in range(h*2)]
    def up3(g,d):
        h,w=len(g),len(g[0])
        return [[g[r//3][c//3] for c in range(w*3)] for r in range(h*3)]
    tools.append(CrossTool('up2x',up2))
    tools.append(CrossTool('up3x',up3))
    
    # スタンプ
    def stamp_lg(g,d):
        if not d.objects or len(d.objects)<2: return None
        objs=sorted(d.objects,key=len,reverse=True)
        tmpl,_,_=_obj_patch(objs[0],d.bg); th,tw=len(tmpl),len(tmpl[0])
        res=copy_grid(g)
        for obj in objs[1:]:
            cr=sum(r for r,c,v in obj)//len(obj)
            cc=sum(c for r,c,v in obj)//len(obj)
            res=_paste(res,tmpl,cr-th//2,cc-tw//2,d.bg)
        return res
    tools.append(CrossTool('stamp_lg',stamp_lg))
    
    # 行列ギャップ充填
    def fill_row_gap(g,d):
        bg=d.bg; res=copy_grid(g)
        for r in range(len(g)):
            ps=[(c,g[r][c]) for c in range(len(g[0])) if g[r][c]!=bg]
            if len(ps)>=2:
                for i in range(len(ps)-1):
                    c1,v1=ps[i]; c2,v2=ps[i+1]
                    if v1==v2:
                        for c in range(c1+1,c2): res[r][c]=v1
        return res
    def fill_col_gap(g,d):
        bg=d.bg; h,w=len(g),len(g[0]); res=copy_grid(g)
        for c in range(w):
            ps=[(r,g[r][c]) for r in range(h) if g[r][c]!=bg]
            if len(ps)>=2:
                for i in range(len(ps)-1):
                    r1,v1=ps[i]; r2,v2=ps[i+1]
                    if v1==v2:
                        for r in range(r1+1,r2): res[r][c]=v1
        return res
    tools.append(CrossTool('fill_row_gap',fill_row_gap))
    tools.append(CrossTool('fill_col_gap',fill_col_gap))
    
    return tools

# ═══════════════════════════════════════
# CrossRouter — ルーティング（探索）
# ═══════════════════════════════════════

class CrossRouter:
    def __init__(self, tools=None):
        self.tools = tools or _build_tools()
    
    def solve(self, train_pairs, test_inputs, timeout=30.0):
        t0=time.time(); deadline=t0+timeout; solutions=[]
        inp0, out0 = train_pairs[0]
        decomps0 = CrossDecomposer.decompose_all(inp0)
        
        # Stage 1: 単体道具
        for decomp in decomps0:
            if time.time()>deadline: break
            for tool in self.tools:
                result = tool.apply(inp0, decomp)
                if result is not None and grid_eq(result, out0):
                    fn = self._bind(decomp.kind, tool)
                    if self._verify(fn, train_pairs):
                        solutions.append((f'{decomp.kind}:{tool.name}', fn))
        
        # Stage 2: 収束
        if not solutions:
            no_conv = {'up2x','up3x'}
            for decomp in decomps0:
                if time.time()>deadline: break
                for tool in self.tools:
                    if tool.name in no_conv: continue
                    cfn = self._bind_conv(decomp.kind, tool)
                    r = cfn(inp0)
                    if r is not None and grid_eq(r, out0):
                        if self._verify(cfn, train_pairs):
                            solutions.append((f'conv:{decomp.kind}:{tool.name}', cfn))
        
        # Stage 3: 2道具合成
        if not solutions and time.time()<deadline:
            for decomp in decomps0:
                if time.time()>deadline or solutions: break
                mids=[]
                for ta in self.tools:
                    mid=ta.apply(inp0,decomp)
                    if mid is not None and not grid_eq(mid,inp0):
                        mids.append((ta,mid))
                mids=mids[:15]
                for ta,mid in mids:
                    if time.time()>deadline or solutions: break
                    mid_ds=CrossDecomposer.decompose_all(mid)
                    for md in mid_ds:
                        if time.time()>deadline or solutions: break
                        for tb in self.tools:
                            r=tb.apply(mid,md)
                            if r is not None and grid_eq(r,out0):
                                fn=self._bind_pipe(decomp.kind,ta,tb)
                                if self._verify(fn, train_pairs):
                                    solutions.append((f'{decomp.kind}:{ta.name}->{tb.name}',fn))
                                    break
        
        # Stage 4: 色マップ
        if not solutions:
            cm=self._learn_cmap(train_pairs)
            if cm:
                m=dict(cm)
                fn=lambda g,m=m:[[m.get(c,c) for c in row] for row in g]
                if self._verify(fn,train_pairs):
                    solutions.append(('cmap',fn))
        
        if not solutions: return None, []
        name,fn=solutions[0]
        preds=[]
        for ti in test_inputs:
            try:
                p=fn(ti)
                if p is None: return None, []
                preds.append(p)
            except: return None, []
        return preds, solutions
    
    def _bind(self, dk, tool):
        def fn(g):
            for d in CrossDecomposer.decompose_all(g):
                if d.kind==dk: return tool.apply(g,d)
            return None
        return fn
    
    def _bind_conv(self, dk, tool, mx=20):
        def fn(g):
            cur=g
            for _ in range(mx):
                for d in CrossDecomposer.decompose_all(cur):
                    if d.kind==dk:
                        nxt=tool.apply(cur,d)
                        if nxt is None or (isinstance(nxt,list) and len(nxt)>100): return None
                        if grid_eq(cur,nxt): return cur
                        cur=nxt; break
                else: return None
            return cur
        return fn
    
    def _bind_pipe(self, dk, ta, tb):
        def fn(g):
            for d in CrossDecomposer.decompose_all(g):
                if d.kind==dk:
                    mid=ta.apply(g,d)
                    if mid is None: return None
                    for md in CrossDecomposer.decompose_all(mid):
                        r=tb.apply(mid,md)
                        if r is not None: return r
                    return None
            return None
        return fn
    
    def _verify(self, fn, pairs):
        for inp,exp in pairs:
            try:
                r=fn(inp)
                if r is None or not grid_eq(r,exp): return False
            except: return False
        return True
    
    def _learn_cmap(self, pairs):
        cm={}
        for inp,out in pairs:
            h,w=len(inp),len(inp[0])
            if len(out)!=h or len(out[0])!=w: return None
            for r in range(h):
                for c in range(w):
                    ic,oc=inp[r][c],out[r][c]
                    if ic in cm:
                        if cm[ic]!=oc: return None
                    else: cm[ic]=oc
        return cm if any(k!=v for k,v in cm.items()) else None

def solve_cross2(train_pairs, test_inputs, timeout=30.0):
    router=CrossRouter()
    preds,sols=router.solve(train_pairs,test_inputs,timeout)
    if preds is None: return None, []
    return [[p] for p in preds], sols

if __name__=='__main__':
    import json, os, sys
    EVAL_DIR='/private/tmp/arc-agi-2/data/evaluation'
    tids=sorted(f.replace('.json','') for f in os.listdir(EVAL_DIR))
    correct=0; total=0
    for tid in tids:
        with open(f'{EVAL_DIR}/{tid}.json') as f:
            task=json.load(f)
        tp=[(ex['input'],ex['output']) for ex in task['train']]
        ti=[t['input'] for t in task['test']]
        t0=time.time()
        preds,sols=solve_cross2(tp,ti,timeout=15)
        elapsed=time.time()-t0; total+=1
        ok=False
        if preds:
            ok=all(grid_eq(preds[j][0],task['test'][j]['output']) for j in range(len(task['test'])))
        if ok: correct+=1
        names=[s[0] for s in sols[:2]]
        st='✅' if ok else '❌' if sols else '  '
        print(f'{total:3d}/{len(tids)} {st} {tid} | {len(sols)}sol {elapsed:.1f}s | {names}',flush=True)
    print(f'\n=== {correct}/{total} ===',flush=True)
