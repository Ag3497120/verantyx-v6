"""
arc/cross_brain_v2.py — Cross Brain V2: 全方向フル強化

断片(勘) → Cross(ルーティング) → フル(大学院知識)
全カテゴリの知識を深化
"""

import numpy as np
from collections import Counter, defaultdict, deque
from scipy.ndimage import label as scipy_label
from itertools import product

def _bg(g): return int(Counter(np.array(g).flatten()).most_common(1)[0][0])

def _objs(g, bg, conn=8):
    struct = np.ones((3,3),dtype=int) if conn==8 else np.array([[0,1,0],[1,1,1],[0,1,0]])
    mask=(np.array(g)!=bg).astype(int); labeled,n=scipy_label(mask,structure=struct)
    objs=[]
    for i in range(1,n+1):
        cells=list(zip(*np.where(labeled==i)))
        colors=[int(g[r,c]) for r,c in cells]
        r1=min(r for r,c in cells);c1=min(c for r,c in cells)
        r2=max(r for r,c in cells);c2=max(c for r,c in cells)
        objs.append({'cells':cells,'size':len(cells),'color':Counter(colors).most_common(1)[0][0],
            'colors':set(colors),'bbox':(r1,c1,r2,c2),'bh':r2-r1+1,'bw':c2-c1+1,
            'shape':frozenset((r-r1,c-c1) for r,c in cells),
            'is_rect':len(cells)==(r2-r1+1)*(c2-c1+1),
            'n_colors':len(set(colors))})
    return objs

def grid_eq(a,b):
    a,b=np.array(a),np.array(b)
    return a.shape==b.shape and np.array_equal(a,b)

def _verify(fn, tp, ti):
    try:
        for inp,out in tp:
            p=fn(inp)
            if p is None or not grid_eq(p,out): return None
        return fn(ti)
    except: return None

# ══════════════════════════════════════════════════════════════
# SMALLER: 出力が入力より小さい（crop/extract/panel比較/条件抽出）
# ══════════════════════════════════════════════════════════════

def crop_by_attr(tp, ti):
    """属性別crop: 最大/最小/仲間はずれ/特定色/矩形/非矩形"""
    gi=np.array(ti); bg=_bg(gi)
    
    for key in ['size','bh','bw','color','n_colors']:
        for which in ['max','min']:
            def mk(k,w):
                def fn(grid):
                    g=np.array(grid);b=_bg(g);objs=_objs(g,b)
                    if not objs: return None
                    o=(max if w=='max' else min)(objs,key=lambda x:x[k])
                    r1,c1,r2,c2=o['bbox']
                    return g[r1:r2+1,c1:c2+1].tolist()
                return fn
            r=_verify(mk(key,which),tp,ti)
            if r is not None: return r
    
    # 仲間はずれcrop
    for attr in ['shape','color','size','is_rect']:
        def mk(a):
            def fn(grid):
                g=np.array(grid);b=_bg(g);objs=_objs(g,b)
                if len(objs)<2: return None
                vals=[o.get(a,o['shape']) for o in objs]
                cc=Counter(tuple(v) if isinstance(v,frozenset) else v for v in vals)
                target_val=cc.most_common()[-1][0]
                uniq=[o for i,o in enumerate(objs) if (tuple(vals[i]) if isinstance(vals[i],frozenset) else vals[i])==target_val]
                if len(uniq)!=1: return None
                o=uniq[0]; r1,c1,r2,c2=o['bbox']
                return g[r1:r2+1,c1:c2+1].tolist()
            return fn
        r=_verify(mk(attr),tp,ti)
        if r is not None: return r
    
    # 枠crop (1px枠 / 最大オブジェクト内部)
    for margin in [1,2]:
        def mk(m):
            def fn(grid):
                g=np.array(grid);b=_bg(g);objs=_objs(g,b)
                if not objs: return None
                o=max(objs,key=lambda x:x['size']); r1,c1,r2,c2=o['bbox']
                inner=g[r1+m:r2+1-m,c1+m:c2+1-m]
                return inner.tolist() if inner.size>0 else None
            return fn
        r=_verify(mk(margin),tp,ti)
        if r is not None: return r
    
    # 特定色crop
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out); b=_bg(ga)
        objs=_objs(ga,b)
        for o in objs:
            r1,c1,r2,c2=o['bbox']
            if grid_eq(ga[r1:r2+1,c1:c2+1].tolist(),out):
                target_color=o['color']
                def mk(tc):
                    def fn(grid):
                        g=np.array(grid);bg2=_bg(g);os2=_objs(g,bg2)
                        match=[o2 for o2 in os2 if o2['color']==tc]
                        if not match: return None
                        o2=max(match,key=lambda x:x['size'])
                        r1,c1,r2,c2=o2['bbox']
                        return g[r1:r2+1,c1:c2+1].tolist()
                    return fn
                r=_verify(mk(target_color),tp,ti)
                if r is not None: return r
        break
    
    return None

def panel_compare(tp, ti):
    """パネル分割→比較結果（差分/共通/XOR/OR/AND/majority/カウント）"""
    for n in [2,3,4,5]:
        for ax in ['h','v']:
            for op in ['or','xor','and','first','last','max','min','majority',
                       'diff','count_nonbg','unique_only']:
                def mk(nn,axx,opp):
                    def fn(grid):
                        g=np.array(grid);h,w=g.shape;bg=_bg(g)
                        if axx=='h' and h%nn==0:
                            ph=h//nn; panels=[g[i*ph:(i+1)*ph,:] for i in range(nn)]
                        elif axx=='v' and w%nn==0:
                            pw=w//nn; panels=[g[:,i*pw:(i+1)*pw] for i in range(nn)]
                        else: return None
                        rh,rw=panels[0].shape; result=np.full((rh,rw),bg,dtype=int)
                        for r in range(rh):
                            for c in range(rw):
                                vals=[int(p[r,c]) for p in panels]
                                nb=[v for v in vals if v!=bg]
                                if opp=='or': result[r,c]=nb[0] if nb else bg
                                elif opp=='xor': result[r,c]=nb[0] if len(nb)==1 else bg
                                elif opp=='and': result[r,c]=nb[0] if len(nb)==len(vals) and len(set(nb))==1 else bg
                                elif opp=='first': result[r,c]=nb[0] if nb else bg
                                elif opp=='last': result[r,c]=nb[-1] if nb else bg
                                elif opp=='max': result[r,c]=max(nb) if nb else bg
                                elif opp=='min': result[r,c]=min(nb) if nb else bg
                                elif opp=='majority': result[r,c]=Counter(nb).most_common(1)[0][0] if nb else bg
                                elif opp=='diff':
                                    if len(set(nb))>1: result[r,c]=max(set(nb),key=nb.count)
                                    elif len(nb)==1: result[r,c]=nb[0]
                                elif opp=='count_nonbg':
                                    result[r,c]=len(nb) if len(nb)>0 else bg
                                elif opp=='unique_only':
                                    uniq=[v for v in nb if nb.count(v)==1]
                                    result[r,c]=uniq[0] if len(uniq)==1 else bg
                        return result.tolist()
                    return fn
                r=_verify(mk(n,ax,op),tp,ti)
                if r is not None: return r
    return None

def conditional_extract(tp, ti):
    """条件付き抽出: セパレータで区切った中の特定パネル"""
    for inp,out in tp:
        ga=np.array(inp); bg=_bg(ga); h,w=ga.shape; go=np.array(out)
        
        # セパレータ行検出
        h_seps=[]; v_seps=[]
        for r in range(h):
            vals=set(int(v) for v in ga[r])
            if len(vals)==1 and vals.pop()!=bg: h_seps.append(r)
        for c in range(w):
            vals=set(int(v) for v in ga[:,c])
            if len(vals)==1 and vals.pop()!=bg: v_seps.append(c)
        
        if not h_seps and not v_seps: break
        
        # パネルに分割
        rows=[-1]+h_seps+[h]; cols=[-1]+v_seps+[w]
        panels=[]
        for i in range(len(rows)-1):
            for j in range(len(cols)-1):
                r1,r2=rows[i]+1,rows[i+1]; c1,c2=cols[j]+1,cols[j+1]
                if r2>r1 and c2>c1: panels.append((ga[r1:r2,c1:c2],(r1,c1)))
        
        # どのパネルが出力に一致するか、その条件は何か
        for idx,(panel,pos) in enumerate(panels):
            if grid_eq(panel.tolist(), out):
                # 条件: 最も非BGセルが多い？少ない？特定色が含まれる？
                for condition in ['most_fg','least_fg','most_colors','least_colors']:
                    def mk(cond, seps_h, seps_v, hh, ww):
                        def fn(grid):
                            g=np.array(grid); bg2=_bg(g); h2,w2=g.shape
                            hs2=[]; vs2=[]
                            for r in range(h2):
                                vals2=set(int(v) for v in g[r])
                                if len(vals2)==1 and vals2.pop()!=bg2: hs2.append(r)
                            for c in range(w2):
                                vals2=set(int(v) for v in g[:,c])
                                if len(vals2)==1 and vals2.pop()!=bg2: vs2.append(c)
                            rs2=[-1]+hs2+[h2]; cs2=[-1]+vs2+[w2]
                            ps2=[]
                            for i in range(len(rs2)-1):
                                for j in range(len(cs2)-1):
                                    r1,r2=rs2[i]+1,rs2[i+1]; c1,c2=cs2[j]+1,cs2[j+1]
                                    if r2>r1 and c2>c1: ps2.append(g[r1:r2,c1:c2])
                            if not ps2: return None
                            if cond=='most_fg': p=max(ps2,key=lambda x:(x!=bg2).sum())
                            elif cond=='least_fg': p=min(ps2,key=lambda x:(x!=bg2).sum())
                            elif cond=='most_colors': p=max(ps2,key=lambda x:len(set(int(v) for v in x.flatten())-{bg2}))
                            elif cond=='least_colors': p=min(ps2,key=lambda x:len(set(int(v) for v in x.flatten())-{bg2}))
                            else: return None
                            return p.tolist()
                        return fn
                    r=_verify(mk(condition,h_seps,v_seps,h,w),tp,ti)
                    if r is not None: return r
        break
    return None


# ══════════════════════════════════════════════════════════════
# ADD: 何かが追加される（リバーシ/接続/対称/flood/砲台/拡張/射影）
# ══════════════════════════════════════════════════════════════

def reversi_variants(tp, ti):
    """リバーシ全バリエーション"""
    for dirs in [
        [(-1,0),(1,0),(0,-1),(0,1)],
        [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)],
    ]:
        for it in [False, True]:
            for fm in ['same','any']:
                def mk(d,i,f):
                    def fn(grid):
                        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g);orig=g.copy()
                        for _ in range(30 if i else 1):
                            ch=False
                            for r in range(h):
                                for c in range(w):
                                    if g[r,c]!=bg: continue
                                    for dr,dc in d:
                                        nr,nc=r+dr,c+dc;c1=None
                                        while 0<=nr<h and 0<=nc<w:
                                            if g[nr,nc]!=bg: c1=int(g[nr,nc]); break
                                            nr+=dr; nc+=dc
                                        nr,nc=r-dr,c-dc;c2=None
                                        while 0<=nr<h and 0<=nc<w:
                                            if g[nr,nc]!=bg: c2=int(g[nr,nc]); break
                                            nr-=dr; nc-=dc
                                        if c1 is not None and c2 is not None:
                                            if f=='same' and c1==c2: g[r,c]=c1; ch=True; break
                                            elif f=='any': g[r,c]=c1; ch=True; break
                            if not ch: break
                        return g.tolist() if not np.array_equal(g,orig) else None
                    return fn
                r=_verify(mk(dirs,it,fm),tp,ti)
                if r is not None: return r
    return None

def connect_variants(tp, ti):
    """点つなぎ全バリエーション（直線/対角/最近点/全ペア）"""
    for diag in [False, True]:
        for mode in ['all_pairs','nearest']:
            def mk(dg,md):
                def fn(grid):
                    g=np.array(grid).copy();h,w=g.shape;bg=_bg(g);orig=g.copy()
                    for color in set(int(v) for v in g.flatten())-{bg}:
                        pts=[(r,c) for r in range(h) for c in range(w) if g[r,c]==color]
                        if len(pts)<2: continue
                        if md=='all_pairs':
                            for i,(r1,c1) in enumerate(pts):
                                for r2,c2 in pts[i+1:]:
                                    _draw_line(g,r1,c1,r2,c2,color,bg,dg)
                        elif md=='nearest':
                            used=set()
                            for r1,c1 in pts:
                                best=None; bd=999
                                for r2,c2 in pts:
                                    if (r1,c1)==(r2,c2): continue
                                    if (r2,c2) in used: continue
                                    d=abs(r2-r1)+abs(c2-c1)
                                    if d<bd: bd=d; best=(r2,c2)
                                if best:
                                    _draw_line(g,r1,c1,best[0],best[1],color,bg,dg)
                                    used.add((r1,c1))
                    return g.tolist() if not np.array_equal(g,orig) else None
                return fn
            r=_verify(mk(diag,mode),tp,ti)
            if r is not None: return r
    return None

def _draw_line(g,r1,c1,r2,c2,color,bg,allow_diag):
    h,w=g.shape
    if r1==r2:
        for c in range(min(c1,c2)+1,max(c1,c2)):
            if g[r1,c]==bg: g[r1,c]=color
    elif c1==c2:
        for r in range(min(r1,r2)+1,max(r1,r2)):
            if g[r,c1]==bg: g[r,c1]=color
    elif allow_diag and abs(r2-r1)==abs(c2-c1):
        dr=1 if r2>r1 else -1; dc=1 if c2>c1 else -1
        r,c=r1+dr,c1+dc
        while (r,c)!=(r2,c2) and 0<=r<h and 0<=c<w:
            if g[r,c]==bg: g[r,c]=color
            r+=dr; c+=dc

def symmetry_variants(tp, ti):
    """対称補完全バリエーション"""
    for mode in ['lr','ud','both','rot180','diag','anti_diag','rot90']:
        def mk(m):
            def fn(grid):
                g=np.array(grid).copy();h,w=g.shape;bg=_bg(g);orig=g.copy()
                for r in range(h):
                    for c in range(w):
                        if g[r,c]!=bg: continue
                        mirrors=[]
                        if m in ('lr','both'): mirrors.append((r,w-1-c))
                        if m in ('ud','both'): mirrors.append((h-1-r,c))
                        if m in ('both','rot180'): mirrors.append((h-1-r,w-1-c))
                        if m=='diag' and h==w: mirrors.append((c,r))
                        if m=='anti_diag' and h==w: mirrors.append((w-1-c,h-1-r))
                        if m=='rot90' and h==w:
                            mirrors+=[(c,h-1-r),(h-1-r,w-1-c),(w-1-c,r)]
                        for mr,mc in mirrors:
                            if 0<=mr<h and 0<=mc<w and g[mr,mc]!=bg:
                                g[r,c]=g[mr,mc]; break
                return g.tolist() if not np.array_equal(g,orig) else None
            return fn
        r=_verify(mk(mode),tp,ti)
        if r is not None: return r
    return None

def flood_variants(tp, ti):
    """Flood fill全バリエーション"""
    # 近傍推定色
    def flood_nb(grid):
        g=np.array(grid);h,w=g.shape;bg=_bg(g)
        vis=np.zeros((h,w),bool);q=[]
        for r in range(h):
            for c in [0,w-1]:
                if g[r,c]==bg and not vis[r,c]: q.append((r,c)); vis[r,c]=True
        for c in range(w):
            for r in [0,h-1]:
                if g[r,c]==bg and not vis[r,c]: q.append((r,c)); vis[r,c]=True
        while q:
            r,c=q.pop(0)
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=r+dr,c+dc
                if 0<=nr<h and 0<=nc<w and not vis[nr,nc] and g[nr,nc]==bg:
                    vis[nr,nc]=True; q.append((nr,nc))
        result=g.copy();ch=False
        for r in range(h):
            for c in range(w):
                if g[r,c]==bg and not vis[r,c]:
                    nb=set()
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=r+dr,c+dc
                        if 0<=nr<h and 0<=nc<w and g[nr,nc]!=bg: nb.add(int(g[nr,nc]))
                    result[r,c]=nb.pop() if len(nb)==1 else (min(nb) if nb else 1)
                    ch=True
        return result.tolist() if ch else None
    
    r=_verify(flood_nb,tp,ti)
    if r is not None: return r
    
    # 固定色flood
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out);bg=_bg(ga)
        for rr in range(ga.shape[0]):
            for cc in range(ga.shape[1]):
                if ga[rr,cc]==bg and go[rr,cc]!=bg:
                    fc=int(go[rr,cc])
                    def mk(fcc):
                        def fn(grid):
                            g=np.array(grid);h,w=g.shape;bg2=_bg(g)
                            vis=np.zeros((h,w),bool);q=[]
                            for r2 in range(h):
                                for c2 in [0,w-1]:
                                    if g[r2,c2]==bg2 and not vis[r2,c2]: q.append((r2,c2)); vis[r2,c2]=True
                            for c2 in range(w):
                                for r2 in [0,h-1]:
                                    if g[r2,c2]==bg2 and not vis[r2,c2]: q.append((r2,c2)); vis[r2,c2]=True
                            while q:
                                r2,c2=q.pop(0)
                                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                    nr,nc=r2+dr,c2+dc
                                    if 0<=nr<h and 0<=nc<w and not vis[nr,nc] and g[nr,nc]==bg2:
                                        vis[nr,nc]=True; q.append((nr,nc))
                            result=g.copy();ch2=False
                            for r2 in range(h):
                                for c2 in range(w):
                                    if g[r2,c2]==bg2 and not vis[r2,c2]: result[r2,c2]=fcc; ch2=True
                            return result.tolist() if ch2 else None
                        return fn
                    r=_verify(mk(fc),tp,ti)
                    if r is not None: return r
                    break
            break
        break
    
    # 囲碁territory: 1色で囲まれたBG
    def go_territory(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g)
        struct4=np.array([[0,1,0],[1,1,1],[0,1,0]])
        bg_mask=(g==bg).astype(int)
        labeled,n=scipy_label(bg_mask,structure=struct4)
        ch=False
        for i in range(1,n+1):
            cells=list(zip(*np.where(labeled==i)))
            if any(r==0 or r==h-1 or c==0 or c==w-1 for r,c in cells): continue
            border=set()
            for r,c in cells:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and g[nr,nc]!=bg: border.add(int(g[nr,nc]))
            if len(border)==1:
                fill=border.pop()
                for r,c in cells: g[r,c]=fill; ch=True
        return g.tolist() if ch else None
    
    r=_verify(go_territory,tp,ti)
    if r is not None: return r
    return None

def morphology_variants(tp, ti):
    """モルフォロジー: 膨張/収縮/エッジ/内部"""
    def dilate(grid,steps=1):
        g=np.array(grid);h,w=g.shape;bg=_bg(g);result=g.copy()
        for _ in range(steps):
            new=result.copy()
            for r in range(h):
                for c in range(w):
                    if result[r,c]!=bg: continue
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=r+dr,c+dc
                        if 0<=nr<h and 0<=nc<w and result[nr,nc]!=bg:
                            new[r,c]=result[nr,nc]; break
                    else:
                        for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                            nr,nc=r+dr,c+dc
                            if 0<=nr<h and 0<=nc<w and result[nr,nc]!=bg:
                                new[r,c]=result[nr,nc]; break
            result=new
        return result.tolist() if not np.array_equal(result,g) else None
    def erode(grid):
        g=np.array(grid);h,w=g.shape;bg=_bg(g);result=g.copy();ch=False
        for r in range(h):
            for c in range(w):
                if g[r,c]==bg: continue
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if not(0<=nr<h and 0<=nc<w) or g[nr,nc]==bg: result[r,c]=bg; ch=True; break
        return result.tolist() if ch else None
    def edge(grid):
        g=np.array(grid);h,w=g.shape;bg=_bg(g);result=np.full_like(g,bg);ch=False
        for r in range(h):
            for c in range(w):
                if g[r,c]==bg: continue
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if not(0<=nr<h and 0<=nc<w) or g[nr,nc]==bg: result[r,c]=g[r,c]; ch=True; break
        return result.tolist() if ch else None
    
    for fn in [lambda g:dilate(g,1),lambda g:dilate(g,2),lambda g:dilate(g,3),erode,edge]:
        r=_verify(fn,tp,ti)
        if r is not None: return r
    return None

def diagonal_draw(tp, ti):
    """対角線引き（ブロック崩し/ピンボール）"""
    def breakout(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g);orig=g.copy()
        objs=_objs(g,bg)
        single=[o for o in objs if o['size']==1]
        if not single: return None
        for o in single:
            r0,c0=o['cells'][0]; color=o['color']
            for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                r,c=r0+dr,c0+dc
                while 0<=r<h and 0<=c<w:
                    if g[r,c]!=bg: break
                    g[r,c]=color; r+=dr; c+=dc
        return g.tolist() if not np.array_equal(g,orig) else None
    r=_verify(breakout,tp,ti)
    if r is not None: return r
    return None

def minesweeper(tp, ti):
    """マインスイーパー: 近傍非BG数をセルに書く"""
    def ms(grid):
        g=np.array(grid);h,w=g.shape;bg=_bg(g);result=g.copy();ch=False
        for r in range(h):
            for c in range(w):
                if g[r,c]!=bg: continue
                cnt=0
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        nr,nc=r+dr,c+dc
                        if 0<=nr<h and 0<=nc<w and g[nr,nc]!=bg: cnt+=1
                if 0<cnt<=9: result[r,c]=cnt; ch=True
        return result.tolist() if ch else None
    r=_verify(ms,tp,ti)
    if r is not None: return r
    return None

def cannon_stamp(tp, ti):
    try:
        from arc.cross3d_projection import cannon_solve
        def fn(grid): return cannon_solve(tp, grid)
        r=_verify(fn,tp,ti)
        if r is not None: return r
    except: pass
    return None


# ══════════════════════════════════════════════════════════════
# ADD_REM: 移動系（重力/壁押し/オブジェクト移動/回転配置）
# ══════════════════════════════════════════════════════════════

def gravity_variants(tp, ti):
    """重力全バリエーション（壁あり/なし）"""
    # 通常重力
    for d in ['down','up','left','right']:
        def mk(dd):
            def fn(grid):
                g=np.array(grid);h,w=g.shape;bg=_bg(g);result=np.full_like(g,bg)
                if dd in ('down','up'):
                    for c in range(w):
                        vals=[int(g[r,c]) for r in range(h) if g[r,c]!=bg]
                        if dd=='down':
                            for i,v in enumerate(reversed(vals)): result[h-1-i,c]=v
                        else:
                            for i,v in enumerate(vals): result[i,c]=v
                else:
                    for r in range(h):
                        vals=[int(g[r,c]) for c in range(w) if g[r,c]!=bg]
                        if dd=='right':
                            for i,v in enumerate(reversed(vals)): result[r,w-1-i]=v
                        else:
                            for i,v in enumerate(vals): result[r,i]=v
                return result.tolist()
            return fn
        r=_verify(mk(d),tp,ti)
        if r is not None: return r
    
    # 壁あり重力
    for d in ['down','up','left','right']:
        def mk(dd):
            def fn(grid):
                g=np.array(grid);h,w=g.shape;bg=_bg(g)
                colors=Counter(int(v) for v in g.flatten() if v!=bg)
                if len(colors)<2: return None
                wall=colors.most_common(1)[0][0]
                result=g.copy()
                dr,dc={'down':(1,0),'up':(-1,0),'left':(0,-1),'right':(0,1)}[dd]
                rows=range(h-1,-1,-1) if dr>0 else range(h)
                cols=range(w-1,-1,-1) if dc>0 else range(w)
                moved=False
                for r in rows:
                    for c in cols:
                        v=int(result[r,c])
                        if v==bg or v==wall: continue
                        nr,nc=r,c
                        while True:
                            nnr,nnc=nr+dr,nc+dc
                            if not(0<=nnr<h and 0<=nnc<w): break
                            if result[nnr,nnc]!=bg: break
                            nr,nc=nnr,nnc
                        if (nr,nc)!=(r,c): result[nr,nc]=v; result[r,c]=bg; moved=True
                return result.tolist() if moved else None
            return fn
        r=_verify(mk(d),tp,ti)
        if r is not None: return r
    return None

def object_translate(tp, ti):
    """オブジェクト全体を一定ベクトルで移動"""
    movements=[]
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=_bg(ga)
        objs_i=_objs(ga,bg); objs_o=_objs(go,bg)
        for oi in objs_i:
            for oo in objs_o:
                if oi['color']==oo['color'] and oi['shape']==oo['shape']:
                    dr=oo['bbox'][0]-oi['bbox'][0]; dc=oo['bbox'][1]-oi['bbox'][1]
                    movements.append((dr,dc)); break
    if not movements or len(set(movements))!=1: return None
    dr,dc=movements[0]
    if dr==0 and dc==0: return None
    gi=np.array(ti);h,w=gi.shape;bg=_bg(gi)
    result=np.full_like(gi,bg)
    for r in range(h):
        for c in range(w):
            if gi[r,c]!=bg:
                nr,nc=r+dr,c+dc
                if 0<=nr<h and 0<=nc<w: result[nr,nc]=gi[r,c]
    return result.tolist()

def rotate_transform(tp, ti):
    """回転/転置/反転"""
    gi=np.array(ti)
    for k in [1,2,3]:
        r=_verify(lambda g,kk=k: np.rot90(np.array(g),kk).tolist(),tp,ti)
        if r is not None: return r
    r=_verify(lambda g: np.array(g).T.tolist(),tp,ti)
    if r is not None: return r
    for ax in [0,1]:
        r=_verify(lambda g,a=ax: np.flip(np.array(g),a).tolist(),tp,ti)
        if r is not None: return r
    return None


# ══════════════════════════════════════════════════════════════
# CHANGE: 色変更のみ（色マップ/近傍ルール/テンプレ/偶奇）
# ══════════════════════════════════════════════════════════════

def colormap_full(tp, ti):
    """色変換マッピング"""
    m={}
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                iv,ov=int(ga[r,c]),int(go[r,c])
                if iv!=ov:
                    if iv in m and m[iv]!=ov: return None
                    m[iv]=ov
    if not m: return None
    gi=np.array(ti);result=gi.copy()
    for o,n in m.items(): result[gi==o]=n
    r=_verify(lambda g: (lambda ga: (lambda r: r.tolist())(np.where(np.isin(ga, list(m.keys())), np.vectorize(m.get)(np.clip(ga,0,9)), ga)))(np.array(g)),tp,ti)
    if r is not None: return r
    # simple apply
    def apply_cm(grid):
        g=np.array(grid);r2=g.copy()
        for o,n in m.items(): r2[g==o]=n
        return r2.tolist()
    return _verify(apply_cm,tp,ti)

def template_match(tp, ti):
    """テンプレートマッチ（3x3/5x5近傍）"""
    for radius in [1,2]:
        rules={};ok2=True
        for inp,out in tp:
            ga,go=np.array(inp),np.array(out)
            if ga.shape!=go.shape: return None
            h,w=ga.shape
            for r in range(h):
                for c in range(w):
                    ctx=tuple(tuple(int(ga[r+dr,c+dc]) if 0<=r+dr<h and 0<=c+dc<w else -1
                        for dc in range(-radius,radius+1)) for dr in range(-radius,radius+1))
                    ov=int(go[r,c])
                    if ctx in rules and rules[ctx]!=ov: ok2=False; break
                    rules[ctx]=ov
                if not ok2: break
            if not ok2: break
        if not ok2 or not rules: continue
        
        def mk(rl,rad):
            def fn(grid):
                g=np.array(grid);h,w=g.shape;result=g.copy();ch=False
                for r in range(h):
                    for c in range(w):
                        ctx=tuple(tuple(int(g[r+dr,c+dc]) if 0<=r+dr<h and 0<=c+dc<w else -1
                            for dc in range(-rad,rad+1)) for dr in range(-rad,rad+1))
                        if ctx in rl and rl[ctx]!=int(g[r,c]): result[r,c]=rl[ctx]; ch=True
                return result.tolist() if ch else None
            return fn
        r=_verify(mk(rules,radius),tp,ti)
        if r is not None: return r
    return None

def neighbor_abstract(tp, ti):
    """近傍抽象ルール（複数抽象化レベル）"""
    for ab in ['count_majority','directional','self_count']:
        rules={};ok2=True
        for inp,out in tp:
            ga,go=np.array(inp),np.array(out)
            if ga.shape!=go.shape: return None
            bg=_bg(ga);h,w=ga.shape
            for r in range(h):
                for c in range(w):
                    sc=int(ga[r,c])
                    nb4=[int(ga[r+dr,c+dc]) if 0<=r+dr<h and 0<=c+dc<w else -1
                         for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]]
                    nb8=[int(ga[r+dr,c+dc]) if 0<=r+dr<h and 0<=c+dc<w else -1
                         for dr in [-1,0,1] for dc in [-1,0,1] if not(dr==0 and dc==0)]
                    nn=[v for v in nb8 if v!=bg and v!=-1]
                    ov=int(go[r,c])
                    if ab=='count_majority':
                        mc=Counter(nn).most_common(1)[0][0] if nn else bg
                        key=(sc,len(nn),mc)
                    elif ab=='directional':
                        key=(sc,tuple(nb4))
                    elif ab=='self_count':
                        key=(sc,len(nn),sum(1 for v in nn if v==sc))
                    if key in rules and rules[key]!=ov: ok2=False; break
                    rules[key]=ov
                if not ok2: break
            if not ok2: break
        if not ok2: continue
        
        def mk(rl,abb):
            def fn(grid):
                g=np.array(grid);h,w=g.shape;bg2=_bg(g);result=g.copy();ch=False
                for r in range(h):
                    for c in range(w):
                        sc=int(g[r,c])
                        nb4=[int(g[r+dr,c+dc]) if 0<=r+dr<h and 0<=c+dc<w else -1
                             for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]]
                        nb8=[int(g[r+dr,c+dc]) if 0<=r+dr<h and 0<=c+dc<w else -1
                             for dr in [-1,0,1] for dc in [-1,0,1] if not(dr==0 and dc==0)]
                        nn=[v for v in nb8 if v!=bg2 and v!=-1]
                        if abb=='count_majority':
                            mc=Counter(nn).most_common(1)[0][0] if nn else bg2
                            key=(sc,len(nn),mc)
                        elif abb=='directional':
                            key=(sc,tuple(nb4))
                        elif abb=='self_count':
                            key=(sc,len(nn),sum(1 for v in nn if v==sc))
                        if key in rl and rl[key]!=sc: result[r,c]=rl[key]; ch=True
                return result.tolist() if ch else None
            return fn
        r=_verify(mk(rules,ab),tp,ti)
        if r is not None: return r
    return None

def evenodd_full(tp, ti):
    """偶奇ルール（行/列/チェッカー）"""
    for axis in ['row','col','checker']:
        def mk(ax):
            def fn(grid):
                g=np.array(grid); r2=g.copy()
                # trainから学習
                em,om={},{}
                for inp,out in tp:
                    ga,go=np.array(inp),np.array(out)
                    if ga.shape!=go.shape: return None
                    for r in range(ga.shape[0]):
                        for c in range(ga.shape[1]):
                            if ga[r,c]!=go[r,c]:
                                if ax=='row': idx=r
                                elif ax=='col': idx=c
                                else: idx=r+c
                                if idx%2==0: em[int(ga[r,c])]=int(go[r,c])
                                else: om[int(ga[r,c])]=int(go[r,c])
                if not em and not om: return None
                for r in range(g.shape[0]):
                    for c in range(g.shape[1]):
                        v=int(g[r,c])
                        if ax=='row': idx=r
                        elif ax=='col': idx=c
                        else: idx=r+c
                        if idx%2==0 and v in em: r2[r,c]=em[v]
                        elif idx%2==1 and v in om: r2[r,c]=om[v]
                return r2.tolist()
            return fn
        r=_verify(mk(axis),tp,ti)
        if r is not None: return r
    return None


# ══════════════════════════════════════════════════════════════
# LARGER: 拡大系（scale/tile/枠付き拡大）
# ══════════════════════════════════════════════════════════════

def scale_variants(tp, ti):
    for inp,out in tp:
        gi2,go2=np.array(inp),np.array(out)
        for s in range(2,8):
            if gi2.shape[0]*s==go2.shape[0] and gi2.shape[1]*s==go2.shape[1]:
                r=_verify(lambda g,ss=s: np.repeat(np.repeat(np.array(g),ss,0),ss,1).tolist(),tp,ti)
                if r is not None: return r
        break
    return None

def tile_variants(tp, ti):
    for inp,out in tp:
        gi2,go2=np.array(inp),np.array(out)
        for rh in range(1,6):
            for rw in range(1,6):
                if rh==1 and rw==1: continue
                if gi2.shape[0]*rh==go2.shape[0] and gi2.shape[1]*rw==go2.shape[1]:
                    r=_verify(lambda g,rrh=rh,rrw=rw: np.tile(np.array(g),(rrh,rrw)).tolist(),tp,ti)
                    if r is not None: return r
        break
    return None

def count_output(tp, ti):
    gi=np.array(ti);bg=_bg(gi)
    for what in ['obj','color','fg']:
        ok=True
        for inp,out in tp:
            ga=np.array(inp);b=_bg(ga)
            if what=='obj': v=len(_objs(ga,b))
            elif what=='color': v=len(set(int(x) for x in ga.flatten())-{b})
            else: v=int((ga!=b).sum())
            if np.array(out).shape!=(1,1) or int(np.array(out)[0,0])!=v: ok=False; break
        if ok:
            if what=='obj': v=len(_objs(gi,bg))
            elif what=='color': v=len(set(int(x) for x in gi.flatten())-{bg})
            else: v=int((gi!=bg).sum())
            return [[v]]
    return None

def repeat_pattern(tp, ti):
    gi=np.array(ti);h,w=gi.shape
    for ax in ['r','c']:
        dim=h if ax=='r' else w
        for p in range(1,dim//2+1):
            if dim%p!=0: continue
            tile=gi[:p,:] if ax=='r' else gi[:,:p]
            result=np.tile(tile,(h//p,1)) if ax=='r' else np.tile(tile,(1,w//p))
            if np.array_equal(result,gi): continue
            def mk(pp,axx):
                def fn(grid):
                    g=np.array(grid);d=g.shape[0] if axx=='r' else g.shape[1]
                    if d%pp!=0: return None
                    t=g[:pp,:] if axx=='r' else g[:,:pp]
                    return (np.tile(t,(g.shape[0]//pp,1)) if axx=='r' else np.tile(t,(1,g.shape[1]//pp))).tolist()
                return fn
            r=_verify(mk(p,ax),tp,ti)
            if r is not None: return r
    return None


# ══════════════════════════════════════════════════════════════
# Cross Brain V2: 勘→ルーティング→フル知識
# ══════════════════════════════════════════════════════════════

def extract_intuition(train_pairs):
    gi0,go0=np.array(train_pairs[0][0]),np.array(train_pairs[0][1])
    bg=_bg(gi0)
    f={}
    f['same_size']=gi0.shape==go0.shape
    f['out_smaller']=go0.size<gi0.size
    f['out_larger']=go0.size>gi0.size
    f['out_1x1']=go0.shape==(1,1)
    
    if f['same_size']:
        a=r2=ch=0
        for r in range(gi0.shape[0]):
            for c in range(gi0.shape[1]):
                iv,ov=int(gi0[r,c]),int(go0[r,c])
                if iv==ov: pass
                elif iv==bg and ov!=bg: a+=1
                elif iv!=bg and ov==bg: r2+=1
                else: ch+=1
        f['only_add']=a>0 and r2==0 and ch==0
        f['only_change']=ch>0 and a==0 and r2==0
        f['add_and_remove']=a>0 and r2>0
        f['mixed']=ch>0 and (a>0 or r2>0)
    else:
        f['only_add']=f['only_change']=f['add_and_remove']=f['mixed']=False
    
    if f['out_smaller'] or f['out_larger']:
        oh,ow=go0.shape; ih,iw=gi0.shape
        f['int_ratio']=(max(oh,ih)%min(oh,ih)==0 and max(ow,iw)%min(ow,iw)==0)
    else:
        f['int_ratio']=True
    
    # セパレータ
    f['has_sep']=False
    for r in range(gi0.shape[0]):
        if len(set(int(v) for v in gi0[r]))==1 and int(gi0[r,0])!=bg:
            f['has_sep']=True; break
    if not f['has_sep']:
        for c in range(gi0.shape[1]):
            if len(set(int(v) for v in gi0[:,c]))==1 and int(gi0[0,c])!=bg:
                f['has_sep']=True; break
    
    return f

# ルーティングテーブル（優先度順）
ROUTES=[
    (lambda f:f.get('out_1x1'), [count_output]),
    (lambda f:f.get('out_smaller') and f.get('has_sep'), [conditional_extract, panel_compare, crop_by_attr]),
    (lambda f:f.get('out_smaller') and f.get('int_ratio'), [panel_compare, crop_by_attr, conditional_extract]),
    (lambda f:f.get('out_smaller'), [crop_by_attr, panel_compare, conditional_extract]),
    (lambda f:f.get('out_larger') and f.get('int_ratio'), [scale_variants, tile_variants]),
    (lambda f:f.get('out_larger'), [tile_variants, scale_variants]),
    (lambda f:f.get('only_change'), [colormap_full, evenodd_full, neighbor_abstract, template_match]),
    (lambda f:f.get('only_add'), [reversi_variants, connect_variants, symmetry_variants, flood_variants,
                                   cannon_stamp, morphology_variants, diagonal_draw, minesweeper]),
    (lambda f:f.get('add_and_remove'), [gravity_variants, object_translate, rotate_transform]),
    (lambda f:f.get('mixed'), [template_match, neighbor_abstract, colormap_full, gravity_variants,
                                reversi_variants, morphology_variants]),
    (lambda f:f.get('same_size'), [template_match, neighbor_abstract, colormap_full, symmetry_variants,
                                    reversi_variants, morphology_variants, evenodd_full, flood_variants,
                                    connect_variants, gravity_variants, rotate_transform, repeat_pattern]),
    (lambda f:not f.get('same_size'), [crop_by_attr, panel_compare, conditional_extract, scale_variants,
                                        tile_variants, rotate_transform, count_output]),
]

ALL_SOLVERS=[
    crop_by_attr, panel_compare, conditional_extract,
    reversi_variants, connect_variants, symmetry_variants, flood_variants,
    cannon_stamp, morphology_variants, diagonal_draw, minesweeper,
    gravity_variants, object_translate, rotate_transform,
    colormap_full, template_match, neighbor_abstract, evenodd_full,
    scale_variants, tile_variants, count_output, repeat_pattern,
]

def cross_brain_v2_solve(train_pairs, test_input):
    features=extract_intuition(train_pairs)
    tried=set()
    
    for cond,solvers in ROUTES:
        try:
            if cond(features):
                for s in solvers:
                    if id(s) in tried: continue
                    tried.add(id(s))
                    try:
                        r=s(train_pairs,test_input)
                        if r is not None: return r, s.__name__
                    except: continue
        except: continue
    
    for s in ALL_SOLVERS:
        if id(s) in tried: continue
        try:
            r=s(train_pairs,test_input)
            if r is not None: return r, s.__name__
        except: continue
    
    return None, None

if __name__=="__main__":
    import sys,json,re
    from pathlib import Path
    split='evaluation' if '--eval' in sys.argv else 'training'
    data_dir=Path(f'/tmp/arc-agi-2/data/{split}')
    existing=set()
    with open('arc_v82.log') as f:
        for l in f:
            m=re.search(r'✓.*?([0-9a-f]{8})',l)
            if m: existing.add(m.group(1))
    synth=set(f.stem for f in Path('synth_results').glob('*.py'))
    all_e=existing|synth
    solved=[]
    for tf in sorted(data_dir.glob('*.json')):
        tid=tf.stem
        with open(tf) as f: task=json.load(f)
        tp=[(e['input'],e['output']) for e in task['train']]
        ti,to=task['test'][0]['input'],task['test'][0].get('output')
        result,name=cross_brain_v2_solve(tp,ti)
        if result and to and grid_eq(result,to):
            tag='NEW' if tid not in all_e else ''
            solved.append((tid,name,tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    total=len(list(data_dir.glob('*.json')))
    new=[t for t,_,tg in solved if tg=='NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
    print('\n手法別:')
    for name,cnt in Counter(n for _,n,_ in solved).most_common():
        print(f'  {name}: {cnt}')
