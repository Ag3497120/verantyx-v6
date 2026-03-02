"""
arc/cross_childhood.py - Cross's life: Elementary school grades 1-6

Cross is "Juji-kun" (十字くん). A child born in the world of colors and shapes.
6 years of life experience creates "ah, I know this!" when seeing problems.
"""
import numpy as np
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label

def _bg(g): return int(Counter(np.array(g).flatten()).most_common(1)[0][0])

def _objs(g, bg, conn=8):
    struct = np.ones((3,3),dtype=int) if conn==8 else np.array([[0,1,0],[1,1,1],[0,1,0]])
    mask=(np.array(g)!=bg).astype(int)
    labeled,n=scipy_label(mask,structure=struct)
    objs=[]
    for i in range(1,n+1):
        cells=list(zip(*np.where(labeled==i)))
        colors=[int(g[r,c]) for r,c in cells]
        r1=min(r for r,c in cells);c1=min(c for r,c in cells)
        r2=max(r for r,c in cells);c2=max(c for r,c in cells)
        objs.append({'cells':cells,'size':len(cells),'color':Counter(colors).most_common(1)[0][0],
            'colors':set(colors),'bbox':(r1,c1,r2,c2),'bh':r2-r1+1,'bw':c2-c1+1,
            'shape':frozenset((r-r1,c-c1) for r,c in cells)})
    return objs

def _overlay(gi,go):
    gi,go=np.array(gi),np.array(go)
    if gi.shape!=go.shape: return {'type':'size_diff','in':gi.shape,'out':go.shape}
    bg=_bg(gi); a=r=ch=0
    for rr in range(gi.shape[0]):
        for cc in range(gi.shape[1]):
            iv,ov=int(gi[rr,cc]),int(go[rr,cc])
            if iv==ov: pass
            elif iv==bg and ov!=bg: a+=1
            elif iv!=bg and ov==bg: r+=1
            else: ch+=1
    if a and not r and not ch: return {'type':'only_add','n':a}
    if r and not a and not ch: return {'type':'only_remove','n':r}
    if ch and not a and not r: return {'type':'only_change','n':ch}
    if a and r and not ch: return {'type':'add_and_remove'}
    return {'type':'mixed'}

class Experience:
    def __init__(self,name,grade,memory,act):
        self.name=name; self.grade=grade; self.memory=memory; self.act=act

class CrossChildhood:
    def __init__(self):
        self.experiences=[]
        self._build_life()

    def _build_life(self):
        E=Experience
        # Grade 1: Count & Compare
        self.experiences.append(E('数える',1,'何個あるかな？',self._act_count))
        self.experiences.append(E('大きい方',1,'どっちが大きい？',self._act_biggest))
        self.experiences.append(E('小さい方',1,'一番小さいのどれ？',self._act_smallest))
        self.experiences.append(E('仲間はずれ',1,'1つだけ違うの',self._act_odd_one))
        # Grade 2: Connect & Fold
        self.experiences.append(E('リバーシ',2,'挟んだら裏返る！',self._act_reversi))
        self.experiences.append(E('点つなぎ',2,'同じ色を線でつなぐ',self._act_connect))
        self.experiences.append(E('折り紙',2,'半分に折ったら同じ',self._act_symmetry))
        self.experiences.append(E('繰り返し',2,'赤青赤青...',self._act_repeat))
        # Grade 3: Move & Shape
        self.experiences.append(E('テトリス',3,'ブロックが落ちる',self._act_gravity))
        self.experiences.append(E('回す',3,'くるっと回す',self._act_rotate))
        self.experiences.append(E('タイル貼り',3,'同じ模様を繰り返す',self._act_tile))
        # Grade 4: Area & Sort
        self.experiences.append(E('ソリティア',4,'色を揃えて並べる',self._act_sort))
        self.experiences.append(E('虫眼鏡',4,'大きくする',self._act_scale))
        self.experiences.append(E('囲み塗り',4,'囲んだ中を塗る',self._act_flood))
        # Grade 5: Map & Mirror
        self.experiences.append(E('色変え',5,'赤→青のルール',self._act_colormap))
        self.experiences.append(E('鏡',5,'左右逆',self._act_mirror))
        self.experiences.append(E('三枚おろし',5,'分けて重ねる',self._act_panel))
        self.experiences.append(E('額縁crop',5,'枠の中を切り取る',self._act_crop))
        # Grade 6: Logic & Advanced
        self.experiences.append(E('リバーシ大会',6,'反復挟み',self._act_reversi_iter))
        self.experiences.append(E('穴埋め',6,'周りから推測',self._act_holefill))
        self.experiences.append(E('砲台',6,'端まで飛ばす',self._act_cannon))
        self.experiences.append(E('壁押し',6,'壁まで滑る',self._act_wallpush))
        self.experiences.append(E('偶数奇数',6,'偶奇で分ける',self._act_evenodd))

    def solve(self, train_pairs, test_input):
        from arc.grid import grid_eq
        for exp in self.experiences:
            try:
                result = exp.act(train_pairs, test_input)
                if result is None: continue
                ok=True
                for inp,out in train_pairs:
                    p=exp.act(train_pairs,inp)
                    if p is None or not grid_eq(p,out): ok=False; break
                if ok: return result, exp.name
            except: continue
        return None, None

    # === Grade 1 ===
    def _act_count(self,tp,ti):
        from arc.grid import grid_eq
        gi=np.array(ti); bg=_bg(gi)
        for what in ['obj','color','fg']:
            ok=True
            for inp,out in tp:
                ga=np.array(inp); b=_bg(ga)
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

    def _act_biggest(self,tp,ti):
        from arc.grid import grid_eq
        for k in ['size','bh','bw']:
            ok=True
            for inp,out in tp:
                ga=np.array(inp); b=_bg(ga); objs=_objs(ga,b)
                if not objs: ok=False; break
                o=max(objs,key=lambda x:x[k]); r1,c1,r2,c2=o['bbox']
                if not grid_eq(ga[r1:r2+1,c1:c2+1].tolist(),out): ok=False; break
            if ok:
                gi=np.array(ti); bg=_bg(gi); objs=_objs(gi,bg)
                if not objs: return None
                o=max(objs,key=lambda x:x[k]); r1,c1,r2,c2=o['bbox']
                return gi[r1:r2+1,c1:c2+1].tolist()
        return None

    def _act_smallest(self,tp,ti):
        from arc.grid import grid_eq
        for k in ['size','bh','bw']:
            ok=True
            for inp,out in tp:
                ga=np.array(inp); b=_bg(ga); objs=_objs(ga,b)
                if not objs: ok=False; break
                o=min(objs,key=lambda x:x[k]); r1,c1,r2,c2=o['bbox']
                if not grid_eq(ga[r1:r2+1,c1:c2+1].tolist(),out): ok=False; break
            if ok:
                gi=np.array(ti); bg=_bg(gi); objs=_objs(gi,bg)
                if not objs: return None
                o=min(objs,key=lambda x:x[k]); r1,c1,r2,c2=o['bbox']
                return gi[r1:r2+1,c1:c2+1].tolist()
        return None

    def _act_odd_one(self,tp,ti):
        from arc.grid import grid_eq
        for mode in ['shape','color','size']:
            ok=True
            for inp,out in tp:
                ga=np.array(inp);b=_bg(ga);objs=_objs(ga,b)
                cc=Counter(o[mode] if mode!='shape' else o['shape'] for o in objs)
                uniq=[o for o in objs if cc[o[mode] if mode!='shape' else o['shape']]==1]
                if not uniq: ok=False; break
                o=uniq[0]; r1,c1,r2,c2=o['bbox']
                if not grid_eq(ga[r1:r2+1,c1:c2+1].tolist(),out): ok=False; break
            if ok:
                gi=np.array(ti);bg=_bg(gi);objs=_objs(gi,bg)
                cc=Counter(o[mode] if mode!='shape' else o['shape'] for o in objs)
                uniq=[o for o in objs if cc[o[mode] if mode!='shape' else o['shape']]==1]
                if uniq:
                    o=uniq[0]; r1,c1,r2,c2=o['bbox']
                    return gi[r1:r2+1,c1:c2+1].tolist()
        return None

    # === Grade 2 ===
    def _act_reversi(self,tp,ti):
        return self._reversi_core(tp,ti,False)

    def _reversi_core(self,tp,ti,iterative):
        from arc.grid import grid_eq
        def apply(grid):
            g=np.array(grid).copy(); h,w=g.shape; bg=_bg(g); orig=g.copy()
            for _ in range(20 if iterative else 1):
                ch=False
                for r in range(h):
                    for c in range(w):
                        if g[r,c]!=bg: continue
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                            nr,nc=r+dr,c+dc; c1=None
                            while 0<=nr<h and 0<=nc<w:
                                if g[nr,nc]!=bg: c1=int(g[nr,nc]); break
                                nr+=dr; nc+=dc
                            nr,nc=r-dr,c-dc; c2=None
                            while 0<=nr<h and 0<=nc<w:
                                if g[nr,nc]!=bg: c2=int(g[nr,nc]); break
                                nr-=dr; nc-=dc
                            if c1 is not None and c2 is not None and c1==c2:
                                g[r,c]=c1; ch=True; break
                if not ch: break
            return g.tolist() if not np.array_equal(g,orig) else None
        ok=True
        for inp,out in tp:
            if not grid_eq(apply(inp),out): ok=False; break
        return apply(ti) if ok else None

    def _act_connect(self,tp,ti):
        from arc.grid import grid_eq
        def apply(grid):
            g=np.array(grid).copy(); h,w=g.shape; bg=_bg(g); orig=g.copy()
            for color in set(int(v) for v in g.flatten())-{bg}:
                pts=[(r,c) for r in range(h) for c in range(w) if g[r,c]==color]
                for i,(r1,c1) in enumerate(pts):
                    for r2,c2 in pts[i+1:]:
                        if r1==r2:
                            for c in range(min(c1,c2)+1,max(c1,c2)):
                                if g[r1,c]==bg: g[r1,c]=color
                        elif c1==c2:
                            for r in range(min(r1,r2)+1,max(r1,r2)):
                                if g[r,c1]==bg: g[r,c1]=color
            return g.tolist() if not np.array_equal(g,orig) else None
        ok=True
        for inp,out in tp:
            if not grid_eq(apply(inp),out): ok=False; break
        return apply(ti) if ok else None

    def _act_symmetry(self,tp,ti):
        from arc.grid import grid_eq
        def apply(grid,mode):
            g=np.array(grid).copy(); h,w=g.shape; bg=_bg(g); orig=g.copy()
            pairs=[]
            if mode=='lr': pairs=[(r,c,r,w-1-c) for r in range(h) for c in range(w)]
            elif mode=='ud': pairs=[(r,c,h-1-r,c) for r in range(h) for c in range(w)]
            elif mode=='both': pairs=[(r,c,r,w-1-c) for r in range(h) for c in range(w)]+[(r,c,h-1-r,c) for r in range(h) for c in range(w)]+[(r,c,h-1-r,w-1-c) for r in range(h) for c in range(w)]
            elif mode=='rot180': pairs=[(r,c,h-1-r,w-1-c) for r in range(h) for c in range(w)]
            elif mode=='diag':
                if h!=w: return None
                pairs=[(r,c,c,r) for r in range(h) for c in range(w)]
            for r,c,rr,cc in pairs:
                if g[r,c]==bg and g[rr,cc]!=bg: g[r,c]=g[rr,cc]
            return g.tolist() if not np.array_equal(g,orig) else None
        for m in ['lr','ud','both','rot180','diag']:
            ok=True
            for inp,out in tp:
                if not grid_eq(apply(inp,m),out): ok=False; break
            if ok: return apply(ti,m)
        return None

    def _act_repeat(self,tp,ti):
        from arc.grid import grid_eq
        gi=np.array(ti); h,w=gi.shape
        for ax in ['r','c']:
            dim=h if ax=='r' else w
            for p in range(1,dim//2+1):
                if dim%p!=0: continue
                tile=gi[:p,:] if ax=='r' else gi[:,:p]
                result=np.tile(tile,(h//p,1)) if ax=='r' else np.tile(tile,(1,w//p))
                if np.array_equal(result,gi): continue
                ok=True
                for inp,out in tp:
                    ga=np.array(inp); d=ga.shape[0] if ax=='r' else ga.shape[1]
                    if d%p!=0: ok=False; break
                    t=ga[:p,:] if ax=='r' else ga[:,:p]
                    pred=np.tile(t,(ga.shape[0]//p,1)) if ax=='r' else np.tile(t,(1,ga.shape[1]//p))
                    if not grid_eq(pred.tolist(),out): ok=False; break
                if ok: return result.tolist()
        return None

    # === Grade 3 ===
    def _act_gravity(self,tp,ti):
        from arc.grid import grid_eq
        def apply(grid,d):
            g=np.array(grid); h,w=g.shape; bg=_bg(g); result=np.full_like(g,bg)
            if d in ('down','up'):
                for c in range(w):
                    vals=[int(g[r,c]) for r in range(h) if g[r,c]!=bg]
                    if d=='down':
                        for i,v in enumerate(reversed(vals)): result[h-1-i,c]=v
                    else:
                        for i,v in enumerate(vals): result[i,c]=v
            else:
                for r in range(h):
                    vals=[int(g[r,c]) for c in range(w) if g[r,c]!=bg]
                    if d=='right':
                        for i,v in enumerate(reversed(vals)): result[r,w-1-i]=v
                    else:
                        for i,v in enumerate(vals): result[r,i]=v
            return result.tolist()
        for d in ['down','up','left','right']:
            ok=True
            for inp,out in tp:
                if not grid_eq(apply(inp,d),out): ok=False; break
            if ok: return apply(ti,d)
        return None

    def _act_rotate(self,tp,ti):
        from arc.grid import grid_eq
        gi=np.array(ti)
        for k in [1,2,3]:
            ok=True
            for inp,out in tp:
                if not grid_eq(np.rot90(np.array(inp),k).tolist(),out): ok=False; break
            if ok: return np.rot90(gi,k).tolist()
        # transpose
        ok=True
        for inp,out in tp:
            if not grid_eq(np.array(inp).T.tolist(),out): ok=False; break
        if ok: return gi.T.tolist()
        return None

    def _act_tile(self,tp,ti):
        from arc.grid import grid_eq
        for inp,out in tp:
            gi,go=np.array(inp),np.array(out)
            for rh in range(1,6):
                for rw in range(1,6):
                    if rh==1 and rw==1: continue
                    if gi.shape[0]*rh==go.shape[0] and gi.shape[1]*rw==go.shape[1]:
                        if grid_eq(np.tile(gi,(rh,rw)).tolist(),out):
                            return np.tile(np.array(ti),(rh,rw)).tolist()
            break
        return None

    # === Grade 4 ===
    def _act_sort(self,tp,ti):
        from arc.grid import grid_eq
        def rsort(grid):
            g=np.array(grid);bg=_bg(g);r=g.copy();ch=False
            for row in range(g.shape[0]):
                vals=sorted([int(v) for v in g[row] if v!=bg])
                ci=0
                for c in range(g.shape[1]):
                    if g[row,c]!=bg:
                        if r[row,c]!=vals[ci]: r[row,c]=vals[ci]; ch=True
                        ci+=1
            return r.tolist() if ch else None
        def csort(grid):
            g=np.array(grid);bg=_bg(g);r=g.copy();ch=False
            for c in range(g.shape[1]):
                vals=sorted([int(v) for v in g[:,c] if v!=bg])
                ri=0
                for row in range(g.shape[0]):
                    if g[row,c]!=bg:
                        if r[row,c]!=vals[ri]: r[row,c]=vals[ri]; ch=True
                        ri+=1
            return r.tolist() if ch else None
        for fn in [rsort,csort]:
            ok=True
            for inp,out in tp:
                if not grid_eq(fn(inp),out): ok=False; break
            if ok: return fn(ti)
        return None

    def _act_scale(self,tp,ti):
        from arc.grid import grid_eq
        for inp,out in tp:
            gi,go=np.array(inp),np.array(out)
            if go.shape[0]>gi.shape[0]:
                s=go.shape[0]//gi.shape[0]
                if s*gi.shape[0]==go.shape[0] and s*gi.shape[1]==go.shape[1]:
                    if grid_eq(np.repeat(np.repeat(gi,s,0),s,1).tolist(),out):
                        t=np.array(ti); return np.repeat(np.repeat(t,s,0),s,1).tolist()
            break
        return None

    def _act_flood(self,tp,ti):
        from arc.grid import grid_eq
        def apply(grid,fc=None):
            g=np.array(grid);h,w=g.shape;bg=_bg(g)
            vis=np.zeros((h,w),bool); q=[]
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
            result=g.copy(); ch=False
            for r in range(h):
                for c in range(w):
                    if g[r,c]==bg and not vis[r,c]:
                        if fc is not None: result[r,c]=fc
                        else:
                            nb=set()
                            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr,nc=r+dr,c+dc
                                if 0<=nr<h and 0<=nc<w and g[nr,nc]!=bg: nb.add(int(g[nr,nc]))
                            result[r,c]=nb.pop() if len(nb)==1 else (min(nb) if nb else 1)
                        ch=True
            return result.tolist() if ch else None
        ok=True
        for inp,out in tp:
            if not grid_eq(apply(inp),out): ok=False; break
        if ok: return apply(ti)
        # learn fill color
        for inp,out in tp:
            ga,go=np.array(inp),np.array(out); bg=_bg(ga)
            for r in range(ga.shape[0]):
                for c in range(ga.shape[1]):
                    if ga[r,c]==bg and go[r,c]!=bg:
                        fc=int(go[r,c])
                        ok2=True
                        for i2,o2 in tp:
                            if not grid_eq(apply(i2,fc),o2): ok2=False; break
                        if ok2: return apply(ti,fc)
            break
        return None

    # === Grade 5 ===
    def _act_colormap(self,tp,ti):
        from arc.grid import grid_eq
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
        gi=np.array(ti); result=gi.copy()
        for o,n in m.items(): result[gi==o]=n
        return result.tolist()

    def _act_mirror(self,tp,ti):
        return self._act_symmetry(tp,ti)

    def _act_panel(self,tp,ti):
        from arc.grid import grid_eq
        for n in [2,3,4]:
            for ax in ['h','v']:
                for op in ['first','or','xor']:
                    ok=True
                    for inp,out in tp:
                        p=self._panel_apply(inp,n,ax,op)
                        if p is None or not grid_eq(p,out): ok=False; break
                    if ok: return self._panel_apply(ti,n,ax,op)
        return None

    def _panel_apply(self,grid,n,ax,op):
        g=np.array(grid); h,w=g.shape; bg=_bg(g)
        if ax=='h' and h%n==0:
            ph=h//n; panels=[g[i*ph:(i+1)*ph,:] for i in range(n)]
        elif ax=='v' and w%n==0:
            pw=w//n; panels=[g[:,i*pw:(i+1)*pw] for i in range(n)]
        else: return None
        rh,rw=panels[0].shape; result=np.full((rh,rw),bg,dtype=int)
        for r in range(rh):
            for c in range(rw):
                vals=[int(p[r,c]) for p in panels]; nb=[v for v in vals if v!=bg]
                if op=='first': result[r,c]=nb[0] if nb else bg
                elif op=='or': result[r,c]=nb[0] if nb else bg
                elif op=='xor': result[r,c]=nb[0] if len(nb)==1 else bg
        return result.tolist()

    def _act_crop(self,tp,ti):
        from arc.grid import grid_eq
        def apply(grid):
            g=np.array(grid); bg=_bg(g); objs=_objs(g,bg)
            if not objs: return None
            o=max(objs,key=lambda x:x['size']); r1,c1,r2,c2=o['bbox']
            inner=g[r1+1:r2,c1+1:c2]
            return inner.tolist() if inner.size>0 else None
        ok=True
        for inp,out in tp:
            if not grid_eq(apply(inp),out): ok=False; break
        return apply(ti) if ok else None

    # === Grade 6 ===
    def _act_reversi_iter(self,tp,ti):
        return self._reversi_core(tp,ti,True)

    def _act_holefill(self,tp,ti):
        from arc.grid import grid_eq
        def apply(grid):
            g=np.array(grid).copy(); h,w=g.shape; bg=_bg(g); orig=g.copy()
            for r in range(h):
                for c in range(w):
                    if g[r,c]!=bg: continue
                    nb=[]
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=r+dr,c+dc
                        if 0<=nr<h and 0<=nc<w and g[nr,nc]!=bg: nb.append(int(g[nr,nc]))
                    if len(nb)>=3: g[r,c]=Counter(nb).most_common(1)[0][0]
            return g.tolist() if not np.array_equal(g,orig) else None
        ok=True
        for inp,out in tp:
            if not grid_eq(apply(inp),out): ok=False; break
        return apply(ti) if ok else None

    def _act_cannon(self,tp,ti):
        try:
            from arc.cross3d_projection import cannon_solve
            from arc.grid import grid_eq
            ok=True
            for inp,out in tp:
                p=cannon_solve(tp,inp)
                if p is None or not grid_eq(p,out): ok=False; break
            if ok: return cannon_solve(tp,ti)
        except: pass
        return None

    def _act_wallpush(self,tp,ti):
        from arc.grid import grid_eq
        def apply(grid,d):
            g=np.array(grid); h,w=g.shape; bg=_bg(g)
            colors=Counter(int(v) for v in g.flatten() if v!=bg)
            if len(colors)<2: return None
            wall=colors.most_common(1)[0][0]
            result=g.copy()
            dr,dc={'down':(1,0),'up':(-1,0),'left':(0,-1),'right':(0,1)}[d]
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
        for d in ['down','up','left','right']:
            ok=True
            for inp,out in tp:
                p=apply(inp,d)
                if p is None or not grid_eq(p,out): ok=False; break
            if ok: return apply(ti,d)
        return None

    def _act_evenodd(self,tp,ti):
        from arc.grid import grid_eq
        for inp,out in tp:
            ga,go=np.array(inp),np.array(out)
            if ga.shape!=go.shape: return None
            em,om={},{}
            for r in range(ga.shape[0]):
                for c in range(ga.shape[1]):
                    if ga[r,c]!=go[r,c]:
                        if r%2==0: em[int(ga[r,c])]=int(go[r,c])
                        else: om[int(ga[r,c])]=int(go[r,c])
            if em or om:
                def apply_eo(grid):
                    g=np.array(grid); r2=g.copy()
                    for r in range(g.shape[0]):
                        for c in range(g.shape[1]):
                            v=int(g[r,c])
                            if r%2==0 and v in em: r2[r,c]=em[v]
                            elif r%2==1 and v in om: r2[r,c]=om[v]
                    return r2.tolist()
                ok=True
                for i2,o2 in tp:
                    if not grid_eq(apply_eo(i2),o2): ok=False; break
                if ok: return apply_eo(ti)
            break
        return None


def childhood_solve(train_pairs, test_input):
    return CrossChildhood().solve(train_pairs, test_input)

if __name__=="__main__":
    import sys,json,re
    from pathlib import Path
    from arc.grid import grid_eq
    split='evaluation' if '--eval' in sys.argv else 'training'
    data_dir=Path(f'/tmp/arc-agi-2/data/{split}')
    existing=set()
    with open('arc_v82.log') as f:
        for l in f:
            m=re.search(r'✓.*?([0-9a-f]{8})',l)
            if m: existing.add(m.group(1))
    synth=set(f.stem for f in Path('synth_results').glob('*.py'))
    all_e=existing|synth
    cross=CrossChildhood(); solved=[]
    for tf in sorted(data_dir.glob('*.json')):
        tid=tf.stem
        with open(tf) as f: task=json.load(f)
        tp=[(e['input'],e['output']) for e in task['train']]
        ti,to=task['test'][0]['input'],task['test'][0].get('output')
        result,name=cross.solve(tp,ti)
        if result and to and grid_eq(result,to):
            tag='NEW' if tid not in all_e else ''
            solved.append((tid,name,tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    total=len(list(data_dir.glob('*.json')))
    new=[t for t,_,tg in solved if tg=='NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
    print('\n経験別:')
    for name,cnt in Counter(n for _,n,_ in solved).most_common():
        print(f'  {name}: {cnt}')
