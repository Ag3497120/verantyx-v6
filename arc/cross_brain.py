"""
arc/cross_brain.py — Cross Brain: 断片×フル二層記憶統合エンジン

断片記憶 = 勘（特徴ベクトルでルーティング）
フル記憶 = 知識（高精度アルゴリズム群）
Cross構造 = 勘→知識のルーティング
"""

import numpy as np
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label

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
            'is_rect': len(cells)==(r2-r1+1)*(c2-c1+1),
            'is_square': (r2-r1)==(c2-c1),
            'patch': np.array(g)[r1:r2+1,c1:c2+1].copy()})
    return objs

# ══════════════════════════════════════════════════════════════
# 勘エンジン: 問題の「顔」を読む（特徴量抽出）
# ══════════════════════════════════════════════════════════════

def extract_intuition(train_pairs):
    """問題から直感的特徴を抽出 → ルーティング用"""
    features = {}
    
    gi0, go0 = np.array(train_pairs[0][0]), np.array(train_pairs[0][1])
    bg = _bg(gi0)
    
    # サイズ関係
    features['same_size'] = gi0.shape == go0.shape
    features['out_smaller'] = go0.size < gi0.size
    features['out_larger'] = go0.size > gi0.size
    features['out_1x1'] = go0.shape == (1,1)
    
    if gi0.shape == go0.shape:
        # overlay分析
        same = added = removed = changed = 0
        for r in range(gi0.shape[0]):
            for c in range(gi0.shape[1]):
                iv,ov = int(gi0[r,c]),int(go0[r,c])
                if iv==ov: same+=1
                elif iv==bg and ov!=bg: added+=1
                elif iv!=bg and ov==bg: removed+=1
                else: changed+=1
        total_diff = added+removed+changed
        features['only_add'] = added>0 and removed==0 and changed==0
        features['only_change'] = changed>0 and added==0 and removed==0
        features['only_remove'] = removed>0 and added==0 and changed==0
        features['add_and_remove'] = added>0 and removed>0 and changed==0
        features['change_ratio'] = total_diff / max(gi0.size, 1)
    else:
        features['only_add'] = features['only_change'] = features['only_remove'] = False
        features['add_and_remove'] = False
        features['change_ratio'] = 1.0
    
    # オブジェクト特徴
    objs = _objs(gi0, bg)
    features['n_objects'] = len(objs)
    features['n_colors'] = len(set(int(v) for v in gi0.flatten()) - {bg})
    features['all_same_shape'] = len(set(o['shape'] for o in objs)) == 1 if objs else False
    features['all_same_color'] = len(set(o['color'] for o in objs)) == 1 if objs else False
    features['has_single_cells'] = any(o['size']==1 for o in objs)
    features['has_rectangles'] = any(o['is_rect'] for o in objs)
    features['all_rectangles'] = all(o['is_rect'] for o in objs) if objs else False
    
    # 対称性チェック
    features['input_lr_sym'] = np.array_equal(gi0, gi0[:,::-1])
    features['input_ud_sym'] = np.array_equal(gi0, gi0[::-1,:])
    features['partial_sym'] = False
    if not features['input_lr_sym']:
        # 半分以上対称？
        h,w = gi0.shape
        match = sum(1 for r in range(h) for c in range(w//2) if gi0[r,c]==gi0[r,w-1-c])
        features['partial_sym'] = match > h*(w//2)*0.5
    
    # サイズ比
    if features['out_smaller'] and go0.size > 0:
        features['size_ratio_h'] = gi0.shape[0] / max(go0.shape[0],1)
        features['size_ratio_w'] = gi0.shape[1] / max(go0.shape[1],1)
        features['is_integer_ratio'] = (gi0.shape[0] % go0.shape[0] == 0 and 
                                         gi0.shape[1] % go0.shape[1] == 0)
    elif features['out_larger']:
        features['size_ratio_h'] = go0.shape[0] / max(gi0.shape[0],1)
        features['size_ratio_w'] = go0.shape[1] / max(gi0.shape[1],1)
        features['is_integer_ratio'] = (go0.shape[0] % gi0.shape[0] == 0 and 
                                         go0.shape[1] % gi0.shape[1] == 0)
    else:
        features['size_ratio_h'] = features['size_ratio_w'] = 1.0
        features['is_integer_ratio'] = True
    
    # セパレータ検出
    features['has_h_separator'] = False
    features['has_v_separator'] = False
    for r in range(gi0.shape[0]):
        vals = set(int(v) for v in gi0[r])
        if len(vals)==1 and vals.pop()!=bg:
            features['has_h_separator'] = True; break
    for c in range(gi0.shape[1]):
        vals = set(int(v) for v in gi0[:,c])
        if len(vals)==1 and vals.pop()!=bg:
            features['has_v_separator'] = True; break
    
    # 一貫性チェック（全trainで同じ傾向か）
    features['consistent_overlay'] = True
    for inp, out in train_pairs[1:]:
        gi2, go2 = np.array(inp), np.array(out)
        if (gi2.shape == go2.shape) != features['same_size']:
            features['consistent_overlay'] = False
    
    return features


# ══════════════════════════════════════════════════════════════
# フル記憶: 高精度アルゴリズム群
# ══════════════════════════════════════════════════════════════

def grid_eq(a, b):
    a, b = np.array(a), np.array(b)
    return a.shape == b.shape and np.array_equal(a, b)

def _verify(fn, train_pairs, test_input):
    """train全通過→test適用"""
    try:
        for inp, out in train_pairs:
            p = fn(inp)
            if p is None or not grid_eq(p, out): return None
        return fn(test_input)
    except: return None

def _verify_with_tp(fn, train_pairs, test_input):
    """train_pairsも渡す版"""
    try:
        for inp, out in train_pairs:
            p = fn(train_pairs, inp)
            if p is None or not grid_eq(p, out): return None
        return fn(train_pairs, test_input)
    except: return None

# --- 色変換（全バリエーション）---
def full_colormap(tp, ti):
    m = {}
    for inp, out in tp:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                iv, ov = int(ga[r,c]), int(go[r,c])
                if iv != ov:
                    if iv in m and m[iv] != ov: return None
                    m[iv] = ov
    if not m: return None
    gi = np.array(ti); result = gi.copy()
    for o,n in m.items(): result[gi==o] = n
    return result.tolist()

# --- 対称補完（全パターン）---
def full_symmetry(tp, ti):
    for mode in ['lr','ud','both','rot180','diag','anti_diag']:
        def make_fn(m):
            def fn(grid):
                g=np.array(grid).copy(); h,w=g.shape; bg=_bg(g); orig=g.copy()
                for r in range(h):
                    for c in range(w):
                        if g[r,c]!=bg: continue
                        mirrors=[]
                        if m in ('lr','both'): mirrors.append((r,w-1-c))
                        if m in ('ud','both'): mirrors.append((h-1-r,c))
                        if m in ('both','rot180'): mirrors.append((h-1-r,w-1-c))
                        if m=='diag' and h==w: mirrors.append((c,r))
                        if m=='anti_diag' and h==w: mirrors.append((w-1-c,h-1-r))
                        for mr,mc in mirrors:
                            if 0<=mr<h and 0<=mc<w and g[mr,mc]!=bg:
                                g[r,c]=g[mr,mc]; break
                return g.tolist() if not np.array_equal(g,orig) else None
            return fn
        r = _verify(make_fn(mode), tp, ti)
        if r is not None: return r
    return None

# --- リバーシ（全バリエーション）---
def full_reversi(tp, ti):
    for dirs in [
        [(-1,0),(1,0),(0,-1),(0,1)],  # 4方向
        [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)],  # 8方向
    ]:
        for iterative in [False, True]:
            for fill_mode in ['same', 'any_pair']:
                def make_fn(d, it, fm):
                    def fn(grid):
                        g=np.array(grid).copy(); h,w=g.shape; bg=_bg(g); orig=g.copy()
                        for _ in range(30 if it else 1):
                            ch=False
                            for r in range(h):
                                for c in range(w):
                                    if g[r,c]!=bg: continue
                                    for dr,dc in d:
                                        nr,nc=r+dr,c+dc; c1=None
                                        while 0<=nr<h and 0<=nc<w:
                                            if g[nr,nc]!=bg: c1=int(g[nr,nc]); break
                                            nr+=dr; nc+=dc
                                        nr,nc=r-dr,c-dc; c2=None
                                        while 0<=nr<h and 0<=nc<w:
                                            if g[nr,nc]!=bg: c2=int(g[nr,nc]); break
                                            nr-=dr; nc-=dc
                                        ok_fill=False
                                        if fm=='same' and c1 is not None and c2 is not None and c1==c2:
                                            g[r,c]=c1; ok_fill=True
                                        elif fm=='any_pair' and c1 is not None and c2 is not None:
                                            g[r,c]=c1; ok_fill=True
                                        if ok_fill: ch=True; break
                            if not ch: break
                        return g.tolist() if not np.array_equal(g,orig) else None
                    return fn
                r = _verify(make_fn(dirs, iterative, fill_mode), tp, ti)
                if r is not None: return r
    return None

# --- 点つなぎ（直線+対角線）---
def full_connect(tp, ti):
    for include_diag in [False, True]:
        def make_fn(diag):
            def fn(grid):
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
                            elif diag and abs(r2-r1)==abs(c2-c1):
                                dr=1 if r2>r1 else -1; dc=1 if c2>c1 else -1
                                r,c=r1+dr,c1+dc
                                while (r,c)!=(r2,c2):
                                    if g[r,c]==bg: g[r,c]=color
                                    r+=dr; c+=dc
                return g.tolist() if not np.array_equal(g,orig) else None
            return fn
        r = _verify(make_fn(include_diag), tp, ti)
        if r is not None: return r
    return None

# --- 重力（壁あり/なし）---
def full_gravity(tp, ti):
    for d in ['down','up','left','right']:
        def make_fn(direction):
            def fn(grid):
                g=np.array(grid); h,w=g.shape; bg=_bg(g); result=np.full_like(g,bg)
                if direction in ('down','up'):
                    for c in range(w):
                        vals=[int(g[r,c]) for r in range(h) if g[r,c]!=bg]
                        if direction=='down':
                            for i,v in enumerate(reversed(vals)): result[h-1-i,c]=v
                        else:
                            for i,v in enumerate(vals): result[i,c]=v
                else:
                    for r in range(h):
                        vals=[int(g[r,c]) for c in range(w) if g[r,c]!=bg]
                        if direction=='right':
                            for i,v in enumerate(reversed(vals)): result[r,w-1-i]=v
                        else:
                            for i,v in enumerate(vals): result[r,i]=v
                return result.tolist()
            return fn
        r = _verify(make_fn(d), tp, ti)
        if r is not None: return r
    
    # 壁あり重力
    for d in ['down','up','left','right']:
        def make_wall_fn(direction):
            def fn(grid):
                g=np.array(grid); h,w=g.shape; bg=_bg(g)
                colors=Counter(int(v) for v in g.flatten() if v!=bg)
                if len(colors)<2: return None
                wall=colors.most_common(1)[0][0]
                result=g.copy()
                dr,dc={'down':(1,0),'up':(-1,0),'left':(0,-1),'right':(0,1)}[direction]
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
        r = _verify(make_wall_fn(d), tp, ti)
        if r is not None: return r
    return None

# --- 回転/転置 ---
def full_rotate(tp, ti):
    gi=np.array(ti)
    for k in [1,2,3]:
        if _verify(lambda g: np.rot90(np.array(g),k).tolist(), tp, ti) is not None:
            return np.rot90(gi,k).tolist()
    if _verify(lambda g: np.array(g).T.tolist(), tp, ti) is not None:
        return gi.T.tolist()
    # flip
    for ax in [0,1]:
        if _verify(lambda g,a=ax: np.flip(np.array(g),a).tolist(), tp, ti) is not None:
            return np.flip(gi,ax).tolist()
    return None

# --- crop系（最大/最小/仲間はずれ/色指定）---
def full_crop(tp, ti):
    gi=np.array(ti); bg=_bg(gi)
    
    for key in ['size','bh','bw']:
        for which in ['max','min']:
            def make_fn(k,w):
                def fn(grid):
                    g=np.array(grid);b=_bg(g);objs=_objs(g,b)
                    if not objs: return None
                    o=(max if w=='max' else min)(objs,key=lambda x:x[k])
                    r1,c1,r2,c2=o['bbox']
                    return g[r1:r2+1,c1:c2+1].tolist()
                return fn
            r = _verify(make_fn(key,which), tp, ti)
            if r is not None: return r
    
    # 仲間はずれcrop
    for attr in ['shape','color','size']:
        def make_fn(a):
            def fn(grid):
                g=np.array(grid);b=_bg(g);objs=_objs(g,b)
                if len(objs)<2: return None
                cc=Counter(o[a] if a!='shape' else o['shape'] for o in objs)
                uniq=[o for o in objs if cc[o[a] if a!='shape' else o['shape']]==1]
                if not uniq: return None
                o=uniq[0]; r1,c1,r2,c2=o['bbox']
                return g[r1:r2+1,c1:c2+1].tolist()
            return fn
        r = _verify(make_fn(attr), tp, ti)
        if r is not None: return r
    
    # 枠内crop
    def frame_crop(grid):
        g=np.array(grid);b=_bg(g);objs=_objs(g,b)
        if not objs: return None
        o=max(objs,key=lambda x:x['size']); r1,c1,r2,c2=o['bbox']
        inner=g[r1+1:r2,c1+1:c2]
        return inner.tolist() if inner.size>0 else None
    r = _verify(frame_crop, tp, ti)
    if r is not None: return r
    
    return None

# --- パネル合成（2-4分割×多演算）---
def full_panel(tp, ti):
    for n in [2,3,4]:
        for ax in ['h','v']:
            for op in ['or','xor','and','first','last','max','min','majority','diff']:
                def make_fn(nn,axx,opp):
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
                                elif opp=='and': result[r,c]=nb[0] if len(nb)==len(vals) else bg
                                elif opp=='first': result[r,c]=nb[0] if nb else bg
                                elif opp=='last': result[r,c]=nb[-1] if nb else bg
                                elif opp=='max': result[r,c]=max(nb) if nb else bg
                                elif opp=='min': result[r,c]=min(nb) if nb else bg
                                elif opp=='majority': result[r,c]=Counter(nb).most_common(1)[0][0] if nb else bg
                                elif opp=='diff':
                                    if len(set(nb))>1: result[r,c]=nb[-1]
                                    elif len(nb)==1: result[r,c]=nb[0]
                        return result.tolist()
                    return fn
                r = _verify(make_fn(n,ax,op), tp, ti)
                if r is not None: return r
    return None

# --- テンプレートマッチ（3x3/5x5近傍ルール学習）---
def full_template(tp, ti):
    for radius in [1, 2]:
        rules = {}; consistent = True
        for inp, out in tp:
            ga,go=np.array(inp),np.array(out)
            if ga.shape!=go.shape: return None
            h,w=ga.shape
            for r in range(h):
                for c in range(w):
                    ctx=[]
                    for dr in range(-radius,radius+1):
                        row=[]
                        for dc in range(-radius,radius+1):
                            nr,nc=r+dr,c+dc
                            row.append(int(ga[nr,nc]) if 0<=nr<h and 0<=nc<w else -1)
                        ctx.append(tuple(row))
                    ctx=tuple(ctx); ov=int(go[r,c])
                    if ctx in rules and rules[ctx]!=ov: consistent=False; break
                    rules[ctx]=ov
                if not consistent: break
            if not consistent: break
        if not consistent or not rules: continue
        
        def make_fn(rl,rad):
            def fn(grid):
                g=np.array(grid);h,w=g.shape;result=g.copy();ch=False
                for r in range(h):
                    for c in range(w):
                        ctx=[]
                        for dr in range(-rad,rad+1):
                            row=[]
                            for dc in range(-rad,rad+1):
                                nr,nc=r+dr,c+dc
                                row.append(int(g[nr,nc]) if 0<=nr<h and 0<=nc<w else -1)
                            ctx.append(tuple(row))
                        ctx=tuple(ctx)
                        if ctx in rl and rl[ctx]!=int(g[r,c]):
                            result[r,c]=rl[ctx]; ch=True
                return result.tolist() if ch else None
            return fn
        r = _verify(make_fn(rules,radius), tp, ti)
        if r is not None: return r
    return None

# --- 近傍抽象ルール ---
def full_neighbor_abstract(tp, ti):
    """近傍を(自色, 非BG数, 最頻色, 4方向色)で抽象化"""
    for abstraction in ['count_majority', 'directional', 'color_count']:
        rules={}; consistent=True
        for inp,out in tp:
            ga,go=np.array(inp),np.array(out); bg=_bg(ga); h,w=ga.shape
            for r in range(h):
                for c in range(w):
                    nb=[]
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=r+dr,c+dc
                        nb.append(int(ga[nr,nc]) if 0<=nr<h and 0<=nc<w else -1)
                    nb8=[]
                    for dr in [-1,0,1]:
                        for dc in [-1,0,1]:
                            if dr==0 and dc==0: continue
                            nr,nc=r+dr,c+dc
                            nb8.append(int(ga[nr,nc]) if 0<=nr<h and 0<=nc<w else -1)
                    
                    sc=int(ga[r,c]); ov=int(go[r,c])
                    nn=[v for v in nb8 if v!=bg and v!=-1]
                    
                    if abstraction=='count_majority':
                        mc=Counter(nn).most_common(1)[0][0] if nn else bg
                        key=(sc,len(nn),mc)
                    elif abstraction=='directional':
                        key=(sc,tuple(nb))
                    elif abstraction=='color_count':
                        key=(sc,tuple(sorted(Counter(nn).items())))
                    
                    if key in rules and rules[key]!=ov: consistent=False; break
                    rules[key]=ov
                if not consistent: break
            if not consistent: break
        if not consistent: continue
        
        def make_fn(rl,ab):
            def fn(grid):
                g=np.array(grid);h,w=g.shape;bg2=_bg(g);result=g.copy();ch=False
                for r in range(h):
                    for c in range(w):
                        nb=[]
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr,nc=r+dr,c+dc
                            nb.append(int(g[nr,nc]) if 0<=nr<h and 0<=nc<w else -1)
                        nb8=[]
                        for dr in [-1,0,1]:
                            for dc in [-1,0,1]:
                                if dr==0 and dc==0: continue
                                nr,nc=r+dr,c+dc
                                nb8.append(int(g[nr,nc]) if 0<=nr<h and 0<=nc<w else -1)
                        sc=int(g[r,c])
                        nn=[v for v in nb8 if v!=bg2 and v!=-1]
                        if ab=='count_majority':
                            mc=Counter(nn).most_common(1)[0][0] if nn else bg2
                            key=(sc,len(nn),mc)
                        elif ab=='directional':
                            key=(sc,tuple(nb))
                        elif ab=='color_count':
                            key=(sc,tuple(sorted(Counter(nn).items())))
                        if key in rl and rl[key]!=sc:
                            result[r,c]=rl[key]; ch=True
                return result.tolist() if ch else None
            return fn
        r = _verify(make_fn(rules,abstraction), tp, ti)
        if r is not None: return r
    return None

# --- Flood fill ---
def full_flood(tp, ti):
    def flood_neighbor(grid):
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
        result=g.copy(); ch=False
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
    
    r = _verify(flood_neighbor, tp, ti)
    if r is not None: return r
    
    # 固定色flood
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out); bg=_bg(ga)
        for rr in range(ga.shape[0]):
            for cc in range(ga.shape[1]):
                if ga[rr,cc]==bg and go[rr,cc]!=bg:
                    fc=int(go[rr,cc])
                    def make_fn(fcc):
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
                                    if g[r2,c2]==bg2 and not vis[r2,c2]:
                                        result[r2,c2]=fcc; ch2=True
                            return result.tolist() if ch2 else None
                        return fn
                    r = _verify(make_fn(fc), tp, ti)
                    if r is not None: return r
                    break
            break
        break
    return None

# --- Scale ---
def full_scale(tp, ti):
    for inp,out in tp:
        gi2,go2=np.array(inp),np.array(out)
        for s in range(2,8):
            if gi2.shape[0]*s==go2.shape[0] and gi2.shape[1]*s==go2.shape[1]:
                if grid_eq(np.repeat(np.repeat(gi2,s,0),s,1).tolist(),out):
                    t=np.array(ti); return np.repeat(np.repeat(t,s,0),s,1).tolist()
        break
    return None

# --- Tile ---
def full_tile(tp, ti):
    for inp,out in tp:
        gi2,go2=np.array(inp),np.array(out)
        for rh in range(1,6):
            for rw in range(1,6):
                if rh==1 and rw==1: continue
                if gi2.shape[0]*rh==go2.shape[0] and gi2.shape[1]*rw==go2.shape[1]:
                    if grid_eq(np.tile(gi2,(rh,rw)).tolist(),out):
                        return np.tile(np.array(ti),(rh,rw)).tolist()
        break
    return None

# --- Count ---
def full_count(tp, ti):
    gi=np.array(ti); bg=_bg(gi)
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

# --- Repeat pattern ---
def full_repeat(tp, ti):
    gi=np.array(ti); h,w=gi.shape
    for ax in ['r','c']:
        dim=h if ax=='r' else w
        for p in range(1,dim//2+1):
            if dim%p!=0: continue
            tile=gi[:p,:] if ax=='r' else gi[:,:p]
            result=np.tile(tile,(h//p,1)) if ax=='r' else np.tile(tile,(1,w//p))
            if np.array_equal(result,gi): continue
            r = _verify(lambda g,pp=p,axx=ax: (
                np.tile(np.array(g)[:pp,:], (np.array(g).shape[0]//pp, 1)) if axx=='r' else
                np.tile(np.array(g)[:,:pp], (1, np.array(g).shape[1]//pp))
            ).tolist() if (np.array(g).shape[0 if axx=='r' else 1] % pp == 0) else None, tp, ti)
            if r is not None: return r
    return None

# --- Morphology ---
def full_morphology(tp, ti):
    def dilate(grid, steps=1):
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
            result=new
        return result.tolist() if not np.array_equal(result,g) else None
    
    def erode(grid):
        g=np.array(grid);h,w=g.shape;bg=_bg(g);result=g.copy();ch=False
        for r in range(h):
            for c in range(w):
                if g[r,c]==bg: continue
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if not(0<=nr<h and 0<=nc<w) or g[nr,nc]==bg:
                        result[r,c]=bg; ch=True; break
        return result.tolist() if ch else None
    
    def edge(grid):
        g=np.array(grid);h,w=g.shape;bg=_bg(g);result=np.full_like(g,bg);ch=False
        for r in range(h):
            for c in range(w):
                if g[r,c]==bg: continue
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if not(0<=nr<h and 0<=nc<w) or g[nr,nc]==bg:
                        result[r,c]=g[r,c]; ch=True; break
        return result.tolist() if ch else None
    
    for fn in [lambda g: dilate(g,1), lambda g: dilate(g,2), lambda g: dilate(g,3),
               erode, edge]:
        r = _verify(fn, tp, ti)
        if r is not None: return r
    return None

# --- Cannon ---
def full_cannon(tp, ti):
    try:
        from arc.cross3d_projection import cannon_solve
        r = _verify_with_tp(cannon_solve, tp, ti)
        if r is not None: return r
    except: pass
    return None

# --- Even/Odd ---
def full_evenodd(tp, ti):
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        em,om={},{}
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                if ga[r,c]!=go[r,c]:
                    if r%2==0: em[int(ga[r,c])]=int(go[r,c])
                    else: om[int(ga[r,c])]=int(go[r,c])
        if not em and not om: return None
        def make_fn(e,o):
            def fn(grid):
                g=np.array(grid); r2=g.copy()
                for r in range(g.shape[0]):
                    for c in range(g.shape[1]):
                        v=int(g[r,c])
                        if r%2==0 and v in e: r2[r,c]=e[v]
                        elif r%2==1 and v in o: r2[r,c]=o[v]
                return r2.tolist()
            return fn
        r = _verify(make_fn(em,om), tp, ti)
        if r is not None: return r
        # col版
        ec,oc={},{}
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                if ga[r,c]!=go[r,c]:
                    if c%2==0: ec[int(ga[r,c])]=int(go[r,c])
                    else: oc[int(ga[r,c])]=int(go[r,c])
        def make_fn2(e,o):
            def fn(grid):
                g=np.array(grid); r2=g.copy()
                for r in range(g.shape[0]):
                    for c in range(g.shape[1]):
                        v=int(g[r,c])
                        if c%2==0 and v in e: r2[r,c]=e[v]
                        elif c%2==1 and v in o: r2[r,c]=o[v]
                return r2.tolist()
            return fn
        r = _verify(make_fn2(ec,oc), tp, ti)
        if r is not None: return r
        break
    return None


# ══════════════════════════════════════════════════════════════
# Cross Brain: ルーティングテーブル
# ══════════════════════════════════════════════════════════════

# 勘→知識のルーティング（優先度付き）
ROUTING = [
    # (条件関数, フル記憶リスト)
    (lambda f: f.get('out_1x1'), [full_count]),
    (lambda f: f.get('out_smaller') and f.get('is_integer_ratio'), [full_panel, full_crop]),
    (lambda f: f.get('out_smaller'), [full_crop, full_panel]),
    (lambda f: f.get('out_larger') and f.get('is_integer_ratio'), [full_scale, full_tile]),
    (lambda f: f.get('out_larger'), [full_tile, full_scale]),
    (lambda f: f.get('only_change'), [full_colormap, full_evenodd, full_neighbor_abstract, full_template]),
    (lambda f: f.get('only_add') and f.get('partial_sym'), [full_symmetry, full_reversi, full_connect, full_flood]),
    (lambda f: f.get('only_add'), [full_reversi, full_connect, full_flood, full_symmetry, full_cannon, full_morphology]),
    (lambda f: f.get('add_and_remove'), [full_gravity, full_rotate]),
    (lambda f: f.get('same_size'), [full_template, full_neighbor_abstract, full_colormap, full_morphology, full_symmetry, full_reversi, full_evenodd]),
    (lambda f: not f.get('same_size'), [full_panel, full_crop, full_scale, full_tile, full_rotate, full_count]),
]

# フォールバック（全部試す）
ALL_FULL = [
    full_colormap, full_symmetry, full_reversi, full_connect, full_gravity,
    full_rotate, full_crop, full_panel, full_template, full_neighbor_abstract,
    full_flood, full_scale, full_tile, full_count, full_repeat,
    full_morphology, full_cannon, full_evenodd,
]


def cross_brain_solve(train_pairs, test_input):
    """
    Cross Brain: 勘でルーティング → フル記憶で実行
    """
    # 勘を読む
    features = extract_intuition(train_pairs)
    
    # ルーティングテーブルで優先順位付き実行
    tried = set()
    for condition, solvers in ROUTING:
        try:
            if condition(features):
                for solver in solvers:
                    if id(solver) in tried: continue
                    tried.add(id(solver))
                    try:
                        result = solver(train_pairs, test_input)
                        if result is not None:
                            return result, solver.__name__
                    except: continue
        except: continue
    
    # フォールバック: 全部試す
    for solver in ALL_FULL:
        if id(solver) in tried: continue
        try:
            result = solver(train_pairs, test_input)
            if result is not None:
                return result, solver.__name__
        except: continue
    
    return None, None


if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    
    split = 'evaluation' if '--eval' in sys.argv else 'training'
    data_dir = Path(f'/tmp/arc-agi-2/data/{split}')
    
    existing = set()
    with open('arc_v82.log') as f:
        for l in f:
            m = re.search(r'✓.*?([0-9a-f]{8})', l)
            if m: existing.add(m.group(1))
    synth = set(f.stem for f in Path('synth_results').glob('*.py'))
    all_e = existing | synth
    
    solved = []
    for tf in sorted(data_dir.glob('*.json')):
        tid = tf.stem
        with open(tf) as f: task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti, to = task['test'][0]['input'], task['test'][0].get('output')
        
        result, name = cross_brain_solve(tp, ti)
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_e else ''
            solved.append((tid, name, tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tg in solved if tg == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
    print('\n手法別:')
    for name, cnt in Counter(n for _,n,_ in solved).most_common():
        print(f'  {name}: {cnt}')
