"""
arc/cross_life_eye.py — 視覚経験エンジン V2

人間が生まれてから見てきたものをルールとして使う:
- 種が散らばってる → 成長する、つながる
- 対称な形がある → 完成させたい（欠けを埋める）
- 地面がある → 重力で落ちる
- 箱がある → 中を塗る、中身を出す
- 虫がいる → 這う、動く
- 蝶がいる → 羽を広げる（対称展開）
- 交差点がある → 4方向に広がる
- 道がある → つなぐ

各「見え方」に対して具体的な操作セットを持つ。
一つ一つ順番に試す（まとめて発火しない）。
"""

import numpy as np
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label


def _bg(g): return int(Counter(np.array(g).flatten()).most_common(1)[0][0])

def _objs(g, bg, conn=8):
    struct=np.ones((3,3),dtype=int) if conn==8 else np.array([[0,1,0],[1,1,1],[0,1,0]])
    mask=(np.array(g)!=bg).astype(int); labeled,n=scipy_label(mask,structure=struct)
    objs=[]
    for i in range(1,n+1):
        cells=list(zip(*np.where(labeled==i)))
        colors=[int(g[r,c]) for r,c in cells]
        r1=min(r for r,c in cells);c1=min(c for r,c in cells)
        r2=max(r for r,c in cells);c2=max(c for r,c in cells)
        bh,bw=r2-r1+1,c2-c1+1
        objs.append({'cells':cells,'size':len(cells),
            'color':Counter(colors).most_common(1)[0][0],'colors':set(colors),
            'bbox':(r1,c1,r2,c2),'bh':bh,'bw':bw,
            'is_rect':len(cells)==bh*bw,
            'shape':frozenset((r-r1,c-c1) for r,c in cells),
            'center':((r1+r2)/2,(c1+c2)/2),
            'fill_ratio':len(cells)/(bh*bw)})
    return objs

def grid_eq(a,b):
    a,b=np.array(a),np.array(b);return a.shape==b.shape and np.array_equal(a,b)


# ══════════════════════════════════════════════════════════════
# 種の経験: 種を見たら「成長する」
# ══════════════════════════════════════════════════════════════

def exp_seed_grow_cross(tp, ti):
    """種(1-2セル)から十字方向に端or障害物まで伸びる"""
    def apply(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g)
        objs=_objs(g,bg)
        seeds=[o for o in objs if o['size']<=2]
        if not seeds: return None
        changed=False
        for seed in seeds:
            for r,c in seed['cells']:
                color=int(g[r,c])
                for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nr,nc=r+dr,c+dc
                    while 0<=nr<h and 0<=nc<w and g[nr,nc]==bg:
                        g[nr,nc]=color;changed=True;nr+=dr;nc+=dc
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp)
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None

def exp_seed_grow_cross_stop(tp, ti):
    """種から十字方向、同色にぶつかったら止まる"""
    def apply(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g)
        objs=_objs(g,bg)
        seeds=[o for o in objs if o['size']<=2]
        if not seeds: return None
        changed=False
        for seed in seeds:
            for r,c in seed['cells']:
                color=int(g[r,c])
                for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nr,nc=r+dr,c+dc
                    while 0<=nr<h and 0<=nc<w:
                        if g[nr,nc]==bg:
                            g[nr,nc]=color;changed=True
                        elif g[nr,nc]==color:
                            break  # 同色で止まる
                        else:
                            break  # 別色で止まる
                        nr+=dr;nc+=dc
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp)
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None

def exp_seed_connect(tp, ti):
    """同色の種同士をH/V線で接続"""
    def apply(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g)
        changed=False
        for color in set(int(v) for v in g.flatten()) - {bg}:
            pts=[(r,c) for r in range(h) for c in range(w) if g[r,c]==color]
            if len(pts)<2: continue
            for i,(r1,c1) in enumerate(pts):
                for r2,c2 in pts[i+1:]:
                    if r1==r2:
                        for c in range(min(c1,c2)+1,max(c1,c2)):
                            if g[r1,c]==bg: g[r1,c]=color;changed=True
                    elif c1==c2:
                        for r in range(min(r1,r2)+1,max(r1,r2)):
                            if g[r,c1]==bg: g[r,c1]=color;changed=True
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp)
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None


# ══════════════════════════════════════════════════════════════
# 対称の経験: 蝶を見たら「羽を広げたい」
# ══════════════════════════════════════════════════════════════

def exp_symmetry_complete_lr(tp, ti):
    """左右対称に補完"""
    def apply(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g);changed=False
        for r in range(h):
            for c in range(w):
                mc=w-1-c
                if g[r,c]!=bg and g[r,mc]==bg:
                    g[r,mc]=g[r,c];changed=True
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp);
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None

def exp_symmetry_complete_ud(tp, ti):
    """上下対称に補完"""
    def apply(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g);changed=False
        for r in range(h):
            mr=h-1-r
            for c in range(w):
                if g[r,c]!=bg and g[mr,c]==bg:
                    g[mr,c]=g[r,c];changed=True
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp);
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None

def exp_symmetry_complete_4fold(tp, ti):
    """4方向対称に補完"""
    def apply(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g);changed=False
        for r in range(h):
            for c in range(w):
                if g[r,c]==bg: continue
                for mr,mc in [(r,w-1-c),(h-1-r,c),(h-1-r,w-1-c)]:
                    if 0<=mr<h and 0<=mc<w and g[mr,mc]==bg:
                        g[mr,mc]=g[r,c];changed=True
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp);
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None

def exp_symmetry_complete_rot90(tp, ti):
    """90度回転対称に補完"""
    def apply(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g)
        if h!=w: return None
        changed=False
        for _ in range(3):  # 3回回転
            for r in range(h):
                for c in range(w):
                    if g[r,c]==bg: continue
                    mr,mc=c,h-1-r
                    if g[mr,mc]==bg: g[mr,mc]=g[r,c];changed=True
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp);
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None

def exp_symmetry_per_object(tp, ti):
    """各オブジェクト内で対称補完"""
    def apply(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g)
        objs=_objs(g,bg)
        changed=False
        for obj in objs:
            r1,c1,r2,c2=obj['bbox']
            bh,bw=obj['bh'],obj['bw']
            cells=set(obj['cells'])
            # 左右対称
            for r,c in list(cells):
                mc=c1+(c2-c)
                if (r,mc) not in cells and 0<=mc<w:
                    if g[r,mc]==bg:
                        g[r,mc]=int(g[r,c]);changed=True
            # 上下対称
            for r,c in list(cells):
                mr=r1+(r2-r)
                if (mr,c) not in cells and 0<=mr<h:
                    if g[mr,c]==bg:
                        g[mr,c]=int(g[r,c]);changed=True
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp);
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None


# ══════════════════════════════════════════════════════════════
# 重力の経験: 地面があったら「落ちる」
# ══════════════════════════════════════════════════════════════

def exp_gravity_down(tp, ti):
    """非BGセルを下に落とす"""
    def apply(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g)
        for c in range(w):
            vals=[int(g[r,c]) for r in range(h) if g[r,c]!=bg]
            g[:,c]=bg
            for i,v in enumerate(reversed(vals)):
                g[h-1-i,c]=v
        if np.array_equal(g,np.array(grid)): return None
        return g.tolist()
    ok=True
    for inp,out in tp:
        p=apply(inp);
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None

def exp_gravity_to_wall(tp, ti):
    """壁/障害物にぶつかるまで落とす"""
    for dr,dc,name in [(1,0,'down'),(-1,0,'up'),(0,1,'right'),(0,-1,'left')]:
        def apply(grid, ddr=dr, ddc=dc):
            g=np.array(grid).copy();h,w=g.shape;bg=_bg(g)
            objs=_objs(g,bg)
            moved=False
            for obj in sorted(objs, key=lambda o: -(o['center'][0]*ddr+o['center'][1]*ddc)):
                cells=sorted(obj['cells'], key=lambda rc: -(rc[0]*ddr+rc[1]*ddc))
                for r,c in cells:
                    nr,nc=r+ddr,c+ddc
                    while 0<=nr<h and 0<=nc<w and g[nr,nc]==bg:
                        g[nr,nc]=g[nr-ddr,nc-ddc]; g[nr-ddr,nc-ddc]=bg
                        moved=True; nr+=ddr; nc+=ddc
            return g.tolist() if moved else None
        ok=True
        for inp,out in tp:
            p=apply(inp)
            if p is None or not grid_eq(p,out): ok=False;break
        if ok: return apply(ti)
    return None


# ══════════════════════════════════════════════════════════════
# 箱の経験: 箱を見たら「中に何かある」「中を塗る」
# ══════════════════════════════════════════════════════════════

def exp_box_fill_enclosed(tp, ti):
    """閉じた領域を囲む色で塗る"""
    def apply(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g)
        bg_mask=(g==bg).astype(int)
        labeled,n=scipy_label(bg_mask)
        edge_labels=set()
        for r in [0,h-1]:
            for c in range(w):
                if labeled[r,c]>0: edge_labels.add(labeled[r,c])
        for c in [0,w-1]:
            for r in range(h):
                if labeled[r,c]>0: edge_labels.add(labeled[r,c])
        changed=False
        for lid in range(1,n+1):
            if lid in edge_labels: continue
            region=list(zip(*np.where(labeled==lid)))
            nc=set()
            for r,c in region:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr2,nc2=r+dr,c+dc
                    if 0<=nr2<h and 0<=nc2<w and g[nr2,nc2]!=bg:
                        nc.add(int(g[nr2,nc2]))
            if len(nc)==1:
                fc=nc.pop()
                for r,c in region: g[r,c]=fc;changed=True
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp);
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None

def exp_box_crop_largest(tp, ti):
    """最大オブジェクトのbboxをcrop"""
    def apply(grid):
        g=np.array(grid);bg=_bg(g);objs=_objs(g,bg)
        if not objs: return None
        o=max(objs,key=lambda x:x['size'])
        r1,c1,r2,c2=o['bbox']
        return g[r1:r2+1,c1:c2+1].tolist()
    ok=True
    for inp,out in tp:
        p=apply(inp);
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None

def exp_box_crop_smallest(tp, ti):
    """最小オブジェクトのbboxをcrop"""
    def apply(grid):
        g=np.array(grid);bg=_bg(g);objs=_objs(g,bg)
        if not objs: return None
        o=min(objs,key=lambda x:x['size'])
        r1,c1,r2,c2=o['bbox']
        return g[r1:r2+1,c1:c2+1].tolist()
    ok=True
    for inp,out in tp:
        p=apply(inp);
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None


# ══════════════════════════════════════════════════════════════
# 色マップの経験: 色の対応を学ぶ
# ══════════════════════════════════════════════════════════════

def exp_colormap(tp, ti):
    """一貫した色マップ"""
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
    for inp,out in tp:
        ga=np.array(inp);r2=ga.copy()
        for o,n in m.items(): r2[ga==o]=n
        if not grid_eq(r2.tolist(),out): return None
    return result.tolist()


# ══════════════════════════════════════════════════════════════
# 視覚経験ソルバー: 一つ一つ順番に試す
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# 追加の視覚経験
# ══════════════════════════════════════════════════════════════

def exp_mirror_across_line(tp, ti):
    """軸線（全行or全列が同色）を基準にミラー"""
    for inp, out in tp:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None
        bg = _bg(ga); h, w = ga.shape
        
        # 水平軸
        for r in range(h):
            if all(ga[r,c] != bg for c in range(w)):
                def mirror_h(grid, axis_r=r):
                    g=np.array(grid).copy();hh,ww=g.shape;bg2=_bg(g)
                    axes=[rr for rr in range(hh) if all(g[rr,cc]!=bg2 for cc in range(ww))]
                    if not axes: return None
                    ar=axes[0]; changed=False
                    for rr in range(hh):
                        mr=2*ar-rr
                        if mr<0 or mr>=hh or mr==rr: continue
                        for cc in range(ww):
                            if g[rr,cc]!=bg2 and g[mr,cc]==bg2:
                                g[mr,cc]=g[rr,cc];changed=True
                    return g.tolist() if changed else None
                ok=True
                for i2,o2 in tp:
                    p=mirror_h(i2)
                    if p is None or not grid_eq(p,o2): ok=False;break
                if ok: return mirror_h(ti)
        
        # 垂直軸
        for c in range(w):
            if all(ga[r,c] != bg for r in range(h)):
                def mirror_v(grid, axis_c=c):
                    g=np.array(grid).copy();hh,ww=g.shape;bg2=_bg(g)
                    axes=[cc for cc in range(ww) if all(g[rr,cc]!=bg2 for rr in range(hh))]
                    if not axes: return None
                    ac=axes[0]; changed=False
                    for cc in range(ww):
                        mc=2*ac-cc
                        if mc<0 or mc>=ww or mc==cc: continue
                        for rr in range(hh):
                            if g[rr,cc]!=bg2 and g[rr,mc]==bg2:
                                g[rr,mc]=g[rr,cc];changed=True
                    return g.tolist() if changed else None
                ok=True
                for i2,o2 in tp:
                    p=mirror_v(i2)
                    if p is None or not grid_eq(p,o2): ok=False;break
                if ok: return mirror_v(ti)
        break
    return None


def exp_flood_color_region(tp, ti):
    """各色のBG隣接セルをflood fill（色が広がる）"""
    for n_iter in [1, 2, 3, 5, 10, 50]:
        def apply(grid, iterations=n_iter):
            g=np.array(grid).copy();h,w=g.shape;bg=_bg(g)
            for _ in range(iterations):
                new_g=g.copy(); any_change=False
                for r in range(h):
                    for c in range(w):
                        if g[r,c]!=bg: continue
                        nc=[]
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr,ncc=r+dr,c+dc
                            if 0<=nr<h and 0<=ncc<w and g[nr,ncc]!=bg:
                                nc.append(int(g[nr,ncc]))
                        if len(nc)==1:
                            new_g[r,c]=nc[0]; any_change=True
                        elif len(nc)>=2:
                            cc=Counter(nc)
                            if cc.most_common(1)[0][1]>1:
                                new_g[r,c]=cc.most_common(1)[0][0]; any_change=True
                g=new_g
                if not any_change: break
            if np.array_equal(g,np.array(grid)): return None
            return g.tolist()
        ok=True
        for inp,out in tp:
            p=apply(inp)
            if p is None or not grid_eq(p,out): ok=False;break
        if ok: return apply(ti)
    return None


def exp_neighbor_count_color(tp, ti):
    """各セルの非BG隣接数に基づいて色を変える"""
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=_bg(ga);h,w=ga.shape
        
        # 変更されたセルで、隣接非BG数→色のマッピングを学習
        nb_map={}; consistent=True
        for r in range(h):
            for c in range(w):
                if ga[r,c]==go[r,c]: continue
                nb=sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
                       if 0<=r+dr<h and 0<=c+dc<w and ga[r+dr,c+dc]!=bg)
                ov=int(go[r,c])
                if nb in nb_map and nb_map[nb]!=ov: consistent=False;break
                nb_map[nb]=ov
            if not consistent: break
        
        if consistent and nb_map:
            def apply(grid):
                g=np.array(grid).copy();h2,w2=g.shape;bg2=_bg(g);changed=False
                for r in range(h2):
                    for c in range(w2):
                        nb=sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
                               if 0<=r+dr<h2 and 0<=c+dc<w2 and g[r+dr,c+dc]!=bg2)
                        if nb in nb_map and g[r,c]!=nb_map[nb]:
                            g[r,c]=nb_map[nb];changed=True
                return g.tolist() if changed else None
            ok=True
            for i2,o2 in tp:
                p=apply(i2)
                if p is None or not grid_eq(p,o2): ok=False;break
            if ok: return apply(ti)
        break
    return None


def exp_tile_pattern(tp, ti):
    """小さなパターンをタイリング"""
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        hi,wi=ga.shape;ho,wo=go.shape
        if ho<hi or wo<wi: continue
        if ho%hi==0 and wo%wi==0:
            tile=ga
            pred=np.tile(tile,(ho//hi,wo//wi))
            if grid_eq(pred.tolist(),out):
                gi=np.array(ti)
                ht,wt=gi.shape
                for rr in range(1,5):
                    for cc in range(1,5):
                        pred_t=np.tile(gi,(rr,cc))
                        ok=True
                        for i2,o2 in tp:
                            g2=np.array(i2);h2,w2=g2.shape
                            go2=np.array(o2);ho2,wo2=go2.shape
                            if ho2%h2!=0 or wo2%w2!=0: ok=False;break
                            p2=np.tile(g2,(ho2//h2,wo2//w2))
                            if not grid_eq(p2.tolist(),o2): ok=False;break
                        if ok:
                            return pred_t.tolist()
        break
    return None


def exp_scale_up(tp, ti):
    """各セルをNxNブロックに拡大"""
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        hi,wi=ga.shape;ho,wo=go.shape
        if ho%hi!=0 or wo%wi!=0: continue
        sr,sc=ho//hi,wo//wi
        if sr!=sc or sr<2: continue
        pred=np.repeat(np.repeat(ga,sr,axis=0),sc,axis=1)
        if grid_eq(pred.tolist(),out):
            def apply(grid,scale=sr):
                g=np.array(grid)
                return np.repeat(np.repeat(g,scale,axis=0),scale,axis=1).tolist()
            ok=True
            for i2,o2 in tp:
                p=apply(i2)
                if not grid_eq(p,o2): ok=False;break
            if ok: return apply(ti)
        break
    return None


def exp_unique_object_extract(tp, ti):
    """ユニークな属性を持つオブジェクトを抽出"""
    for attr_fn in [lambda o:o['color'], lambda o:o['size'], lambda o:o['shape'],
                    lambda o:(o['bh'],o['bw']), lambda o:o['is_rect']]:
        ok=True
        for inp,out in tp:
            g=np.array(inp);bg=_bg(g);objs=_objs(g,bg)
            if len(objs)<2: ok=False;break
            cc=Counter(attr_fn(o) for o in objs)
            if len(cc)<2: ok=False;break
            uniq=[o for o in objs if cc[attr_fn(o)]==1]
            if len(uniq)!=1: ok=False;break
            o=uniq[0];r1,c1,r2,c2=o['bbox']
            crop=g[r1:r2+1,c1:c2+1]
            if not grid_eq(crop.tolist(),out): ok=False;break
        if ok:
            g=np.array(ti);bg=_bg(g);objs=_objs(g,bg)
            if len(objs)<2: continue
            cc=Counter(attr_fn(o) for o in objs)
            if len(cc)<2: continue
            uniq=[o for o in objs if cc[attr_fn(o)]==1]
            if len(uniq)==1:
                o=uniq[0];r1,c1,r2,c2=o['bbox']
                return g[r1:r2+1,c1:c2+1].tolist()
    return None



# ══════════════════════════════════════════════════════════════
# 形→意味→操作の深化バリエーション
# ══════════════════════════════════════════════════════════════

def exp_shape_to_stamp(tp, ti):
    """形の意味に基づくスタンプ: 最大=テンプレ→各小オブジェクトの位置にスタンプ"""
    for recolor in [True, False]:
        def apply(grid, rc=recolor):
            g=np.array(grid);bg=_bg(g);objs=_objs(g,bg)
            if len(objs)<2: return None
            tmpl=max(objs,key=lambda o:o['size'])
            targets=[o for o in objs if o is not tmpl]
            tr1,tc1=tmpl['bbox'][0],tmpl['bbox'][1]
            tshape=[(r-tr1,c-tc1,int(g[r,c])) for r,c in tmpl['cells']]
            result=g.copy()
            for t in targets:
                or1,oc1=t['bbox'][0],t['bbox'][1]
                tc=t['color']
                for dr,dc,tv in tshape:
                    nr,nc=or1+dr,oc1+dc
                    if 0<=nr<g.shape[0] and 0<=nc<g.shape[1]:
                        result[nr,nc]=tc if rc and tv!=bg else tv
            if np.array_equal(result,g): return None
            return result.tolist()
        ok=True
        for inp,out in tp:
            p=apply(inp)
            if p is None or not grid_eq(p,out): ok=False;break
        if ok: return apply(ti)
    return None


def exp_shape_to_fill_color(tp, ti):
    """形の意味: オブジェクトのbbox内部をそのオブジェクトの色で塗りつぶし"""
    def apply(grid):
        g=np.array(grid).copy();bg=_bg(g);objs=_objs(g,bg);changed=False
        for obj in objs:
            if obj['is_rect']: continue
            r1,c1,r2,c2=obj['bbox']
            for r in range(r1,r2+1):
                for c in range(c1,c2+1):
                    if g[r,c]==bg: g[r,c]=obj['color'];changed=True
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp)
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None


def exp_shape_remove_noise(tp, ti):
    """形の意味: 各オブジェクト内の少数派色セルを多数派色に置換"""
    def apply(grid):
        g=np.array(grid).copy();bg=_bg(g);objs=_objs(g,bg);changed=False
        for obj in objs:
            if len(obj['colors'])<=1: continue
            maj=obj['color']
            for r,c in obj['cells']:
                if int(g[r,c])!=maj and int(g[r,c])!=bg:
                    g[r,c]=maj;changed=True
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp)
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None


def exp_shape_sort_by_size(tp, ti):
    """形の意味: オブジェクトをサイズ順に左→右 or 上→下に並べ替え"""
    for direction in ['lr','ud']:
        def apply(grid, d=direction):
            g=np.array(grid);bg=_bg(g);objs=_objs(g,bg)
            if len(objs)<2: return None
            # 各オブジェクトのcrop
            crops=[]
            for o in sorted(objs,key=lambda x:x['size']):
                r1,c1,r2,c2=o['bbox']
                crops.append(g[r1:r2+1,c1:c2+1].copy())
            # 出力サイズ計算
            if d=='lr':
                h=max(c.shape[0] for c in crops)
                w=sum(c.shape[1] for c in crops)
                result=np.full((h,w),bg,dtype=int)
                cc=0
                for crop in crops:
                    ch,cw=crop.shape
                    result[:ch,cc:cc+cw]=crop
                    cc+=cw
            else:
                h=sum(c.shape[0] for c in crops)
                w=max(c.shape[1] for c in crops)
                result=np.full((h,w),bg,dtype=int)
                rr=0
                for crop in crops:
                    ch,cw=crop.shape
                    result[rr:rr+ch,:cw]=crop
                    rr+=ch
            return result.tolist()
        ok=True
        for inp,out in tp:
            p=apply(inp)
            if p is None or not grid_eq(p,out): ok=False;break
        if ok: return apply(ti)
    return None


def exp_shape_color_by_size(tp, ti):
    """形の意味: オブジェクトのサイズに応じて色を割り当て"""
    # trainから size→output_color のマッピングを学習
    size_to_color = {}
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=_bg(ga);objs=_objs(ga,bg)
        for obj in objs:
            # このオブジェクトのセルが出力で何色になったか
            out_colors=[int(go[r,c]) for r,c in obj['cells'] if go[r,c]!=ga[r,c]]
            if not out_colors: continue
            oc=Counter(out_colors).most_common(1)[0][0]
            if obj['size'] in size_to_color and size_to_color[obj['size']]!=oc:
                return None
            size_to_color[obj['size']]=oc
    if not size_to_color: return None
    def apply(grid):
        g=np.array(grid).copy();bg2=_bg(g);objs=_objs(g,bg2);changed=False
        for obj in objs:
            if obj['size'] in size_to_color:
                nc=size_to_color[obj['size']]
                for r,c in obj['cells']:
                    if g[r,c]!=nc: g[r,c]=nc;changed=True
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp)
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None


def exp_shape_outline(tp, ti):
    """形の意味: 塗りつぶし矩形→枠線のみ（内部をBGに）"""
    def apply(grid):
        g=np.array(grid).copy();bg=_bg(g);objs=_objs(g,bg);changed=False
        for obj in objs:
            if not obj['is_rect'] or obj['bh']<3 or obj['bw']<3: continue
            r1,c1,r2,c2=obj['bbox']
            for r in range(r1+1,r2):
                for c in range(c1+1,c2):
                    if g[r,c]!=bg: g[r,c]=bg;changed=True
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp)
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None


def exp_shape_diagonal_fill(tp, ti):
    """種から斜め4方向に端まで伸びる"""
    def apply(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g)
        objs=_objs(g,bg)
        seeds=[o for o in objs if o['size']<=2]
        if not seeds: return None
        changed=False
        for seed in seeds:
            for r,c in seed['cells']:
                color=int(g[r,c])
                for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr,nc=r+dr,c+dc
                    while 0<=nr<h and 0<=nc<w and g[nr,nc]==bg:
                        g[nr,nc]=color;changed=True;nr+=dr;nc+=dc
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp)
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None


def exp_shape_cross_and_diag(tp, ti):
    """種から8方向（十字+斜め）に端まで伸びる"""
    def apply(grid):
        g=np.array(grid).copy();h,w=g.shape;bg=_bg(g)
        objs=_objs(g,bg)
        seeds=[o for o in objs if o['size']<=2]
        if not seeds: return None
        changed=False
        for seed in seeds:
            for r,c in seed['cells']:
                color=int(g[r,c])
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        nr,nc=r+dr,c+dc
                        while 0<=nr<h and 0<=nc<w and g[nr,nc]==bg:
                            g[nr,nc]=color;changed=True;nr+=dr;nc+=dc
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp)
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None


def exp_object_count_output(tp, ti):
    """オブジェクト数を出力サイズに反映（1xN or Nx1）"""
    for inp,out in tp:
        ga=np.array(inp);go=np.array(out);bg=_bg(ga)
        objs=_objs(ga,bg)
        n=len(objs)
        if go.shape==(1,n) or go.shape==(n,1):
            # 各オブジェクトの色を並べる
            for sort_key in ['center_r','center_c','size','color']:
                if sort_key=='center_r': sorted_objs=sorted(objs,key=lambda o:o['center'][0])
                elif sort_key=='center_c': sorted_objs=sorted(objs,key=lambda o:o['center'][1])
                elif sort_key=='size': sorted_objs=sorted(objs,key=lambda o:o['size'])
                else: sorted_objs=sorted(objs,key=lambda o:o['color'])
                colors=[o['color'] for o in sorted_objs]
                if go.shape==(1,n): pred=np.array([colors])
                else: pred=np.array([[c] for c in colors])
                if grid_eq(pred.tolist(),out):
                    def make_apply(sk=sort_key,horiz=go.shape==(1,n)):
                        def apply(grid):
                            g=np.array(grid);bg2=_bg(g);objs2=_objs(g,bg2)
                            if not objs2: return None
                            if sk=='center_r': so=sorted(objs2,key=lambda o:o['center'][0])
                            elif sk=='center_c': so=sorted(objs2,key=lambda o:o['center'][1])
                            elif sk=='size': so=sorted(objs2,key=lambda o:o['size'])
                            else: so=sorted(objs2,key=lambda o:o['color'])
                            colors2=[o['color'] for o in so]
                            nn=len(colors2)
                            if horiz: return [[c for c in colors2]]
                            else: return [[c] for c in colors2]
                        return apply
                    af=make_apply()
                    ok2=True
                    for i2,o2 in tp:
                        p=af(i2)
                        if p is None or not grid_eq(p,o2): ok2=False;break
                    if ok2: return af(ti)
        break
    return None


def exp_conditional_color_swap(tp, ti):
    """条件付き色入れ替え: 特定色Aの近くにある色Bを色Cに"""
    # trainから変更セルを分析
    swap_rules = []
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=_bg(ga);h,w=ga.shape
        for r in range(h):
            for c in range(w):
                if ga[r,c]!=go[r,c]:
                    old_c=int(ga[r,c]);new_c=int(go[r,c])
                    # 近傍に何色があるか
                    nbs=[int(ga[r+dr,c+dc]) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                         if 0<=r+dr<h and 0<=c+dc<w and ga[r+dr,c+dc]!=bg and ga[r+dr,c+dc]!=old_c]
                    if nbs:
                        trigger=Counter(nbs).most_common(1)[0][0]
                        swap_rules.append((old_c,trigger,new_c))
    if not swap_rules: return None
    # 最頻ルールを使用
    rule=Counter(swap_rules).most_common(1)[0][0]
    old_c,trigger,new_c=rule
    def apply(grid):
        g=np.array(grid).copy();h2,w2=g.shape;bg2=_bg(g);changed=False
        for r in range(h2):
            for c in range(w2):
                if int(g[r,c])!=old_c: continue
                nbs=[int(g[r+dr,c+dc]) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                     if 0<=r+dr<h2 and 0<=c+dc<w2]
                if trigger in nbs:
                    g[r,c]=new_c;changed=True
        return g.tolist() if changed else None
    ok=True
    for inp,out in tp:
        p=apply(inp)
        if p is None or not grid_eq(p,out): ok=False;break
    return apply(ti) if ok else None



ALL_EXPERIENCES = [
    # 色（最初に試す — kofdaiの思考プロセス）
    ('色マップ', exp_colormap),
    
    # 対称（蝶・鏡）
    ('対称LR', exp_symmetry_complete_lr),
    ('対称UD', exp_symmetry_complete_ud),
    ('対称4方向', exp_symmetry_complete_4fold),
    ('対称90度', exp_symmetry_complete_rot90),
    ('対称オブジェクト内', exp_symmetry_per_object),
    
    # 種の成長
    ('種→十字', exp_seed_grow_cross),
    ('種→十字止まり', exp_seed_grow_cross_stop),
    ('種→接続', exp_seed_connect),
    
    # 重力
    ('重力↓', exp_gravity_down),
    ('重力→壁', exp_gravity_to_wall),
    
    # 箱
    ('箱→閉じ塗り', exp_box_fill_enclosed),
    ('箱→最大crop', exp_box_crop_largest),
    ('箱→最小crop', exp_box_crop_smallest),
    
    # 軸ミラー
    ('軸ミラー', exp_mirror_across_line),
    
    # 色の広がり
    ('色flood', exp_flood_color_region),
    
    # 隣接ルール
    ('隣接数→色', exp_neighbor_count_color),
    
    # 拡大
    ('タイル', exp_tile_pattern),
    ('拡大', exp_scale_up),
    
    # 仲間はずれ
    ('仲間はずれ抽出', exp_unique_object_extract),
    
    # 形→意味→操作の深化
    ('形→スタンプ', exp_shape_to_stamp),
    ('形→内部塗り', exp_shape_to_fill_color),
    ('形→ノイズ除去', exp_shape_remove_noise),
    ('形→サイズ順並替', exp_shape_sort_by_size),
    ('形→サイズ色分け', exp_shape_color_by_size),
    ('形→枠線化', exp_shape_outline),
    ('種→斜め', exp_shape_diagonal_fill),
    ('種→8方向', exp_shape_cross_and_diag),
    ('オブジェクト数→出力', exp_object_count_output),
    ('条件付き色替え', exp_conditional_color_swap),
]


def cross_life_eye_solve(train_pairs, test_input):
    """一つ一つ順番に経験を試す"""
    for name, exp_fn in ALL_EXPERIENCES:
        try:
            result = exp_fn(train_pairs, test_input)
            if result is not None:
                return result, f'life:{name}'
        except:
            pass
    
    # フォールバック: Brain V2
    from arc.cross_brain_v2 import cross_brain_v2_solve
    return cross_brain_v2_solve(train_pairs, test_input)


if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    
    split = 'evaluation' if '--eval' in sys.argv else 'training'
    data_dir = Path(f'/tmp/arc-agi-2/data/{split}')
    
    existing = set()
    with open('arc_v82.log') as f:
        for l in f:
            m2 = re.search(r'✓.*?([0-9a-f]{8})', l)
            if m2: existing.add(m2.group(1))
    synth = set(f.stem for f in Path('synth_results').glob('*.py'))
    all_e = existing | synth
    
    solved = []
    for tf in sorted(data_dir.glob('*.json')):
        tid = tf.stem
        with open(tf) as f: task = json.load(f)
        tp2 = [(e['input'], e['output']) for e in task['train']]
        ti2, to2 = task['test'][0]['input'], task['test'][0].get('output')
        
        result, sname = cross_life_eye_solve(tp2, ti2)
        if result and to2 and grid_eq(result, to2):
            tag = 'NEW' if tid not in all_e else ''
            solved.append((tid, sname, tag))
            if tag: print(f'  ✓ {tid} [{sname}] NEW')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tg in solved if tg == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
    
    solver_stats = Counter()
    for _, sname, _ in solved:
        solver_stats[sname] += 1
    print('\n経験別:')
    for s, c in solver_stats.most_common():
        print(f'  {s}: {c}')


