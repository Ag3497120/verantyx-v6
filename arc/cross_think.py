"""
arc/cross_think.py — Crossの思考プロセス

人間の問題解決:
1. まず見る（何がある？）
2. 色の順番・並びを見る（規則性ある？）
3. なければシンボル的意味を探す（この色は壁？マーカー？）
4. 一つ一つ仮説を検証する（trainで合う？）
5. 全部ダメなら組み合わせて試す

「まとめて発火」ではなく「一つ一つ見る」
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
        objs.append({'cells':cells,'size':len(cells),'color':Counter(colors).most_common(1)[0][0],
            'colors':set(colors),'bbox':(r1,c1,r2,c2),'bh':r2-r1+1,'bw':c2-c1+1,
            'shape':frozenset((r-r1,c-c1) for r,c in cells),
            'is_rect':len(cells)==(r2-r1+1)*(c2-c1+1)})
    return objs

def grid_eq(a,b):
    a,b=np.array(a),np.array(b)
    return a.shape==b.shape and np.array_equal(a,b)


# ══════════════════════════════════════════════════════════════
# Step 0: 見る（知覚）— 何がある？
# ══════════════════════════════════════════════════════════════

class Observation:
    """問題を「見る」結果"""
    def __init__(self, train_pairs):
        self.train_pairs = train_pairs
        self.n_train = len(train_pairs)
        
        gi0, go0 = np.array(train_pairs[0][0]), np.array(train_pairs[0][1])
        self.bg = _bg(gi0)
        
        # サイズ関係
        self.same_size = gi0.shape == go0.shape
        self.out_smaller = go0.size < gi0.size
        self.out_larger = go0.size > gi0.size
        self.out_1x1 = go0.shape == (1,1)
        
        # 色の情報
        self.in_colors = sorted(set(int(v) for v in gi0.flatten()))
        self.out_colors = sorted(set(int(v) for v in go0.flatten()))
        self.fg_colors = [c for c in self.in_colors if c != self.bg]
        self.n_colors = len(self.fg_colors)
        
        # 色のCrossサイズ（希少度ベース）
        total = gi0.size
        color_counts = Counter(int(v) for v in gi0.flatten())
        self.cross_sizes = {}
        for color, count in color_counts.items():
            if color == self.bg:
                self.cross_sizes[color] = 0
                continue
            ratio = count / total
            if ratio < 0.01: self.cross_sizes[color] = 7
            elif ratio < 0.03: self.cross_sizes[color] = 5
            elif ratio < 0.1: self.cross_sizes[color] = 3
            elif ratio < 0.3: self.cross_sizes[color] = 2
            else: self.cross_sizes[color] = 1
        
        # 色の役割
        self.color_roles = {}
        for color in self.in_colors:
            self.color_roles[color] = self._detect_role(gi0, color, color_counts.get(color,0))
        
        # オブジェクト
        self.objs = _objs(gi0, self.bg)
        self.n_objs = len(self.objs)
        
        # overlay
        if self.same_size:
            a=r=ch=0
            for rr in range(gi0.shape[0]):
                for cc in range(gi0.shape[1]):
                    iv,ov=int(gi0[rr,cc]),int(go0[rr,cc])
                    if iv==ov: pass
                    elif iv==self.bg and ov!=self.bg: a+=1
                    elif iv!=self.bg and ov==self.bg: r+=1
                    else: ch+=1
            self.overlay = 'only_add' if a and not r and not ch else \
                           'only_remove' if r and not a and not ch else \
                           'only_change' if ch and not a and not r else \
                           'add_and_remove' if a and r else 'mixed'
        else:
            self.overlay = 'size_diff'
        
        # セパレータ
        self.has_separator = False
        self.sep_color = None
        for r in range(gi0.shape[0]):
            vals=set(int(v) for v in gi0[r])
            if len(vals)==1 and vals.pop()!=self.bg:
                self.has_separator=True; self.sep_color=int(gi0[r,0]); break
        if not self.has_separator:
            for c in range(gi0.shape[1]):
                vals=set(int(v) for v in gi0[:,c])
                if len(vals)==1 and vals.pop()!=self.bg:
                    self.has_separator=True; self.sep_color=int(gi0[0,c]); break
    
    def _detect_role(self, g, color, count):
        h,w=g.shape; total=h*w; ratio=count/total
        if ratio>0.5: return 'bg'
        edge=sum(1 for r in range(h) for c in [0,w-1] if g[r,c]==color) + \
             sum(1 for c in range(w) for r in [0,h-1] if g[r,c]==color)
        if edge>(2*(h+w)-4)*0.4: return 'frame'
        if count<=2: return 'marker'
        for r in range(h):
            if all(g[r,c]==color for c in range(w)): return 'separator'
        for c in range(w):
            if all(g[r,c]==color for r in range(h)): return 'separator'
        if ratio<0.05: return 'accent'
        return 'fill'


# ══════════════════════════════════════════════════════════════
# Step 1: 色の順番を見る（最初に試すこと）
# ══════════════════════════════════════════════════════════════

def think_color_order(obs, tp, ti):
    """色の値の順番に意味がある？ (1<2<3...とか)"""
    
    # 1a. 色の値そのまま変換（色マップ）
    result = _try_colormap(tp, ti)
    if result: return result, '色の順番:色マップ'
    
    # 1b. 色の大小で並べ替え
    result = _try_color_sort(tp, ti, obs.bg)
    if result: return result, '色の順番:ソート'
    
    # 1c. 色値を演算（+1, *2, mod）
    result = _try_color_arithmetic(tp, ti)
    if result: return result, '色の順番:演算'
    
    # 1d. 偶数色/奇数色で分ける
    result = _try_color_parity(tp, ti, obs.bg)
    if result: return result, '色の順番:偶奇'
    
    return None, None

def _try_colormap(tp, ti):
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
    # verify
    for inp,out in tp:
        ga=np.array(inp);r2=ga.copy()
        for o,n in m.items(): r2[ga==o]=n
        if not grid_eq(r2.tolist(),out): return None
    return result.tolist()

def _try_color_sort(tp, ti, bg):
    for axis in ['row','col']:
        ok=True
        for inp,out in tp:
            ga,go=np.array(inp),np.array(out)
            if ga.shape!=go.shape: ok=False; break
            r2=ga.copy()
            if axis=='row':
                for r in range(ga.shape[0]):
                    vals=sorted([int(v) for v in ga[r] if v!=bg])
                    ci=0
                    for c in range(ga.shape[1]):
                        if ga[r,c]!=bg: r2[r,c]=vals[ci]; ci+=1
            else:
                for c in range(ga.shape[1]):
                    vals=sorted([int(v) for v in ga[:,c] if v!=bg])
                    ri=0
                    for r in range(ga.shape[0]):
                        if ga[r,c]!=bg: r2[r,c]=vals[ri]; ri+=1
            if not grid_eq(r2.tolist(),out): ok=False; break
        if ok:
            gi=np.array(ti);result=gi.copy()
            if axis=='row':
                for r in range(gi.shape[0]):
                    vals=sorted([int(v) for v in gi[r] if v!=bg])
                    ci=0
                    for c in range(gi.shape[1]):
                        if gi[r,c]!=bg: result[r,c]=vals[ci]; ci+=1
            else:
                for c in range(gi.shape[1]):
                    vals=sorted([int(v) for v in gi[:,c] if v!=bg])
                    ri=0
                    for r in range(gi.shape[0]):
                        if gi[r,c]!=bg: result[r,c]=vals[ri]; ri+=1
            return result.tolist()
    return None

def _try_color_arithmetic(tp, ti):
    for op_name, op in [('add1',lambda v:(v+1)%10), ('sub1',lambda v:(v-1)%10),
                        ('mul2',lambda v:(v*2)%10), ('mod3',lambda v:v%3)]:
        ok=True
        for inp,out in tp:
            ga,go=np.array(inp),np.array(out)
            if ga.shape!=go.shape: ok=False; break
            r2=np.vectorize(op)(ga)
            if not grid_eq(r2.tolist(),out): ok=False; break
        if ok:
            return np.vectorize(op)(np.array(ti)).tolist()
    return None

def _try_color_parity(tp, ti, bg):
    # 偶数色→1色、奇数色→別色
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        even_map={}; odd_map={}
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                iv,ov=int(ga[r,c]),int(go[r,c])
                if iv==ov: continue
                if iv%2==0: even_map[iv]=ov
                else: odd_map[iv]=ov
        if even_map or odd_map:
            def apply(grid):
                g=np.array(grid);r2=g.copy()
                for o,n in even_map.items(): r2[g==o]=n
                for o,n in odd_map.items(): r2[g==o]=n
                return r2.tolist()
            ok=True
            for i2,o2 in tp:
                if not grid_eq(apply(i2),o2): ok=False; break
            if ok: return apply(ti)
        break
    return None


# ══════════════════════════════════════════════════════════════
# Step 2: シンボル的意味を探す
# ══════════════════════════════════════════════════════════════

def think_symbolic(obs, tp, ti):
    """色にシンボル的意味がある？（壁、マーカー、方向指標...）"""
    
    # 2a. マーカー色が方向を示す
    result = _try_marker_direction(tp, ti, obs)
    if result: return result, 'シンボル:マーカー方向'
    
    # 2b. フレーム色が構造を示す
    result = _try_frame_structure(tp, ti, obs)
    if result: return result, 'シンボル:フレーム構造'
    
    # 2c. アクセント色がルールを示す
    result = _try_accent_rule(tp, ti, obs)
    if result: return result, 'シンボル:アクセント'
    
    # 2d. Crossサイズの支配
    result = _try_cross_dominance(tp, ti, obs)
    if result: return result, 'シンボル:Cross支配'
    
    # 2e. Crossサイズの競合
    result = _try_cross_competition(tp, ti, obs)
    if result: return result, 'シンボル:Cross競合'
    
    # 2f. Cross境界で何かが起きる
    result = _try_cross_boundary(tp, ti, obs)
    if result: return result, 'シンボル:Cross境界'
    
    return None, None

def _build_influence(grid, cross_sizes):
    g=np.array(grid);h,w=g.shape;bg=_bg(g)
    influence={}
    for color,size in cross_sizes.items():
        if size==0: continue
        inf=np.zeros((h,w),dtype=float); half=size//2
        for r in range(h):
            for c in range(w):
                if int(g[r,c])!=color: continue
                for dr in range(-half,half+1):
                    for dc in range(-half,half+1):
                        nr,nc=r+dr,c+dc
                        if 0<=nr<h and 0<=nc<w:
                            dist=max(abs(dr),abs(dc))
                            inf[nr,nc]=max(inf[nr,nc], 1.0-dist/(half+1))
        influence[color]=inf
    return influence

def _try_marker_direction(tp, ti, obs):
    """マーカー(1-2セル)が方向を指す → その方向に操作"""
    markers = [c for c,role in obs.color_roles.items() if role=='marker']
    if not markers: return None
    
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=obs.bg
        
        for mc in markers:
            mpos = [(r,c) for r in range(ga.shape[0]) for c in range(ga.shape[1]) if ga[r,c]==mc]
            if len(mpos)!=1: continue
            mr,mcc = mpos[0]
            
            # マーカーの周囲の非BGオブジェクトとの関係
            objs = _objs(ga, bg)
            for obj in objs:
                if obj['color']==mc: continue
                # マーカーからオブジェクトへの方向
                cr = sum(r for r,c in obj['cells'])/len(obj['cells'])
                cc2 = sum(c for r,c in obj['cells'])/len(obj['cells'])
                dr = 1 if cr > mr else (-1 if cr < mr else 0)
                dc = 1 if cc2 > mcc else (-1 if cc2 < mcc else 0)
                
                # その方向にオブジェクトを移動して出力と一致するか
                if dr==0 and dc==0: continue
        break
    return None

def _try_frame_structure(tp, ti, obs):
    """枠色がグリッド構造を定義"""
    frames = [c for c,role in obs.color_roles.items() if role=='frame']
    if not frames: return None
    return None

def _try_accent_rule(tp, ti, obs):
    """少数色がルールの鍵"""
    accents = [c for c,role in obs.color_roles.items() if role=='accent']
    if not accents: return None
    return None

def _try_cross_dominance(tp, ti, obs):
    """大きなCrossが小さなCrossを支配する"""
    if len(obs.fg_colors) < 2: return None
    
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=obs.bg; h,w=ga.shape
        
        inf = _build_influence(inp, obs.cross_sizes)
        
        # 各セルで最も影響力の大きい色が出力色を決めるか
        ok=True
        for r in range(h):
            for c in range(w):
                if ga[r,c]==go[r,c]: continue
                
                # このセルの支配色（最大影響力）
                best_color=None; best_str=0
                for color,inf_map in inf.items():
                    if inf_map[r,c]>best_str:
                        best_str=inf_map[r,c]; best_color=color
                
                if best_color!=int(go[r,c]):
                    ok=False; break
            if not ok: break
        
        if ok:
            # test適用
            gi=np.array(ti); result=gi.copy()
            inf_t = _build_influence(ti, obs.cross_sizes)
            for r in range(gi.shape[0]):
                for c in range(gi.shape[1]):
                    best_color=None; best_str=0
                    for color,inf_map in inf_t.items():
                        if inf_map[r,c]>best_str:
                            best_str=inf_map[r,c]; best_color=color
                    if best_color is not None and best_color!=int(gi[r,c]):
                        result[r,c]=best_color
            
            # 全train検証
            ok2=True
            for i2,o2 in tp:
                ga2=np.array(i2);r2=ga2.copy()
                inf2=_build_influence(i2,obs.cross_sizes)
                for r in range(ga2.shape[0]):
                    for c in range(ga2.shape[1]):
                        bc=None;bs=0
                        for col,im in inf2.items():
                            if im[r,c]>bs: bs=im[r,c]; bc=col
                        if bc is not None and bc!=int(ga2[r,c]): r2[r,c]=bc
                if not grid_eq(r2.tolist(),o2): ok2=False; break
            if ok2: return result.tolist()
        break
    return None

def _try_cross_competition(tp, ti, obs):
    """同サイズのCrossが競合 → 境界線が生まれる"""
    # 同じCrossサイズの色ペアを探す
    size_groups = defaultdict(list)
    for color, size in obs.cross_sizes.items():
        if size > 0: size_groups[size].append(color)
    
    competing = [(c1,c2) for sz,colors in size_groups.items() if len(colors)>=2 
                 for i,c1 in enumerate(colors) for c2 in colors[i+1:]]
    
    if not competing: return None
    
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=obs.bg; h,w=ga.shape
        
        inf=_build_influence(inp, obs.cross_sizes)
        
        # 競合する色の影響力が等しい場所 = 境界
        for c1,c2 in competing:
            if c1 not in inf or c2 not in inf: continue
            boundary = np.abs(inf[c1] - inf[c2]) < 0.15
            
            # 境界セルで出力に新しい色が追加されるか
            boundary_fills = {}
            for r in range(h):
                for c in range(w):
                    if boundary[r,c] and ga[r,c]!=go[r,c]:
                        boundary_fills[int(ga[r,c])] = int(go[r,c])
            
            if boundary_fills:
                def apply(grid, fills, cs):
                    g=np.array(grid);hh,ww=g.shape;bg2=_bg(g)
                    inf2=_build_influence(grid,cs)
                    if c1 not in inf2 or c2 not in inf2: return None
                    bd=np.abs(inf2[c1]-inf2[c2])<0.15
                    result=g.copy()
                    for r in range(hh):
                        for cc in range(ww):
                            if bd[r,cc] and int(g[r,cc]) in fills:
                                result[r,cc]=fills[int(g[r,cc])]
                    return result.tolist()
                
                ok=True
                for i2,o2 in tp:
                    p=apply(i2,boundary_fills,obs.cross_sizes)
                    if p is None or not grid_eq(p,o2): ok=False; break
                if ok:
                    return apply(ti,boundary_fills,obs.cross_sizes)
        break
    return None

def _try_cross_boundary(tp, ti, obs):
    """Cross影響圏の端で何かが起きる"""
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=obs.bg; h,w=ga.shape
        
        inf=_build_influence(inp, obs.cross_sizes)
        
        # 変更セルが影響圏の端にあるか
        for r in range(h):
            for c in range(w):
                if ga[r,c]==go[r,c]: continue
                # このセルの各色の影響力
                for color,inf_map in inf.items():
                    if 0.05 < inf_map[r,c] < 0.3:
                        # 端にいる → この色の影響圏の端で出力色に変わる
                        pass
        break
    return None


# ══════════════════════════════════════════════════════════════
# Step 3: 一つ一つ仮説を検証する
# ══════════════════════════════════════════════════════════════

def think_one_by_one(obs, tp, ti):
    """個別の操作仮説を順番に検証"""
    from arc.cross_brain_v2 import (
        crop_by_attr, panel_compare, conditional_extract,
        reversi_variants, connect_variants, symmetry_variants, flood_variants,
        cannon_stamp, morphology_variants, diagonal_draw, minesweeper,
        gravity_variants, object_translate, rotate_transform,
        colormap_full, template_match, neighbor_abstract, evenodd_full,
        scale_variants, tile_variants, count_output, repeat_pattern,
    )
    
    # overlayに基づく優先順序（一つ一つ試す）
    if obs.out_1x1:
        order = [count_output]
    elif obs.out_smaller:
        if obs.has_separator:
            order = [conditional_extract, panel_compare, crop_by_attr]
        else:
            order = [crop_by_attr, panel_compare, conditional_extract]
    elif obs.out_larger:
        order = [scale_variants, tile_variants]
    elif obs.overlay == 'only_change':
        order = [colormap_full, evenodd_full, neighbor_abstract, template_match]
    elif obs.overlay == 'only_add':
        order = [reversi_variants, connect_variants, symmetry_variants, 
                 flood_variants, cannon_stamp, morphology_variants, 
                 diagonal_draw, minesweeper]
    elif obs.overlay == 'add_and_remove':
        order = [gravity_variants, object_translate, rotate_transform]
    else:
        order = [template_match, neighbor_abstract, colormap_full,
                 symmetry_variants, reversi_variants, morphology_variants,
                 gravity_variants, rotate_transform, flood_variants,
                 connect_variants, evenodd_full, repeat_pattern]
    
    for solver in order:
        try:
            result = solver(tp, ti)
            if result is not None:
                return result, f'仮説検証:{solver.__name__}'
        except:
            continue
    
    return None, None


# ══════════════════════════════════════════════════════════════
# Step 4: 組み合わせて試す
# ══════════════════════════════════════════════════════════════

def think_combine(obs, tp, ti):
    """これまで試したものを組み合わせる"""
    from arc.cross_brain_v2 import (
        crop_by_attr, panel_compare, conditional_extract,
        reversi_variants, connect_variants, symmetry_variants, flood_variants,
        morphology_variants, gravity_variants, rotate_transform,
        colormap_full, template_match, neighbor_abstract, evenodd_full,
        scale_variants, tile_variants,
    )
    
    # 操作を「前処理」「本処理」「後処理」に分類
    pre_ops = [colormap_full, evenodd_full]  # 色を変えてから
    main_ops = [reversi_variants, connect_variants, symmetry_variants, 
                flood_variants, morphology_variants, gravity_variants,
                template_match, neighbor_abstract, rotate_transform]
    post_ops = [crop_by_attr, panel_compare, conditional_extract,
                scale_variants, tile_variants]
    
    # pre → main
    if obs.same_size:
        for pre in pre_ops:
            intermediates = []
            ok = True
            for inp, out in tp:
                try:
                    mid = pre(tp, inp)
                    if mid is None: ok=False; break
                    intermediates.append((mid, out))
                except: ok=False; break
            if not ok: continue
            
            for main in main_ops:
                try:
                    ok2 = True
                    for mid, out in intermediates:
                        p = main(intermediates, mid)
                        if p is None or not grid_eq(p, out): ok2=False; break
                    if ok2:
                        mid_t = pre(tp, ti)
                        if mid_t:
                            result = main(intermediates, mid_t)
                            if result: return result, f'組合せ:{pre.__name__}→{main.__name__}'
                except: continue
    
    # main → post
    if obs.out_smaller:
        for main in main_ops:
            intermediates = []
            ok = True
            for inp, out in tp:
                try:
                    mid = main(tp, inp)
                    if mid is None: ok=False; break
                    intermediates.append((mid, out))
                except: ok=False; break
            if not ok: continue
            
            for post in post_ops:
                try:
                    ok2 = True
                    for mid, out in intermediates:
                        p = post(intermediates, mid)
                        if p is None or not grid_eq(p, out): ok2=False; break
                    if ok2:
                        mid_t = main(tp, ti)
                        if mid_t:
                            result = post(intermediates, mid_t)
                            if result: return result, f'組合せ:{main.__name__}→{post.__name__}'
                except: continue
    
    return None, None


# ══════════════════════════════════════════════════════════════
# Cross Think: 人間の思考プロセスを再現
# ══════════════════════════════════════════════════════════════

def cross_think_solve(train_pairs, test_input):
    """
    人間の思考プロセス:
    1. まず見る
    2. 色の順番を見る
    3. シンボル的意味を探す
    4. 一つ一つ仮説を検証
    5. 組み合わせて試す
    """
    
    # Step 0: 見る
    obs = Observation(train_pairs)
    
    # Step 1: 色の順番
    result, name = think_color_order(obs, train_pairs, test_input)
    if result is not None:
        return result, f'[Step1]{name}'
    
    # Step 2: シンボル的意味
    result, name = think_symbolic(obs, train_pairs, test_input)
    if result is not None:
        return result, f'[Step2]{name}'
    
    # Step 3: 一つ一つ仮説検証
    result, name = think_one_by_one(obs, train_pairs, test_input)
    if result is not None:
        return result, f'[Step3]{name}'
    
    # Step 4: 組み合わせ
    result, name = think_combine(obs, train_pairs, test_input)
    if result is not None:
        return result, f'[Step4]{name}'
    
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
        
        result, name = cross_think_solve(tp, ti)
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_e else ''
            solved.append((tid, name, tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tg in solved if tg == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
    print('\nStep別:')
    steps = defaultdict(int)
    for _,name,_ in solved:
        step = name.split(']')[0]+']' if '[' in name else 'other'
        steps[step] += 1
    for step, cnt in sorted(steps.items()):
        print(f'  {step}: {cnt}')
