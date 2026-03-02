"""
arc/cross_qr_deep.py — QR Deep Decoder: 構造パターンごとの専用デコーダー

構造→操作の深いマッピング:
- framed: 枠の中を塗る/crop/変換
- odd_one_out: 属性差分で抽出/変換
- templated: テンプレ→スタンプ/繰り返し
- aligned: 並び→ソート/接続
- recolored: 色対応変換
- grid_divided: パネル演算(XOR/OR/AND/overlay)
- cross_junction: 十字から展開
- marked: マーカー指示移動/塗り
"""

import numpy as np
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label
from arc.cross_qr import (detect_anchors, build_connections, StructureCode, 
                           _bg, _objs, grid_eq, Anchor)


# ══════════════════════════════════════════════════════════════
# Framed: 枠の操作（50問未解決）
# ══════════════════════════════════════════════════════════════

def decode_framed_deep(tp, ti, structure):
    """枠に対する深い操作"""
    results = []
    
    # A. 枠内部をそのまま crop
    r = _framed_crop(tp, ti)
    if r: results.append((r, 'framed:crop'))
    
    # B. 枠の内部を枠の色で flood fill
    r = _framed_flood(tp, ti)
    if r: results.append((r, 'framed:flood'))
    
    # C. 枠の色で囲まれた領域を塗る
    r = _framed_enclosed_fill(tp, ti)
    if r: results.append((r, 'framed:enclosed'))
    
    # D. 枠の形を他の場所にスタンプ
    r = _framed_stamp(tp, ti)
    if r: results.append((r, 'framed:stamp'))
    
    # E. 枠同士のoverlay
    r = _framed_overlay(tp, ti)
    if r: results.append((r, 'framed:overlay'))
    
    # F. 枠の中の色パターンを外に反映
    r = _framed_mirror_out(tp, ti)
    if r: results.append((r, 'framed:mirror_out'))
    
    return results

def _framed_crop(tp, ti):
    """枠の内部をcrop（各サイズの枠、padding 0-2）"""
    g0 = np.array(tp[0][0]); bg = _bg(g0)
    objs = _objs(g0, bg)
    
    # 枠候補: 矩形でないオブジェクト（辺のみ）
    for obj in sorted(objs, key=lambda o: -o['size']):
        if obj['is_rect']: continue
        r1,c1,r2,c2 = obj['bbox']
        if obj['bh'] < 3 or obj['bw'] < 3: continue
        
        # 辺上のセルの割合
        border = sum(1 for r,c in obj['cells'] if r==r1 or r==r2 or c==c1 or c==c2)
        if border / obj['size'] < 0.5: continue
        
        for pad in [1, 0, 2]:
            def crop_frame(grid, color=obj['color'], p=pad):
                g=np.array(grid); bg2=_bg(g); objs2=_objs(g,bg2)
                for o in sorted(objs2,key=lambda x:-x['size']):
                    if o['is_rect']: continue
                    if o['color']!=color: continue
                    r1,c1,r2,c2=o['bbox']
                    if o['bh']<3 or o['bw']<3: continue
                    inner=g[r1+p:r2+1-p,c1+p:c2+1-p]
                    if inner.size>0: return inner.tolist()
                return None
            
            ok=True
            for inp,out in tp:
                p=crop_frame(inp)
                if p is None or not grid_eq(p,out): ok=False; break
            if ok: return crop_frame(ti)
    return None

def _framed_flood(tp, ti):
    """枠の内部をflood fill"""
    for fill_mode in ['frame_color', 'inner_color', 'bg_to_frame']:
        ok=True
        for inp,out in tp:
            p=_apply_flood(inp, fill_mode)
            if p is None or not grid_eq(p,out): ok=False; break
        if ok: return _apply_flood(ti, fill_mode)
    return None

def _apply_flood(grid, mode):
    g=np.array(grid).copy(); h,w=g.shape; bg=_bg(g)
    objs=_objs(g,bg)
    frames=[o for o in objs if not o['is_rect'] and o['bh']>=3 and o['bw']>=3
            and sum(1 for r,c in o['cells'] if r==o['bbox'][0] or r==o['bbox'][2] or c==o['bbox'][1] or c==o['bbox'][3])/o['size']>0.5]
    if not frames: return None
    
    changed=False
    for frame in frames:
        r1,c1,r2,c2=frame['bbox']
        fc=frame['color']
        for r in range(r1+1,r2):
            for c in range(c1+1,c2):
                if g[r,c]==bg:
                    if mode=='frame_color': g[r,c]=fc; changed=True
                    elif mode=='bg_to_frame': g[r,c]=fc; changed=True
    
    if mode=='inner_color':
        for frame in frames:
            r1,c1,r2,c2=frame['bbox']
            inner_colors=[int(g[r,c]) for r in range(r1+1,r2) for c in range(c1+1,c2) if g[r,c]!=bg and g[r,c]!=frame['color']]
            if inner_colors:
                fill_c=Counter(inner_colors).most_common(1)[0][0]
                for r in range(r1+1,r2):
                    for c in range(c1+1,c2):
                        if g[r,c]==bg: g[r,c]=fill_c; changed=True
    
    return g.tolist() if changed else None

def _framed_enclosed_fill(tp, ti):
    """閉じた領域のflood fill（枠でない矩形含む）"""
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=_bg(ga)
        
        # 追加されたセル
        added=[(r,c,int(go[r,c])) for r in range(ga.shape[0]) for c in range(ga.shape[1]) if ga[r,c]==bg and go[r,c]!=bg]
        if not added: return None
        
        # 追加セルは何色？それは何の色？
        add_colors=Counter(c for _,_,c in added)
        
        # 周囲を全て同色で囲まれた空白領域を塗る
        def enclosed_fill(grid):
            g=np.array(grid).copy(); h,w=g.shape; bg2=_bg(g)
            # 各非BG色について、その色で完全に囲まれたBG領域を探す
            changed=False
            for color in set(int(v) for v in g.flatten()) - {bg2}:
                # BFS: 端に接続しないBG連結成分を探す
                visited=np.zeros((h,w),dtype=bool)
                mask=(g!=bg2)&(g!=color)  # この色以外の非BGも壁
                
                for sr in range(h):
                    for sc in range(w):
                        if visited[sr,sc] or g[sr,sc]!=bg2: continue
                        # BFS
                        queue=[(sr,sc)]; region=[(sr,sc)]; visited[sr,sc]=True
                        touches_edge=False; all_neighbors_same_color=True
                        
                        while queue:
                            r,c=queue.pop(0)
                            if r==0 or r==h-1 or c==0 or c==w-1: touches_edge=True
                            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr,nc=r+dr,c+dc
                                if 0<=nr<h and 0<=nc<w:
                                    if not visited[nr,nc] and g[nr,nc]==bg2:
                                        visited[nr,nc]=True; queue.append((nr,nc)); region.append((nr,nc))
                                    elif g[nr,nc]!=bg2 and g[nr,nc]!=color:
                                        all_neighbors_same_color=False
                                else:
                                    touches_edge=True
                        
                        if not touches_edge and len(region)>0:
                            for r,c in region: g[r,c]=color; changed=True
            return g.tolist() if changed else None
        
        ok=True
        for i2,o2 in tp:
            p=enclosed_fill(i2)
            if p is None or not grid_eq(p,o2): ok=False; break
        if ok: return enclosed_fill(ti)
        break
    return None

def _framed_stamp(tp, ti):
    """枠の形を他のオブジェクトに適用"""
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=_bg(ga)
        objs=_objs(ga,bg)
        
        # 最大オブジェクト=テンプレ、小オブジェクト=スタンプ先
        if len(objs)<2: return None
        objs.sort(key=lambda o:-o['size'])
        template=objs[0]; targets=objs[1:]
        
        # テンプレの形状(相対座標)
        tr1,tc1,_,_=template['bbox']
        tshape={(r-tr1,c-tc1):int(ga[r,c]) for r,c in template['cells']}
        th,tw=template['bh'],template['bw']
        
        # 各ターゲット位置にテンプレをスタンプして出力と一致するか
        # ターゲットの色でリカラー？
        break
    return None

def _framed_overlay(tp, ti):
    """複数の枠のoverlay"""
    return None

def _framed_mirror_out(tp, ti):
    """枠内パターンを外に反映"""
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=_bg(ga); h,w=ga.shape
        objs=_objs(ga,bg)
        
        # 矩形オブジェクト（枠含む）を見つける
        rects=[o for o in objs if o['is_rect'] and o['bh']>=2 and o['bw']>=2]
        if not rects: return None
        
        # 最大矩形のパターンを繰り返し（タイリング）
        rect=max(rects,key=lambda o:o['size'])
        r1,c1,r2,c2=rect['bbox']
        tile=ga[r1:r2+1,c1:c2+1]
        th,tw=tile.shape
        
        # タイリングで出力を作る
        pred=np.full((h,w),bg,dtype=int)
        for r in range(h):
            for c in range(w):
                tr,tc=r%th,c%tw
                pred[r,c]=tile[tr,tc]
        
        if grid_eq(pred.tolist(),out):
            gi=np.array(ti); h2,w2=gi.shape
            objs2=_objs(gi,_bg(gi))
            rects2=[o for o in objs2 if o['is_rect'] and o['bh']>=2 and o['bw']>=2]
            if rects2:
                rect2=max(rects2,key=lambda o:o['size'])
                r1,c1,r2,c2=rect2['bbox']
                tile2=gi[r1:r2+1,c1:c2+1]
                th2,tw2=tile2.shape
                result=np.full((h2,w2),_bg(gi),dtype=int)
                for r in range(h2):
                    for c in range(w2):
                        result[r,c]=tile2[r%th2,c%tw2]
                return result.tolist()
        break
    return None


# ══════════════════════════════════════════════════════════════
# Odd One Out: 仲間はずれ操作（61問未解決）
# ══════════════════════════════════════════════════════════════

def decode_odd_deep(tp, ti, structure):
    """仲間はずれの深い操作"""
    results = []
    
    # A. 仲間はずれを抽出
    r = _odd_extract(tp, ti)
    if r: results.append((r, 'odd:extract'))
    
    # B. 仲間はずれを削除
    r = _odd_remove(tp, ti)
    if r: results.append((r, 'odd:remove'))
    
    # C. 仲間はずれの属性で変換
    r = _odd_transform(tp, ti)
    if r: results.append((r, 'odd:transform'))
    
    # D. 仲間はずれの色を他に適用
    r = _odd_recolor(tp, ti)
    if r: results.append((r, 'odd:recolor'))
    
    return results

def _odd_extract(tp, ti):
    """仲間はずれを抽出（形/色/サイズで判定）"""
    bg0 = _bg(np.array(tp[0][0]))
    
    for attr_fn, attr_name in [
        (lambda o: o['shape'], 'shape'),
        (lambda o: o['color'], 'color'),
        (lambda o: o['size'], 'size'),
        (lambda o: o['bh']*1000+o['bw'], 'bbox'),
        (lambda o: o['is_rect'], 'rect'),
        (lambda o: o['n_colors'], 'ncolors'),
    ]:
        ok=True
        for inp,out in tp:
            ga=np.array(inp); bg=_bg(ga); objs=_objs(ga,bg)
            if len(objs)<2: ok=False; break
            
            cc=Counter(attr_fn(o) for o in objs)
            
            # 最も少ない属性値を持つオブジェクト
            if len(cc)<2: ok=False; break
            min_count=min(cc.values())
            oddballs=[o for o in objs if cc[attr_fn(o)]==min_count]
            
            if not oddballs: ok=False; break
            
            # oddball の bbox を crop
            if len(oddballs)==1:
                o=oddballs[0]; r1,c1,r2,c2=o['bbox']
                crop=ga[r1:r2+1,c1:c2+1]
                if not grid_eq(crop.tolist(),out): ok=False; break
            else:
                ok=False; break
        
        if ok:
            gi=np.array(ti); bg=_bg(gi); objs=_objs(gi,bg)
            if len(objs)<2: continue
            cc=Counter(attr_fn(o) for o in objs)
            if len(cc)<2: continue
            min_count=min(cc.values())
            oddballs=[o for o in objs if cc[attr_fn(o)]==min_count]
            if len(oddballs)==1:
                o=oddballs[0]; r1,c1,r2,c2=o['bbox']
                return gi[r1:r2+1,c1:c2+1].tolist()
    return None

def _odd_remove(tp, ti):
    """仲間はずれを削除"""
    for attr_fn in [
        lambda o: o['shape'], lambda o: o['color'], lambda o: o['size'],
    ]:
        ok=True
        for inp,out in tp:
            ga=np.array(inp).copy(); bg=_bg(ga); objs=_objs(ga,bg)
            if len(objs)<2: ok=False; break
            cc=Counter(attr_fn(o) for o in objs)
            if len(cc)<2: ok=False; break
            min_count=min(cc.values())
            oddballs=[o for o in objs if cc[attr_fn(o)]==min_count]
            for o in oddballs:
                for r,c in o['cells']: ga[r,c]=bg
            if not grid_eq(ga.tolist(),out): ok=False; break
        if ok:
            gi=np.array(ti).copy(); bg=_bg(gi); objs=_objs(gi,bg)
            cc=Counter(attr_fn(o) for o in objs)
            if len(cc)<2: continue
            min_count=min(cc.values())
            oddballs=[o for o in objs if cc[attr_fn(o)]==min_count]
            for o in oddballs:
                for r,c in o['cells']: gi[r,c]=bg
            return gi.tolist()
    return None

def _odd_transform(tp, ti):
    """仲間はずれの属性を使って変換"""
    # 仲間はずれの色を全体に適用、等
    return None

def _odd_recolor(tp, ti):
    """仲間はずれの色で他を塗る"""
    return None


# ══════════════════════════════════════════════════════════════
# Aligned: 整列操作（44問未解決）
# ══════════════════════════════════════════════════════════════

def decode_aligned_deep(tp, ti, structure):
    """整列したオブジェクトの深い操作"""
    results = []
    
    # A. 整列方向にソート
    r = _aligned_sort(tp, ti)
    if r: results.append((r, 'aligned:sort'))
    
    # B. 整列方向に接続（線を引く）
    r = _aligned_connect(tp, ti)
    if r: results.append((r, 'aligned:connect'))
    
    # C. 整列方向に移動
    r = _aligned_move(tp, ti)
    if r: results.append((r, 'aligned:move'))
    
    return results

def _aligned_sort(tp, ti):
    """オブジェクトを属性でソート"""
    return None

def _aligned_connect(tp, ti):
    """同色オブジェクト間を線で接続"""
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=_bg(ga)
        
        # 出力で追加されたセルが直線上にあるか
        added=[(r,c,int(go[r,c])) for r in range(ga.shape[0]) for c in range(ga.shape[1]) 
               if ga[r,c]==bg and go[r,c]!=bg]
        if not added: return None
        
        # 追加色ごとにグループ化
        by_color=defaultdict(list)
        for r,c,col in added: by_color[col].append((r,c))
        
        # 各追加色について: 同色の既存オブジェクト間を結んでいるか
        break
    
    # 簡易版: 同色の1x1セル間をH/V線で結ぶ
    def connect(grid):
        g=np.array(grid).copy(); h,w=g.shape; bg2=_bg(g)
        changed=False
        for color in set(int(v) for v in g.flatten()) - {bg2}:
            pts=[(r,c) for r in range(h) for c in range(w) if g[r,c]==color]
            if len(pts)<2: continue
            for i,(r1,c1) in enumerate(pts):
                for r2,c2 in pts[i+1:]:
                    if r1==r2:  # 水平
                        for c in range(min(c1,c2)+1,max(c1,c2)):
                            if g[r1,c]==bg2: g[r1,c]=color; changed=True
                    elif c1==c2:  # 垂直
                        for r in range(min(r1,r2)+1,max(r1,r2)):
                            if g[r,c1]==bg2: g[r,c1]=color; changed=True
        return g.tolist() if changed else None
    
    ok=True
    for inp,out in tp:
        p=connect(inp)
        if p is None or not grid_eq(p,out): ok=False; break
    if ok: return connect(ti)
    return None

def _aligned_move(tp, ti):
    """オブジェクトを整列方向に移動"""
    return None


# ══════════════════════════════════════════════════════════════
# Cross Junction: 十字から展開（40問未解決）
# ══════════════════════════════════════════════════════════════

def decode_cross_junction_deep(tp, ti, structure):
    """十字形を使った操作"""
    results = []
    
    # A. 十字の4方向にflood
    r = _cross_flood(tp, ti)
    if r: results.append((r, 'cross:flood'))
    
    # B. 十字の色で十字線を引く
    r = _cross_lines(tp, ti)
    if r: results.append((r, 'cross:lines'))
    
    return results

def _cross_flood(tp, ti):
    """十字の中心から4方向にflood"""
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=_bg(ga); h,w=ga.shape
        
        # 十字形オブジェクトを見つける
        objs=_objs(ga,bg)
        for obj in objs:
            cr,cc=int(round(obj['center'][0])),int(round(obj['center'][1]))
            cells_set=set(obj['cells'])
            # 4方向にセルが伸びてるか
            dirs={} # direction -> length
            for d,name in [((0,1),'right'),((-1,0),'up'),((0,-1),'left'),((1,0),'down')]:
                length=0
                while (cr+d[0]*(length+1),cc+d[1]*(length+1)) in cells_set: length+=1
                dirs[name]=length
            if sum(v>0 for v in dirs.values())>=2:
                # 十字の各腕の先からグリッド端までflood
                def cross_flood_apply(grid):
                    g=np.array(grid).copy(); h2,w2=g.shape; bg2=_bg(g)
                    objs2=_objs(g,bg2)
                    for o in objs2:
                        cr2,cc2=int(round(o['center'][0])),int(round(o['center'][1]))
                        cs=set(o['cells'])
                        for dr,dc in [(0,1),(-1,0),(0,-1),(1,0)]:
                            l=0
                            while (cr2+dr*(l+1),cc2+dc*(l+1)) in cs: l+=1
                            if l>0:
                                # 腕の先からflood
                                r,c=cr2+dr*(l+1),cc2+dc*(l+1)
                                while 0<=r<h2 and 0<=c<w2 and g[r,c]==bg2:
                                    g[r,c]=o['color']; r+=dr; c+=dc
                    return g.tolist()
                
                ok=True
                for i2,o2 in tp:
                    p=cross_flood_apply(i2)
                    if not grid_eq(p,o2): ok=False; break
                if ok: return cross_flood_apply(ti)
        break
    return None

def _cross_lines(tp, ti):
    """十字形セルの位置からH/V線を端まで引く"""
    def draw_lines(grid):
        g=np.array(grid).copy(); h,w=g.shape; bg2=_bg(g)
        changed=False
        for color in set(int(v) for v in g.flatten()) - {bg2}:
            pts=[(r,c) for r in range(h) for c in range(w) if g[r,c]==color]
            for r,c in pts:
                # 水平と垂直に拡張
                for dc in [-1,1]:
                    nc=c+dc
                    while 0<=nc<w and g[r,nc]==bg2:
                        g[r,nc]=color; nc+=dc; changed=True
                for dr in [-1,1]:
                    nr=r+dr
                    while 0<=nr<h and g[nr,c]==bg2:
                        g[nr,c]=color; nr+=dr; changed=True
        return g.tolist() if changed else None
    
    ok=True
    for inp,out in tp:
        p=draw_lines(inp)
        if p is None or not grid_eq(p,out): ok=False; break
    if ok: return draw_lines(ti)
    return None


# ══════════════════════════════════════════════════════════════
# Templated: テンプレート操作（35問未解決）
# ══════════════════════════════════════════════════════════════

def decode_templated_deep(tp, ti, structure):
    """テンプレートの深い操作"""
    results = []
    
    # A. テンプレの差分で変換
    r = _template_diff(tp, ti)
    if r: results.append((r, 'template:diff'))
    
    # B. テンプレの色をカウント
    r = _template_count(tp, ti)
    if r: results.append((r, 'template:count'))
    
    return results

def _template_diff(tp, ti):
    """同じ形のオブジェクト間の差分を見つける"""
    for inp,out in tp:
        ga=np.array(inp); bg=_bg(ga); objs=_objs(ga,bg)
        shape_groups=defaultdict(list)
        for o in objs: shape_groups[o['shape']].append(o)
        
        # 2つ以上のグループ
        for shape,group in shape_groups.items():
            if len(group)<2: continue
            
            # グループ内の差分（色の違い）
            base=group[0]
            diffs=[]
            for other in group[1:]:
                # 相対座標での色差分
                base_map={pos:int(ga[r,c]) for (r,c),pos in zip(base['cells'],[(r-base['bbox'][0],c-base['bbox'][1]) for r,c in base['cells']])}
                other_map={pos:int(ga[r,c]) for (r,c),pos in zip(other['cells'],[(r-other['bbox'][0],c-other['bbox'][1]) for r,c in other['cells']])}
                
                diff_cells=[(pos,base_map.get(pos),other_map.get(pos)) for pos in set(base_map)|set(other_map) 
                           if base_map.get(pos)!=other_map.get(pos)]
                diffs.append(diff_cells)
        break
    return None

def _template_count(tp, ti):
    """テンプレートの数を数えて出力"""
    return None


# ══════════════════════════════════════════════════════════════
# Grid Divided: パネル演算（7問未解決）
# ══════════════════════════════════════════════════════════════

def decode_grid_divided_deep(tp, ti, structure):
    """パネル間の演算"""
    results = []
    
    r = _panel_ops(tp, ti)
    if r: results.append((r, 'grid:panel_ops'))
    
    return results

def _panel_ops(tp, ti):
    """セパレータで分割してパネル間演算"""
    def find_panels(grid):
        g=np.array(grid); h,w=g.shape; bg=_bg(g)
        h_seps=[]; v_seps=[]
        for r in range(h):
            vals=set(int(v) for v in g[r])
            if len(vals)==1 and vals.pop()!=bg: h_seps.append(r)
        for c in range(w):
            vals=set(int(v) for v in g[:,c])
            if len(vals)==1 and vals.pop()!=bg: v_seps.append(c)
        
        rows=[-1]+sorted(h_seps)+[h]; cols=[-1]+sorted(v_seps)+[w]
        panels=[]
        for i in range(len(rows)-1):
            for j in range(len(cols)-1):
                r1,r2=rows[i]+1,rows[i+1]; c1,c2=cols[j]+1,cols[j+1]
                if r2>r1 and c2>c1: panels.append(g[r1:r2,c1:c2])
        return panels
    
    panels0=find_panels(tp[0][0])
    if len(panels0)<2: return None
    shapes=set(p.shape for p in panels0)
    if len(shapes)!=1: return None
    ph,pw=panels0[0].shape
    go=np.array(tp[0][1])
    
    for op in ['xor','or','and','first_nonbg','last_nonbg','majority']:
        bg=_bg(np.array(tp[0][0]))
        def apply_op(panels, bg2, operation=op):
            result=np.full((ph,pw),bg2,dtype=int)
            for r in range(ph):
                for c in range(pw):
                    vals=[int(p[r,c]) for p in panels]
                    nb=[v for v in vals if v!=bg2]
                    if operation=='xor': result[r,c]=nb[0] if len(nb)==1 else bg2
                    elif operation=='or': result[r,c]=nb[0] if nb else bg2
                    elif operation=='and': result[r,c]=nb[0] if len(nb)==len(vals) and len(set(nb))==1 else bg2
                    elif operation=='first_nonbg': result[r,c]=nb[0] if nb else bg2
                    elif operation=='last_nonbg': result[r,c]=nb[-1] if nb else bg2
                    elif operation=='majority': result[r,c]=Counter(nb).most_common(1)[0][0] if nb else bg2
            return result
        
        pred=apply_op(panels0,bg)
        if go.shape==(ph,pw) and grid_eq(pred.tolist(),go.tolist()):
            ok=True
            for inp,out in tp:
                ps=find_panels(inp); bg2=_bg(np.array(inp))
                if not ps or set(p.shape for p in ps)!=shapes: ok=False; break
                p=apply_op(ps,bg2)
                if not grid_eq(p.tolist(),out): ok=False; break
            if ok:
                ps_t=find_panels(ti); bg_t=_bg(np.array(ti))
                if ps_t: return apply_op(ps_t,bg_t).tolist()
    return None


# ══════════════════════════════════════════════════════════════
# Marked: マーカー指示操作（11問未解決）
# ══════════════════════════════════════════════════════════════

def decode_marked_deep(tp, ti, structure):
    """マーカーの深い操作"""
    results = []
    
    # マーカーの色がオブジェクトのどの位置にあるかで方向指示
    r = _marker_gravity(tp, ti)
    if r: results.append((r, 'marked:gravity'))
    
    return results

def _marker_gravity(tp, ti):
    """マーカーの位置方向にオブジェクトを移動"""
    for inp,out in tp:
        ga,go=np.array(inp),np.array(out)
        if ga.shape!=go.shape: return None
        bg=_bg(ga); h,w=ga.shape
        objs=_objs(ga,bg)
        markers=[o for o in objs if o['size']<=2]
        bigs=[o for o in objs if o['size']>2]
        
        if not markers or not bigs: return None
        
        # マーカーが大きいオブジェクトのどの方向にあるか→その方向に移動
        for big in bigs:
            br,bc=big['center']
            for marker in markers:
                mr,mc=marker['center']
                # 方向
                dr=1 if mr>br else (-1 if mr<br else 0)
                dc=1 if mc>bc else (-1 if mc<bc else 0)
                if dr==0 and dc==0: continue
                
                # その方向にbigを端まで移動
                # ...
        break
    return None


# ══════════════════════════════════════════════════════════════
# QR Deep Master
# ══════════════════════════════════════════════════════════════

def qr_deep_solve(train_pairs, test_input):
    """QR Deep Decoder: 構造パターンごとの専用デコーダー"""
    
    # 構造読み取り
    anchors = detect_anchors(train_pairs[0][0])
    conns = build_connections(anchors)
    structure = StructureCode(anchors, conns, train_pairs[0][0])
    
    all_results = []
    
    # 構造パターンごとのデコーダー（優先度順）
    if 'grid_divided' in structure.patterns:
        all_results.extend(decode_grid_divided_deep(train_pairs, test_input, structure))
    
    if 'framed' in structure.patterns:
        all_results.extend(decode_framed_deep(train_pairs, test_input, structure))
    
    if 'odd_one_out' in structure.patterns:
        all_results.extend(decode_odd_deep(train_pairs, test_input, structure))
    
    if 'aligned' in structure.patterns:
        all_results.extend(decode_aligned_deep(train_pairs, test_input, structure))
    
    if 'cross_junction' in structure.patterns:
        all_results.extend(decode_cross_junction_deep(train_pairs, test_input, structure))
    
    if 'templated' in structure.patterns:
        all_results.extend(decode_templated_deep(train_pairs, test_input, structure))
    
    if 'marked' in structure.patterns:
        all_results.extend(decode_marked_deep(train_pairs, test_input, structure))
    
    # train検証
    for result, name in all_results:
        ok = True
        for inp, out in train_pairs:
            # 同じデコーダーで解く
            a = detect_anchors(inp)
            c = build_connections(a)
            s = StructureCode(a, c, inp)
            
            candidates = []
            if 'grid_divided' in s.patterns:
                candidates.extend(decode_grid_divided_deep(train_pairs, inp, s))
            if 'framed' in s.patterns:
                candidates.extend(decode_framed_deep(train_pairs, inp, s))
            if 'odd_one_out' in s.patterns:
                candidates.extend(decode_odd_deep(train_pairs, inp, s))
            if 'aligned' in s.patterns:
                candidates.extend(decode_aligned_deep(train_pairs, inp, s))
            if 'cross_junction' in s.patterns:
                candidates.extend(decode_cross_junction_deep(train_pairs, inp, s))
            if 'templated' in s.patterns:
                candidates.extend(decode_templated_deep(train_pairs, inp, s))
            if 'marked' in s.patterns:
                candidates.extend(decode_marked_deep(train_pairs, inp, s))
            
            found = any(grid_eq(r, out) for r, _ in candidates)
            if not found:
                ok = False; break
        
        if ok:
            return result, name
    
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
        
        result, name = qr_deep_solve(tp, ti)
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_e else ''
            solved.append((tid, name, tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tg in solved if tg == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
    if new:
        print(f'NEW: {new}')
