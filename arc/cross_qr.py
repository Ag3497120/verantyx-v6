"""
arc/cross_qr.py — QRコード×Cross構造: 構造を読んでから解く

Step 1: アンカー検出（ランドマーク）
Step 2: アンカー間のCross接続
Step 3: 構造コード読み取り（問題の「型」）
Step 4: データ領域デコード
Step 5: エラー訂正（補完）
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
            'is_rect':len(cells)==(r2-r1+1)*(c2-c1+1),
            'center':((r1+r2)/2,(c1+c2)/2),
            'n_colors':len(set(colors)),
            'id':i})
    return objs

def grid_eq(a,b):
    a,b=np.array(a),np.array(b)
    return a.shape==b.shape and np.array_equal(a,b)


# ══════════════════════════════════════════════════════════════
# Step 1: アンカー検出
# ══════════════════════════════════════════════════════════════

class Anchor:
    """グリッド内のランドマーク"""
    def __init__(self, obj, anchor_type, strength):
        self.obj = obj
        self.type = anchor_type  # 'corner','cross','frame','marker','separator','unique','repeated'
        self.strength = strength  # 0-1, ランドマークとしての確信度
        self.center = obj['center']
        self.color = obj['color']
        self.size = obj['size']
        self.bbox = obj['bbox']
        self.shape = obj['shape']

def detect_anchors(grid):
    """グリッドからアンカー（ランドマーク）を検出"""
    g = np.array(grid)
    h, w = g.shape
    bg = _bg(g)
    objs = _objs(g, bg)
    
    if not objs:
        return []
    
    anchors = []
    
    # 形状の出現頻度
    shape_counts = Counter(o['shape'] for o in objs)
    color_counts = Counter(o['color'] for o in objs)
    size_counts = Counter(o['size'] for o in objs)
    
    for obj in objs:
        scores = []
        
        # 1. 角検出: オブジェクトがグリッドの角にある
        cr, cc = obj['center']
        corner_dist = min(cr, h-1-cr, cc, w-1-cc)
        if corner_dist < min(h, w) * 0.15:
            scores.append(('corner', 0.7))
        
        # 2. 十字形検出: 十字っぽい形
        cells_set = set(obj['cells'])
        cr_int, cc_int = int(round(cr)), int(round(cc))
        cross_cells = {(cr_int+d, cc_int) for d in range(-2,3)} | \
                      {(cr_int, cc_int+d) for d in range(-2,3)}
        cross_overlap = len(cells_set & cross_cells) / max(len(cells_set), 1)
        if cross_overlap > 0.6 and obj['size'] >= 3:
            scores.append(('cross', 0.8))
        
        # 3. 枠検出: 矩形の辺だけ
        r1, c1, r2, c2 = obj['bbox']
        bh, bw = obj['bh'], obj['bw']
        if bh >= 3 and bw >= 3 and not obj['is_rect']:
            # 辺上のセルの割合
            border_cells = {(r,c) for r,c in obj['cells'] 
                          if r==r1 or r==r2 or c==c1 or c==c2}
            if len(border_cells) / obj['size'] > 0.7:
                scores.append(('frame', 0.85))
        
        # 4. マーカー: 1-2セルの小さなオブジェクト
        if obj['size'] <= 2:
            scores.append(('marker', 0.6))
        
        # 5. セパレータ: 1行/列を占める
        if bh == 1 and bw >= w * 0.8:
            scores.append(('separator', 0.9))
        if bw == 1 and bh >= h * 0.8:
            scores.append(('separator', 0.9))
        
        # 6. ユニーク: 他にない形
        if shape_counts[obj['shape']] == 1:
            scores.append(('unique', 0.5 + 0.3 * min(obj['size'] / 10, 1.0)))
        
        # 7. 繰り返し: 同じ形が複数ある → テンプレート
        if shape_counts[obj['shape']] >= 2:
            scores.append(('repeated', 0.4 + 0.1 * shape_counts[obj['shape']]))
        
        # 8. 最大オブジェクト
        if obj['size'] == max(o['size'] for o in objs):
            scores.append(('largest', 0.7))
        
        # 9. 正方形
        if obj['is_rect'] and obj['bh'] == obj['bw'] and obj['size'] >= 4:
            scores.append(('square', 0.6))
        
        # 10. 対称オブジェクト
        shape_list = list(obj['shape'])
        max_r = max(r for r,c in shape_list)
        max_c = max(c for r,c in shape_list)
        lr_mirror = frozenset((r, max_c-c) for r,c in shape_list)
        ud_mirror = frozenset((max_r-r, c) for r,c in shape_list)
        if obj['shape'] == lr_mirror:
            scores.append(('symmetric_lr', 0.5))
        if obj['shape'] == ud_mirror:
            scores.append(('symmetric_ud', 0.5))
        
        if scores:
            best_type, best_score = max(scores, key=lambda x: x[1])
            anchors.append(Anchor(obj, best_type, best_score))
    
    # 強度順にソート
    anchors.sort(key=lambda a: -a.strength)
    
    return anchors


# ══════════════════════════════════════════════════════════════
# Step 2: Cross接続構築
# ══════════════════════════════════════════════════════════════

class CrossConnection:
    """2つのアンカー間のCross接続"""
    def __init__(self, a1, a2):
        self.a1 = a1
        self.a2 = a2
        self.distance = ((a1.center[0]-a2.center[0])**2 + (a1.center[1]-a2.center[1])**2)**0.5
        
        # 方向
        dr = a2.center[0] - a1.center[0]
        dc = a2.center[1] - a1.center[1]
        if abs(dr) < 0.5: self.direction = 'horizontal'
        elif abs(dc) < 0.5: self.direction = 'vertical'
        elif abs(abs(dr)-abs(dc)) < 1: self.direction = 'diagonal'
        else: self.direction = 'oblique'
        
        # 色関係
        self.same_color = a1.color == a2.color
        self.same_shape = a1.shape == a2.shape
        self.same_size = a1.size == a2.size
        
        # 接続の種類
        self.connection_type = self._classify()
    
    def _classify(self):
        if self.same_shape and self.same_color:
            return 'clone'  # 完全コピー
        elif self.same_shape and not self.same_color:
            return 'recolor'  # 形は同じ色違い
        elif self.same_color and not self.same_shape:
            return 'reshape'  # 色は同じ形違い
        elif self.same_size:
            return 'sibling'  # サイズ同じ
        else:
            return 'related'  # その他

def build_connections(anchors):
    """アンカー間のCross接続を構築"""
    connections = []
    for i, a1 in enumerate(anchors):
        for a2 in anchors[i+1:]:
            conn = CrossConnection(a1, a2)
            connections.append(conn)
    
    # 距離順にソート（近い接続が重要）
    connections.sort(key=lambda c: c.distance)
    
    return connections


# ══════════════════════════════════════════════════════════════
# Step 3: 構造コード読み取り
# ══════════════════════════════════════════════════════════════

class StructureCode:
    """問題の構造コード"""
    def __init__(self, anchors, connections, grid):
        g = np.array(grid)
        self.h, self.w = g.shape
        self.bg = _bg(g)
        self.anchors = anchors
        self.connections = connections
        
        # 構造パターンを分類
        self.patterns = self._detect_patterns()
    
    def _detect_patterns(self):
        patterns = set()
        
        # アンカータイプの分布
        types = Counter(a.type for a in self.anchors)
        
        # セパレータがある → グリッド分割型
        if 'separator' in types:
            patterns.add('grid_divided')
        
        # 枠がある → 枠の中身型
        if 'frame' in types:
            patterns.add('framed')
        
        # マーカーがある → 指標型
        if 'marker' in types:
            patterns.add('marked')
        
        # repeatedが多い → テンプレート型
        if types.get('repeated', 0) >= 2:
            patterns.add('templated')
        
        # uniqueがある → 仲間はずれ型
        if 'unique' in types:
            patterns.add('odd_one_out')
        
        # 角にアンカーがある → QR型（位置検出）
        if types.get('corner', 0) >= 2:
            patterns.add('qr_positioned')
        
        # 十字がある → 交差点型
        if 'cross' in types:
            patterns.add('cross_junction')
        
        # クローン接続が多い → コピー/タイル型
        clone_conns = [c for c in self.connections if c.connection_type == 'clone']
        if len(clone_conns) >= 2:
            patterns.add('tiled')
        
        # recolor接続 → 色置換型
        recolor_conns = [c for c in self.connections if c.connection_type == 'recolor']
        if recolor_conns:
            patterns.add('recolored')
        
        # 整列した接続 → 並び型
        h_conns = [c for c in self.connections if c.direction == 'horizontal']
        v_conns = [c for c in self.connections if c.direction == 'vertical']
        if len(h_conns) >= 2 or len(v_conns) >= 2:
            patterns.add('aligned')
        
        # 対称アンカー → 対称型
        sym_anchors = [a for a in self.anchors if 'symmetric' in a.type]
        if sym_anchors:
            patterns.add('symmetric')
        
        return patterns


# ══════════════════════════════════════════════════════════════
# Step 4: データ領域デコード（構造→操作マッピング）
# ══════════════════════════════════════════════════════════════

def decode_and_solve(structure_in, structure_out, train_pairs, test_input):
    """構造コードに基づいてデータを読み取り変換"""
    
    # 入出力の構造変化を分析
    in_patterns = structure_in.patterns
    out_patterns = structure_out.patterns if structure_out else set()
    
    results = []
    
    # grid_divided → パネル操作
    if 'grid_divided' in in_patterns:
        r = _decode_grid_divided(train_pairs, test_input, structure_in)
        if r: results.append((r, 'qr:grid_divided'))
    
    # framed → 枠の中身操作
    if 'framed' in in_patterns:
        r = _decode_framed(train_pairs, test_input, structure_in)
        if r: results.append((r, 'qr:framed'))
    
    # marked → マーカー指示操作
    if 'marked' in in_patterns:
        r = _decode_marked(train_pairs, test_input, structure_in)
        if r: results.append((r, 'qr:marked'))
    
    # templated → テンプレートスタンプ
    if 'templated' in in_patterns:
        r = _decode_templated(train_pairs, test_input, structure_in)
        if r: results.append((r, 'qr:templated'))
    
    # odd_one_out → 仲間はずれ抽出
    if 'odd_one_out' in in_patterns:
        r = _decode_odd_one(train_pairs, test_input, structure_in)
        if r: results.append((r, 'qr:odd_one'))
    
    # recolored → 色の対応関係
    if 'recolored' in in_patterns:
        r = _decode_recolored(train_pairs, test_input, structure_in)
        if r: results.append((r, 'qr:recolored'))
    
    # aligned → 整列操作
    if 'aligned' in in_patterns:
        r = _decode_aligned(train_pairs, test_input, structure_in)
        if r: results.append((r, 'qr:aligned'))
    
    # tiled → タイル/繰り返し
    if 'tiled' in in_patterns:
        r = _decode_tiled(train_pairs, test_input, structure_in)
        if r: results.append((r, 'qr:tiled'))
    
    # symmetric → 対称補完
    if 'symmetric' in in_patterns:
        r = _decode_symmetric(train_pairs, test_input, structure_in)
        if r: results.append((r, 'qr:symmetric'))
    
    return results


def _decode_grid_divided(tp, ti, structure):
    """セパレータで分割されたグリッドの操作"""
    from arc.cross_brain_v2 import conditional_extract, panel_compare
    
    # まずconditional_extractを試す
    try:
        r = conditional_extract(tp, ti)
        if r: return r
    except: pass
    
    # panel_compare
    try:
        r = panel_compare(tp, ti)
        if r: return r
    except: pass
    
    # セパレータで分割→各パネルを個別分析
    g = np.array(ti)
    h, w = g.shape
    bg = structure.bg
    
    sep_anchors = [a for a in structure.anchors if a.type == 'separator']
    if not sep_anchors: return None
    
    # セパレータの位置
    h_seps = []; v_seps = []
    for a in sep_anchors:
        r1,c1,r2,c2 = a.bbox
        if r2-r1 == 0:  # 水平
            h_seps.append(r1)
        elif c2-c1 == 0:  # 垂直
            v_seps.append(c1)
    
    if not h_seps and not v_seps: return None
    
    # パネル分割
    rows = [-1] + sorted(h_seps) + [h]
    cols = [-1] + sorted(v_seps) + [w]
    panels = []
    for i in range(len(rows)-1):
        for j in range(len(cols)-1):
            r1, r2 = rows[i]+1, rows[i+1]
            c1, c2 = cols[j]+1, cols[j+1]
            if r2>r1 and c2>c1:
                panels.append(g[r1:r2, c1:c2])
    
    if len(panels) < 2: return None
    
    # パネル間の関係で操作を推定
    # 全パネル同サイズ？
    shapes = set(p.shape for p in panels)
    if len(shapes) == 1:
        ph, pw = panels[0].shape
        # XOR, OR, AND等を試す
        for op in ['xor', 'or', 'and', 'diff']:
            result = np.full((ph, pw), bg, dtype=int)
            for r in range(ph):
                for c in range(pw):
                    vals = [int(p[r,c]) for p in panels]
                    nb = [v for v in vals if v != bg]
                    if op == 'xor':
                        result[r,c] = nb[0] if len(nb)==1 else bg
                    elif op == 'or':
                        result[r,c] = nb[0] if nb else bg
                    elif op == 'and':
                        result[r,c] = nb[0] if len(nb)==len(vals) and len(set(nb))==1 else bg
                    elif op == 'diff':
                        if len(set(nb))>1: result[r,c]=max(set(nb),key=nb.count)
                        elif len(nb)==1: result[r,c]=nb[0]
            
            # train検証
            ok = True
            for inp, out in tp:
                ga = np.array(inp); bg2 = _bg(ga)
                hs2 = []; vs2 = []
                for rr in range(ga.shape[0]):
                    vals2 = set(int(v) for v in ga[rr])
                    if len(vals2)==1 and vals2.pop()!=bg2: hs2.append(rr)
                for cc in range(ga.shape[1]):
                    vals2 = set(int(v) for v in ga[:,cc])
                    if len(vals2)==1 and vals2.pop()!=bg2: vs2.append(cc)
                rs2 = [-1]+sorted(hs2)+[ga.shape[0]]
                cs2 = [-1]+sorted(vs2)+[ga.shape[1]]
                ps2 = []
                for ii in range(len(rs2)-1):
                    for jj in range(len(cs2)-1):
                        rr1,rr2=rs2[ii]+1,rs2[ii+1]; cc1,cc2=cs2[jj]+1,cs2[jj+1]
                        if rr2>rr1 and cc2>cc1: ps2.append(ga[rr1:rr2,cc1:cc2])
                if not ps2 or set(p.shape for p in ps2)!=shapes:
                    ok=False; break
                pred = np.full((ph,pw),bg2,dtype=int)
                for r in range(ph):
                    for c in range(pw):
                        vals = [int(p[r,c]) for p in ps2]
                        nb = [v for v in vals if v!=bg2]
                        if op=='xor': pred[r,c]=nb[0] if len(nb)==1 else bg2
                        elif op=='or': pred[r,c]=nb[0] if nb else bg2
                        elif op=='and': pred[r,c]=nb[0] if len(nb)==len(vals) and len(set(nb))==1 else bg2
                        elif op=='diff':
                            if len(set(nb))>1: pred[r,c]=max(set(nb),key=nb.count)
                            elif len(nb)==1: pred[r,c]=nb[0]
                if not grid_eq(pred.tolist(),out): ok=False; break
            if ok: return result.tolist()
    
    return None


def _decode_framed(tp, ti, structure):
    """枠の中身を操作"""
    frame_anchors = [a for a in structure.anchors if a.type=='frame' or a.type=='largest']
    if not frame_anchors: return None
    
    # 最大枠の内部crop
    for margin in [1, 2]:
        def mk(m):
            def fn(grid):
                g=np.array(grid);bg2=_bg(g);objs=_objs(g,bg2)
                if not objs: return None
                o=max(objs,key=lambda x:x['size']); r1,c1,r2,c2=o['bbox']
                inner=g[r1+m:r2+1-m,c1+m:c2+1-m]
                return inner.tolist() if inner.size>0 else None
            return fn
        ok=True
        for inp,out in tp:
            p=mk(margin)(inp)
            if p is None or not grid_eq(p,out): ok=False; break
        if ok: return mk(margin)(ti)
    
    # 枠の中を塗りつぶし
    def frame_fill(grid):
        g=np.array(grid).copy();h,w=g.shape;bg2=_bg(g)
        objs=_objs(g,bg2)
        if not objs: return None
        frame_obj=max(objs,key=lambda x:x['size'])
        r1,c1,r2,c2=frame_obj['bbox']
        changed=False
        for r in range(r1+1,r2):
            for c in range(c1+1,c2):
                if g[r,c]==bg2:
                    g[r,c]=frame_obj['color']; changed=True
        return g.tolist() if changed else None
    
    ok=True
    for inp,out in tp:
        p=frame_fill(inp)
        if p is None or not grid_eq(p,out): ok=False; break
    if ok: return frame_fill(ti)
    
    return None


def _decode_marked(tp, ti, structure):
    """マーカーが指示する操作"""
    marker_anchors = [a for a in structure.anchors if a.type=='marker']
    if not marker_anchors: return None
    
    gi = np.array(ti)
    h, w = gi.shape
    bg = structure.bg
    
    # マーカーの色が「方向」を示すパターン
    # マーカーの位置が「どのオブジェクトに対して何をするか」を示す
    
    for inp, out in tp:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: continue
        bg2 = _bg(ga)
        
        objs = _objs(ga, bg2)
        markers = [o for o in objs if o['size'] <= 2]
        non_markers = [o for o in objs if o['size'] > 2]
        
        if not markers or not non_markers: continue
        
        # マーカーが非マーカーオブジェクトの近くにある
        for marker in markers:
            mr, mc = marker['center']
            for nm in non_markers:
                nr, nc = nm['center']
                dr = int(round(nr - mr))
                dc = int(round(nc - mc))
                
                # マーカーがオブジェクトの方向を指す → そっちに移動
                # または、マーカーの色でオブジェクトを塗る
        break
    
    return None


def _decode_templated(tp, ti, structure):
    """テンプレートパターン: 同じ形の繰り返し"""
    repeated = [a for a in structure.anchors if a.type=='repeated']
    if len(repeated) < 2: return None
    
    # 同じ形のグループ
    shape_groups = defaultdict(list)
    for a in repeated:
        shape_groups[a.shape].append(a)
    
    # 最も多いグループ = テンプレート
    template_group = max(shape_groups.values(), key=len)
    if len(template_group) < 2: return None
    
    # テンプレートの色が異なる → 色がデータ
    colors = [a.color for a in template_group]
    if len(set(colors)) > 1:
        # 各テンプレートの色が「データ」
        # 出力でこのデータがどう変換されるかを学習
        pass
    
    return None


def _decode_odd_one(tp, ti, structure):
    """仲間はずれを見つけて抽出"""
    unique_anchors = [a for a in structure.anchors if a.type=='unique']
    if not unique_anchors: return None
    
    for attr in ['shape','color','size']:
        ok=True
        for inp,out in tp:
            ga=np.array(inp);bg2=_bg(ga);objs=_objs(ga,bg2)
            if len(objs)<2: ok=False; break
            cc=Counter(o[attr] if attr!='shape' else o['shape'] for o in objs)
            uniq=[o for o in objs if cc[o[attr] if attr!='shape' else o['shape']]==1]
            if not uniq: ok=False; break
            o=uniq[0]; r1,c1,r2,c2=o['bbox']
            if not grid_eq(ga[r1:r2+1,c1:c2+1].tolist(),out): ok=False; break
        if ok:
            gi=np.array(ti);bg2=_bg(gi);objs=_objs(gi,bg2)
            cc=Counter(o[attr] if attr!='shape' else o['shape'] for o in objs)
            uniq=[o for o in objs if cc[o[attr] if attr!='shape' else o['shape']]==1]
            if uniq:
                o=uniq[0]; r1,c1,r2,c2=o['bbox']
                return gi[r1:r2+1,c1:c2+1].tolist()
    return None


def _decode_recolored(tp, ti, structure):
    """色の対応関係を使った変換"""
    recolor_conns = [c for c in structure.connections if c.connection_type=='recolor']
    if not recolor_conns: return None
    
    # 同じ形で色が異なるペア → 色のマッピング
    color_map = {}
    for conn in recolor_conns:
        c1, c2 = conn.a1.color, conn.a2.color
        if c1 != c2:
            color_map[c1] = c2
    
    if not color_map: return None
    
    # 適用
    def apply(grid):
        g=np.array(grid); r=g.copy()
        for o,n in color_map.items(): r[g==o]=n
        return r.tolist()
    
    ok=True
    for inp,out in tp:
        if not grid_eq(apply(inp),out): ok=False; break
    return apply(ti) if ok else None


def _decode_aligned(tp, ti, structure):
    """整列したオブジェクトの操作"""
    # 水平/垂直に並んだオブジェクト → ソート、移動など
    h_conns = [c for c in structure.connections if c.direction=='horizontal' and c.same_shape]
    v_conns = [c for c in structure.connections if c.direction=='vertical' and c.same_shape]
    
    if not h_conns and not v_conns: return None
    
    return None


def _decode_tiled(tp, ti, structure):
    """タイルパターン"""
    from arc.cross_brain_v2 import tile_variants, scale_variants
    try:
        r = tile_variants(tp, ti)
        if r: return r
    except: pass
    try:
        r = scale_variants(tp, ti)
        if r: return r
    except: pass
    return None


def _decode_symmetric(tp, ti, structure):
    """対称パターンの補完"""
    from arc.cross_brain_v2 import symmetry_variants
    try:
        r = symmetry_variants(tp, ti)
        if r: return r
    except: pass
    return None


# ══════════════════════════════════════════════════════════════
# Step 5: エラー訂正（補完）
# ══════════════════════════════════════════════════════════════

def error_correct(grid, structure):
    """パターンの不完全な部分を補完"""
    g = np.array(grid)
    h, w = g.shape
    bg = structure.bg
    
    result = g.copy()
    changed = False
    
    # 5a. 対称性補完
    # 入力がほぼ対称なら、対称になるように補完
    for mode in ['lr', 'ud', 'both']:
        test = g.copy()
        match = mismatch = 0
        
        for r in range(h):
            for c in range(w):
                if mode in ('lr', 'both'):
                    mc = w-1-c
                    if g[r,c] != bg and g[r,mc] != bg:
                        if g[r,c] == g[r,mc]: match += 1
                        else: mismatch += 1
                    elif g[r,c] != bg and g[r,mc] == bg:
                        test[r,mc] = g[r,c]
                
                if mode in ('ud', 'both'):
                    mr = h-1-r
                    if g[r,c] != bg and g[mr,c] != bg:
                        if g[r,c] == g[mr,c]: match += 1
                        else: mismatch += 1
                    elif g[r,c] != bg and g[mr,c] == bg:
                        test[mr,c] = g[r,c]
        
        # 80%以上対称なら補完
        if match > 0 and match / (match + mismatch + 0.001) > 0.8:
            if not np.array_equal(test, g):
                return test.tolist()
    
    # 5b. 周期性補完
    for period in range(1, h//2+1):
        if h % period != 0: continue
        tile = g[:period, :]
        pred = np.tile(tile, (h//period, 1))
        if not np.array_equal(pred, g):
            # 部分一致をチェック
            match = np.sum(pred == g)
            if match / g.size > 0.7:
                return pred.tolist()
    
    return None


# ══════════════════════════════════════════════════════════════
# QR Master: 全ステップ統合
# ══════════════════════════════════════════════════════════════

def qr_solve(train_pairs, test_input):
    """
    QRコード読み取りプロセスでARC問題を解く:
    1. アンカー検出
    2. Cross接続
    3. 構造コード
    4. デコード
    5. エラー訂正
    """
    
    # Step 1-3: 入力の構造を読む
    anchors_in = detect_anchors(train_pairs[0][0])
    connections_in = build_connections(anchors_in)
    structure_in = StructureCode(anchors_in, connections_in, train_pairs[0][0])
    
    anchors_out = detect_anchors(train_pairs[0][1])
    connections_out = build_connections(anchors_out)
    structure_out = StructureCode(anchors_out, connections_out, train_pairs[0][1])
    
    # Step 4: デコード
    results = decode_and_solve(structure_in, structure_out, train_pairs, test_input)
    
    # train検証
    for result, name in results:
        ok = True
        for inp, out in train_pairs:
            # 同じプロセスで各trainを解く
            a_in = detect_anchors(inp)
            c_in = build_connections(a_in)
            s_in = StructureCode(a_in, c_in, inp)
            
            a_out = detect_anchors(out)
            c_out = build_connections(a_out)
            s_out = StructureCode(a_out, c_out, out)
            
            rl = decode_and_solve(s_in, s_out, train_pairs, inp)
            found = False
            for r, _ in rl:
                if grid_eq(r, out):
                    found = True; break
            if not found:
                ok = False; break
        
        if ok:
            return result, name
    
    # Step 5: エラー訂正
    anchors_test = detect_anchors(test_input)
    connections_test = build_connections(anchors_test)
    structure_test = StructureCode(anchors_test, connections_test, test_input)
    
    ec_result = error_correct(np.array(test_input), structure_test)
    if ec_result is not None:
        ok = True
        for inp, out in train_pairs:
            a_t = detect_anchors(inp)
            c_t = build_connections(a_t)
            s_t = StructureCode(a_t, c_t, inp)
            p = error_correct(np.array(inp), s_t)
            if p is None or not grid_eq(p, out):
                ok = False; break
        if ok:
            return ec_result, 'qr:error_correct'
    
    return None, None


def cross_qr_solve(train_pairs, test_input):
    """Cross Brain V2 + QRプロセス統合"""
    from arc.cross_brain_v2 import cross_brain_v2_solve
    
    # まずQRで構造を読む
    result, name = qr_solve(train_pairs, test_input)
    if result is not None:
        return result, name
    
    # QRで解けなかったらBrain V2にフォールバック
    result, name = cross_brain_v2_solve(train_pairs, test_input)
    if result is not None:
        return result, name
    
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
        
        result, name = cross_qr_solve(tp, ti)
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_e else ''
            solved.append((tid, name, tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tg in solved if tg == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
    
    # アンカー統計
    print('\nアンカー統計:')
    anchor_types = Counter()
    pattern_types = Counter()
    for tf in sorted(data_dir.glob('*.json'))[:100]:
        with open(tf) as f: task = json.load(f)
        gi = task['train'][0]['input']
        anchors = detect_anchors(gi)
        for a in anchors: anchor_types[a.type] += 1
        conns = build_connections(anchors)
        sc = StructureCode(anchors, conns, gi)
        for p in sc.patterns: pattern_types[p] += 1
    
    print('  アンカータイプ:')
    for t, c in anchor_types.most_common():
        print(f'    {t}: {c}')
    print('  構造パターン:')
    for t, c in pattern_types.most_common():
        print(f'    {t}: {c}')
