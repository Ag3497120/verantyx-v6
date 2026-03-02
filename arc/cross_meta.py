"""
arc/cross_meta.py — メタソルバー: trainから操作を「発見」する

個別ソルバーを呼ぶのではなく、入出力の差分パターンを分析して
操作ルールを自動推論する。

分析手順:
1. 差分の構造を見る（追加/削除/変更/サイズ変化）
2. 追加セルと既存オブジェクトの空間関係を分析
3. 全trainで一貫するルールを抽出
4. testに適用
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
            'center':((r1+r2)/2,(c1+c2)/2)})
    return objs

def grid_eq(a,b):
    a,b=np.array(a),np.array(b);return a.shape==b.shape and np.array_equal(a,b)


# ══════════════════════════════════════════════════════════════
# Rule 1: 最近オブジェクトの色で塗る
# ══════════════════════════════════════════════════════════════

def rule_nearest_object_color(tp, ti):
    """追加セルを最も近いオブジェクトの色で塗る"""
    # 全trainで確認
    for dist_metric in ['manhattan', 'chebyshev']:
        ok = True
        for inp, out in tp:
            ga, go = np.array(inp), np.array(out)
            if ga.shape != go.shape: ok=False; break
            bg = _bg(ga); h, w = ga.shape
            objs = _objs(ga, bg)
            if not objs: ok=False; break
            
            for r in range(h):
                for c in range(w):
                    if ga[r,c] == bg and go[r,c] != bg:
                        # 最近オブジェクト
                        best_d = 9999; best_c = -1
                        for obj in objs:
                            for or2, oc in obj['cells']:
                                if dist_metric == 'manhattan':
                                    d = abs(r-or2) + abs(c-oc)
                                else:
                                    d = max(abs(r-or2), abs(c-oc))
                                if d < best_d:
                                    best_d = d; best_c = obj['color']
                        if best_c != int(go[r,c]):
                            ok = False; break
                    elif ga[r,c] != go[r,c]:
                        ok = False; break
                if not ok: break
            if not ok: break
        
        if ok:
            gi = np.array(ti).copy()
            h, w = gi.shape; bg = _bg(gi)
            objs = _objs(gi, bg)
            if not objs: continue
            for r in range(h):
                for c in range(w):
                    if gi[r,c] == bg:
                        best_d = 9999; best_c = bg
                        for obj in objs:
                            for or2, oc in obj['cells']:
                                if dist_metric == 'manhattan':
                                    d = abs(r-or2) + abs(c-oc)
                                else:
                                    d = max(abs(r-or2), abs(c-oc))
                                if d < best_d:
                                    best_d = d; best_c = obj['color']
                        gi[r,c] = best_c
            return gi.tolist(), 'nearest_color'
    return None, None


# ══════════════════════════════════════════════════════════════
# Rule 2: オブジェクトの形をスタンプ
# ══════════════════════════════════════════════════════════════

def rule_stamp_shape(tp, ti):
    """あるオブジェクトの形を他のオブジェクトの位置にスタンプ"""
    for inp, out in tp:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None, None
        bg = _bg(ga)
        objs = _objs(ga, bg)
        if len(objs) < 2: return None, None
        
        # 最大オブジェクト = テンプレート
        template = max(objs, key=lambda o: o['size'])
        others = [o for o in objs if o is not template]
        
        # テンプレートの相対形状
        tr1, tc1 = template['bbox'][0], template['bbox'][1]
        t_shape = [(r-tr1, c-tc1, int(ga[r,c])) for r,c in template['cells']]
        
        # 各otherの位置にテンプレをスタンプして出力と一致するか
        # otherの色でリカラー
        for recolor in [True, False]:
            pred = ga.copy()
            for other in others:
                or1, oc1 = other['bbox'][0], other['bbox'][1]
                oc = other['color']
                for dr, dc, tv in t_shape:
                    nr, nc = or1+dr, oc1+dc
                    if 0 <= nr < ga.shape[0] and 0 <= nc < ga.shape[1]:
                        if recolor:
                            pred[nr, nc] = oc if tv != bg else bg
                        else:
                            pred[nr, nc] = tv
            
            if grid_eq(pred.tolist(), out):
                def apply(grid, recolor=recolor):
                    g = np.array(grid).copy()
                    bg2 = _bg(g); objs2 = _objs(g, bg2)
                    if len(objs2) < 2: return None
                    tmpl = max(objs2, key=lambda o: o['size'])
                    otrs = [o for o in objs2 if o is not tmpl]
                    tr1b, tc1b = tmpl['bbox'][0], tmpl['bbox'][1]
                    tshape = [(r-tr1b, c-tc1b, int(g[r,c])) for r,c in tmpl['cells']]
                    for ot in otrs:
                        or1b, oc1b = ot['bbox'][0], ot['bbox'][1]
                        occ = ot['color']
                        for dr, dc, tv in tshape:
                            nr, nc = or1b+dr, oc1b+dc
                            if 0<=nr<g.shape[0] and 0<=nc<g.shape[1]:
                                if recolor:
                                    g[nr,nc] = occ if tv != bg2 else bg2
                                else:
                                    g[nr,nc] = tv
                    return g.tolist()
                
                ok = True
                for i2, o2 in tp:
                    p = apply(i2)
                    if p is None or not grid_eq(p, o2): ok=False; break
                if ok: return apply(ti), 'stamp_shape'
        break
    return None, None


# ══════════════════════════════════════════════════════════════
# Rule 3: 行/列の周期パターン（色の繰り返し）
# ══════════════════════════════════════════════════════════════

def rule_row_col_period(tp, ti):
    """行/列ごとに周期パターンを検出して補完/修復"""
    from arc.cross_repair import period_repair_solve_v2
    return period_repair_solve_v2(tp, ti)


# ══════════════════════════════════════════════════════════════
# Rule 4: オブジェクト間の相対パターン
# ══════════════════════════════════════════════════════════════

def rule_object_relative(tp, ti):
    """trainのオブジェクト間の相対関係を学習してtestに適用"""
    # 各trainで入力オブジェクト→出力の変化を分析
    # 同じ形のオブジェクト間で一貫した変換があるか
    
    for inp, out in tp:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None, None
        bg = _bg(ga)
        objs_in = _objs(ga, bg)
        objs_out = _objs(go, bg)
        
        # 各入力オブジェクトについて、出力での変化
        for obj_in in objs_in:
            r1,c1,r2,c2 = obj_in['bbox']
            # この領域の出力
            region_out = go[r1:r2+1, c1:c2+1]
            region_in = ga[r1:r2+1, c1:c2+1]
            
            if np.array_equal(region_in, region_out):
                continue  # 変化なし
            
            # 変化パターンを記録
            # ...
        break
    return None, None


# ══════════════════════════════════════════════════════════════
# Rule 5: BGセルの隣接パターンで色決定
# ══════════════════════════════════════════════════════════════

def _get_nb8(g, r, c, h, w):
    """8近傍の値をタプルで返す"""
    nb = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0: continue
            nr, nc = r+dr, c+dc
            nb.append(int(g[nr, nc]) if 0<=nr<h and 0<=nc<w else -1)
    return tuple(nb)


def _get_nb4(g, r, c, h, w):
    """4近傍の値をタプルで返す（上右下左）"""
    nb = []
    for dr, dc in [(-1,0),(0,1),(1,0),(0,-1)]:
        nr, nc = r+dr, c+dc
        nb.append(int(g[nr, nc]) if 0<=nr<h and 0<=nc<w else -1)
    return tuple(nb)


def _abstract_nb(nb, cv, bg):
    """近傍を抽象化: B=BG, S=同色, D=異色非BG, E=境界外"""
    return tuple('E' if v==-1 else ('B' if v==bg else ('S' if v==cv else 'D')) for v in nb)


def _abstract_nb_count(nb, cv, bg):
    """近傍をカウントベースで抽象化"""
    n_nonbg = sum(1 for v in nb if v != -1 and v != bg)
    n_same = sum(1 for v in nb if v == cv and v != -1)
    n_diff = n_nonbg - n_same
    return (n_nonbg, n_same, n_diff)


def _abstract_nb_color_set(nb, cv, bg):
    """近傍の色を正規化: BG=0, 色1→1, 色2→2... (出現順)"""
    color_map = {}; idx = 0; result = []
    for v in nb:
        if v == -1: result.append(-1)
        elif v == bg: result.append(0)
        else:
            if v not in color_map:
                idx += 1; color_map[v] = idx
            result.append(color_map[v])
    return tuple(result)


def rule_neighbor_pattern(tp, ti):
    """各セルの近傍パターン→出力色のマッピング（複数抽象化レベル）"""
    
    strategies = [
        ('raw8', _get_nb8, lambda nb, cv, bg: nb, lambda ov, cv, bg: ov),
        ('raw4', _get_nb4, lambda nb, cv, bg: nb, lambda ov, cv, bg: ov),
        ('abs8', _get_nb8, _abstract_nb, lambda ov, cv, bg: 'B' if ov==bg else ('S' if ov==cv else 'new')),
        ('abs4', _get_nb4, _abstract_nb, lambda ov, cv, bg: 'B' if ov==bg else ('S' if ov==cv else 'new')),
        ('cnt8', _get_nb8, _abstract_nb_count, lambda ov, cv, bg: ov),
        ('cnt4', _get_nb4, _abstract_nb_count, lambda ov, cv, bg: ov),
        ('cset8', _get_nb8, lambda nb, cv, bg: _abstract_nb_color_set(nb, cv, bg), lambda ov, cv, bg: ov),
        ('center_abs8', _get_nb8,
         lambda nb, cv, bg: (cv==bg, _abstract_nb(nb, cv, bg)),
         lambda ov, cv, bg: ov),
        ('cv_raw8', _get_nb8, lambda nb, cv, bg: (cv, nb), lambda ov, cv, bg: ov),
        ('all_raw8', _get_nb8, lambda nb, cv, bg: nb, lambda ov, cv, bg: ov),
        ('all_abs8', _get_nb8, _abstract_nb, lambda ov, cv, bg: 'B' if ov==bg else ('S' if ov==cv else 'new')),
        ('all_cnt8', _get_nb8, _abstract_nb_count, lambda ov, cv, bg: ov),
    ]
    
    for strat_name, nb_fn, key_fn, val_fn in strategies:
        all_cells = strat_name.startswith('all_')
        pattern_map = {}
        consistent = True
        new_color_candidates = []
        
        for inp, out in tp:
            ga, go = np.array(inp), np.array(out)
            if ga.shape != go.shape:
                consistent = False; break
            bg = _bg(ga); h, w = ga.shape
            
            for r in range(h):
                for c in range(w):
                    if not all_cells and ga[r,c] == go[r,c]:
                        continue
                    cv = int(ga[r,c])
                    nb = nb_fn(ga, r, c, h, w)
                    key = key_fn(nb, cv, bg)
                    val = val_fn(int(go[r,c]), cv, bg)
                    
                    if key in pattern_map:
                        if pattern_map[key] != val:
                            consistent = False; break
                    else:
                        pattern_map[key] = val
                    
                    if val == 'new':
                        non_bg_nbs = [v for v in nb if v != -1 and v != bg and v != cv]
                        if non_bg_nbs:
                            new_color_candidates.append(Counter(non_bg_nbs).most_common(1)[0][0])
                if not consistent: break
            if not consistent: break
        
        if not consistent or not pattern_map:
            continue
        if not all_cells and len(pattern_map) > 100:
            continue
        
        default_new_color = Counter(new_color_candidates).most_common(1)[0][0] if new_color_candidates else None
        
        def apply_pattern(grid, pm=pattern_map, nfn=nb_fn, kfn=key_fn,
                         all_c=all_cells, dnc=default_new_color):
            g = np.array(grid).copy()
            h, w = g.shape; bg2 = _bg(g); changed = False
            for r in range(h):
                for c in range(w):
                    cv = int(g[r,c])
                    nb = nfn(g, r, c, h, w)
                    key = kfn(nb, cv, bg2)
                    if key in pm:
                        val = pm[key]
                        if isinstance(val, str):
                            if val == 'B': new_val = bg2
                            elif val == 'S': new_val = cv
                            elif val == 'new':
                                non_bg_nbs = [v for v in nb if v != -1 and v != bg2 and v != cv]
                                new_val = Counter(non_bg_nbs).most_common(1)[0][0] if non_bg_nbs else (dnc if dnc is not None else cv)
                            else: continue
                        else:
                            new_val = val
                        if g[r,c] != new_val:
                            g[r,c] = new_val; changed = True
            return g.tolist() if changed else None
        
        ok = True
        for inp2, out2 in tp:
            p = apply_pattern(inp2)
            if p is None or not grid_eq(p, out2): ok = False; break
        
        if ok:
            result = apply_pattern(ti)
            if result is not None:
                return result, f'neighbor_pattern:{strat_name}'
    
    return None, None


# ══════════════════════════════════════════════════════════════
# Rule 6: 抽象化した近傍パターン
# ══════════════════════════════════════════════════════════════

def rule_abstract_neighbor(tp, ti):
    """近傍を色そのものではなく「同色/異色/BG」に抽象化"""
    
    pattern_map = {}
    
    for inp, out in tp:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: return None, None
        bg = _bg(ga); h, w = ga.shape
        
        for r in range(h):
            for c in range(w):
                if ga[r,c] != go[r,c]:
                    cv = int(ga[r,c])
                    nb = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r+dr, c+dc
                            if 0<=nr<h and 0<=nc<w:
                                nv = int(ga[nr,nc])
                                if nv == bg: nb.append('B')
                                elif nv == cv: nb.append('S')
                                else: nb.append('D')
                            else:
                                nb.append('E')
                    
                    key = (cv == bg, tuple(nb))
                    ov = int(go[r,c])
                    # 色を抽象化: BG, 自分色, 他色 → 出力は何？
                    out_abs = 'B' if ov == bg else ('S' if ov == cv else 'new')
                    
                    if key in pattern_map and pattern_map[key] != out_abs:
                        return None, None
                    pattern_map[key] = out_abs
    
    # パターン数が適切か
    if not pattern_map or len(pattern_map) > 50: return None, None
    
    return None, None  # 抽象パターンから具体色への逆変換が必要 — 後で実装


# ══════════════════════════════════════════════════════════════
# Master: 全ルール + 全経験を順番に
# ══════════════════════════════════════════════════════════════

def cross_meta_solve(train_pairs, test_input):
    """メタソルバー"""
    
    # メタルール
    for rule_fn in [rule_row_col_period, rule_neighbor_pattern, rule_stamp_shape, rule_nearest_object_color]:
        try:
            r, name = rule_fn(train_pairs, test_input)
            if r is not None:
                return r, f'meta:{name}'
        except:
            pass
    
    # 視覚経験
    from arc.cross_life_eye import cross_life_eye_solve
    return cross_life_eye_solve(train_pairs, test_input)


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
        tpx = [(e['input'], e['output']) for e in task['train']]
        tix, tox = task['test'][0]['input'], task['test'][0].get('output')
        
        result, name = cross_meta_solve(tpx, tix)
        if result and tox and grid_eq(result, tox):
            tag = 'NEW' if tid not in all_e else ''
            solved.append((tid, name, tag))
            if tag: print(f'  ✓ {tid} [{name}] NEW')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tg in solved if tg == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
    
    solver_stats = Counter()
    for _, name, _ in solved: solver_stats[name] += 1
    print('\nソルバー別:')
    for s, c in solver_stats.most_common(): print(f'  {s}: {c}')
