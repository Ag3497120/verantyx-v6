"""
Phase A: Verantyx全力分析 — 170K行の真価
=========================================
94モジュール中間結果、cross2構造対応、216コマンド合成、
世界法則、対称性、周期性、600B SVD推論型分類
"""
import sys, os, time, json, traceback
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))


def grid_eq(a, b):
    if len(a) != len(b): return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]): return False
        for j in range(len(a[i])):
            if a[i][j] != b[i][j]: return False
    return True


def grid_match_pct(pred, expected):
    """2グリッド間の一致率（same sizeの場合）"""
    if not pred or not expected: return 0.0
    if len(pred) != len(expected): return 0.0
    if not pred[0] or not expected[0]: return 0.0
    if len(pred[0]) != len(expected[0]): return 0.0
    h, w = len(expected), len(expected[0])
    match = sum(1 for r in range(h) for c in range(w) if pred[r][c] == expected[r][c])
    return match / (h * w) * 100


def grid_diff(pred, expected):
    """差分セルのリスト"""
    if len(pred) != len(expected): return []
    diffs = []
    for r in range(len(expected)):
        if len(pred[r]) != len(expected[r]): return []
        for c in range(len(expected[r])):
            if pred[r][c] != expected[r][c]:
                diffs.append((r, c, pred[r][c], expected[r][c]))
    return diffs


def grid_str(g, max_rows=8):
    rows = [' '.join(str(c) for c in row) for row in g[:max_rows]]
    if len(g) > max_rows:
        rows.append(f"... ({len(g)} rows total)")
    return '\n'.join(rows)


# ═══════════════════════════════════════
# A1: Cross Engine 94モジュール中間結果
# ═══════════════════════════════════════

def analyze_cross_engine(task, timeout_sec=15):
    """Cross Engineの各モジュールをtrain[0]に対して実行、部分一致を収集"""
    results = []
    try:
        from arc.cross_engine import solve_cross_engine
        from arc.cross_solver import CrossSolver
        from arc.grid import grid_eq as geq
        
        train_pairs = [(e['input'], e['output']) for e in task['train']]
        inp0, out0 = train_pairs[0]
        
        # CrossSolverの各フェーズを個別実行
        solver = CrossSolver()
        
        # Phase 1: piece生成を試みる
        try:
            pieces = solver.generate_pieces(inp0, out0)
            if pieces:
                for p in pieces[:10]:
                    name = getattr(p, 'name', str(type(p).__name__))
                    # pieceをtrain[0] inputに適用
                    try:
                        pred = p.apply(inp0)
                        if pred:
                            pct = grid_match_pct(pred, out0)
                            if pct > 20:
                                diffs = grid_diff(pred, out0)
                                results.append({
                                    'source': 'cross_solver',
                                    'piece': name,
                                    'match_pct': round(pct, 1),
                                    'diff_count': len(diffs),
                                    'diffs': diffs[:5],
                                })
                    except: pass
        except: pass
        
    except Exception as e:
        results.append({'source': 'cross_engine', 'error': str(e)[:100]})
    
    return results


# ═══════════════════════════════════════
# A2: Cross2 構造対応分析
# ═══════════════════════════════════════

def analyze_cross2_structure(task):
    """入力と出力の両方をcross2で分解し、構造対応を取る"""
    results = []
    try:
        from arc.cross2 import CrossDecomposer
        
        for i, ex in enumerate(task['train']):
            inp, out = ex['input'], ex['output']
            
            # 入力の分解
            try:
                inp_decomps = CrossDecomposer.decompose_all(inp)
            except:
                inp_decomps = []
            
            # 出力の分解
            try:
                out_decomps = CrossDecomposer.decompose_all(out)
            except:
                out_decomps = []
            
            for d_in in inp_decomps:
                kind = d_in.kind
                
                # オブジェクト分解の場合: 入出力のオブジェクト対応を検出
                if kind in ('obj4', 'obj8', 'mono'):
                    in_objs = d_in.objects if hasattr(d_in, 'objects') else []
                    # 出力で同じkindの分解を探す
                    out_d = next((d for d in out_decomps if d.kind == kind), None)
                    out_objs = out_d.objects if out_d and hasattr(out_d, 'objects') else []
                    
                    if in_objs and out_objs:
                        result = {
                            'train_idx': i,
                            'decomp': kind,
                            'in_obj_count': len(in_objs),
                            'out_obj_count': len(out_objs),
                        }
                        
                        # オブジェクトの特徴抽出
                        def obj_features(objs):
                            feats = []
                            for o in objs:
                                cells = list(o) if hasattr(o, '__iter__') else []
                                if cells and len(cells[0]) >= 3:
                                    rows = [c[0] for c in cells]
                                    cols = [c[1] for c in cells]
                                    colors = [c[2] for c in cells]
                                    feats.append({
                                        'size': len(cells),
                                        'bbox': (min(rows), min(cols), max(rows), max(cols)),
                                        'colors': sorted(set(colors)),
                                        'main_color': Counter(colors).most_common(1)[0][0],
                                    })
                            return feats
                        
                        result['in_objects'] = obj_features(in_objs)[:5]
                        result['out_objects'] = obj_features(out_objs)[:5]
                        
                        # 1:1対応の検出
                        if len(in_objs) == len(out_objs):
                            result['correspondence'] = '1:1'
                            # 各オブジェクトの変化を検出
                            changes = []
                            for j, (io, oo) in enumerate(zip(result['in_objects'], result['out_objects'])):
                                change = {}
                                if io['main_color'] != oo['main_color']:
                                    change['color'] = f"{io['main_color']}->{oo['main_color']}"
                                if io['size'] != oo['size']:
                                    change['size'] = f"{io['size']}->{oo['size']}"
                                ib, ob = io['bbox'], oo['bbox']
                                if ib != ob:
                                    change['moved'] = f"({ib[0]},{ib[1]})->({ob[0]},{ob[1]})"
                                if change:
                                    changes.append(change)
                            if changes:
                                result['object_changes'] = changes
                        
                        results.append(result)
                
                # パネル分解
                elif kind in ('pan_h', 'pan_v'):
                    panels = d_in.panels if hasattr(d_in, 'panels') else []
                    out_d = next((d for d in out_decomps if d.kind == kind), None)
                    out_panels = out_d.panels if out_d and hasattr(out_d, 'panels') else []
                    results.append({
                        'train_idx': i,
                        'decomp': kind,
                        'in_panels': len(panels),
                        'out_panels': len(out_panels),
                    })
    
    except Exception as e:
        results.append({'error': f'cross2: {e}'})
    
    return results


# ═══════════════════════════════════════
# A3: World Commands スキャン（216コマンド + 2段合成）
# ═══════════════════════════════════════

def analyze_world_commands(task, threshold=30, top_n=20):
    """216コマンドの単体+2段合成スキャン"""
    results = {'single': [], 'compose2': [], 'converge': []}
    
    try:
        from arc.world_commands import build_all_commands
        
        train_pairs = [(e['input'], e['output']) for e in task['train']]
        cmds = build_all_commands(train_pairs)
        inp0, out0 = train_pairs[0]
        
        # 単体スキャン
        single_scores = []
        for name, fn in cmds:
            try:
                r = fn(inp0)
                if r is None: continue
                pct = grid_match_pct(r, out0)
                if pct > threshold:
                    single_scores.append((name, pct, r))
            except: pass
        
        single_scores.sort(key=lambda x: -x[1])
        for name, pct, r in single_scores[:10]:
            diffs = grid_diff(r, out0)
            results['single'].append({
                'cmd': name,
                'match_pct': round(pct, 1),
                'diff_count': len(diffs),
                'diffs': diffs[:5],
            })
        
        # 収束スキャン: 単体top命令を繰り返し適用
        for name, pct, _ in single_scores[:5]:
            fn = dict(cmds)[name] if name in dict(cmds) else None
            if not fn: continue
            try:
                g = [row[:] for row in inp0]
                for step in range(10):
                    g2 = fn(g)
                    if g2 is None or grid_eq(g2, g): break
                    g = g2
                conv_pct = grid_match_pct(g, out0)
                if conv_pct > pct + 5:
                    results['converge'].append({
                        'cmd': name,
                        'steps': step + 1,
                        'match_pct': round(conv_pct, 1),
                    })
            except: pass
        
        # 2段合成: top-N × 全コマンド
        if single_scores:
            top_cmds = [(n, dict(cmds).get(n)) for n, _, _ in single_scores[:top_n]]
            for name1, fn1 in top_cmds:
                if fn1 is None: continue
                try:
                    mid = fn1(inp0)
                    if mid is None: continue
                except: continue
                
                for name2, fn2 in cmds[:50]:  # 上位50のみ（速度制限）
                    if name2 == name1: continue
                    try:
                        r = fn2(mid)
                        if r is None: continue
                        pct = grid_match_pct(r, out0)
                        if pct > max(threshold + 20, 60):
                            results['compose2'].append({
                                'cmd1': name1,
                                'cmd2': name2,
                                'match_pct': round(pct, 1),
                            })
                    except: pass
            
            results['compose2'].sort(key=lambda x: -x['match_pct'])
            results['compose2'] = results['compose2'][:5]
    
    except Exception as e:
        results['error'] = str(e)[:100]
    
    return results


# ═══════════════════════════════════════
# A4: World Priors / Laws
# ═══════════════════════════════════════

def analyze_world_priors(task):
    """43世界法則ピースの試行"""
    results = []
    try:
        from arc.world_priors import generate_world_prior_pieces
        
        train_pairs = [(e['input'], e['output']) for e in task['train']]
        pieces = generate_world_prior_pieces(train_pairs)
        
        if pieces:
            inp0, out0 = train_pairs[0]
            for p in pieces:
                try:
                    pred = p.apply(inp0)
                    if pred:
                        pct = grid_match_pct(pred, out0)
                        if pct > 20:
                            results.append({
                                'prior': getattr(p, 'name', str(p)),
                                'match_pct': round(pct, 1),
                            })
                except: pass
    except Exception as e:
        results.append({'error': f'world_priors: {e}'})
    
    return results


# ═══════════════════════════════════════
# A5: 対称性・周期性・構造分析
# ═══════════════════════════════════════

def analyze_structure(task):
    """対称性、周期性、セパレータ、色マップ等の構造分析"""
    results = {}
    train = task['train']
    
    # サイズ関係
    in_sizes = [(len(e['input']), len(e['input'][0])) for e in train]
    out_sizes = [(len(e['output']), len(e['output'][0])) for e in train]
    results['in_sizes'] = in_sizes
    results['out_sizes'] = out_sizes
    results['same_size'] = in_sizes == out_sizes
    
    # サイズ比率
    ratios = []
    for (ih, iw), (oh, ow) in zip(in_sizes, out_sizes):
        if ih > 0 and iw > 0:
            ratios.append((oh/ih, ow/iw))
    if ratios and len(set(ratios)) == 1:
        results['size_ratio'] = ratios[0]
    
    # 色分析
    color_info = []
    for i, e in enumerate(train):
        in_colors = set(v for row in e['input'] for v in row)
        out_colors = set(v for row in e['output'] for v in row)
        in_cnt = Counter(v for row in e['input'] for v in row)
        out_cnt = Counter(v for row in e['output'] for v in row)
        color_info.append({
            'in_colors': sorted(in_colors),
            'out_colors': sorted(out_colors),
            'in_bg': in_cnt.most_common(1)[0][0],
            'out_bg': out_cnt.most_common(1)[0][0],
            'new_colors': sorted(out_colors - in_colors),
            'removed_colors': sorted(in_colors - out_colors),
        })
    results['colors'] = color_info
    
    # セル差分（same sizeの場合）
    if results['same_size']:
        diffs_info = []
        for i, e in enumerate(train):
            h, w = len(e['input']), len(e['input'][0])
            changed = []
            for r in range(h):
                for c in range(w):
                    if e['input'][r][c] != e['output'][r][c]:
                        changed.append((r, c, e['input'][r][c], e['output'][r][c]))
            diffs_info.append({
                'changed': len(changed),
                'total': h * w,
                'pct': round(len(changed) / (h * w) * 100, 1),
                'details': changed[:15],
            })
        results['diffs'] = diffs_info
    
    # 色マップ検出
    cmap = {}; cmap_ok = True
    if results['same_size']:
        for e in train:
            for r in range(len(e['input'])):
                for c in range(len(e['input'][r])):
                    ic, oc = e['input'][r][c], e['output'][r][c]
                    if ic in cmap:
                        if cmap[ic] != oc: cmap_ok = False; break
                    else: cmap[ic] = oc
                if not cmap_ok: break
            if not cmap_ok: break
        if cmap_ok and any(k != v for k, v in cmap.items()):
            results['color_map'] = cmap
    
    # 対称性検出
    symmetries = []
    for i, e in enumerate(train):
        g = e['output']
        h, w = len(g), len(g[0])
        # 水平対称
        h_sym = all(g[r][c] == g[h-1-r][c] for r in range(h//2) for c in range(w))
        # 垂直対称
        v_sym = all(g[r][c] == g[r][w-1-c] for r in range(h) for c in range(w//2))
        # 90度回転対称
        rot_sym = (h == w) and all(g[r][c] == g[c][h-1-r] for r in range(h) for c in range(w))
        if h_sym: symmetries.append(f"train{i+1}_output: horizontal")
        if v_sym: symmetries.append(f"train{i+1}_output: vertical")
        if rot_sym: symmetries.append(f"train{i+1}_output: rotational_90")
    if symmetries:
        results['symmetries'] = symmetries
    
    # セパレータ検出
    separators = []
    for e in train:
        g = e['input']
        h, w = len(g), len(g[0])
        # 水平セパレータ（全セル同色の行）
        for r in range(h):
            vals = set(g[r])
            if len(vals) == 1 and vals != {0}:
                separators.append(f"h_sep row={r} color={g[r][0]}")
        # 垂直セパレータ
        for c in range(w):
            vals = set(g[r][c] for r in range(h))
            if len(vals) == 1 and vals != {0}:
                separators.append(f"v_sep col={c} color={g[0][c]}")
        break  # train[0]のみ
    if separators:
        results['separators'] = separators[:10]
    
    # 周期性検出（出力グリッド）
    for e in train:
        g = e['output']
        h, w = len(g), len(g[0])
        for ph in range(1, h//2 + 1):
            if h % ph != 0: continue
            periodic = True
            for r in range(ph, h):
                if g[r] != g[r % ph]:
                    periodic = False; break
            if periodic:
                results.setdefault('periodicity', []).append(f"vertical period={ph}")
                break
        break
    
    return results


# ═══════════════════════════════════════
# A6: 統合分析関数
# ═══════════════════════════════════════

def full_analysis(task, timeout_sec=30):
    """全分析を実行して統合結果を返す"""
    analysis = {}
    t0 = time.time()
    
    # A8: 基本構造分析（必ず実行）
    analysis['structure'] = analyze_structure(task)
    
    # A2: Cross2構造対応
    try:
        analysis['cross2'] = analyze_cross2_structure(task)
    except Exception as e:
        analysis['cross2'] = [{'error': str(e)[:100]}]
    
    # A3: World Commands
    if time.time() - t0 < timeout_sec:
        try:
            analysis['world_commands'] = analyze_world_commands(task, threshold=30)
        except Exception as e:
            analysis['world_commands'] = {'error': str(e)[:100]}
    
    # A4: World Priors
    if time.time() - t0 < timeout_sec:
        try:
            analysis['world_priors'] = analyze_world_priors(task)
        except Exception as e:
            analysis['world_priors'] = [{'error': str(e)[:100]}]
    
    # A1: Cross Engine（重い、最後に）
    if time.time() - t0 < timeout_sec:
        try:
            analysis['cross_engine'] = analyze_cross_engine(task, timeout_sec=10)
        except Exception as e:
            analysis['cross_engine'] = [{'error': str(e)[:100]}]
    
    analysis['analysis_time'] = round(time.time() - t0, 2)
    return analysis
