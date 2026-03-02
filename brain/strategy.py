"""
Phase B: 戦略決定 — cross構造によるルーティング
================================================
分析結果から最適な戦略を選択し、V3に渡す情報を構造化する
"""


def select_strategy(analysis):
    """
    分析結果から戦略を選択。
    Returns: (strategy_name, strategy_data)
    
    戦略:
    1. 'repair'   — 既存ソルバーが50%+一致、差分修正を指示
    2. 'compose'  — 2段合成が60%+一致、合成パイプラインを指示
    3. 'object'   — オブジェクト1:1対応あり、各objの変換を指示
    4. 'panel'    — パネル分解成功、パネル間関係を指示
    5. 'colormap' — 色マップ検出、マップ適用+αを指示
    6. 'symmetry' — 対称性検出、対称性ベースの変換を指示
    7. 'guided'   — 部分的なヒントあり、ヒント付きフルスクラッチ
    8. 'scratch'  — 何も引っかからない、基本分析のみ
    """
    
    struct = analysis.get('structure', {})
    wc = analysis.get('world_commands', {})
    cross2 = analysis.get('cross2', [])
    ce = analysis.get('cross_engine', [])
    wp = analysis.get('world_priors', [])
    
    best_strategy = ('scratch', {})
    best_score = 0
    
    # 1. Cross Engine / World Priors で高一致
    for item in (ce or []):
        pct = item.get('match_pct', 0)
        if pct > 50 and pct > best_score:
            best_score = pct
            best_strategy = ('repair', {
                'source': item.get('source', 'cross_engine'),
                'piece': item.get('piece', ''),
                'match_pct': pct,
                'diffs': item.get('diffs', []),
                'diff_count': item.get('diff_count', 0),
            })
    
    for item in (wp or []):
        pct = item.get('match_pct', 0)
        if pct > 50 and pct > best_score:
            best_score = pct
            best_strategy = ('repair', {
                'source': 'world_prior',
                'piece': item.get('prior', ''),
                'match_pct': pct,
            })
    
    # 2. World Commands 2段合成
    compose2 = wc.get('compose2', [])
    if compose2:
        top = compose2[0]
        pct = top.get('match_pct', 0)
        if pct > 60 and pct > best_score:
            best_score = pct
            best_strategy = ('compose', {
                'cmd1': top['cmd1'],
                'cmd2': top['cmd2'],
                'match_pct': pct,
            })
    
    # World Commands 単体
    singles = wc.get('single', [])
    if singles:
        top = singles[0]
        pct = top.get('match_pct', 0)
        if pct > 50 and pct > best_score:
            best_score = pct
            best_strategy = ('repair', {
                'source': 'world_command',
                'piece': top['cmd'],
                'match_pct': pct,
                'diffs': top.get('diffs', []),
                'diff_count': top.get('diff_count', 0),
            })
    
    # Converge
    converges = wc.get('converge', [])
    if converges:
        top = converges[0]
        pct = top.get('match_pct', 0)
        if pct > best_score:
            best_score = pct
            best_strategy = ('repair', {
                'source': 'world_command_converge',
                'piece': top['cmd'],
                'steps': top['steps'],
                'match_pct': pct,
            })
    
    # 3. オブジェクト1:1対応
    for item in (cross2 or []):
        if isinstance(item, dict) and item.get('correspondence') == '1:1':
            changes = item.get('object_changes', [])
            if changes:
                obj_score = 40  # オブジェクト対応のベーススコア
                if obj_score > best_score or best_strategy[0] == 'scratch':
                    best_strategy = ('object', {
                        'decomp': item.get('decomp', ''),
                        'obj_count': item.get('in_obj_count', 0),
                        'in_objects': item.get('in_objects', []),
                        'out_objects': item.get('out_objects', []),
                        'changes': changes,
                    })
                    best_score = max(best_score, obj_score)
    
    # 4. パネル分解
    for item in (cross2 or []):
        if isinstance(item, dict) and item.get('decomp', '').startswith('pan_'):
            if item.get('in_panels', 0) > 1:
                panel_score = 35
                if panel_score > best_score or best_strategy[0] == 'scratch':
                    best_strategy = ('panel', {
                        'direction': 'horizontal' if 'h' in item['decomp'] else 'vertical',
                        'in_panels': item['in_panels'],
                        'out_panels': item.get('out_panels', 0),
                    })
                    best_score = max(best_score, panel_score)
    
    # 5. 色マップ
    cmap = struct.get('color_map')
    if cmap:
        cmap_score = 45
        if cmap_score > best_score or best_strategy[0] == 'scratch':
            best_strategy = ('colormap', {'map': cmap})
            best_score = max(best_score, cmap_score)
    
    # 6. 対称性
    syms = struct.get('symmetries', [])
    if syms:
        sym_score = 30
        if sym_score > best_score or best_strategy[0] == 'scratch':
            best_strategy = ('symmetry', {'symmetries': syms})
            best_score = max(best_score, sym_score)
    
    # guided: 何かしらヒントがある場合
    if best_strategy[0] == 'scratch':
        hints = []
        if singles: hints.extend([f"world_cmd:{s['cmd']} ({s['match_pct']}%)" for s in singles[:3]])
        if cross2:
            for item in cross2:
                if isinstance(item, dict) and 'decomp' in item:
                    hints.append(f"decomp:{item['decomp']} in={item.get('in_obj_count', '?')} out={item.get('out_obj_count', '?')}")
        if struct.get('separators'):
            hints.append(f"separators: {struct['separators'][:3]}")
        if struct.get('periodicity'):
            hints.append(f"periodicity: {struct['periodicity']}")
        if struct.get('size_ratio'):
            hints.append(f"size_ratio: {struct['size_ratio']}")
        
        if hints:
            best_strategy = ('guided', {'hints': hints})
    
    return best_strategy
