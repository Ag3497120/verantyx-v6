"""
Phase C: 戦略別プロンプト生成
==============================
各戦略に最適化されたプロンプトをV3に送る
"""


def grid_str(g):
    return '\n'.join(' '.join(str(c) for c in row) for row in g)


def build_system_prompt(strategy_name):
    """戦略に応じたシステムプロンプト"""
    base = """You solve ARC-AGI puzzles by writing Python transform functions.
grid = list[list[int]], colors 0-9. Pure Python only, no numpy.
You receive automated analysis from Verantyx (170,000+ lines of ARC-solving code).
Output ONLY a ```python block with def transform(grid) -> list[list[int]]."""
    
    extras = {
        'repair': "\nVerantyx found a CLOSE MATCH solver. Your job is to understand WHY it's close and write the CORRECT version. Focus on the specific cells that differ.",
        'compose': "\nVerantyx found that composing two transformations gets close. Think about what each step does and write a clean combined version.",
        'object': "\nVerantyx detected objects with 1:1 correspondence between input and output. Focus on what transformation applies to EACH object (move, recolor, rotate, resize, etc).",
        'panel': "\nVerantyx detected panel structure (grid divided by separator lines). Focus on how panels relate to each other and how the output is constructed from them.",
        'colormap': "\nVerantyx detected a systematic color mapping. This is a strong hint but there may be additional spatial transformations.",
        'symmetry': "\nVerantyx detected symmetry in the output. The transformation likely involves creating or restoring symmetry.",
        'guided': "\nVerantyx found partial hints. Use them as starting points but think independently about the pattern.",
        'scratch': "\nNo strong automated hints. Analyze the examples carefully. Look for objects, patterns, symmetry, color relationships.",
    }
    
    return base + extras.get(strategy_name, '')


def build_user_prompt(task, analysis, strategy_name, strategy_data):
    """タスクデータ + 分析結果 + 戦略情報を統合したプロンプト"""
    parts = []
    
    # ═══ 戦略固有のヒント（最重要、最上部に配置） ═══
    parts.append(f"## Verantyx Strategy: {strategy_name.upper()}\n")
    
    if strategy_name == 'repair':
        src = strategy_data.get('source', '')
        piece = strategy_data.get('piece', '')
        pct = strategy_data.get('match_pct', 0)
        parts.append(f"Verantyx's `{src}/{piece}` solver achieves **{pct}% cell accuracy** on train[0].")
        diffs = strategy_data.get('diffs', [])
        dc = strategy_data.get('diff_count', 0)
        if diffs:
            parts.append(f"Remaining {dc} incorrect cells:")
            for r, c, got, want in diffs[:10]:
                parts.append(f"  ({r},{c}): solver outputs {got}, correct is {want}")
        if strategy_data.get('steps'):
            parts.append(f"(Achieved by applying `{piece}` {strategy_data['steps']} times iteratively)")
        parts.append(f"\nThis solver is CLOSE. Understand its approach, then write the CORRECT logic.\n")
    
    elif strategy_name == 'compose':
        parts.append(f"Two-step composition gets **{strategy_data['match_pct']}% accuracy**:")
        parts.append(f"  Step 1: `{strategy_data['cmd1']}`")
        parts.append(f"  Step 2: `{strategy_data['cmd2']}`")
        parts.append("Write a clean version that captures what these two steps do.\n")
    
    elif strategy_name == 'object':
        parts.append(f"Decomposition `{strategy_data['decomp']}` found **{strategy_data['obj_count']}** objects with 1:1 input→output correspondence.")
        parts.append("Object details:")
        for j, (io, oo) in enumerate(zip(strategy_data.get('in_objects', []), strategy_data.get('out_objects', []))):
            parts.append(f"  Object {j}: {io} → {oo}")
        changes = strategy_data.get('changes', [])
        if changes:
            parts.append("Detected changes per object:")
            for j, ch in enumerate(changes):
                parts.append(f"  Object {j}: {ch}")
        parts.append("")
    
    elif strategy_name == 'panel':
        d = strategy_data.get('direction', '?')
        parts.append(f"Grid has **{strategy_data['in_panels']} {d} panels** (separated by colored lines).")
        parts.append(f"Output has {strategy_data.get('out_panels', '?')} panels.")
        parts.append("Figure out the relationship between panels (overlay, difference, union, XOR, etc).\n")
    
    elif strategy_name == 'colormap':
        cmap = strategy_data.get('map', {})
        parts.append(f"Systematic color mapping detected: {cmap}")
        parts.append("This mapping is consistent across all train examples. There may be additional spatial changes.\n")
    
    elif strategy_name == 'symmetry':
        for s in strategy_data.get('symmetries', []):
            parts.append(f"  {s}")
        parts.append("The output has symmetry. The transform likely creates/restores symmetry.\n")
    
    elif strategy_name == 'guided':
        parts.append("Partial hints from Verantyx analysis:")
        for h in strategy_data.get('hints', []):
            parts.append(f"  - {h}")
        parts.append("")
    
    # ═══ 構造分析サマリー ═══
    struct = analysis.get('structure', {})
    parts.append("## Structure Analysis\n")
    
    in_sz = struct.get('in_sizes', [])
    out_sz = struct.get('out_sizes', [])
    parts.append(f"Input sizes: {in_sz}")
    parts.append(f"Output sizes: {out_sz}")
    if struct.get('same_size'):
        parts.append("→ Same dimensions (in-place transformation)")
    if struct.get('size_ratio'):
        parts.append(f"→ Constant size ratio: {struct['size_ratio']}")
    
    # 色情報
    for i, ci in enumerate(struct.get('colors', [])):
        nc = ci.get('new_colors', [])
        rc = ci.get('removed_colors', [])
        extra = ""
        if nc: extra += f" new={nc}"
        if rc: extra += f" removed={rc}"
        parts.append(f"Train {i+1}: in={ci['in_colors']} out={ci['out_colors']} bg={ci['in_bg']}->{ci['out_bg']}{extra}")
    
    # 差分
    if 'diffs' in struct:
        parts.append("")
        for i, di in enumerate(struct['diffs']):
            parts.append(f"Train {i+1}: {di['changed']}/{di['total']} cells changed ({di['pct']}%)")
            for r, c, old, new in di.get('details', [])[:8]:
                parts.append(f"  ({r},{c}): {old} → {new}")
    
    # セパレータ、周期性
    if struct.get('separators'):
        parts.append(f"\nSeparators: {struct['separators']}")
    if struct.get('periodicity'):
        parts.append(f"Periodicity: {struct['periodicity']}")
    
    # World Commands ヒント（repair以外でも上位を表示）
    wc = analysis.get('world_commands', {})
    singles = wc.get('single', [])
    if singles and strategy_name != 'repair':
        parts.append("\n## Partial-match world commands (applied to train[0]):")
        for s in singles[:5]:
            parts.append(f"  {s['cmd']}: {s['match_pct']}% match ({s['diff_count']} cells off)")
    
    # Cross2 構造情報（object以外でも表示）
    cross2 = analysis.get('cross2', [])
    if cross2 and strategy_name not in ('object', 'panel'):
        parts.append("\n## Object/Panel decomposition:")
        for item in cross2[:5]:
            if isinstance(item, dict) and 'decomp' in item:
                parts.append(f"  {item['decomp']}: in={item.get('in_obj_count', item.get('in_panels', '?'))} out={item.get('out_obj_count', item.get('out_panels', '?'))}")
    
    # ═══ タスクデータ ═══
    parts.append("\n## Task Data\n")
    for i, ex in enumerate(task['train']):
        h, w = len(ex['input']), len(ex['input'][0])
        parts.append(f"Train {i+1} Input ({h}x{w}):")
        parts.append(grid_str(ex['input']))
        oh, ow = len(ex['output']), len(ex['output'][0])
        parts.append(f"Train {i+1} Output ({oh}x{ow}):")
        parts.append(grid_str(ex['output']))
        parts.append("")
    
    for i, ex in enumerate(task['test']):
        h, w = len(ex['input']), len(ex['input'][0])
        parts.append(f"Test {i+1} Input ({h}x{w}):")
        parts.append(grid_str(ex['input']))
        parts.append("")
    
    parts.append("Write def transform(grid) -> list[list[int]]. Pure Python, no numpy.")
    parts.append("Use the Verantyx analysis and strategy hints above.")
    
    return '\n'.join(parts)


def build_feedback(train_errors, analysis=None):
    """フェーズDフィードバック（分析結果との照合込み）"""
    parts = [f"Wrong: {train_errors}"]
    
    # 分析結果との方向性フィードバック
    if analysis:
        wc = analysis.get('world_commands', {})
        singles = wc.get('single', [])
        if singles:
            parts.append(f"\nHint: Verantyx's '{singles[0]['cmd']}' gets {singles[0]['match_pct']}% right. Consider what that command does.")
    
    parts.append("\nFix your logic. Output ```python block with def transform(grid).")
    return '\n'.join(parts)
