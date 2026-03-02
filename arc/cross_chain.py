"""
arc/cross_chain.py — 操作チェイン: 複数操作の組み合わせ

単発で解けない → 2-3段の操作チェインで解く
例: crop → colormap, symmetry → flood, gravity → colormap
"""

import numpy as np
from collections import Counter
from scipy.ndimage import label as scipy_label
from arc.cross_brain_v2 import (
    _bg, _objs, grid_eq, _verify,
    # 全ソルバー（単発操作として使う）
    crop_by_attr, panel_compare, conditional_extract,
    reversi_variants, connect_variants, symmetry_variants, flood_variants,
    cannon_stamp, morphology_variants, diagonal_draw, minesweeper,
    gravity_variants, object_translate, rotate_transform,
    colormap_full, template_match, neighbor_abstract, evenodd_full,
    scale_variants, tile_variants, count_output, repeat_pattern,
    extract_intuition,
)

# ══════════════════════════════════════════════════════════════
# 単発操作を「ステップ関数」に変換
# ══════════════════════════════════════════════════════════════

def _make_step(solver, train_pairs):
    """ソルバーを「grid→grid」関数に変換（train_pairsをバインド）"""
    def step(grid):
        try:
            result = solver(train_pairs, grid)
            return result
        except:
            return None
    return step


# ══════════════════════════════════════════════════════════════
# 差分からチェイン候補を推測
# ══════════════════════════════════════════════════════════════

def _analyze_chain_hints(train_pairs):
    """trainの入出力から必要な操作を推測"""
    hints = set()
    
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        bg = _bg(gi)
        
        # サイズ変化
        if gi.shape != go.shape:
            if go.size < gi.size:
                hints.add('crop'); hints.add('panel'); hints.add('extract')
            else:
                hints.add('scale'); hints.add('tile')
        
        if gi.shape == go.shape:
            # overlay分析
            a = r = ch = 0
            for rr in range(gi.shape[0]):
                for cc in range(gi.shape[1]):
                    iv, ov = int(gi[rr,cc]), int(go[rr,cc])
                    if iv == ov: pass
                    elif iv == bg and ov != bg: a += 1
                    elif iv != bg and ov == bg: r += 1
                    else: ch += 1
            
            if a > 0: hints.add('add')
            if r > 0: hints.add('remove')
            if ch > 0: hints.add('change')
            
            # 対称性チェック
            if np.array_equal(go, go[:, ::-1]) and not np.array_equal(gi, gi[:, ::-1]):
                hints.add('symmetry')
            if np.array_equal(go, go[::-1, :]) and not np.array_equal(gi, gi[::-1, :]):
                hints.add('symmetry')
    
    return hints


# ══════════════════════════════════════════════════════════════
# サイズ変更操作（チェインの前処理/後処理用）
# ══════════════════════════════════════════════════════════════

# same_size操作群（チェインの中間ステップ）
SAME_SIZE_OPS = [
    ('colormap', colormap_full),
    ('symmetry', symmetry_variants),
    ('reversi', reversi_variants),
    ('connect', connect_variants),
    ('flood', flood_variants),
    ('gravity', gravity_variants),
    ('morphology', morphology_variants),
    ('template', template_match),
    ('neighbor', neighbor_abstract),
    ('evenodd', evenodd_full),
    ('diagonal', diagonal_draw),
    ('minesweeper', minesweeper),
    ('rotate', rotate_transform),
    ('repeat', repeat_pattern),
]

# サイズ縮小操作群
SHRINK_OPS = [
    ('crop', crop_by_attr),
    ('panel', panel_compare),
    ('extract', conditional_extract),
    ('count', count_output),
]

# サイズ拡大操作群  
GROW_OPS = [
    ('scale', scale_variants),
    ('tile', tile_variants),
]


def chain_2step(train_pairs, test_input):
    """2段チェイン: op1 → op2"""
    
    gi0, go0 = np.array(train_pairs[0][0]), np.array(train_pairs[0][1])
    same_size = gi0.shape == go0.shape
    smaller = go0.size < gi0.size
    larger = go0.size > gi0.size
    
    if same_size:
        # same→same チェイン
        candidates = _gen_same_same_chains(train_pairs, test_input)
        if candidates: return candidates
    
    elif smaller:
        # same→shrink または shrink→same チェイン
        candidates = _gen_preshrink_chains(train_pairs, test_input)
        if candidates: return candidates
        candidates = _gen_postshrink_chains(train_pairs, test_input)
        if candidates: return candidates
    
    elif larger:
        # grow→same または same→grow チェイン
        candidates = _gen_pregrow_chains(train_pairs, test_input)
        if candidates: return candidates
    
    return None, None


def _gen_same_same_chains(train_pairs, test_input):
    """same_size → same_size チェイン"""
    
    # 各op1のtrain中間結果を計算
    for name1, op1 in SAME_SIZE_OPS:
        # op1でtrainを変換
        intermediates = []
        ok1 = True
        for inp, out in train_pairs:
            try:
                mid = op1(train_pairs, inp)
                if mid is None:
                    ok1 = False; break
                intermediates.append((mid, out))
            except:
                ok1 = False; break
        
        if not ok1 or not intermediates:
            continue
        
        # 中間結果→出力を解くop2を探す
        for name2, op2 in SAME_SIZE_OPS:
            if name2 == name1: continue
            
            try:
                ok2 = True
                for mid, out in intermediates:
                    p = op2(intermediates, mid)
                    if p is None or not grid_eq(p, out):
                        ok2 = False; break
                
                if ok2:
                    # test適用
                    mid_test = op1(train_pairs, test_input)
                    if mid_test is not None:
                        result = op2(intermediates, mid_test)
                        if result is not None:
                            return result, f'{name1}→{name2}'
            except:
                continue
    
    return None, None


def _gen_preshrink_chains(train_pairs, test_input):
    """same_size操作 → shrink チェイン"""
    
    for name1, op1 in SAME_SIZE_OPS:
        intermediates = []
        ok1 = True
        for inp, out in train_pairs:
            try:
                mid = op1(train_pairs, inp)
                if mid is None:
                    ok1 = False; break
                intermediates.append((mid, out))
            except:
                ok1 = False; break
        
        if not ok1: continue
        
        for name2, op2 in SHRINK_OPS:
            try:
                ok2 = True
                for mid, out in intermediates:
                    p = op2(intermediates, mid)
                    if p is None or not grid_eq(p, out):
                        ok2 = False; break
                
                if ok2:
                    mid_test = op1(train_pairs, test_input)
                    if mid_test is not None:
                        result = op2(intermediates, mid_test)
                        if result is not None:
                            return result, f'{name1}→{name2}'
            except:
                continue
    
    return None, None


def _gen_postshrink_chains(train_pairs, test_input):
    """shrink → same_size操作 チェイン"""
    
    # まずshrinkを適用
    for name1, op1 in SHRINK_OPS:
        intermediates = []
        ok1 = True
        for inp, out in train_pairs:
            try:
                mid = op1(train_pairs, inp)
                if mid is None:
                    ok1 = False; break
                intermediates.append((mid, out))
            except:
                ok1 = False; break
        
        if not ok1: continue
        
        # 中間結果と出力のサイズが同じか
        if not all(np.array(mid).shape == np.array(out).shape for mid, out in intermediates):
            continue
        
        for name2, op2 in SAME_SIZE_OPS:
            try:
                ok2 = True
                for mid, out in intermediates:
                    p = op2(intermediates, mid)
                    if p is None or not grid_eq(p, out):
                        ok2 = False; break
                
                if ok2:
                    mid_test = op1(train_pairs, test_input)
                    if mid_test is not None:
                        result = op2(intermediates, mid_test)
                        if result is not None:
                            return result, f'{name1}→{name2}'
            except:
                continue
    
    return None, None


def _gen_pregrow_chains(train_pairs, test_input):
    """same_size → grow, or grow → same_size"""
    
    # grow → same
    for name1, op1 in GROW_OPS:
        intermediates = []
        ok1 = True
        for inp, out in train_pairs:
            try:
                mid = op1(train_pairs, inp)
                if mid is None:
                    ok1 = False; break
                intermediates.append((mid, out))
            except:
                ok1 = False; break
        
        if not ok1: continue
        
        if not all(np.array(mid).shape == np.array(out).shape for mid, out in intermediates):
            continue
        
        for name2, op2 in SAME_SIZE_OPS:
            try:
                ok2 = True
                for mid, out in intermediates:
                    p = op2(intermediates, mid)
                    if p is None or not grid_eq(p, out):
                        ok2 = False; break
                
                if ok2:
                    mid_test = op1(train_pairs, test_input)
                    if mid_test is not None:
                        result = op2(intermediates, mid_test)
                        if result is not None:
                            return result, f'{name1}→{name2}'
            except:
                continue
    
    # same → grow
    for name1, op1 in SAME_SIZE_OPS:
        intermediates = []
        ok1 = True
        for inp, out in train_pairs:
            try:
                mid = op1(train_pairs, inp)
                if mid is None:
                    ok1 = False; break
                intermediates.append((mid, out))
            except:
                ok1 = False; break
        
        if not ok1: continue
        
        for name2, op2 in GROW_OPS:
            try:
                ok2 = True
                for mid, out in intermediates:
                    p = op2(intermediates, mid)
                    if p is None or not grid_eq(p, out):
                        ok2 = False; break
                
                if ok2:
                    mid_test = op1(train_pairs, test_input)
                    if mid_test is not None:
                        result = op2(intermediates, mid_test)
                        if result is not None:
                            return result, f'{name1}→{name2}'
            except:
                continue
    
    return None, None


def cross_chain_solve(train_pairs, test_input):
    """
    1. まず単発で試す (cross_brain_v2)
    2. 解けなかったら2段チェインで試す
    """
    from arc.cross_brain_v2 import cross_brain_v2_solve
    
    # 単発
    result, name = cross_brain_v2_solve(train_pairs, test_input)
    if result is not None:
        return result, name
    
    # 2段チェイン
    result, name = chain_2step(train_pairs, test_input)
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
        
        result, name = cross_chain_solve(tp, ti)
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
