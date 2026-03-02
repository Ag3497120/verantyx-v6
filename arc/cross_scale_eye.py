"""
arc/cross_scale_eye.py — 色ごとにCrossの粒度を変えるマルチスケール知覚

色=数字ではなく、色=空間的な存在感の大きさ

各色のCrossサイズ:
- 少数色 → 大きなCross（目立つ、影響範囲が広い）
- 多数色 → 小さなCross（地味、局所的）
- 背景 → Crossなし

マルチスケール特徴:
- Scale 1: 各セル単位（生データ）
- Scale 3: 3x3ブロック単位（局所パターン）
- Scale 5: 5x5ブロック単位（中域パターン）
- Scale N: オブジェクト単位（全体パターン）
"""

import numpy as np
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label


def _bg(g): return int(Counter(np.array(g).flatten()).most_common(1)[0][0])

def _objs(g, bg, conn=8):
    struct = np.ones((3,3),dtype=int) if conn==8 else np.array([[0,1,0],[1,1,1],[0,1,0]])
    mask = (np.array(g)!=bg).astype(int)
    labeled, n = scipy_label(mask, structure=struct)
    objs = []
    for i in range(1, n+1):
        cells = list(zip(*np.where(labeled==i)))
        colors = [int(g[r,c]) for r,c in cells]
        r1=min(r for r,c in cells);c1=min(c for r,c in cells)
        r2=max(r for r,c in cells);c2=max(c for r,c in cells)
        objs.append({'cells':cells,'size':len(cells),'color':Counter(colors).most_common(1)[0][0],
            'colors':set(colors),'bbox':(r1,c1,r2,c2),'bh':r2-r1+1,'bw':c2-c1+1,
            'shape':frozenset((r-r1,c-c1) for r,c in cells)})
    return objs


def compute_cross_sizes(grid):
    """色ごとのCrossサイズを計算: 希少色ほど大きなCross"""
    g = np.array(grid)
    h, w = g.shape
    total = h * w
    bg = _bg(g)
    
    color_counts = Counter(int(v) for v in g.flatten())
    
    sizes = {}
    for color, count in color_counts.items():
        if color == bg:
            sizes[color] = 0  # 背景はCrossなし
            continue
        
        ratio = count / total
        
        # 希少なほど大きなCross
        # 1セル → size 5 (大きな存在感)
        # 10% → size 3 (中程度)
        # 30%+ → size 1 (控えめ)
        if ratio < 0.01:    # 極希少
            sizes[color] = 7
        elif ratio < 0.03:  # 希少
            sizes[color] = 5
        elif ratio < 0.1:   # 少数
            sizes[color] = 3
        elif ratio < 0.3:   # 中程度
            sizes[color] = 2
        else:               # 多数
            sizes[color] = 1
    
    return sizes


def build_influence_map(grid, cross_sizes):
    """各色のCrossを展開して影響マップを作る"""
    g = np.array(grid)
    h, w = g.shape
    bg = _bg(g)
    
    # 各色の影響マップ (float, 0-1)
    influence = {}
    for color, size in cross_sizes.items():
        if size == 0: continue
        
        inf_map = np.zeros((h, w), dtype=float)
        half = size // 2
        
        for r in range(h):
            for c in range(w):
                if int(g[r, c]) != color: continue
                
                # 十字型のCrossを展開（中心が最強、端が弱い）
                for dr in range(-half, half+1):
                    for dc in range(-half, half+1):
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w:
                            # 距離に応じて減衰
                            dist = max(abs(dr), abs(dc))
                            strength = 1.0 - dist / (half + 1)
                            inf_map[nr, nc] = max(inf_map[nr, nc], strength)
        
        influence[color] = inf_map
    
    return influence


def build_multiscale_features(grid):
    """マルチスケール特徴量: 各スケールでグリッドを見る"""
    g = np.array(grid)
    h, w = g.shape
    bg = _bg(g)
    
    features = {}
    
    # Scale 1: 生データ
    features['scale1'] = g.copy()
    
    # Scale 3: 3x3ブロックの「支配色」
    s3h, s3w = (h+2)//3, (w+2)//3
    scale3 = np.full((s3h, s3w), bg, dtype=int)
    for r in range(s3h):
        for c in range(s3w):
            r1, c1 = r*3, c*3
            r2, c2 = min(r1+3, h), min(c1+3, w)
            block = g[r1:r2, c1:c2]
            vals = [int(v) for v in block.flatten() if v != bg]
            if vals:
                scale3[r, c] = Counter(vals).most_common(1)[0][0]
    features['scale3'] = scale3
    
    # Scale 5: 5x5ブロック
    s5h, s5w = (h+4)//5, (w+4)//5
    scale5 = np.full((s5h, s5w), bg, dtype=int)
    for r in range(s5h):
        for c in range(s5w):
            r1, c1 = r*5, c*5
            r2, c2 = min(r1+5, h), min(c1+5, w)
            block = g[r1:r2, c1:c2]
            vals = [int(v) for v in block.flatten() if v != bg]
            if vals:
                scale5[r, c] = Counter(vals).most_common(1)[0][0]
    features['scale5'] = scale5
    
    # Object scale: オブジェクト→1セルに圧縮
    objs = _objs(g, bg)
    features['objects'] = objs
    features['n_objs'] = len(objs)
    
    # 色影響マップ
    cross_sizes = compute_cross_sizes(grid)
    features['cross_sizes'] = cross_sizes
    features['influence'] = build_influence_map(grid, cross_sizes)
    
    # 各色の支配領域（influenceが最大の色）
    dominance = np.full((h, w), bg, dtype=int)
    max_inf = np.zeros((h, w))
    for color, inf_map in features['influence'].items():
        mask = inf_map > max_inf
        dominance[mask] = color
        max_inf[mask] = inf_map[mask]
    features['dominance'] = dominance
    
    return features


def multiscale_match(features_in, features_out, train_pairs):
    """マルチスケール特徴のマッチングで変換ルールを推論"""
    
    # 各スケールで変換ルールを学習
    rules = {}
    
    for scale_name in ['scale3', 'scale5']:
        sin = features_in.get(scale_name)
        sout = features_out.get(scale_name)
        
        if sin is None or sout is None: continue
        if sin.shape != sout.shape: continue
        
        # 色の対応
        mapping = {}
        consistent = True
        for r in range(sin.shape[0]):
            for c in range(sin.shape[1]):
                iv, ov = int(sin[r,c]), int(sout[r,c])
                if iv != ov:
                    if iv in mapping and mapping[iv] != ov:
                        consistent = False; break
                    mapping[iv] = ov
            if not consistent: break
        
        if consistent and mapping:
            rules[scale_name] = mapping
    
    # 支配領域の変換
    dom_in = features_in.get('dominance')
    dom_out = features_out.get('dominance')
    if dom_in is not None and dom_out is not None and dom_in.shape == dom_out.shape:
        mapping = {}
        consistent = True
        for r in range(dom_in.shape[0]):
            for c in range(dom_in.shape[1]):
                iv, ov = int(dom_in[r,c]), int(dom_out[r,c])
                if iv != ov:
                    if iv in mapping and mapping[iv] != ov:
                        consistent = False; break
                    mapping[iv] = ov
            if not consistent: break
        if consistent and mapping:
            rules['dominance'] = mapping
    
    return rules


def influence_based_solve(train_pairs, test_input):
    """影響マップベースのソルバー: Crossの大きさで問題を見る"""
    from arc.cross_brain_v2 import grid_eq
    
    gi = np.array(test_input)
    h, w = gi.shape
    bg = _bg(gi)
    
    # 全trainの影響マップを分析
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: continue
        
        fi = build_multiscale_features(inp)
        fo = build_multiscale_features(out)
        
        # 影響マップの重なりが出力を決めるか
        dom_in = fi['dominance']
        
        # 出力で変わったセル: 影響マップの何と一致する?
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                if ga[r,c] != go[r,c]:
                    # このセルを支配している色は?
                    dominant_color = int(dom_in[r,c])
                    out_color = int(go[r,c])
        break
    
    # 影響マップの交差点を塗るパターン
    ft = build_multiscale_features(test_input)
    
    # 2色の影響が重なる場所を特定色で塗る
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape: continue
        bg_t = _bg(ga)
        fi = build_multiscale_features(inp)
        
        # 追加されたセルの位置で、2色以上の影響が重なっているか
        overlap_fills = {}  # (color1, color2) → fill_color
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                if ga[r,c] == bg_t and go[r,c] != bg_t:
                    # このセルに影響している色ペア
                    influencers = []
                    for color, inf_map in fi['influence'].items():
                        if inf_map[r,c] > 0.1:
                            influencers.append((color, inf_map[r,c]))
                    
                    if len(influencers) >= 2:
                        influencers.sort(key=lambda x: -x[1])
                        key = (influencers[0][0], influencers[1][0])
                        fill = int(go[r,c])
                        if key in overlap_fills and overlap_fills[key] != fill:
                            overlap_fills = {}; break
                        overlap_fills[key] = fill
        
        if overlap_fills:
            # test適用
            result = gi.copy()
            ft2 = build_multiscale_features(test_input)
            changed = False
            
            for r in range(h):
                for c in range(w):
                    if result[r,c] != bg: continue
                    influencers = []
                    for color, inf_map in ft2['influence'].items():
                        if inf_map[r,c] > 0.1:
                            influencers.append((color, inf_map[r,c]))
                    
                    if len(influencers) >= 2:
                        influencers.sort(key=lambda x: -x[1])
                        key = (influencers[0][0], influencers[1][0])
                        if key in overlap_fills:
                            result[r,c] = overlap_fills[key]
                            changed = True
                        # 逆順も試す
                        key_rev = (influencers[1][0], influencers[0][0])
                        if key_rev in overlap_fills:
                            result[r,c] = overlap_fills[key_rev]
                            changed = True
            
            if changed:
                # train検証
                ok = True
                for inp2, out2 in train_pairs:
                    pred = np.array(inp2).copy()
                    bg2 = _bg(pred)
                    fi2 = build_multiscale_features(inp2)
                    for r in range(pred.shape[0]):
                        for c in range(pred.shape[1]):
                            if pred[r,c] != bg2: continue
                            infs = []
                            for col, im in fi2['influence'].items():
                                if im[r,c] > 0.1: infs.append((col, im[r,c]))
                            if len(infs) >= 2:
                                infs.sort(key=lambda x: -x[1])
                                k = (infs[0][0], infs[1][0])
                                if k in overlap_fills: pred[r,c] = overlap_fills[k]
                                k2 = (infs[1][0], infs[0][0])
                                if k2 in overlap_fills: pred[r,c] = overlap_fills[k2]
                    if not grid_eq(pred.tolist(), out2): ok = False; break
                
                if ok: return result.tolist(), 'influence_overlap'
        break
    
    return None, None


def scale_eye_solve(train_pairs, test_input):
    """マルチスケール知覚ソルバー"""
    from arc.cross_brain_v2 import grid_eq, cross_brain_v2_solve
    
    # まず通常のbrain_v2
    result, name = cross_brain_v2_solve(train_pairs, test_input)
    if result is not None:
        return result, name
    
    # 影響マップベース
    result, name = influence_based_solve(train_pairs, test_input)
    if result is not None:
        return result, name
    
    return None, None


if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    from arc.cross_brain_v2 import grid_eq
    
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
        
        result, name = scale_eye_solve(tp, ti)
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_e else ''
            solved.append((tid, name, tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tg in solved if tg == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
