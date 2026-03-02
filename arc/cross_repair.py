"""
arc/cross_repair.py — 2D周期パターン修復（QRエラー訂正）

1. パネル検出（セパレータ/枠で分割）
2. パネル内の枠を検出（ネスト枠）
3. 枠内部の2D周期パターンを多数決修復
"""

import numpy as np
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label


def _bg(g): return int(Counter(np.array(g).flatten()).most_common(1)[0][0])

def grid_eq(a, b):
    a, b = np.array(a), np.array(b)
    return a.shape == b.shape and np.array_equal(a, b)


def _find_separators(g):
    """行・列のセパレータ（全セルが同色の行/列）"""
    h, w = g.shape
    h_seps = set(); v_seps = set()
    for r in range(h):
        vals = set(int(v) for v in g[r])
        if len(vals) == 1: h_seps.add(r)
    for c in range(w):
        vals = set(int(v) for v in g[:,c])
        if len(vals) == 1: v_seps.add(c)
    return sorted(h_seps), sorted(v_seps)


def _split_by_seps(h, w, h_seps, v_seps):
    """セパレータでパネルに分割"""
    row_ranges = []
    prev = -1
    for s in h_seps:
        if s - prev > 1: row_ranges.append((prev+1, s-1))
        prev = s
    if prev < h-1: row_ranges.append((prev+1, h-1))
    if not row_ranges: row_ranges = [(0, h-1)]
    
    col_ranges = []
    prev = -1
    for s in v_seps:
        if s - prev > 1: col_ranges.append((prev+1, s-1))
        prev = s
    if prev < w-1: col_ranges.append((prev+1, w-1))
    if not col_ranges: col_ranges = [(0, w-1)]
    
    return [(r1,c1,r2,c2) for r1,r2 in row_ranges for c1,c2 in col_ranges]


def _strip_frame(panel):
    """パネル周囲の枠を除去して内部を返す"""
    h, w = panel.shape
    if h < 3 or w < 3: return panel, 0, 0
    
    # 枠色候補: 周囲の最頻色
    border_vals = []
    for c in range(w): border_vals.extend([int(panel[0,c]), int(panel[h-1,c])])
    for r in range(h): border_vals.extend([int(panel[r,0]), int(panel[r,w-1])])
    
    if not border_vals: return panel, 0, 0
    fc = Counter(border_vals).most_common(1)[0][0]
    
    # 上下左右の枠厚さ
    top = 0
    for r in range(h):
        if all(int(panel[r,c]) == fc for c in range(w)): top = r+1
        else: break
    
    bot = h
    for r in range(h-1, -1, -1):
        if all(int(panel[r,c]) == fc for c in range(w)): bot = r
        else: break
    
    left = 0
    for c in range(w):
        if all(int(panel[r,c]) == fc for r in range(top, bot)): left = c+1
        else: break
    
    right = w
    for c in range(w-1, -1, -1):
        if all(int(panel[r,c]) == fc for r in range(top, bot)): right = c
        else: break
    
    if top < bot and left < right:
        return panel[top:bot, left:right], top, left
    return panel, 0, 0


def _repair_2d_period(interior):
    """2D多数決周期修復"""
    ph, pw = interior.shape
    if ph < 1 or pw < 1: return None
    
    best_fixed = None
    best_errors = ph * pw + 1
    
    # 行周期と列周期を独立に探索
    max_pr = min(ph // 2 + 1, ph, 8)
    max_pc = min(pw, 8)
    
    for pr in range(1, max_pr + 1):
        for pc in range(1, max_pc + 1):
            if pr == ph and pc == pw: continue  # 全体=テンプレは無意味
            if pr * pc > ph * pw: continue
            
            # テンプレート: 各(r%pr, c%pc)の多数決
            template = defaultdict(list)
            for r in range(ph):
                for c in range(pw):
                    template[(r%pr, c%pc)].append(int(interior[r,c]))
            
            tmpl = {}; errors = 0
            for key, vals in template.items():
                majority = Counter(vals).most_common(1)[0][0]
                count = Counter(vals).most_common(1)[0][1]
                tmpl[key] = majority
                errors += len(vals) - count
            
            # エラーが少なくて、かつ修復の意味がある
            if 0 < errors <= max(ph * pw * 0.2, 2) and errors < best_errors:
                best_errors = errors
                fixed = interior.copy()
                for r in range(ph):
                    for c in range(pw):
                        fixed[r,c] = tmpl[(r%pr, c%pc)]
                best_fixed = fixed
    
    return best_fixed


def period_repair_solve(train_pairs, test_input):
    """2D周期修復ソルバー"""
    
    def apply(grid):
        g = np.array(grid).copy()
        h, w = g.shape
        
        # Level 1: セパレータでパネル分割
        h_seps, v_seps = _find_separators(g)
        panels = _split_by_seps(h, w, h_seps, v_seps)
        
        changed = False
        
        for r1, c1, r2, c2 in panels:
            panel = g[r1:r2+1, c1:c2+1]
            
            # Level 2: パネル内の枠を除去
            interior, offset_r, offset_c = _strip_frame(panel)
            
            # Level 3: 内部の2D周期修復
            fixed = _repair_2d_period(interior)
            if fixed is not None:
                g[r1+offset_r:r1+offset_r+fixed.shape[0],
                  c1+offset_c:c1+offset_c+fixed.shape[1]] = fixed
                changed = True
            
            # 枠なしでもパネル全体を試す
            if fixed is None:
                fixed2 = _repair_2d_period(panel)
                if fixed2 is not None:
                    g[r1:r2+1, c1:c2+1] = fixed2
                    changed = True
        
        # パネル分割なし（全体）も試す
        if not changed:
            interior, offset_r, offset_c = _strip_frame(g)
            fixed = _repair_2d_period(interior)
            if fixed is not None:
                g[offset_r:offset_r+fixed.shape[0],
                  offset_c:offset_c+fixed.shape[1]] = fixed
                changed = True
        
        return g.tolist() if changed else None
    
    # train検証
    ok = True
    for inp, out in train_pairs:
        p = apply(inp)
        if p is None or not grid_eq(p, out):
            ok = False; break
    
    if ok:
        return apply(test_input), 'period_repair_2d'
    
    return None, None


if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    
    split = 'evaluation' if '--eval' in sys.argv else 'training'
    data_dir = Path(f'/tmp/arc-agi-2/data/{split}')
    
    existing = set()
    try:
        with open('arc_v82.log') as f:
            for l in f:
                m = re.search(r'✓.*?([0-9a-f]{8})', l)
                if m: existing.add(m.group(1))
    except: pass
    synth = set(f.stem for f in Path('synth_results').glob('*.py')) if Path('synth_results').exists() else set()
    all_e = existing | synth
    
    solved = []
    for tf in sorted(data_dir.glob('*.json')):
        tid = tf.stem
        with open(tf) as f: task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti, to = task['test'][0]['input'], task['test'][0].get('output')
        
        result, name = period_repair_solve(tp, ti)
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_e else ''
            solved.append((tid, name, tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tg in solved if tg == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')


def _repair_diagonal_bounce(interior):
    """対角線バウンスパターンの修復: 1が対角線上を往復"""
    ph, pw = interior.shape
    if ph < 4 or pw < 2: return None
    
    bg = int(Counter(interior.flatten()).most_common(1)[0][0])
    
    # 非BG色
    fg_colors = set(int(v) for v in interior.flatten()) - {bg}
    if len(fg_colors) != 1: return None
    fg = fg_colors.pop()
    
    # 各行で非BGセルの列位置（各行ちょうど1セルでなければ対角線ではない）
    positions = []
    single_count = 0
    for r in range(ph):
        cols = [c for c in range(pw) if int(interior[r,c]) != bg]
        if len(cols) == 1:
            positions.append(cols[0])
            single_count += 1
        else:
            positions.append(None)
    
    # 80%以上の行が「ちょうど1セル」でなければ対角線パターンではない
    if single_count < ph * 0.8: return None
    
    # バウンスパターンを推定: 0→1→2→...→(pw-1)→(pw-2)→...→0→...
    # 周期 = 2*(pw-1)
    period = 2 * (pw - 1) if pw > 1 else 1
    if period < 2: return None
    
    bounce = list(range(pw)) + list(range(pw-2, 0, -1))  # [0,1,2,3,2,1]
    
    # 最適な位相を探す
    best_phase = 0
    best_matches = 0
    for phase in range(period):
        matches = 0
        for r in range(ph):
            expected_col = bounce[(r + phase) % period]
            if positions[r] is not None and positions[r] == expected_col:
                matches += 1
        if matches > best_matches:
            best_matches = matches
            best_phase = phase
    
    # 80%以上一致なら修復
    valid = sum(1 for p in positions if p is not None)
    if best_matches < valid * 0.75: return None
    
    fixed = np.full_like(interior, bg)
    for r in range(ph):
        c = bounce[(r + best_phase) % period]
        fixed[r, c] = fg
    
    if np.array_equal(fixed, interior): return None
    return fixed


def _repair_repeating_block(interior):
    """繰り返しブロックパターンの修復: NxMブロックが繰り返し"""
    ph, pw = interior.shape
    
    for bh in range(2, ph//2+1):
        if ph % bh != 0: continue
        for bw in range(1, pw+1):
            if pw % bw != 0: continue
            n_blocks_r = ph // bh
            n_blocks_c = pw // bw
            if n_blocks_r * n_blocks_c < 2: continue
            
            # 各ブロックを収集
            blocks = []
            for br in range(n_blocks_r):
                for bc in range(n_blocks_c):
                    blocks.append(interior[br*bh:(br+1)*bh, bc*bw:(bc+1)*bw])
            
            # 多数決テンプレート
            template = np.zeros((bh, bw), dtype=int)
            for r in range(bh):
                for c in range(bw):
                    vals = [int(b[r,c]) for b in blocks]
                    template[r,c] = Counter(vals).most_common(1)[0][0]
            
            # エラー数
            errors = 0
            for b in blocks:
                errors += np.sum(b != template)
            
            if 0 < errors <= max(ph*pw*0.15, 1):
                fixed = np.tile(template, (n_blocks_r, n_blocks_c))
                return fixed
    
    return None


def period_repair_solve_v2(train_pairs, test_input):
    """改良版: 2D周期 + 対角線バウンス + ブロック繰り返し"""
    
    def apply(grid):
        g = np.array(grid).copy()
        h, w = g.shape
        
        h_seps, v_seps = _find_separators(g)
        panels = _split_by_seps(h, w, h_seps, v_seps)
        
        changed = False
        
        for r1, c1, r2, c2 in panels:
            panel = g[r1:r2+1, c1:c2+1]
            interior, off_r, off_c = _strip_frame(panel)
            
            # 複数の修復を試す
            for repair_fn in [_repair_2d_period, _repair_diagonal_bounce, _repair_repeating_block]:
                fixed = repair_fn(interior)
                if fixed is not None:
                    g[r1+off_r:r1+off_r+fixed.shape[0],
                      c1+off_c:c1+off_c+fixed.shape[1]] = fixed
                    changed = True
                    break
        
        # パネル分割なしでも試す
        if not changed:
            interior, off_r, off_c = _strip_frame(g)
            for repair_fn in [_repair_2d_period, _repair_diagonal_bounce, _repair_repeating_block]:
                fixed = repair_fn(interior)
                if fixed is not None:
                    g[off_r:off_r+fixed.shape[0], off_c:off_c+fixed.shape[1]] = fixed
                    changed = True
                    break
        
        return g.tolist() if changed else None
    
    ok = True
    for inp, out in train_pairs:
        p = apply(inp)
        if p is None or not grid_eq(p, out):
            ok = False; break
    
    if ok:
        return apply(test_input), 'period_repair_v2'
    return None, None
