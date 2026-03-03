"""
arc/cross_simulator.py — Cross Simulator: 差分駆動の動的ルール生成

差分(DiffCross)から「条件+操作」を逆推定してプログラムを合成。

設計:
  1. 差分分析: 入力→出力の変更セルを分類
  2. 操作逆推定: 変更パターンから「何をしたか」を推定
  3. 条件逆推定: 「どこに適用したか」を推定
  4. プログラム合成: 条件+操作の組を構築
  5. 検証+進化: train検証→失敗フィードバック→条件追加

操作カタログ:
  - recolor(old, new): 色変換
  - copy_patch(src, dst): パッチコピー
  - fill_region(region, color): 領域塗り
  - extend_line(direction): 線延長
  - reflect(axis): 反射
  - rotate(angle): 回転

条件カタログ:
  - at_distance(color, d): 色からの距離
  - in_object(property): オブジェクト属性
  - in_region(type): 領域タイプ（enclosed, border, etc.）
  - neighbor_pattern(pattern): 近傍パターン
  - relative_to(obj1, obj2, relation): オブジェクト間関係
"""

import numpy as np
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label
from typing import List, Tuple, Optional, Dict, Set, Any
from itertools import combinations

Grid = List[List[int]]


def _bg(g):
    return int(Counter(np.array(g).flatten()).most_common(1)[0][0])


def grid_eq(a, b):
    a, b = np.array(a), np.array(b)
    return a.shape == b.shape and np.array_equal(a, b)


# ═══════════════════════════════════════
# 差分分析
# ═══════════════════════════════════════

class DiffCell:
    """変更セルの情報"""
    __slots__ = ['r', 'c', 'old', 'new']
    def __init__(self, r, c, old, new):
        self.r = r; self.c = c; self.old = old; self.new = new

class DiffAnalysis:
    """入力→出力の差分を多角的に分析"""
    
    def __init__(self, inp, out):
        self.ga = np.array(inp)
        self.go = np.array(out)
        self.h, self.w = self.ga.shape
        self.bg = _bg(self.ga)
        self.same_size = self.ga.shape == self.go.shape
        
        self.changes = []
        if self.same_size:
            for r in range(self.h):
                for c in range(self.w):
                    if self.ga[r,c] != self.go[r,c]:
                        self.changes.append(DiffCell(r, c, int(self.ga[r,c]), int(self.go[r,c])))
        
        # 色遷移の集計
        self.transitions = Counter((d.old, d.new) for d in self.changes)
        
        # 変更セルの空間分布
        self.changed_positions = set((d.r, d.c) for d in self.changes)
        
        # 入力のオブジェクト
        mask = (self.ga != self.bg).astype(int)
        self.labeled, self.n_objects = scipy_label(mask, structure=np.ones((3,3),dtype=int))
        
        self.objects = []
        for i in range(1, self.n_objects + 1):
            cells = list(zip(*np.where(self.labeled == i)))
            colors = Counter(int(self.ga[r,c]) for r,c in cells)
            r1 = min(r for r,c in cells); c1 = min(c for r,c in cells)
            r2 = max(r for r,c in cells); c2 = max(c for r,c in cells)
            self.objects.append({
                'id': i, 'cells': cells, 'size': len(cells),
                'colors': colors, 'main_color': colors.most_common(1)[0][0],
                'bbox': (r1, c1, r2, c2),
                'center': ((r1+r2)/2, (c1+c2)/2),
            })


# ═══════════════════════════════════════
# 操作逆推定
# ═══════════════════════════════════════

class Operation:
    """実行可能な操作"""
    def __init__(self, name, params, apply_fn):
        self.name = name
        self.params = params
        self.apply_fn = apply_fn
    
    def __repr__(self):
        return f"Op({self.name}, {self.params})"


def infer_operations(diffs: List[DiffAnalysis]) -> List[Operation]:
    """差分から可能な操作を逆推定"""
    ops = []
    
    if not diffs or not diffs[0].same_size:
        return ops
    
    # === Op1: 条件付きrecolor ===
    # 一貫した色遷移があるか
    all_trans = Counter()
    for d in diffs:
        all_trans.update(d.transitions)
    
    # 各遷移(old→new)について
    for (old_c, new_c), count in all_trans.items():
        ops.append(Operation(
            'recolor', {'old': old_c, 'new': new_c},
            lambda g, oc=old_c, nc=new_c, **kw: _apply_recolor(g, oc, nc, kw.get('condition'))
        ))
    
    # === Op2: パッチコピー ===
    # 出力に新しいパターンが現れるか
    for d in diffs:
        if not d.changes: continue
        # 変更セルが連結成分を形成するか
        change_mask = np.zeros((d.h, d.w), dtype=int)
        for dc in d.changes:
            change_mask[dc.r, dc.c] = 1
        labeled, n = scipy_label(change_mask, structure=np.ones((3,3),dtype=int))
        
        for i in range(1, n+1):
            cells = list(zip(*np.where(labeled == i)))
            if len(cells) < 2: continue
            r1 = min(r for r,c in cells); c1 = min(c for r,c in cells)
            r2 = max(r for r,c in cells); c2 = max(c for r,c in cells)
            # 出力のパッチ
            out_patch = d.go[r1:r2+1, c1:c2+1].copy()
            
            # このパッチが入力のどこかに存在するか（色違い含む）
            ph, pw = out_patch.shape
            for sr in range(d.h - ph + 1):
                for sc in range(d.w - pw + 1):
                    in_patch = d.ga[sr:sr+ph, sc:sc+pw]
                    # 形状一致（色はマッピング可能）
                    if _patches_match_up_to_color(in_patch, out_patch, d.bg):
                        ops.append(Operation(
                            'copy_patch', {'src': (sr, sc), 'dst': (r1, c1), 'size': (ph, pw)},
                            None  # apply_fnは後で構築
                        ))
    
    # === Op3: 反射/回転 ===
    for d in diffs:
        if not d.changes: continue
        # 変更領域のパッチが入力の反射/回転か
        change_cells = [(dc.r, dc.c) for dc in d.changes]
        if not change_cells: continue
        cr1 = min(r for r,c in change_cells); cc1 = min(c for r,c in change_cells)
        cr2 = max(r for r,c in change_cells); cc2 = max(c for r,c in change_cells)
        out_region = d.go[cr1:cr2+1, cc1:cc2+1]
        
        for tname, tfn in [('fliph', np.fliplr), ('flipv', np.flipud), 
                           ('rot90', lambda x: np.rot90(x,1)), ('rot180', lambda x: np.rot90(x,2))]:
            in_region = d.ga[cr1:cr2+1, cc1:cc2+1]
            transformed = tfn(in_region)
            if transformed.shape == out_region.shape and np.array_equal(transformed, out_region):
                ops.append(Operation('transform_region', {'type': tname, 'region': (cr1,cc1,cr2,cc2)}, None))
    
    return ops


def _apply_recolor(grid, old_color, new_color, condition=None):
    """条件付きrecolor"""
    g = np.array(grid)
    result = g.copy()
    h, w = g.shape
    
    if condition is None:
        # 全セル
        result[g == old_color] = new_color
    else:
        for r in range(h):
            for c in range(w):
                if g[r,c] == old_color and condition(g, r, c):
                    result[r,c] = new_color
    
    return result.tolist()


def _patches_match_up_to_color(p1, p2, bg):
    """2つのパッチが色マッピングで一致するか"""
    if p1.shape != p2.shape: return False
    cmap = {}
    for r in range(p1.shape[0]):
        for c in range(p1.shape[1]):
            v1, v2 = int(p1[r,c]), int(p2[r,c])
            if v1 == bg and v2 == bg: continue
            if v1 in cmap:
                if cmap[v1] != v2: return False
            else:
                cmap[v1] = v2
    return bool(cmap)


# ═══════════════════════════════════════
# 条件逆推定
# ═══════════════════════════════════════

class Condition:
    """セルに対する条件"""
    def __init__(self, name, check_fn):
        self.name = name
        self.check = check_fn
    
    def __repr__(self):
        return f"Cond({self.name})"


def infer_conditions(diffs: List[DiffAnalysis], op: Operation) -> List[Condition]:
    """操作が適用されたセルとされなかったセルから条件を推定"""
    
    if op.name != 'recolor':
        return []
    
    old_c = op.params['old']
    new_c = op.params['new']
    bg = diffs[0].bg
    
    # 変更セル(positive)と非変更同色セル(negative)を収集
    pos_cells = []  # (ga, r, c)
    neg_cells = []
    
    for d in diffs:
        for r in range(d.h):
            for c in range(d.w):
                if int(d.ga[r,c]) != old_c: continue
                if (r, c) in d.changed_positions:
                    pos_cells.append((d.ga, r, c))
                else:
                    neg_cells.append((d.ga, r, c))
    
    if not pos_cells: return []
    if not neg_cells:
        # 全セル変更 → 無条件
        return [Condition('always', lambda g, r, c: True)]
    
    # === 条件候補生成 ===
    conditions = []
    
    # C1: 色Tからの距離
    for target_c in range(10):
        if target_c == bg or target_c == old_c: continue
        
        for max_d in range(1, 8):
            def make_dist_cond(tc, md):
                def check(g, r, c):
                    h, w = g.shape
                    for rr in range(h):
                        for cc in range(w):
                            if int(g[rr,cc]) == tc and abs(r-rr)+abs(c-cc) <= md:
                                return True
                    return False
                return check
            conditions.append(Condition(f'dist_{target_c}_le_{max_d}', make_dist_cond(target_c, max_d)))
    
    # C2: 4近傍に色Tがある
    for target_c in range(10):
        if target_c == bg: continue
        def make_adj_cond(tc):
            def check(g, r, c):
                h, w = g.shape
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w and int(g[nr,nc]) == tc:
                        return True
                return False
            return check
        conditions.append(Condition(f'adj4_{target_c}', make_adj_cond(target_c)))
    
    # C3: 同色連結成分のサイズ
    for sz in range(1, 20):
        def make_csz_cond(s):
            def check(g, r, c):
                mask = (g == g[r,c]).astype(int)
                labeled, _ = scipy_label(mask, structure=np.ones((3,3),dtype=int))
                return int(np.sum(labeled == labeled[r,c])) == s
            return check
        conditions.append(Condition(f'csize_{sz}', make_csz_cond(sz)))
    
    # C4: 境界かどうか
    conditions.append(Condition('on_border', lambda g, r, c: r==0 or r==g.shape[0]-1 or c==0 or c==g.shape[1]-1))
    conditions.append(Condition('not_border', lambda g, r, c: not (r==0 or r==g.shape[0]-1 or c==0 or c==g.shape[1]-1)))
    
    # C5: 同行/同列に色Tがある
    for target_c in range(10):
        if target_c == bg or target_c == old_c: continue
        def make_row_cond(tc):
            def check(g, r, c):
                return any(int(g[r,cc]) == tc for cc in range(g.shape[1]))
            return check
        def make_col_cond(tc):
            def check(g, r, c):
                return any(int(g[rr,c]) == tc for rr in range(g.shape[0]))
            return check
        conditions.append(Condition(f'row_has_{target_c}', make_row_cond(target_c)))
        conditions.append(Condition(f'col_has_{target_c}', make_col_cond(target_c)))
    
    # C6: オブジェクト内に色Tが含まれる
    for target_c in range(10):
        if target_c == bg or target_c == old_c: continue
        def make_obj_cond(tc, bg_):
            def check(g, r, c):
                obj_mask = (g != bg_).astype(int)
                labeled, _ = scipy_label(obj_mask, structure=np.ones((3,3),dtype=int))
                if labeled[r,c] == 0: return False
                obj_cells = list(zip(*np.where(labeled == labeled[r,c])))
                return any(int(g[rr,cc]) == tc for rr,cc in obj_cells)
            return check
        conditions.append(Condition(f'in_obj_with_{target_c}', make_obj_cond(target_c, bg)))
    
    # C7: enclosed (端から到達不可能)
    def make_enclosed_cond(bg_):
        def check(g, r, c):
            h, w = g.shape
            visited = np.zeros((h,w), dtype=bool)
            queue = []
            for rr in range(h):
                for cc in [0, w-1]:
                    if g[rr,cc] == bg_ and not visited[rr,cc]:
                        visited[rr,cc] = True; queue.append((rr,cc))
            for cc in range(w):
                for rr in [0, h-1]:
                    if g[rr,cc] == bg_ and not visited[rr,cc]:
                        visited[rr,cc] = True; queue.append((rr,cc))
            while queue:
                cr, cc = queue.pop(0)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and g[nr,nc] == bg_:
                        visited[nr,nc] = True; queue.append((nr,nc))
            return not visited[r,c]
        return check
    conditions.append(Condition('enclosed', make_enclosed_cond(bg)))
    
    # C8: 2色の間にある（同行）
    for tc1 in range(10):
        for tc2 in range(tc1+1, 10):
            if tc1 == bg or tc2 == bg: continue
            def make_between_cond(t1, t2):
                def check(g, r, c):
                    w = g.shape[1]
                    row = g[r]
                    p1 = [cc for cc in range(w) if int(row[cc])==t1]
                    p2 = [cc for cc in range(w) if int(row[cc])==t2]
                    if not p1 or not p2: return False
                    return (min(p1) < c < max(p2)) or (min(p2) < c < max(p1))
                return check
            conditions.append(Condition(f'between_row_{tc1}_{tc2}', make_between_cond(tc1, tc2)))
    
    # === 条件評価: precision=1, recall=1 を探す ===
    best = []
    for cond in conditions:
        tp = sum(1 for ga, r, c in pos_cells if cond.check(ga, r, c))
        fp = sum(1 for ga, r, c in neg_cells if cond.check(ga, r, c))
        fn = len(pos_cells) - tp
        
        if tp == len(pos_cells) and fp == 0:
            best.append(cond)
    
    if best:
        return best
    
    # 2条件AND
    # 高recallかつ低fpの条件を選んでAND
    scored = []
    for cond in conditions:
        tp = sum(1 for ga, r, c in pos_cells if cond.check(ga, r, c))
        fp = sum(1 for ga, r, c in neg_cells if cond.check(ga, r, c))
        if tp > len(pos_cells) * 0.5:
            scored.append((cond, tp, fp))
    
    scored.sort(key=lambda x: x[2])  # fp少ない順
    
    for i in range(min(len(scored), 10)):
        for j in range(i+1, min(len(scored), 10)):
            c1, _, _ = scored[i]
            c2, _, _ = scored[j]
            tp = sum(1 for ga, r, c in pos_cells if c1.check(ga, r, c) and c2.check(ga, r, c))
            fp = sum(1 for ga, r, c in neg_cells if c1.check(ga, r, c) and c2.check(ga, r, c))
            if tp == len(pos_cells) and fp == 0:
                def make_and(ca, cb):
                    return Condition(f'{ca.name}&{cb.name}', lambda g,r,c,a=ca,b=cb: a.check(g,r,c) and b.check(g,r,c))
                best.append(make_and(c1, c2))
    
    return best


# ═══════════════════════════════════════
# プログラム合成
# ═══════════════════════════════════════

class Program:
    """条件+操作のペア"""
    def __init__(self, name, steps):
        self.name = name
        self.steps = steps  # [(condition, operation), ...]
    
    def apply(self, grid):
        result = [row[:] for row in grid]
        for cond, op in self.steps:
            result = op.apply_fn(result, condition=cond.check if cond else None)
        return result
    
    def __repr__(self):
        return f"Prog({self.name})"


def synthesize(train_pairs: List[Tuple[Grid, Grid]]) -> List[Tuple[str, callable]]:
    """差分駆動プログラム合成のメインエントリ"""
    
    # Step 1: 差分分析
    diffs = []
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        if ga.shape != go.shape:
            return []  # サイズ変更は別のソルバーに任せる
        diffs.append(DiffAnalysis(inp, out))
    
    if not diffs: return []
    
    # Step 2: 操作逆推定
    ops = infer_operations(diffs)
    
    # Step 3: 各操作に対して条件を推定
    programs = []
    
    for op in ops:
        if op.name != 'recolor': continue
        
        conditions = infer_conditions(diffs, op)
        
        for cond in conditions:
            # プログラム構築
            def make_apply(oc, nc, check_fn):
                def apply(grid):
                    g = np.array(grid)
                    result = g.copy()
                    h, w = g.shape
                    for r in range(h):
                        for c in range(w):
                            if int(g[r,c]) == oc:
                                if check_fn is None or check_fn(g, r, c):
                                    result[r,c] = nc
                    return result.tolist()
                return apply
            
            check_fn = cond.check if cond.name != 'always' else None
            apply_fn = make_apply(op.params['old'], op.params['new'], check_fn)
            
            # train検証
            ok = True
            for inp, out in train_pairs:
                pred = apply_fn(inp)
                if not grid_eq(pred, out):
                    ok = False; break
            
            if ok:
                prog_name = f"sim:{op.params['old']}→{op.params['new']}|{cond.name}"
                programs.append((prog_name, apply_fn))
    
    # Step 4: 複数操作の組み合わせ（順次適用）
    if not programs:
        # 単一操作では解けない → 複数遷移を順次適用
        recolor_ops = [op for op in ops if op.name == 'recolor']
        
        if len(recolor_ops) >= 2:
            # 全遷移に対して条件なしで適用してみる
            def make_multi_apply(transitions):
                def apply(grid):
                    g = np.array(grid)
                    result = g.copy()
                    orig = g.copy()
                    for old_c, new_c in transitions:
                        for r in range(g.shape[0]):
                            for c in range(g.shape[1]):
                                if int(orig[r,c]) == old_c:
                                    result[r,c] = new_c
                    return result.tolist()
                return apply
            
            trans = [(op.params['old'], op.params['new']) for op in recolor_ops]
            fn = make_multi_apply(trans)
            
            ok = all(grid_eq(fn(inp), out) for inp, out in train_pairs)
            if ok:
                tstr = '+'.join(f'{o}→{n}' for o,n in trans)
                programs.append((f'sim:multi_recolor|{tstr}', fn))
    
    return programs


# ═══════════════════════════════════════
# エントリポイント
# ═══════════════════════════════════════

if __name__ == '__main__':
    import json, re, time
    from pathlib import Path
    
    data_dir = Path('/tmp/arc-agi-2/data/training')
    eval_dir = Path('/tmp/arc-agi-2/data/evaluation')
    
    existing = set()
    try:
        with open('arc_v82.log') as f:
            for l in f:
                m = re.search(r'✓.*?([0-9a-f]{8})', l)
                if m: existing.add(m.group(1))
    except: pass
    synth = set(f.stem for f in Path('synth_results').glob('*.py')) if Path('synth_results').exists() else set()
    all_solved = existing | synth
    
    t0 = time.time()
    new_solved = []
    
    for split, sdir in [('train', data_dir), ('eval', eval_dir)]:
        for tf in sorted(sdir.glob('*.json')):
            tid = tf.stem
            if split == 'train' and tid in all_solved: continue
            
            with open(tf) as f: task = json.load(f)
            tp = [(e['input'], e['output']) for e in task['train']]
            ti = task['test'][0]['input']
            to = task['test'][0].get('output')
            
            programs = synthesize(tp)
            for name, fn in programs:
                try:
                    pred = fn(ti)
                    test_ok = to and grid_eq(pred, to)
                    tag = 'NEW' if tid not in all_solved else ''
                    print(f"✓ {split} {tid[:8]}: {name} test={'✓' if test_ok else '✗'} {tag}")
                    if test_ok and tag:
                        new_solved.append((tid[:8], name))
                except:
                    pass
                break
    
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"New solved: {len(new_solved)}")
    for t, n in new_solved:
        print(f"  {t}: {n}")
