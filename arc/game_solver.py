"""
arc/game_solver.py — ゲームルール強化ソルバー

overlay分類 × ゲームルール で未解決問題を攻略

=== overlay → ゲームルール マッピング ===
only_add (28問): リバーシ挟み、囲碁陣地、マインスイーパー数え、砲台スタンプ、
                  迷路パス、ブロック崩し反射、flood fill、線引き(connect)
add_and_remove (16問): テトリス重力、チェス移動、将棋駒打ち
only_change (12問): ライフゲーム、数独補完、色交換
size_diff (36問): ジグソー組合せ、切り出し、タイル
mixed (20問): 複合ルール
"""

import numpy as np
from typing import List, Optional, Tuple
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label


def _bg(g):
    return int(Counter(g.flatten()).most_common(1)[0][0])

def _objs8(g, bg):
    struct = np.ones((3,3), dtype=int)
    mask = (g != bg).astype(int)
    labeled, n = scipy_label(mask, structure=struct)
    objs = []
    for i in range(1, n+1):
        cells = list(zip(*np.where(labeled == i)))
        color = int(g[cells[0]])
        objs.append({'cells': cells, 'size': len(cells), 'color': color,
                     'center': (sum(r for r,c in cells)/len(cells), sum(c for r,c in cells)/len(cells))})
    return objs


# ══════════════════════════════════════════════════════════════
# only_add ソルバー群（何も消えない、増えるだけ）
# ══════════════════════════════════════════════════════════════

def reversi_8dir(train_pairs, test_input):
    """リバーシ8方向: 同色2点間のBGを全部塗る"""
    from arc.grid import grid_eq
    
    gi = np.array(test_input)
    h, w = gi.shape
    bg = _bg(gi)
    
    # trainから: 追加セルが「2点の間にある」パターンを検出
    for fill_mode in ['same_color', 'line_color']:
        ok_all = True
        for inp, out in train_pairs:
            pred = _reversi_apply(inp, bg, fill_mode)
            if pred is None or not grid_eq(pred, out):
                ok_all = False; break
        if ok_all:
            return _reversi_apply(test_input, bg, fill_mode)
    return None

def _reversi_apply(grid, bg, mode):
    g = np.array(grid).copy()
    h, w = g.shape
    changed = False
    
    # 8方向で同色挟み（距離制限なし）
    for r in range(h):
        for c in range(w):
            if g[r,c] != bg: continue
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                # この方向と逆方向に最初の非BGを探す
                # 方向1
                nr, nc = r+dr, c+dc
                c1 = None
                while 0 <= nr < h and 0 <= nc < w:
                    if g[nr, nc] != bg:
                        c1 = int(g[nr, nc]); break
                    nr += dr; nc += dc
                # 逆方向
                nr, nc = r-dr, c-dc
                c2 = None
                while 0 <= nr < h and 0 <= nc < w:
                    if g[nr, nc] != bg:
                        c2 = int(g[nr, nc]); break
                    nr -= dr; nc -= dc
                
                if c1 is not None and c2 is not None and c1 == c2:
                    g[r, c] = c1
                    changed = True
                    break
    
    return g.tolist() if changed else None


def connect_lines(train_pairs, test_input):
    """同色点を直線(水平/垂直)で結ぶ"""
    from arc.grid import grid_eq
    gi = np.array(test_input)
    h, w = gi.shape
    bg = _bg(gi)
    
    result = gi.copy()
    changed = False
    
    # 各色の全セル
    for color in set(int(v) for v in gi.flatten()) - {bg}:
        points = [(r, c) for r in range(h) for c in range(w) if gi[r,c] == color]
        if len(points) < 2: continue
        
        for i, (r1, c1) in enumerate(points):
            for r2, c2 in points[i+1:]:
                if r1 == r2:
                    for c in range(min(c1,c2)+1, max(c1,c2)):
                        if result[r1, c] == bg:
                            result[r1, c] = color; changed = True
                elif c1 == c2:
                    for r in range(min(r1,r2)+1, max(r1,r2)):
                        if result[r, c1] == bg:
                            result[r, c1] = color; changed = True
    
    if not changed: return None
    # train検証
    ok = True
    for inp, out in train_pairs:
        ga = np.array(inp); bg_t = _bg(ga)
        p = ga.copy(); ch = False
        for color in set(int(v) for v in ga.flatten()) - {bg_t}:
            pts = [(r, c) for r in range(ga.shape[0]) for c in range(ga.shape[1]) if ga[r,c] == color]
            for i, (r1,c1) in enumerate(pts):
                for r2,c2 in pts[i+1:]:
                    if r1==r2:
                        for c in range(min(c1,c2)+1, max(c1,c2)):
                            if p[r1,c]==bg_t: p[r1,c]=color; ch=True
                    elif c1==c2:
                        for r in range(min(r1,r2)+1, max(r1,r2)):
                            if p[r,c1]==bg_t: p[r,c1]=color; ch=True
        if not grid_eq(p.tolist(), out):
            ok = False; break
    return result.tolist() if ok else None


def go_territory(train_pairs, test_input):
    """囲碁: 囲まれたBG領域を囲む色で塗る"""
    from arc.grid import grid_eq
    
    def _apply(grid):
        g = np.array(grid).copy()
        h, w = g.shape
        bg = _bg(g)
        bg_mask = (g == bg).astype(int)
        struct4 = np.array([[0,1,0],[1,1,1],[0,1,0]])
        labeled, n = scipy_label(bg_mask, structure=struct4)
        
        changed = False
        for i in range(1, n+1):
            cells = list(zip(*np.where(labeled == i)))
            if any(r==0 or r==h-1 or c==0 or c==w-1 for r,c in cells):
                continue
            border = set()
            for r, c in cells:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w and g[nr,nc] != bg:
                        border.add(int(g[nr,nc]))
            if len(border) == 1:
                fill = border.pop()
                for r, c in cells:
                    g[r,c] = fill; changed = True
        return g.tolist() if changed else None
    
    ok = True
    for inp, out in train_pairs:
        p = _apply(inp)
        if p is None or not grid_eq(p, out):
            ok = False; break
    if ok:
        return _apply(test_input)
    return None


def minesweeper_count(train_pairs, test_input):
    """マインスイーパー: 8近傍の非BG数を色として書く"""
    from arc.grid import grid_eq
    
    def _apply(grid):
        g = np.array(grid)
        h, w = g.shape
        bg = _bg(g)
        result = g.copy()
        changed = False
        for r in range(h):
            for c in range(w):
                if g[r,c] != bg: continue
                count = 0
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and g[nr,nc] != bg:
                            count += 1
                if 0 < count <= 9:
                    result[r,c] = count; changed = True
        return result.tolist() if changed else None
    
    ok = True
    for inp, out in train_pairs:
        if _apply(inp) is None or not grid_eq(_apply(inp), out):
            ok = False; break
    return _apply(test_input) if ok else None


def flood_fill_enclosed(train_pairs, test_input):
    """囲まれたBG領域を特定色で塗る（囲む色が複数でもOK）"""
    from arc.grid import grid_eq
    
    # trainから塗り色を学習
    fill_colors = []
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        bg = _bg(ga)
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                if ga[r,c] == bg and go[r,c] != bg:
                    fill_colors.append(int(go[r,c]))
    
    if not fill_colors: return None
    fill_color = Counter(fill_colors).most_common(1)[0][0]
    
    def _apply(grid, fc):
        g = np.array(grid)
        h, w = g.shape
        bg = _bg(g)
        
        visited = np.zeros((h,w), dtype=bool)
        q = []
        for r in range(h):
            for c in [0, w-1]:
                if g[r,c] == bg and not visited[r,c]:
                    q.append((r,c)); visited[r,c] = True
        for c in range(w):
            for r in [0, h-1]:
                if g[r,c] == bg and not visited[r,c]:
                    q.append((r,c)); visited[r,c] = True
        while q:
            r, c = q.pop(0)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and g[nr,nc] == bg:
                    visited[nr,nc] = True; q.append((nr,nc))
        
        result = g.copy()
        changed = False
        for r in range(h):
            for c in range(w):
                if g[r,c] == bg and not visited[r,c]:
                    result[r,c] = fc; changed = True
        return result.tolist() if changed else None
    
    ok = True
    for inp, out in train_pairs:
        if not grid_eq(_apply(inp, fill_color), out):
            ok = False; break
    return _apply(test_input, fill_color) if ok else None


def breakout_diagonal(train_pairs, test_input):
    """ブロック崩し: 単独点から4対角方向に反射線"""
    from arc.grid import grid_eq
    
    def _apply(grid):
        g = np.array(grid).copy()
        h, w = g.shape
        bg = _bg(g)
        
        objs = _objs8(g, bg)
        single = [o for o in objs if o['size'] == 1]
        if not single: return None
        
        changed = False
        for o in single:
            r0, c0 = o['cells'][0]
            color = o['color']
            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                r, c = r0+dr, c0+dc
                while 0<=r<h and 0<=c<w:
                    if g[r,c] != bg: break
                    g[r,c] = color; changed = True
                    r += dr; c += dc
        return g.tolist() if changed else None
    
    ok = True
    for inp, out in train_pairs:
        p = _apply(inp)
        if p is None or not grid_eq(p, out):
            ok = False; break
    return _apply(test_input) if ok else None


def maze_bfs(train_pairs, test_input):
    """迷路: 2つの特殊点間の最短経路をBGに描く"""
    from arc.grid import grid_eq
    from collections import deque
    
    def _apply(grid):
        g = np.array(grid)
        h, w = g.shape
        bg = _bg(g)
        
        objs = _objs8(g, bg)
        single = [o for o in objs if o['size'] == 1]
        if len(single) < 2: return None
        
        start = single[0]['cells'][0]
        end = single[1]['cells'][0]
        path_color = single[0]['color']
        
        visited = set(); parent = {}
        q = deque([start]); visited.add(start)
        while q:
            r, c = q.popleft()
            if (r,c) == end:
                result = g.copy()
                cur = end; changed = False
                while cur in parent:
                    if result[cur] == bg:
                        result[cur] = path_color; changed = True
                    cur = parent[cur]
                return result.tolist() if changed else None
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<h and 0<=nc<w and (nr,nc) not in visited:
                    if g[nr,nc] == bg or (nr,nc) == end:
                        visited.add((nr,nc)); parent[(nr,nc)] = (r,c); q.append((nr,nc))
        return None
    
    ok = True
    for inp, out in train_pairs:
        p = _apply(inp)
        if p is None or not grid_eq(p, out):
            ok = False; break
    return _apply(test_input) if ok else None


# ══════════════════════════════════════════════════════════════
# add_and_remove ソルバー群（移動系）
# ══════════════════════════════════════════════════════════════

def gravity_4dir(train_pairs, test_input):
    """テトリス重力: 4方向"""
    from arc.grid import grid_eq
    
    for direction in ['down', 'up', 'left', 'right']:
        ok = True
        for inp, out in train_pairs:
            p = _gravity(inp, direction)
            if not grid_eq(p, out):
                ok = False; break
        if ok:
            return _gravity(test_input, direction)
    return None

def _gravity(grid, direction):
    g = np.array(grid)
    h, w = g.shape
    bg = _bg(g)
    result = np.full_like(g, bg)
    
    if direction in ('down', 'up'):
        for c in range(w):
            colors = [int(g[r,c]) for r in range(h) if g[r,c] != bg]
            if direction == 'down':
                for i, col in enumerate(reversed(colors)):
                    result[h-1-i, c] = col
            else:
                for i, col in enumerate(colors):
                    result[i, c] = col
    else:
        for r in range(h):
            colors = [int(g[r,c]) for c in range(w) if g[r,c] != bg]
            if direction == 'right':
                for i, col in enumerate(reversed(colors)):
                    result[r, w-1-i] = col
            else:
                for i, col in enumerate(colors):
                    result[r, i] = col
    return result.tolist()


def object_move(train_pairs, test_input):
    """オブジェクト移動: 全trainで共通の移動ベクトル"""
    from arc.grid import grid_eq
    
    movements = []
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        bg = _bg(ga)
        objs_in = _objs8(ga, bg)
        objs_out = _objs8(go, bg)
        
        for oi in objs_in:
            shape_i = frozenset((r-int(round(oi['center'][0])), c-int(round(oi['center'][1]))) for r,c in oi['cells'])
            for oo in objs_out:
                if oi['color'] != oo['color'] or oi['size'] != oo['size']: continue
                shape_o = frozenset((r-int(round(oo['center'][0])), c-int(round(oo['center'][1]))) for r,c in oo['cells'])
                if shape_i == shape_o:
                    dr = round(oo['center'][0] - oi['center'][0])
                    dc = round(oo['center'][1] - oi['center'][1])
                    movements.append((int(dr), int(dc)))
                    break
    
    if not movements or len(set(movements)) != 1:
        return None
    
    dr, dc = movements[0]
    if dr == 0 and dc == 0: return None
    
    gi = np.array(test_input)
    bg = _bg(gi)
    h, w = gi.shape
    result = np.full_like(gi, bg)
    for r in range(h):
        for c in range(w):
            if gi[r,c] != bg:
                nr, nc = r+dr, c+dc
                if 0<=nr<h and 0<=nc<w:
                    result[nr, nc] = gi[r, c]
    return result.tolist()


# ══════════════════════════════════════════════════════════════
# only_change ソルバー群（色変更のみ）
# ══════════════════════════════════════════════════════════════

def color_swap(train_pairs, test_input):
    """色の入れ替え"""
    from arc.grid import grid_eq
    mapping = {}
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        for r in range(ga.shape[0]):
            for c in range(ga.shape[1]):
                iv, ov = int(ga[r,c]), int(go[r,c])
                if iv != ov:
                    if iv in mapping and mapping[iv] != ov:
                        return None
                    mapping[iv] = ov
    
    if not mapping: return None
    gi = np.array(test_input)
    result = gi.copy()
    for old, new in mapping.items():
        result[gi == old] = new
    return result.tolist()


def life_step(train_pairs, test_input):
    """セルオートマトン1ステップ（汎用: 近傍カウント→生死）"""
    from arc.grid import grid_eq
    
    # trainから生存/誕生ルールを学習
    for inp, out in train_pairs:
        ga, go = np.array(inp), np.array(out)
        bg = _bg(ga)
        h, w = ga.shape
        fg = None
        for v in ga.flatten():
            if int(v) != bg: fg = int(v); break
        if fg is None: return None
        
        survive = set()
        birth = set()
        for r in range(h):
            for c in range(w):
                count = 0
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and ga[nr,nc] != bg:
                            count += 1
                if ga[r,c] != bg and go[r,c] != bg:
                    survive.add(count)
                elif ga[r,c] == bg and go[r,c] != bg:
                    birth.add(count)
        
        # ルールを適用
        def _apply(grid, surv, bir, fg_c, bg_c):
            g = np.array(grid)
            h2, w2 = g.shape
            result = np.full_like(g, bg_c)
            for r in range(h2):
                for c in range(w2):
                    cnt = 0
                    for dr in [-1,0,1]:
                        for dc in [-1,0,1]:
                            if dr==0 and dc==0: continue
                            nr, nc = r+dr, c+dc
                            if 0<=nr<h2 and 0<=nc<w2 and g[nr,nc] != bg_c:
                                cnt += 1
                    if g[r,c] != bg_c and cnt in surv:
                        result[r,c] = g[r,c]
                    elif g[r,c] == bg_c and cnt in bir:
                        result[r,c] = fg_c
            return result.tolist()
        
        ok = True
        for inp2, out2 in train_pairs:
            if not grid_eq(_apply(inp2, survive, birth, fg, bg), out2):
                ok = False; break
        if ok:
            return _apply(test_input, survive, birth, fg, bg)
        break
    
    return None


# ══════════════════════════════════════════════════════════════
# 砲台パターン (from cross3d_projection)
# ══════════════════════════════════════════════════════════════

def cannon_stamp(train_pairs, test_input):
    """砲台: テンプレートをドット方向に端までスタンプ"""
    from arc.cross3d_projection import cannon_solve
    from arc.grid import grid_eq
    
    ok = True
    for inp, out in train_pairs:
        p = cannon_solve(train_pairs, inp)
        if p is None or not grid_eq(p, out):
            ok = False; break
    if ok:
        return cannon_solve(train_pairs, test_input)
    return None


# ══════════════════════════════════════════════════════════════
# マスターソルバー
# ══════════════════════════════════════════════════════════════

def classify_overlay(grid_in, grid_out):
    """overlay分類"""
    gi, go = np.array(grid_in), np.array(grid_out)
    if gi.shape != go.shape: return 'size_diff'
    bg = _bg(gi)
    a = b = ch = 0
    for r in range(gi.shape[0]):
        for c in range(gi.shape[1]):
            iv, ov = int(gi[r,c]), int(go[r,c])
            if iv == bg and ov == bg: pass
            elif iv == ov: pass
            elif iv != bg and ov == bg: b += 1
            elif iv == bg and ov != bg: a += 1
            else: ch += 1
    if a > 0 and b == 0 and ch == 0: return 'only_add'
    if b > 0 and a == 0 and ch == 0: return 'only_remove'
    if a > 0 and b > 0 and ch == 0: return 'add_and_remove'
    if ch > 0 and a == 0 and b == 0: return 'only_change'
    return 'mixed'


SOLVER_MAP = {
    'only_add': [cannon_stamp, reversi_8dir, connect_lines, go_territory, 
                 flood_fill_enclosed, minesweeper_count, breakout_diagonal, maze_bfs],
    'add_and_remove': [gravity_4dir, object_move],
    'only_change': [color_swap, life_step],
    'mixed': [reversi_8dir, go_territory, flood_fill_enclosed, gravity_4dir, 
              color_swap, cannon_stamp, life_step],
    'size_diff': [],  # crop/scale は別モジュール
    'only_remove': [],
}


def game_solve(train_pairs, test_input):
    """overlay分類→ゲームルール優先ソルバー"""
    from arc.grid import grid_eq
    
    # 分類
    cat = classify_overlay(train_pairs[0][0], train_pairs[0][1])
    solvers = SOLVER_MAP.get(cat, [])
    
    for solver in solvers:
        try:
            result = solver(train_pairs, test_input)
            if result is not None:
                return result, solver.__name__
        except Exception:
            continue
    
    # 全ソルバーfallback
    all_solvers = set()
    for v in SOLVER_MAP.values():
        all_solvers.update(v)
    
    for solver in all_solvers:
        if solver in solvers: continue
        try:
            result = solver(train_pairs, test_input)
            if result is not None:
                return result, solver.__name__
        except Exception:
            continue
    
    return None, None


if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    from arc.grid import grid_eq
    
    split = 'evaluation' if '--eval' in sys.argv else 'training'
    data_dir = Path(f'/tmp/arc-agi-2/data/{split}')
    
    existing = set()
    with open('arc_v82.log') as f:
        for line in f:
            m = re.search(r'✓.*?([0-9a-f]{8})', line)
            if m: existing.add(m.group(1))
    synth = set(f.stem for f in Path('synth_results').glob('*.py'))
    all_existing = existing | synth
    
    solved = []
    for tf in sorted(data_dir.glob('*.json')):
        tid = tf.stem
        with open(tf) as f: task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti, to = task['test'][0]['input'], task['test'][0].get('output')
        
        result, name = game_solve(tp, ti)
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_existing else ''
            solved.append((tid, name, tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tag in solved if tag == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
