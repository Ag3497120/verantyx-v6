"""
arc/cross6_ops.py — 6軸Cross塊操作ソルバー

kofdai設計の核心:
1. 全セルに6軸Cross記述子を付与
2. 同色連結領域 = Cross塊（背景含む）
3. 各Cross塊に6軸ベースの特徴量を付与
4. 入力→出力のCross塊対応を学習
5. 操作を検出: 移動、色変更、カット、複製、反転、etc.
6. テスト入力に操作を適用

Cross塊の特徴（6軸ベース）:
- shape_sig: 正規化された形状
- cross_profile: 塊内の全セルのCross記述子のヒストグラム
- relative_pos: グリッド内の相対位置
- adjacency: 隣接する他塊との関係
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter
from scipy.ndimage import label as scipy_label
from arc.grid import grid_eq, most_common_color


# ─── 8方向run length ───

def _runs8(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    runs = np.zeros((h, w, 8), dtype=np.int16)
    DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    for d, (dr, dc) in enumerate(DIRS):
        for r in range(h):
            for c in range(w):
                color = g[r, c]
                n = 0
                r2, c2 = r + dr, c + dc
                while 0 <= r2 < h and 0 <= c2 < w and g[r2, c2] == color:
                    n += 1; r2 += dr; c2 += dc
                runs[r, c, d] = n
    return runs


# ─── Cross塊 ───

class CrossChunk:
    """同色連結領域 = 1つのCross塊"""
    __slots__ = ['color', 'cells', 'bbox', 'is_bg', 'h', 'w',
                 '_shape_sig', '_norm_cells', '_cross_profile']
    
    def __init__(self, color: int, cells: Set[Tuple[int,int]], 
                 is_bg: bool = False, grid_h: int = 0, grid_w: int = 0):
        self.color = color
        self.cells = cells
        self.is_bg = is_bg
        self.h = grid_h
        self.w = grid_w
        self._shape_sig = None
        self._norm_cells = None
        self._cross_profile = None
        
        rows = [r for r,c in cells]
        cols = [c for r,c in cells]
        self.bbox = (min(rows), min(cols), max(rows), max(cols))
    
    @property
    def size(self): return len(self.cells)
    
    @property
    def center(self):
        rows = [r for r,c in self.cells]
        cols = [c for r,c in self.cells]
        return (sum(rows)/len(rows), sum(cols)/len(cols))
    
    @property
    def shape_sig(self) -> tuple:
        """平行移動不変の形状シグネチャ"""
        if self._shape_sig is None:
            r0, c0 = self.bbox[0], self.bbox[1]
            self._norm_cells = tuple(sorted((r-r0, c-c0) for r,c in self.cells))
            self._shape_sig = self._norm_cells
        return self._shape_sig
    
    @property  
    def bh(self): return self.bbox[2] - self.bbox[0] + 1
    @property
    def bw(self): return self.bbox[3] - self.bbox[1] + 1
    
    def cross_profile(self, runs: np.ndarray) -> tuple:
        """塊内全セルのCross記述子の統計"""
        if self._cross_profile is None:
            vals = []
            for r, c in self.cells:
                vals.append(tuple(int(x) for x in runs[r, c]))
            # Sort for consistency
            self._cross_profile = tuple(sorted(vals))
        return self._cross_profile
    
    def translate(self, dr: int, dc: int) -> 'CrossChunk':
        new_cells = {(r+dr, c+dc) for r,c in self.cells}
        ch = CrossChunk(self.color, new_cells, self.is_bg, self.h, self.w)
        return ch
    
    def recolor(self, new_color: int) -> 'CrossChunk':
        ch = CrossChunk(new_color, set(self.cells), self.is_bg, self.h, self.w)
        return ch
    
    def flip_h(self) -> 'CrossChunk':
        """左右反転 (bbox基準)"""
        r0, c0, r1, c1 = self.bbox
        new_cells = {(r, c0 + c1 - c) for r,c in self.cells}
        return CrossChunk(self.color, new_cells, self.is_bg, self.h, self.w)
    
    def flip_v(self) -> 'CrossChunk':
        """上下反転"""
        r0, c0, r1, c1 = self.bbox
        new_cells = {(r0 + r1 - r, c) for r,c in self.cells}
        return CrossChunk(self.color, new_cells, self.is_bg, self.h, self.w)
    
    def rotate90(self) -> 'CrossChunk':
        """90度回転"""
        r0, c0 = self.bbox[0], self.bbox[1]
        new_cells = {(r0 + (c - c0), c0 + (self.bh - 1 - (r - r0))) 
                     for r,c in self.cells}
        return CrossChunk(self.color, new_cells, self.is_bg, self.h, self.w)
    
    def overlap(self, other: 'CrossChunk') -> int:
        return len(self.cells & other.cells)
    
    def __repr__(self):
        tag = "BG" if self.is_bg else f"C{self.color}"
        return f"Chunk({tag}, n={self.size}, bbox={self.bbox})"


# ─── グリッド → Cross塊分解 ───

def decompose(grid: List[List[int]]) -> Tuple[List[CrossChunk], int, np.ndarray]:
    """グリッドをCross塊に分解"""
    g = np.array(grid)
    h, w = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    runs = _runs8(g)
    
    colors = sorted(set(int(x) for x in g.flatten()))
    chunks = []
    
    for color in colors:
        mask = (g == color)
        labeled, n = scipy_label(mask)
        for i in range(1, n + 1):
            cells = set(zip(*np.where(labeled == i)))
            ch = CrossChunk(color, cells, is_bg=(color==bg), grid_h=h, grid_w=w)
            chunks.append(ch)
    
    # Sort: foreground first, then by size desc
    chunks.sort(key=lambda ch: (ch.is_bg, -ch.size))
    
    return chunks, bg, runs


def recompose(chunks: List[CrossChunk], h: int, w: int, bg: int) -> List[List[int]]:
    """Cross塊からグリッドを再構成"""
    grid = [[bg] * w for _ in range(h)]
    # Paint bg chunks first, then fg (larger first)
    bg_chunks = [ch for ch in chunks if ch.is_bg]
    fg_chunks = [ch for ch in chunks if not ch.is_bg]
    fg_chunks.sort(key=lambda ch: -ch.size)
    
    for ch in fg_chunks:
        for r, c in ch.cells:
            if 0 <= r < h and 0 <= c < w:
                grid[r][c] = ch.color
    
    return grid


# ─── Cross塊対応付け ───

def match_chunks(src: List[CrossChunk], dst: List[CrossChunk], 
                 runs_src: np.ndarray, runs_dst: np.ndarray) -> List[Dict]:
    """入力と出力のCross塊の対応を検出
    
    マッチング基準 (優先順):
    1. 形状一致 + 色一致 → identity / translation
    2. 形状一致 + 色不一致 → recolor
    3. 位置重複 → cut / merge / split
    4. Cross profile 一致 → 構造的対応
    """
    matches = []
    used_dst = set()
    
    src_fg = [ch for ch in src if not ch.is_bg]
    dst_fg = [ch for ch in dst if not ch.is_bg]
    
    # Pass 1: exact shape + color match (possibly translated)
    for si, s in enumerate(src_fg):
        for di, d in enumerate(dst_fg):
            if di in used_dst:
                continue
            if s.shape_sig == d.shape_sig and s.color == d.color:
                dr = d.bbox[0] - s.bbox[0]
                dc = d.bbox[1] - s.bbox[1]
                matches.append({
                    'type': 'identity' if (dr==0 and dc==0) else 'translate',
                    'src_idx': si, 'dst_idx': di,
                    'src': s, 'dst': d,
                    'delta': (dr, dc),
                })
                used_dst.add(di)
                break
    
    # Pass 2: shape match + recolor
    matched_src = {m['src_idx'] for m in matches}
    for si, s in enumerate(src_fg):
        if si in matched_src:
            continue
        for di, d in enumerate(dst_fg):
            if di in used_dst:
                continue
            if s.shape_sig == d.shape_sig and s.color != d.color:
                dr = d.bbox[0] - s.bbox[0]
                dc = d.bbox[1] - s.bbox[1]
                matches.append({
                    'type': 'recolor' if (dr==0 and dc==0) else 'recolor+translate',
                    'src_idx': si, 'dst_idx': di,
                    'src': s, 'dst': d,
                    'delta': (dr, dc),
                    'color_change': (s.color, d.color),
                })
                used_dst.add(di)
                break
    
    # Pass 3: overlap-based
    matched_src = {m['src_idx'] for m in matches}
    for si, s in enumerate(src_fg):
        if si in matched_src:
            continue
        best_overlap = 0
        best_di = None
        for di, d in enumerate(dst_fg):
            if di in used_dst:
                continue
            ov = s.overlap(d)
            if ov > best_overlap:
                best_overlap = ov
                best_di = di
        if best_di is not None and best_overlap > 0:
            d = dst_fg[best_di]
            matches.append({
                'type': 'transform',
                'src_idx': si, 'dst_idx': best_di,
                'src': s, 'dst': d,
                'overlap': best_overlap,
            })
            used_dst.add(best_di)
    
    # Unmatched dst = additions
    for di, d in enumerate(dst_fg):
        if di not in used_dst:
            matches.append({
                'type': 'addition',
                'dst_idx': di,
                'dst': d,
            })
    
    # Unmatched src = deletions
    matched_src = {m['src_idx'] for m in matches if 'src_idx' in m}
    for si, s in enumerate(src_fg):
        if si not in matched_src:
            matches.append({
                'type': 'deletion',
                'src_idx': si,
                'src': s,
            })
    
    return matches


# ─── 操作パターンの学習 ───

def learn_ops(train_pairs: List[Tuple[List[List[int]], List[List[int]]]]) -> Optional[Dict]:
    """全train例から一貫した操作パターンを学習"""
    
    all_match_sets = []
    
    for inp, out in train_pairs:
        src_chunks, bg_s, runs_s = decompose(inp)
        dst_chunks, bg_d, runs_d = decompose(out)
        matches = match_chunks(src_chunks, dst_chunks, runs_s, runs_d)
        all_match_sets.append({
            'matches': matches,
            'src_chunks': src_chunks,
            'dst_chunks': dst_chunks,
            'bg_s': bg_s, 'bg_d': bg_d,
            'runs_s': runs_s, 'runs_d': runs_d,
            'h_in': len(inp), 'w_in': len(inp[0]),
            'h_out': len(out), 'w_out': len(out[0]),
        })
    
    # 一貫パターン検出
    
    # Pattern A: 全オブジェクトが同じ量だけ移動
    consistent_translate = _check_consistent_translate(all_match_sets)
    if consistent_translate:
        return {'op': 'translate', **consistent_translate}
    
    # Pattern B: 全オブジェクトが同じ色変換
    consistent_recolor = _check_consistent_recolor(all_match_sets)
    if consistent_recolor:
        return {'op': 'recolor', **consistent_recolor}
    
    # Pattern C: 条件付き操作（色/サイズ/位置による分岐）
    conditional = _check_conditional_ops(all_match_sets)
    if conditional:
        return conditional
    
    # Pattern D: 塊のソート・再配置
    rearrange = _check_rearrangement(all_match_sets)
    if rearrange:
        return rearrange
    
    return None


def _check_consistent_translate(match_sets) -> Optional[Dict]:
    """全例で同じ移動量"""
    all_deltas = []
    for ms in match_sets:
        deltas = set()
        for m in ms['matches']:
            if m['type'] in ('translate', 'identity'):
                deltas.add(m['delta'])
        if len(deltas) == 1:
            all_deltas.append(deltas.pop())
        elif len(deltas) == 0:
            return None
        else:
            return None
    
    if not all_deltas:
        return None
    
    if len(set(all_deltas)) == 1:
        return {'delta': all_deltas[0], 'match_sets': match_sets}
    
    return None


def _check_consistent_recolor(match_sets) -> Optional[Dict]:
    """全例で同じ色変換マップ"""
    color_maps = []
    for ms in match_sets:
        cmap = {}
        for m in ms['matches']:
            if 'color_change' in m:
                f, t = m['color_change']
                if f in cmap and cmap[f] != t:
                    return None
                cmap[f] = t
        color_maps.append(cmap)
    
    if not color_maps or not color_maps[0]:
        return None
    
    # Check consistency
    ref = color_maps[0]
    for cm in color_maps[1:]:
        for k, v in cm.items():
            if k in ref and ref[k] != v:
                return None
        ref.update(cm)
    
    return {'color_map': ref, 'match_sets': match_sets}


def _check_conditional_ops(match_sets) -> Optional[Dict]:
    """条件付き操作: 塊の属性で異なる操作"""
    
    # Strategy 1: 色→操作マップ
    color_ops = _learn_color_ops(match_sets)
    if color_ops:
        return color_ops
    
    # Strategy 2: per-object translate by shape
    per_obj = _learn_per_shape_translate(match_sets)
    if per_obj:
        return per_obj
    
    return None


def _learn_color_ops(match_sets) -> Optional[Dict]:
    """色ごとの操作ルールを学習"""
    color_rules = {}
    
    for ms in match_sets:
        per_color = {}
        for m in ms['matches']:
            if 'src' not in m:
                continue
            per_color.setdefault(m['src'].color, []).append(m)
        
        for color, ops in per_color.items():
            types = set(m['type'] for m in ops)
            if len(types) != 1:
                rule = None
            else:
                t = types.pop()
                if t == 'identity':
                    rule = ('keep',)
                elif t == 'translate':
                    deltas = set(m['delta'] for m in ops)
                    rule = ('translate', deltas.pop()) if len(deltas) == 1 else None
                elif t in ('recolor', 'recolor+translate'):
                    recolors = set(m.get('color_change', (0,0)) for m in ops)
                    rule = ('recolor', recolors.pop()[1]) if len(recolors) == 1 else None
                elif t == 'deletion':
                    rule = ('delete',)
                else:
                    rule = None
            
            if rule is None:
                continue
            if color in color_rules and color_rules[color] != rule:
                color_rules[color] = None
            else:
                color_rules[color] = rule
    
    color_rules = {k: v for k, v in color_rules.items() if v is not None}
    if not color_rules:
        return None
    return {'op': 'color_conditional', 'color_rules': color_rules, 'match_sets': match_sets}


def _learn_per_shape_translate(match_sets) -> Optional[Dict]:
    """形状シグネチャ → 移動量"""
    shape_to_delta = {}
    
    for ms in match_sets:
        for m in ms['matches']:
            if m['type'] not in ('translate', 'identity', 'recolor+translate'):
                continue
            if 'src' not in m:
                continue
            sig = m['src'].shape_sig
            delta = m.get('delta', (0, 0))
            
            if sig in shape_to_delta:
                if shape_to_delta[sig] != delta:
                    return None
            else:
                shape_to_delta[sig] = delta
    
    if not shape_to_delta:
        return None
    return {'op': 'per_shape_translate', 'shape_to_delta': shape_to_delta, 'match_sets': match_sets}


def _check_rearrangement(match_sets) -> Optional[Dict]:
    """塊のソート・再配置パターン"""
    # e.g., 色でソートして再配置
    
    for ms in match_sets:
        src_fg = [ch for ch in ms['src_chunks'] if not ch.is_bg]
        dst_fg = [ch for ch in ms['dst_chunks'] if not ch.is_bg]
        
        if len(src_fg) != len(dst_fg):
            return None
        
        # Check if same shapes, just rearranged
        src_sigs = sorted([ch.shape_sig for ch in src_fg])
        dst_sigs = sorted([ch.shape_sig for ch in dst_fg])
        
        if src_sigs != dst_sigs:
            return None
    
    # All examples have same shapes rearranged — detect sorting rule
    # TODO: implement sorting rule detection
    
    return None


# ─── 操作の適用 ───

def apply_ops(inp: List[List[int]], ops: Dict) -> Optional[List[List[int]]]:
    """学習した操作を適用"""
    chunks, bg, runs = decompose(inp)
    h, w = len(inp), len(inp[0])
    fg = [ch for ch in chunks if not ch.is_bg]
    
    if ops['op'] == 'translate':
        dr, dc = ops['delta']
        new_chunks = []
        for ch in fg:
            new_chunks.append(ch.translate(dr, dc))
        return recompose(new_chunks, h, w, bg)
    
    elif ops['op'] == 'recolor':
        cmap = ops['color_map']
        new_chunks = []
        for ch in fg:
            new_color = cmap.get(ch.color, ch.color)
            new_chunks.append(ch.recolor(new_color))
        return recompose(new_chunks, h, w, bg)
    
    elif ops['op'] == 'color_conditional':
        color_rules = ops['color_rules']
        new_chunks = []
        for ch in fg:
            rule = color_rules.get(ch.color)
            if rule is None:
                new_chunks.append(ch)  # keep
            elif rule[0] == 'keep':
                new_chunks.append(ch)
            elif rule[0] == 'translate':
                dr, dc = rule[1]
                new_chunks.append(ch.translate(dr, dc))
            elif rule[0] == 'recolor':
                new_chunks.append(ch.recolor(rule[1]))
            elif rule[0] == 'delete':
                pass  # don't add
            else:
                new_chunks.append(ch)
        return recompose(new_chunks, h, w, bg)
    
    elif ops['op'] == 'per_shape_translate':
        shape_to_delta = ops['shape_to_delta']
        new_chunks = []
        for ch in fg:
            sig = ch.shape_sig
            delta = shape_to_delta.get(sig)
            if delta:
                new_chunks.append(ch.translate(delta[0], delta[1]))
            else:
                new_chunks.append(ch)
        return recompose(new_chunks, h, w, bg)
    
    return None


# ─── メインソルバー ───

def cross6_ops_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """6軸Cross塊操作でタスクを解く"""
    ops = learn_ops(train_pairs)
    if ops is None:
        return None
    
    # Verify on train
    for inp, out in train_pairs:
        pred = apply_ops(inp, ops)
        if pred is None or not grid_eq(pred, out):
            return None
    
    return apply_ops(test_input, ops)


# ─── セル単位ソルバー (v1/v2) も統合 ───

from arc.cross6axis import cross6_solve
from arc.cross6axis_v2 import cross6v2_solve
from arc.cross6_fill import cross6_fill_solve
from arc.cross6_fill_v2 import cross6_fill_v2_solve


def _loo_validate(solver, train_pairs) -> bool:
    """Leave-one-out cross-validation"""
    if len(train_pairs) < 3:
        return True  # not enough for LOO
    for i in range(len(train_pairs)):
        loo_train = train_pairs[:i] + train_pairs[i+1:]
        loo_in, loo_out = train_pairs[i]
        pred = solver(loo_train, loo_in)
        if pred is None or not grid_eq(pred, loo_out):
            return False
    return True


def cross6_combined_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """全Cross6ソルバーを統合
    
    信頼度順: ops > fill_v2(LOO付き) > cell_v1 > cell_v2
    fill_v1は偽陽性が多いので除外
    """
    # 1. 塊操作 (最も信頼性が高い)
    r = cross6_ops_solve(train_pairs, test_input)
    if r is not None:
        return r
    
    # 2. Fill v2 (LOO付き, 汎化性が高い)
    r = cross6_fill_v2_solve(train_pairs, test_input)
    if r is not None:
        return r
    
    # 3. セル単位 exact (v1) — 汎化しにくいが正確
    r = cross6_solve(train_pairs, test_input)
    if r is not None:
        return r
    
    # 4. セル単位 階層 (v2)
    r = cross6v2_solve(train_pairs, test_input)
    if r is not None:
        return r
    
    return None


if __name__ == "__main__":
    import sys, json
    
    if len(sys.argv) < 2:
        print("Usage: python3 -m arc.cross6_ops <task.json>")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        task = json.load(f)
    
    train_pairs = [(ex['input'], ex['output']) for ex in task['train']]
    test_input = task['test'][0]['input']
    test_output = task['test'][0].get('output')
    
    # Analyze
    for i, (inp, out) in enumerate(train_pairs[:3]):
        src_chunks, bg_s, runs_s = decompose(inp)
        dst_chunks, bg_d, runs_d = decompose(out)
        matches = match_chunks(src_chunks, dst_chunks, runs_s, runs_d)
        
        print(f"\n=== Example {i} ===")
        print(f"Src: {len([c for c in src_chunks if not c.is_bg])} fg chunks")
        print(f"Dst: {len([c for c in dst_chunks if not c.is_bg])} fg chunks")
        for m in matches:
            if m['type'] == 'identity':
                print(f"  = {m['src']} stays")
            elif m['type'] == 'translate':
                print(f"  → {m['src']} moves by {m['delta']}")
            elif m['type'] == 'recolor':
                print(f"  🎨 {m['src']} → color {m['color_change']}")
            elif m['type'] == 'deletion':
                print(f"  ✗ {m['src']} deleted")
            elif m['type'] == 'addition':
                print(f"  + {m['dst']} added")
            else:
                print(f"  ? {m['type']}: {m.get('src','?')} → {m.get('dst','?')}")
    
    # Solve
    result = cross6_combined_solve(train_pairs, test_input)
    if result:
        if test_output and grid_eq(result, test_output):
            print("\n✅ SOLVED!")
        else:
            print("\n⚠️ Solution (unverified)")
            for row in result:
                print(' '.join(str(c) for c in row))
    else:
        print("\n✗ No solution")
