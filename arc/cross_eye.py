"""
arc/cross_eye.py — Crossの瞳: 色を人間のように知覚する

AIは色を0-9の数字として見る。人間は:
- 赤が「飛び出して」見える
- 青が「静かに沈んで」見える  
- 黄色が「明るく光って」見える
- 同じ色が「群れ」に見える
- コントラストで「境界」が見える
- 派手な色が「主役」、地味な色が「脇役」

=== ARC 10色の知覚マップ ===
0: 黒 — 闇、背景、無、空虚
1: 青 — 冷たい、深い、静か、空・海
2: 赤 — 熱い、危険、目立つ、血・火
3: 緑 — 自然、安全、中間、葉・草
4: 黄 — 明るい、警告、光、太陽
5: 灰 — 地味、中性、壁、石
6: マゼンタ — 華やか、特別、花
7: オレンジ — 暖かい、活発、果実
8: 水色 — 涼しい、軽い、氷・空
9: 茶 — 重い、古い、土・木
"""

import numpy as np
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label


# ══════════════════════════════════════════════════════════════
# 色の知覚属性（人間の目が感じるもの）
# ══════════════════════════════════════════════════════════════

# 明度 (0=暗い, 1=明るい)
BRIGHTNESS = {0: 0.0, 1: 0.3, 2: 0.5, 3: 0.45, 4: 0.9, 
              5: 0.5, 6: 0.6, 7: 0.7, 8: 0.7, 9: 0.35}

# 暖かさ (-1=冷, 0=中性, 1=暖)
WARMTH = {0: 0.0, 1: -0.8, 2: 1.0, 3: -0.3, 4: 0.8,
          5: 0.0, 6: 0.5, 7: 0.9, 8: -0.7, 9: 0.4}

# 目立ち度 (0=地味, 1=派手)
SALIENCE = {0: 0.0, 1: 0.4, 2: 0.9, 3: 0.5, 4: 0.85,
            5: 0.15, 6: 0.75, 7: 0.7, 8: 0.5, 9: 0.3}

# 重さ (0=軽い, 1=重い)
WEIGHT = {0: 0.1, 1: 0.5, 2: 0.7, 3: 0.4, 4: 0.2,
          5: 0.6, 6: 0.3, 7: 0.4, 8: 0.15, 9: 0.8}

# 色相 (色相環上の位置, 0-360度的な、0=無彩色)
HUE = {0: -1, 1: 240, 2: 0, 3: 120, 4: 60,
       5: -1, 6: 300, 7: 30, 8: 200, 9: 20}

# 彩度 (0=無彩色, 1=鮮やか)
SATURATION = {0: 0.0, 1: 0.7, 2: 0.9, 3: 0.6, 4: 0.8,
              5: 0.05, 6: 0.8, 7: 0.7, 8: 0.5, 9: 0.4}

# RGB近似 (可視化・距離計算用)
RGB = {
    0: (0, 0, 0),       # 黒
    1: (0, 74, 173),     # 青
    2: (255, 0, 0),      # 赤
    3: (0, 160, 0),      # 緑
    4: (255, 224, 32),    # 黄
    5: (128, 128, 128),   # 灰
    6: (207, 0, 207),     # マゼンタ
    7: (255, 127, 0),     # オレンジ
    8: (0, 200, 255),     # 水色
    9: (139, 69, 19),     # 茶
}

# 色の「役割」（ゲシュタルト的認知）
ROLE_NAMES = {
    'background': '背景',     # 最も多い色 → 「地」
    'frame': '枠',           # 端に多い色 → 構造
    'accent': 'アクセント',    # 少数で目立つ → 注目点
    'fill': '塗り',          # 中間量 → 領域
    'marker': 'マーカー',     # 1-2セルのみ → 指標
    'separator': '仕切り',    # 1行/列を占める → 区切り
}


def color_distance(c1, c2):
    """2色間の知覚的距離（人間の目にとっての「違い」）"""
    if c1 == c2: return 0.0
    
    r1, g1, b1 = RGB.get(c1, (128,128,128))
    r2, g2, b2 = RGB.get(c2, (128,128,128))
    
    # 加重ユークリッド距離（人間の目は緑に敏感）
    dr = (r1-r2) * 0.3
    dg = (g1-g2) * 0.59
    db = (b1-b2) * 0.11
    
    return (dr**2 + dg**2 + db**2) ** 0.5 / 255.0


def color_similarity(c1, c2):
    """2色間の知覚的類似度（0=全然違う, 1=同じ）"""
    return 1.0 - min(color_distance(c1, c2), 1.0)


def color_contrast(c1, c2):
    """2色間のコントラスト（明暗差）"""
    return abs(BRIGHTNESS.get(c1, 0.5) - BRIGHTNESS.get(c2, 0.5))


# ══════════════════════════════════════════════════════════════
# Crossの瞳: グリッドを「見る」
# ══════════════════════════════════════════════════════════════

class CrossEye:
    """人間の目のようにグリッドを知覚する"""
    
    def see(self, grid):
        """グリッドを見て知覚情報を返す"""
        g = np.array(grid)
        h, w = g.shape
        
        perception = {}
        
        # 1. 色の分布（何が「地」で何が「図」か）
        color_counts = Counter(int(v) for v in g.flatten())
        total = h * w
        
        perception['colors'] = {}
        for color, count in color_counts.items():
            ratio = count / total
            perception['colors'][color] = {
                'count': count,
                'ratio': ratio,
                'brightness': BRIGHTNESS.get(color, 0.5),
                'warmth': WARMTH.get(color, 0.0),
                'salience': SALIENCE.get(color, 0.5),
                'weight': WEIGHT.get(color, 0.5),
                'role': self._detect_role(g, color, count, ratio),
            }
        
        # 2. 背景色（最も面積が大きい = 「地」）
        bg = color_counts.most_common(1)[0][0]
        perception['background'] = bg
        
        # 3. 目立つ色（salience × ratio のバランス）
        fg_colors = {c for c in color_counts if c != bg}
        if fg_colors:
            perception['most_salient'] = max(fg_colors, 
                key=lambda c: SALIENCE.get(c, 0.5) * min(color_counts[c] / total * 10, 1.0))
        
        # 4. コントラストマップ（各セルの「境界感」）
        contrast_map = np.zeros((h, w))
        for r in range(h):
            for c in range(w):
                max_contrast = 0
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w:
                        ct = color_contrast(int(g[r,c]), int(g[nr,nc]))
                        max_contrast = max(max_contrast, ct)
                contrast_map[r, c] = max_contrast
        perception['contrast_map'] = contrast_map
        
        # 5. 注目領域（salience × contrastが高い場所）
        salience_map = np.zeros((h, w))
        for r in range(h):
            for c in range(w):
                salience_map[r,c] = SALIENCE.get(int(g[r,c]), 0.5) * (1 + contrast_map[r,c])
        perception['attention_map'] = salience_map
        
        # 6. 色のグループ（ゲシュタルト: 同色は群れ）
        struct8 = np.ones((3,3), dtype=int)
        groups = {}
        for color in fg_colors:
            mask = (g == color).astype(int)
            labeled, n = scipy_label(mask, structure=struct8)
            color_groups = []
            for i in range(1, n+1):
                cells = list(zip(*np.where(labeled == i)))
                r1 = min(r for r,c in cells); c1 = min(c for r,c in cells)
                r2 = max(r for r,c in cells); c2 = max(c for r,c in cells)
                color_groups.append({
                    'cells': cells, 'size': len(cells),
                    'bbox': (r1, c1, r2, c2),
                    'center': ((r1+r2)/2, (c1+c2)/2),
                })
            groups[color] = color_groups
        perception['groups'] = groups
        
        # 7. 空間的特徴（対称性、規則性）
        perception['symmetry'] = {
            'lr': np.array_equal(g, g[:, ::-1]),
            'ud': np.array_equal(g, g[::-1, :]),
            'rot90': h == w and np.array_equal(g, np.rot90(g, 1)),
            'rot180': np.array_equal(g, g[::-1, ::-1]),
        }
        
        # 8. 「暖かい領域」と「冷たい領域」
        warmth_map = np.zeros((h, w))
        for r in range(h):
            for c in range(w):
                warmth_map[r, c] = WARMTH.get(int(g[r,c]), 0.0)
        perception['warmth_map'] = warmth_map
        
        return perception
    
    def _detect_role(self, g, color, count, ratio):
        """色の役割を推定"""
        h, w = g.shape
        
        if ratio > 0.5:
            return 'background'
        
        # 端に多い？ → 枠
        edge_count = 0
        for r in range(h):
            if g[r, 0] == color: edge_count += 1
            if g[r, w-1] == color: edge_count += 1
        for c in range(w):
            if g[0, c] == color: edge_count += 1
            if g[h-1, c] == color: edge_count += 1
        edge_total = 2 * (h + w) - 4
        if edge_count > edge_total * 0.5:
            return 'frame'
        
        # 1-2セルのみ → マーカー
        if count <= 2:
            return 'marker'
        
        # 1行を占める → セパレータ
        for r in range(h):
            if all(g[r, c] == color for c in range(w)):
                return 'separator'
        for c in range(w):
            if all(g[r, c] == color for r in range(h)):
                return 'separator'
        
        if ratio < 0.1:
            return 'accent'
        
        return 'fill'
    
    def compare_perceptions(self, perc_in, perc_out):
        """入力と出力の知覚を比較 → 何が変わった？"""
        changes = {}
        
        # 色の役割変化
        in_roles = {c: info['role'] for c, info in perc_in['colors'].items()}
        out_roles = {c: info['role'] for c, info in perc_out['colors'].items()}
        
        changes['new_colors'] = set(out_roles.keys()) - set(in_roles.keys())
        changes['removed_colors'] = set(in_roles.keys()) - set(out_roles.keys())
        changes['role_changes'] = {c: (in_roles.get(c), out_roles.get(c)) 
                                    for c in set(in_roles) & set(out_roles)
                                    if in_roles.get(c) != out_roles.get(c)}
        
        # 注目点の移動
        changes['bg_changed'] = perc_in['background'] != perc_out['background']
        
        # 対称性の変化
        changes['symmetry_gained'] = {k for k, v in perc_out['symmetry'].items() 
                                       if v and not perc_in['symmetry'].get(k)}
        
        return changes
    
    def perceive_relationship(self, grid_in, grid_out):
        """入出力間の知覚的関係を分析"""
        pi = self.see(grid_in)
        po = self.see(grid_out)
        changes = self.compare_perceptions(pi, po)
        
        relationships = []
        
        # 「対称性が生まれた」
        if changes['symmetry_gained']:
            relationships.append(('symmetry_completion', changes['symmetry_gained']))
        
        # 「新しい色が現れた」
        if changes['new_colors']:
            for nc in changes['new_colors']:
                role = po['colors'][nc]['role']
                relationships.append(('new_color_appeared', nc, role))
        
        # 「背景が変わった」
        if changes['bg_changed']:
            relationships.append(('background_changed', pi['background'], po['background']))
        
        # 「マーカーが消えた → 何かの指標だった」
        for c in changes['removed_colors']:
            if pi['colors'][c]['role'] == 'marker':
                relationships.append(('marker_consumed', c))
        
        # 「アクセントが増えた → 何かが強調された」
        for c, (old_role, new_role) in changes['role_changes'].items():
            if new_role == 'accent' and old_role != 'accent':
                relationships.append(('became_accent', c))
        
        return relationships, pi, po


# ══════════════════════════════════════════════════════════════
# 知覚ベースのソルバー
# ══════════════════════════════════════════════════════════════

def perceptual_solve(train_pairs, test_input):
    """知覚を使って解く"""
    from arc.cross_brain_v2 import grid_eq
    
    eye = CrossEye()
    
    # 全trainの知覚分析
    all_relations = []
    for inp, out in train_pairs:
        rels, pi, po = eye.perceive_relationship(inp, out)
        all_relations.append(rels)
    
    # 共通する関係を抽出
    if not all_relations:
        return None, None
    
    # 「マーカーが消費される」パターン
    marker_consumed = all(any(r[0] == 'marker_consumed' for r in rels) for rels in all_relations)
    if marker_consumed:
        result = _solve_marker_consumed(train_pairs, test_input, eye)
        if result is not None:
            return result, 'marker_consumed'
    
    # 「対称補完」パターン
    sym_complete = all(any(r[0] == 'symmetry_completion' for r in rels) for rels in all_relations)
    if sym_complete:
        result = _solve_symmetry_perceptual(train_pairs, test_input, eye)
        if result is not None:
            return result, 'perceptual_symmetry'
    
    # 「目立つ色がルールを決める」パターン
    result = _solve_salience_rules(train_pairs, test_input, eye)
    if result is not None:
        return result, 'salience_rules'
    
    # 「暖色/寒色の分離」パターン
    result = _solve_warmth_separation(train_pairs, test_input, eye)
    if result is not None:
        return result, 'warmth_separation'
    
    # 「コントラスト境界の操作」パターン
    result = _solve_contrast_boundary(train_pairs, test_input, eye)
    if result is not None:
        return result, 'contrast_boundary'
    
    return None, None


def _solve_marker_consumed(train_pairs, test_input, eye):
    """マーカー(1-2セル)が方向や色の指標として使われるパターン"""
    from arc.cross_brain_v2 import grid_eq, _bg, _objs
    
    # マーカーの色と位置 → 何をするかの対応を学習
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        bg = _bg(gi)
        pi = eye.see(inp)
        
        # マーカーを特定
        markers = []
        for color, info in pi['colors'].items():
            if info['role'] == 'marker':
                markers.append(color)
        
        if not markers:
            return None
    
    return None


def _solve_symmetry_perceptual(train_pairs, test_input, eye):
    """知覚的対称補完: 「不完全だから完成させたい」という感覚"""
    from arc.cross_brain_v2 import grid_eq
    
    # trainから対称の種類を特定
    sym_types = set()
    for inp, out in train_pairs:
        rels, pi, po = eye.perceive_relationship(inp, out)
        for r in rels:
            if r[0] == 'symmetry_completion':
                sym_types.update(r[1])
    
    if not sym_types:
        return None
    
    # 対称補完を適用
    gi = np.array(test_input)
    h, w = gi.shape
    bg = _bg(gi)
    result = gi.copy()
    changed = False
    
    for r in range(h):
        for c in range(w):
            if result[r, c] != bg:
                continue
            
            mirrors = []
            if 'lr' in sym_types: mirrors.append((r, w-1-c))
            if 'ud' in sym_types: mirrors.append((h-1-r, c))
            if 'rot180' in sym_types: mirrors.append((h-1-r, w-1-c))
            if 'rot90' in sym_types and h == w:
                mirrors.extend([(c, h-1-r), (h-1-r, w-1-c), (w-1-c, r)])
            
            for mr, mc in mirrors:
                if 0 <= mr < h and 0 <= mc < w and result[mr, mc] != bg:
                    result[r, c] = result[mr, mc]
                    changed = True
                    break
    
    if not changed:
        return None
    
    # train検証
    for inp2, out2 in train_pairs:
        gi2 = np.array(inp2)
        bg2 = _bg(gi2)
        r2 = gi2.copy()
        for r in range(gi2.shape[0]):
            for c in range(gi2.shape[1]):
                if r2[r,c] != bg2:
                    continue
                mirrors = []
                if 'lr' in sym_types: mirrors.append((r, gi2.shape[1]-1-c))
                if 'ud' in sym_types: mirrors.append((gi2.shape[0]-1-r, c))
                if 'rot180' in sym_types: mirrors.append((gi2.shape[0]-1-r, gi2.shape[1]-1-c))
                for mr, mc in mirrors:
                    if 0<=mr<gi2.shape[0] and 0<=mc<gi2.shape[1] and r2[mr,mc]!=bg2:
                        r2[r,c]=r2[mr,mc]; break
        if not grid_eq(r2.tolist(), out2):
            return None
    
    return result.tolist()

def _bg(g): return int(Counter(np.array(g).flatten()).most_common(1)[0][0])

def _solve_salience_rules(train_pairs, test_input, eye):
    """目立つ色がルールの鍵になるパターン"""
    from arc.cross_brain_v2 import grid_eq
    
    # 各trainで最も目立つ色(マーカー/アクセント)が何を決定しているか
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        bg = _bg(gi)
        pi = eye.see(inp)
        
        # アクセント/マーカー色を特定
        key_colors = [c for c, info in pi['colors'].items() 
                      if info['role'] in ('accent', 'marker') and c != bg]
        
        if not key_colors:
            return None
        
        # key_colorの位置が変換ルールを決めるか
        for kc in key_colors:
            # この色の位置
            positions = [(r, c) for r in range(gi.shape[0]) for c in range(gi.shape[1]) 
                        if gi[r,c] == kc]
            if not positions:
                continue
            
            # 出力でこの色の周辺がどう変わったか
            for r, c in positions:
                if r < go.shape[0] and c < go.shape[1]:
                    out_val = int(go[r, c])
                    if out_val != kc and out_val != bg:
                        pass  # 色が変わった → 何かのルール
        break
    
    return None


def _solve_warmth_separation(train_pairs, test_input, eye):
    """暖色/寒色で分離するパターン"""
    from arc.cross_brain_v2 import grid_eq
    
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        if gi.shape != go.shape:
            return None
        bg = _bg(gi)
        
        # 変更されたセルが暖色→寒色 or 寒色→暖色か
        warm_to_cold = cold_to_warm = 0
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                iv, ov = int(gi[r,c]), int(go[r,c])
                if iv != ov:
                    wi, wo = WARMTH.get(iv, 0), WARMTH.get(ov, 0)
                    if wi > 0 and wo < 0: warm_to_cold += 1
                    elif wi < 0 and wo > 0: cold_to_warm += 1
        
        if warm_to_cold == 0 and cold_to_warm == 0:
            return None
        break
    
    return None


def _solve_contrast_boundary(train_pairs, test_input, eye):
    """コントラスト境界に基づく操作"""
    from arc.cross_brain_v2 import grid_eq
    
    pi = eye.see(test_input)
    contrast_map = pi['contrast_map']
    
    # 高コントラスト位置 = 境界
    threshold = np.percentile(contrast_map, 75)
    boundary_mask = contrast_map > threshold
    
    # trainで境界セルがどう変換されるか
    for inp, out in train_pairs:
        gi, go = np.array(inp), np.array(out)
        if gi.shape != go.shape:
            return None
        
        pi_train = eye.see(inp)
        cm = pi_train['contrast_map']
        thresh = np.percentile(cm, 75)
        
        # 境界セルの変換ルール
        boundary_changes = {}
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                if cm[r, c] > thresh and gi[r,c] != go[r,c]:
                    iv = int(gi[r,c])
                    ov = int(go[r,c])
                    if iv in boundary_changes and boundary_changes[iv] != ov:
                        return None
                    boundary_changes[iv] = ov
        break
    
    return None


if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    from arc.cross_brain_v2 import grid_eq, cross_brain_v2_solve
    
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
        
        # まずcross_brain_v2
        result, name = cross_brain_v2_solve(tp, ti)
        
        # 解けなかったら知覚ベース
        if result is None:
            result, name = perceptual_solve(tp, ti)
        
        if result and to and grid_eq(result, to):
            tag = 'NEW' if tid not in all_e else ''
            solved.append((tid, name, tag))
            print(f'  ✓ {tid} [{name}] {tag}')
    
    total = len(list(data_dir.glob('*.json')))
    new = [t for t,_,tg in solved if tg == 'NEW']
    print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
