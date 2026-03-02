"""
arc/episodic_memory.py — エピソード記憶: 断片的連想ネットワーク

=== kofdaiの洞察 ===
人間の記憶は断片的。フル情報ではなく劣化した断片が連想で結びつく。
"24" → 数字パターン → ナンバープレート → 小豆島の教室
この「断片 → 連想 → 別の断片」のチェーンがパターン認識の本質。

=== 設計 ===

1. 断片（Fragment）: グリッドから抽出した劣化情報
   - 色の断片: 「暖色2つ、寒色1つ」（具体色IDではない）
   - 形の断片: 「L字っぽいもの」（座標ではない）
   - 関係の断片: 「大きいのと小さいのが隣」
   - 変化の断片: 「塗りつぶされた」「移動した」

2. エピソード（Episode）: 断片 + 文脈 + 解法
   - training 1000問の各問題が1エピソード
   - 断片群 + どの操作で解けたか

3. 連想（Association）: 断片間のCross構造リンク
   - 同じ断片を持つエピソード同士がリンク
   - 活性化が伝播する（spreading activation）

4. 想起（Recall）: 新問題 → 断片抽出 → 連想検索 → 解法候補
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set, FrozenSet
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from scipy.ndimage import label as scipy_label
import hashlib


# ──────────────────────────────────────────────────────────────
# 断片（Fragment）: 意図的に劣化させた情報
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Fragment:
    """記憶の断片 — 劣化した特徴"""
    category: str       # 断片のカテゴリ
    key: str           # 断片の識別子（劣化済み）
    
    def __repr__(self):
        return f"<{self.category}:{self.key}>"


def extract_fragments(grid) -> Set[Fragment]:
    """グリッドから断片を抽出（意図的に劣化させる）"""
    g = np.array(grid)
    h, w = g.shape
    frags = set()
    
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    
    # ─── 空間断片: サイズ感 ───
    if h <= 5: frags.add(Fragment('size', 'tiny'))
    elif h <= 10: frags.add(Fragment('size', 'small'))
    elif h <= 20: frags.add(Fragment('size', 'medium'))
    else: frags.add(Fragment('size', 'large'))
    
    if h == w: frags.add(Fragment('shape', 'square'))
    elif h > w: frags.add(Fragment('shape', 'tall'))
    else: frags.add(Fragment('shape', 'wide'))
    
    # ─── 色断片: 何色あるか、どんな感じか ───
    colors = set(int(v) for v in g.flatten())
    n_colors = len(colors)
    
    if n_colors <= 2: frags.add(Fragment('palette', 'simple'))  # 2色
    elif n_colors <= 4: frags.add(Fragment('palette', 'few'))    # 3-4色
    elif n_colors <= 7: frags.add(Fragment('palette', 'several')) # 5-7色
    else: frags.add(Fragment('palette', 'many'))                  # 8+色
    
    # 色の偏り
    color_counts = Counter(g.flatten())
    bg_ratio = color_counts[bg] / g.size
    if bg_ratio > 0.8: frags.add(Fragment('density', 'sparse'))
    elif bg_ratio > 0.5: frags.add(Fragment('density', 'normal'))
    else: frags.add(Fragment('density', 'dense'))
    
    # ─── オブジェクト断片: 何がどうある ───
    mask = (g != bg).astype(int)
    labeled, n_obj = scipy_label(mask)
    
    if n_obj == 0: frags.add(Fragment('objects', 'empty'))
    elif n_obj == 1: frags.add(Fragment('objects', 'single'))
    elif n_obj <= 3: frags.add(Fragment('objects', 'few'))
    elif n_obj <= 8: frags.add(Fragment('objects', 'several'))
    else: frags.add(Fragment('objects', 'many'))
    
    # オブジェクトのサイズ分布
    if n_obj >= 2:
        sizes = []
        for i in range(1, n_obj + 1):
            sizes.append(int((labeled == i).sum()))
        sizes.sort()
        
        if sizes[-1] > sizes[0] * 3:
            frags.add(Fragment('obj_sizes', 'varied'))
        else:
            frags.add(Fragment('obj_sizes', 'uniform'))
        
        # 同じサイズのオブジェクトがあるか
        size_counts = Counter(sizes)
        if max(size_counts.values()) >= 2:
            frags.add(Fragment('obj_pattern', 'same_size_exists'))
    
    # ─── 構造断片: 対称性、周期性、区切り ───
    # 対称性（雑に検出 — 断片的）
    if h > 1 and np.sum(g != g[::-1]) < g.size * 0.1:
        frags.add(Fragment('symmetry', 'vertical'))
    if w > 1 and np.sum(g != g[:,::-1]) < g.size * 0.1:
        frags.add(Fragment('symmetry', 'horizontal'))
    
    # 区切り線
    for r in range(h):
        row_vals = set(int(v) for v in g[r])
        if len(row_vals) == 1 and row_vals.pop() != bg:
            frags.add(Fragment('structure', 'h_separator'))
            break
    
    for c in range(w):
        col_vals = set(int(v) for v in g[:, c])
        if len(col_vals) == 1 and col_vals.pop() != bg:
            frags.add(Fragment('structure', 'v_separator'))
            break
    
    # ─── パターン断片: 繰り返し、規則性 ───
    # 行の繰り返し
    for period in [1, 2, 3]:
        if h > period and h % period == 0:
            periodic = all(np.array_equal(g[r], g[r % period]) for r in range(period, h))
            if periodic:
                frags.add(Fragment('pattern', f'row_period_{period}'))
                break
    
    # ─── 形状断片: オブジェクトの形の「印象」 ───
    if n_obj > 0:
        for i in range(1, min(n_obj + 1, 6)):  # 最大5オブジェクト
            cells = list(zip(*np.where(labeled == i)))
            obj_h = max(r for r, c in cells) - min(r for r, c in cells) + 1
            obj_w = max(c for r, c in cells) - min(c for r, c in cells) + 1
            area = len(cells)
            
            # 形の断片（劣化: 正確な形ではなくアスペクト比と充填率）
            if obj_h == obj_w: shape_vibe = "square"
            elif obj_h > obj_w * 2: shape_vibe = "tall_thin"
            elif obj_w > obj_h * 2: shape_vibe = "wide_thin"
            elif obj_h > obj_w: shape_vibe = "tallish"
            else: shape_vibe = "widish"
            
            fill_ratio = area / (obj_h * obj_w) if obj_h * obj_w else 0
            if fill_ratio > 0.9: fill_vibe = "solid"
            elif fill_ratio > 0.5: fill_vibe = "chunky"
            else: fill_vibe = "sparse"
            
            frags.add(Fragment("obj_shape", f"{shape_vibe}_{fill_vibe}"))
            
            # サイズの断片
            if area <= 3: frags.add(Fragment("obj_size", "dot"))
            elif area <= 9: frags.add(Fragment("obj_size", "small"))
            elif area <= 25: frags.add(Fragment("obj_size", "medium"))
            else: frags.add(Fragment("obj_size", "big"))
    
    # ─── 近傍パターン断片: 局所的な色の配置 ───
    # 「この問題は市松模様っぽい」とか「ストライプっぽい」
    # 1000問を区別するために重要
    adj_same = 0; adj_diff = 0
    for r in range(h):
        for c in range(w):
            if c + 1 < w:
                if g[r, c] == g[r, c+1]: adj_same += 1
                else: adj_diff += 1
            if r + 1 < h:
                if g[r, c] == g[r+1, c]: adj_same += 1
                else: adj_diff += 1
    
    total_adj = adj_same + adj_diff
    if total_adj > 0:
        homogeneity = adj_same / total_adj
        if homogeneity > 0.9: frags.add(Fragment("texture", "uniform"))
        elif homogeneity > 0.7: frags.add(Fragment("texture", "smooth"))
        elif homogeneity > 0.4: frags.add(Fragment("texture", "mixed"))
        else: frags.add(Fragment("texture", "noisy"))
    
    # ─── 境界断片: 端に何があるか ───
    edge_colors = set()
    for c in range(w):
        edge_colors.add(int(g[0, c]))
        edge_colors.add(int(g[h-1, c]))
    for r in range(h):
        edge_colors.add(int(g[r, 0]))
        edge_colors.add(int(g[r, w-1]))
    
    if edge_colors == {bg}:
        frags.add(Fragment("border", "all_bg"))
    elif bg not in edge_colors:
        frags.add(Fragment("border", "no_bg"))
    else:
        frags.add(Fragment("border", "mixed"))
    
    # ─── 「雰囲気」断片: 全体の印象（最も劣化した情報）───
    # これが小豆島の教室の「雰囲気」に相当
    fg_cells = [(r, c) for r in range(h) for c in range(w) if g[r, c] != bg]
    if fg_cells:
        # 重心
        cr = np.mean([r for r, c in fg_cells])
        cc = np.mean([c for r, c in fg_cells])
        if cr < h * 0.3: frags.add(Fragment('vibe', 'top_heavy'))
        elif cr > h * 0.7: frags.add(Fragment('vibe', 'bottom_heavy'))
        else: frags.add(Fragment('vibe', 'centered'))
        
        if cc < w * 0.3: frags.add(Fragment('vibe', 'left_heavy'))
        elif cc > w * 0.7: frags.add(Fragment('vibe', 'right_heavy'))
        
        # 散らばり
        if len(fg_cells) > 1:
            spread = np.std([r for r, c in fg_cells]) + np.std([c for r, c in fg_cells])
            if spread < 2: frags.add(Fragment('vibe', 'clustered'))
            elif spread < 5: frags.add(Fragment('vibe', 'grouped'))
            else: frags.add(Fragment('vibe', 'scattered'))
    
    return frags


def extract_delta_fragments(inp, out) -> Set[Fragment]:
    """入力→出力の変化から断片を抽出"""
    gi = np.array(inp)
    go = np.array(out)
    frags = set()
    
    # サイズ変化
    if gi.shape == go.shape:
        frags.add(Fragment('delta_size', 'same'))
    elif go.shape[0] > gi.shape[0] or go.shape[1] > gi.shape[1]:
        frags.add(Fragment('delta_size', 'grew'))
    else:
        frags.add(Fragment('delta_size', 'shrank'))
    
    if gi.shape != go.shape:
        # サイズ比
        r = go.shape[0] / gi.shape[0] if gi.shape[0] else 1
        c = go.shape[1] / gi.shape[1] if gi.shape[1] else 1
        if abs(r - round(r)) < 0.01 and abs(c - round(c)) < 0.01:
            frags.add(Fragment('delta_scale', f'{int(round(r))}x{int(round(c))}'))
        return frags
    
    # 変化量
    diff = gi != go
    change_ratio = diff.sum() / diff.size if diff.size else 0
    
    if change_ratio == 0:
        frags.add(Fragment('delta', 'none'))
    elif change_ratio < 0.05:
        frags.add(Fragment('delta', 'tiny'))
    elif change_ratio < 0.2:
        frags.add(Fragment('delta', 'small'))
    elif change_ratio < 0.5:
        frags.add(Fragment('delta', 'medium'))
    else:
        frags.add(Fragment('delta', 'major'))
    
    # 変化の種類
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    added = int(((gi == bg) & (go != bg)).sum())
    removed = int(((gi != bg) & (go == bg)).sum())
    recolored = int(((gi != bg) & (go != bg) & diff).sum())
    
    if added > 0 and removed == 0 and recolored == 0:
        frags.add(Fragment('delta_type', 'additive'))  # 足し算
    elif removed > 0 and added == 0 and recolored == 0:
        frags.add(Fragment('delta_type', 'subtractive'))  # 引き算
    elif recolored > 0 and added == 0 and removed == 0:
        frags.add(Fragment('delta_type', 'recolor'))  # 色変え
    elif added > 0 and removed > 0:
        frags.add(Fragment('delta_type', 'mixed'))  # 混合
    
    # 変化のパターン（断片的）
    changed_rows = set(r for r, c in zip(*np.where(diff)))
    changed_cols = set(c for r, c in zip(*np.where(diff)))
    
    if len(changed_rows) == 1:
        frags.add(Fragment('delta_shape', 'single_row'))
    elif len(changed_cols) == 1:
        frags.add(Fragment('delta_shape', 'single_col'))
    elif len(changed_rows) <= 3 and len(changed_cols) <= 3:
        frags.add(Fragment('delta_shape', 'localized'))
    else:
        frags.add(Fragment('delta_shape', 'distributed'))
    
    # 新色が出現するか
    in_colors = set(int(v) for v in gi.flatten())
    out_colors = set(int(v) for v in go.flatten())
    new_colors = out_colors - in_colors
    if new_colors:
        frags.add(Fragment('delta_color', 'new_color'))
    
    return frags


# ──────────────────────────────────────────────────────────────
# エピソード（Episode）: 断片 + 文脈 + 解法
# ──────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """一つの問題の記憶 — 断片の集合 + 解法"""
    task_id: str
    input_fragments: Set[Fragment]
    delta_fragments: Set[Fragment]
    all_fragments: Set[Fragment]
    solution_method: Optional[str] = None
    solution_fragments: Set = field(default_factory=set)  # 解法に関する断片


# ──────────────────────────────────────────────────────────────
# エピソード記憶（EpisodicMemory）: 連想ネットワーク
# ──────────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    断片的連想記憶ネットワーク
    
    Cross構造でルーティング:
    - 断片がノード
    - 共起がエッジ（同じエピソード内の断片同士）
    - 活性化がエッジを伝播（spreading activation）
    """
    
    def __init__(self):
        self.episodes: List[Episode] = []
        # 断片 → エピソードのインデックス
        self.fragment_index: Dict[Fragment, List[int]] = defaultdict(list)
        # 断片の共起（連想リンク）
        self.associations: Dict[Fragment, Counter] = defaultdict(Counter)
    
    def store(self, episode: Episode):
        """エピソードを記憶に格納"""
        idx = len(self.episodes)
        self.episodes.append(episode)
        
        # インデックス更新
        for frag in episode.all_fragments:
            self.fragment_index[frag].append(idx)
        
        # 共起（連想リンク）を構築
        frags = list(episode.all_fragments)
        for i, f1 in enumerate(frags):
            for f2 in frags[i+1:]:
                self.associations[f1][f2] += 1
                self.associations[f2][f1] += 1
    
    def recall(self, query_fragments: Set[Fragment], top_k=5) -> List[Tuple[Episode, float]]:
        """
        断片クエリから関連エピソードを想起
        
        spreading activation:
        1. クエリ断片を直接持つエピソードを見つける
        2. そのエピソードの他の断片を通じて間接的に関連するエピソードも活性化
        """
        # Phase 1: 直接マッチ
        episode_scores = Counter()
        
        for frag in query_fragments:
            for idx in self.fragment_index.get(frag, []):
                episode_scores[idx] += 1.0
        
        # Phase 2: 連想による間接活性化（1ホップ）
        # クエリ断片から連想される断片を集める
        associated_frags = Counter()
        for frag in query_fragments:
            for assoc_frag, count in self.associations.get(frag, {}).items():
                if assoc_frag not in query_fragments:
                    associated_frags[assoc_frag] += count * 0.3  # 減衰
        
        # 間接断片でエピソードを活性化
        for frag, weight in associated_frags.most_common(20):
            for idx in self.fragment_index.get(frag, []):
                episode_scores[idx] += weight * 0.5  # さらに減衰
        
        # スコア正規化
        results = []
        for idx, score in episode_scores.most_common(top_k):
            ep = self.episodes[idx]
            # Jaccard的な正規化
            overlap = len(query_fragments & ep.all_fragments)
            union = len(query_fragments | ep.all_fragments)
            normalized = overlap / union if union else 0
            # 生スコアとJaccard の加重平均
            final_score = score * 0.5 + normalized * 10 * 0.5
            results.append((ep, final_score))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    def explain_recall(self, query_fragments, top_k=3) -> str:
        """想起結果を説明"""
        results = self.recall(query_fragments, top_k)
        if not results:
            return "想起: なし"
        
        lines = ["想起されたエピソード:"]
        for ep, score in results:
            shared = query_fragments & ep.all_fragments
            lines.append(f"  {ep.task_id} (score={score:.2f}, method={ep.solution_method})")
            lines.append(f"    共有断片: {', '.join(str(f) for f in sorted(shared, key=str)[:5])}")
        
        return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────
# 記憶の構築: training問題から学習
# ──────────────────────────────────────────────────────────────

def build_episodic_memory(training_dir: str, solution_log: str = None) -> EpisodicMemory:
    """training問題からエピソード記憶を構築"""
    import json, re
    from pathlib import Path
    
    memory = EpisodicMemory()
    task_dir = Path(training_dir)
    
    # 解法ログがあれば読む
    solutions = {}
    if solution_log:
        try:
            with open(solution_log) as f:
                for line in f:
                    m = re.search(r'✓.*?([0-9a-f]{8}).*rule=(.+?)$', line)
                    if m:
                        solutions[m.group(1)] = m.group(2).strip()
        except:
            pass
    
    for tf in sorted(task_dir.glob('*.json')):
        tid = tf.stem
        with open(tf) as f:
            task = json.load(f)
        
        train = [(e['input'], e['output']) for e in task['train']]
        
        # 断片抽出
        input_frags = set()
        delta_frags = set()
        
        for inp, out in train:
            input_frags |= extract_fragments(inp)
            delta_frags |= extract_delta_fragments(inp, out)
        
        # 出力の断片も（劣化して）
        for _, out in train:
            out_frags = extract_fragments(out)
            # 出力断片はプレフィックス付きで
            for f in out_frags:
                input_frags.add(Fragment(f'out_{f.category}', f.key))
        
        all_frags = input_frags | delta_frags
        
        episode = Episode(
            task_id=tid,
            input_fragments=input_frags,
            delta_fragments=delta_frags,
            all_fragments=all_frags,
            solution_method=solutions.get(tid),
        )
        
        memory.store(episode)
    
    return memory


# ──────────────────────────────────────────────────────────────
# 記憶ベースソルバー
# ──────────────────────────────────────────────────────────────

def episodic_solve(memory: EpisodicMemory, train_pairs, test_input) -> Tuple[Optional[list], List[str]]:
    """
    エピソード記憶から類似問題を想起し、その解法を試行
    
    Returns: (prediction, recommended_operations)
    """
    # クエリ断片の抽出
    query_frags = set()
    for inp, out in train_pairs:
        query_frags |= extract_fragments(inp)
        query_frags |= extract_delta_fragments(inp, out)
    query_frags |= extract_fragments(test_input)
    
    # 想起
    recalled = memory.recall(query_frags, top_k=10)
    
    # 想起されたエピソードの解法を集計
    method_votes = Counter()
    for ep, score in recalled:
        if ep.solution_method:
            method_votes[ep.solution_method] += score
    
    # 推薦される操作（解法メソッド名から）
    recommended = [method for method, _ in method_votes.most_common(5)]
    
    return None, recommended


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json, time
    from pathlib import Path
    
    if sys.argv[1] == '--build':
        print("Building episodic memory from training set...")
        t0 = time.time()
        memory = build_episodic_memory(
            '/tmp/arc-agi-2/data/training',
            solution_log='arc_cross_engine_v9.log' if Path('arc_cross_engine_v9.log').exists() else None
        )
        print(f"  Episodes: {len(memory.episodes)}")
        print(f"  Unique fragments: {len(memory.fragment_index)}")
        print(f"  Associations: {sum(len(v) for v in memory.associations.values())}")
        print(f"  Time: {time.time()-t0:.1f}s")
        
        # テスト: 最初のエピソードを想起
        if memory.episodes:
            ep0 = memory.episodes[0]
            print(f"\nTest recall for {ep0.task_id}:")
            print(memory.explain_recall(ep0.all_fragments))
    
    elif sys.argv[1] == '--recall':
        tf = sys.argv[2]
        print("Building memory...")
        memory = build_episodic_memory(
            '/tmp/arc-agi-2/data/training',
            solution_log='arc_cross_engine_v9.log' if Path('arc_cross_engine_v9.log').exists() else None
        )
        
        with open(tf) as f:
            task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti = task['test'][0]['input']
        
        query_frags = set()
        for inp, out in tp:
            query_frags |= extract_fragments(inp)
            query_frags |= extract_delta_fragments(inp, out)
        query_frags |= extract_fragments(ti)
        
        print(f"\nQuery fragments ({len(query_frags)}):")
        for f in sorted(query_frags, key=str):
            print(f"  {f}")
        
        print(f"\n{memory.explain_recall(query_frags, top_k=5)}")
    
    elif sys.argv[1] == '--stats':
        print("Building memory...")
        memory = build_episodic_memory(
            '/tmp/arc-agi-2/data/training',
            solution_log='arc_cross_engine_v9.log' if Path('arc_cross_engine_v9.log').exists() else None
        )
        
        # 断片の分布
        print(f"\nFragment distribution:")
        cat_counts = Counter()
        for frag in memory.fragment_index:
            cat_counts[frag.category] += 1
        for cat, count in cat_counts.most_common():
            print(f"  {cat}: {count} unique fragments")
        
        # 解法の分布
        print(f"\nSolution methods:")
        method_counts = Counter()
        for ep in memory.episodes:
            if ep.solution_method:
                method_counts[ep.solution_method] += 1
        for method, count in method_counts.most_common(10):
            print(f"  {method}: {count}")
