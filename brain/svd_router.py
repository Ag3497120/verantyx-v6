"""
Phase A7: 600B SVD Expert活性化ルーター
========================================
DeepSeek V3の15104 Expert × 4 SVD方向ベクトルを使って
ARC問題の「認知カテゴリ」を推定し、V3のプロンプトを最適化する

concept_dirs.npy: (15104, 4, 7168) — 全ExpertのSVD方向
embed_tokens.npy: (129280, 7168) — トークン埋め込み
"""
import os, sys, time
import numpy as np
from pathlib import Path

SVD_DIR = Path(os.path.expanduser("~/avh_math/avh_math/db/moe_sparse_cross_600b_real"))

# グローバルキャッシュ（1.6GB、一度だけロード）
_concept_dirs = None
_embed_tokens = None


def _load_svd():
    global _concept_dirs, _embed_tokens
    if _concept_dirs is None:
        t0 = time.time()
        _concept_dirs = np.load(SVD_DIR / "concept_dirs.npy")  # (15104, 4, 7168)
        _embed_tokens = np.load(SVD_DIR / "embed_tokens.npy")  # (129280, 7168)
        print(f"  [SVD] Loaded in {time.time()-t0:.1f}s: dirs={_concept_dirs.shape} embed={_embed_tokens.shape}", flush=True)
    return _concept_dirs, _embed_tokens


def text_to_embedding(text, embed_tokens):
    """テキスト → トークン埋め込みの平均（簡易版）"""
    # 簡易トークナイズ: ASCII文字単位でembed_tokensのインデックスに変換
    # DeepSeek V3のtokenizerがないのでバイト単位で近似
    indices = [min(b, embed_tokens.shape[0]-1) for b in text.encode('utf-8')[:512]]
    if not indices:
        return np.zeros(embed_tokens.shape[1])
    vecs = embed_tokens[indices]  # (N, 7168)
    return vecs.mean(axis=0)  # (7168,)


def get_expert_activations(text, top_k=20):
    """
    テキスト → 各Expertとのcos sim → top-k Expertを返す
    Returns: [(expert_id, layer, score), ...]
    """
    concept_dirs, embed_tokens = _load_svd()
    
    emb = text_to_embedding(text, embed_tokens)  # (7168,)
    emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
    
    # concept_dirs: (15104, 4, 7168) → 各expertの4方向との内積
    # max over 4 directions per expert
    scores = np.einsum('edk,k->ed', concept_dirs, emb_norm)  # (15104, 4)
    max_scores = scores.max(axis=1)  # (15104,)
    
    # top-k
    top_ids = np.argsort(max_scores)[-top_k:][::-1]
    
    results = []
    for eid in top_ids:
        # eid → (layer, expert_within_layer)
        layer = eid // 256 + 3  # flat_id = (layer-3)*256 + expert_id
        expert = eid % 256
        results.append({
            'expert_id': int(eid),
            'layer': int(layer),
            'expert_in_layer': int(expert),
            'score': float(max_scores[eid]),
        })
    
    return results


def classify_arc_problem(task_description, expert_activations):
    """
    Expert活性化パターンからARC問題の認知カテゴリを推定
    
    認知カテゴリ:
    - spatial_transform: 回転、反転、移動、スケーリング
    - color_operation: 色置換、色マップ、塗りつぶし
    - object_manipulation: オブジェクト抽出、合成、分解
    - pattern_replication: タイリング、周期パターン、スタンプ
    - structural_reasoning: パネル関係、セパレータ、階層構造
    - neighborhood_rule: セル近傍ルール、CA、成長
    """
    # Expert層の分布で分類
    # 浅い層(3-20): 低レベル特徴（色、形状）
    # 中間層(20-40): 構造的特徴（オブジェクト、関係）
    # 深い層(40-62): 高レベル推論（ルール、パターン）
    
    shallow = sum(1 for e in expert_activations if e['layer'] < 20)
    mid = sum(1 for e in expert_activations if 20 <= e['layer'] < 40)
    deep = sum(1 for e in expert_activations if e['layer'] >= 40)
    
    avg_score = np.mean([e['score'] for e in expert_activations]) if expert_activations else 0
    max_layer = max((e['layer'] for e in expert_activations), default=0)
    
    # ヒューリスティック分類
    categories = {}
    
    # 浅い層が強い → 色操作/空間変換
    if shallow > mid and shallow > deep:
        categories['color_operation'] = 0.4
        categories['spatial_transform'] = 0.3
    
    # 中間層が強い → オブジェクト操作/構造推論
    if mid > shallow and mid > deep:
        categories['object_manipulation'] = 0.4
        categories['structural_reasoning'] = 0.3
    
    # 深い層が強い → パターン複製/近傍ルール
    if deep > shallow and deep > mid:
        categories['pattern_replication'] = 0.35
        categories['neighborhood_rule'] = 0.35
    
    # テキストキーワードによる補正
    desc_lower = task_description.lower()
    if any(w in desc_lower for w in ['rotate', 'flip', 'mirror', 'symmetry']):
        categories['spatial_transform'] = categories.get('spatial_transform', 0) + 0.3
    if any(w in desc_lower for w in ['color', 'fill', 'replace', 'swap']):
        categories['color_operation'] = categories.get('color_operation', 0) + 0.3
    if any(w in desc_lower for w in ['object', 'move', 'extract', 'crop']):
        categories['object_manipulation'] = categories.get('object_manipulation', 0) + 0.3
    if any(w in desc_lower for w in ['tile', 'repeat', 'stamp', 'period']):
        categories['pattern_replication'] = categories.get('pattern_replication', 0) + 0.3
    if any(w in desc_lower for w in ['panel', 'separator', 'grid', 'section']):
        categories['structural_reasoning'] = categories.get('structural_reasoning', 0) + 0.3
    if any(w in desc_lower for w in ['neighbor', 'grow', 'cellular', 'spread']):
        categories['neighborhood_rule'] = categories.get('neighborhood_rule', 0) + 0.3
    
    if not categories:
        categories = {'unknown': 1.0}
    
    # 正規化
    total = sum(categories.values())
    categories = {k: round(v/total, 2) for k, v in categories.items()}
    
    # ソート
    sorted_cats = sorted(categories.items(), key=lambda x: -x[1])
    
    return sorted_cats, {
        'layer_dist': {'shallow': shallow, 'mid': mid, 'deep': deep},
        'avg_score': round(float(avg_score), 4),
        'max_layer': max_layer,
    }


def analyze_arc_task_svd(task, structure_analysis=None):
    """
    ARC問題 → 特徴言語化 → SVD Expert分析 → 認知カテゴリ
    """
    # 問題の特徴を言語化
    desc_parts = []
    
    train = task['train']
    in0 = train[0]['input']
    out0 = train[0]['output']
    ih, iw = len(in0), len(in0[0])
    oh, ow = len(out0), len(out0[0])
    
    desc_parts.append(f"ARC puzzle: input {ih}x{iw} output {oh}x{ow}")
    
    if ih == oh and iw == ow:
        desc_parts.append("same size transformation")
    elif oh > ih or ow > iw:
        desc_parts.append("output larger than input, scaling or tiling")
    else:
        desc_parts.append("output smaller, cropping or extraction")
    
    # structure_analysisからの情報を追加
    if structure_analysis:
        if structure_analysis.get('color_map'):
            desc_parts.append(f"color mapping detected: {structure_analysis['color_map']}")
        if structure_analysis.get('symmetries'):
            desc_parts.append(f"symmetry: {structure_analysis['symmetries']}")
        if structure_analysis.get('separators'):
            desc_parts.append(f"separators: {structure_analysis['separators'][:3]}")
        if structure_analysis.get('periodicity'):
            desc_parts.append(f"periodicity: {structure_analysis['periodicity']}")
        if structure_analysis.get('size_ratio'):
            desc_parts.append(f"size ratio: {structure_analysis['size_ratio']}")
        
        diffs = structure_analysis.get('diffs', [])
        if diffs:
            pct = diffs[0].get('pct', 0)
            desc_parts.append(f"{pct}% cells change")
        
        colors = structure_analysis.get('colors', [])
        if colors:
            nc = colors[0].get('new_colors', [])
            rc = colors[0].get('removed_colors', [])
            if nc: desc_parts.append(f"new colors appear: {nc}")
            if rc: desc_parts.append(f"colors removed: {rc}")
    
    description = '. '.join(desc_parts)
    
    # SVD分析
    experts = get_expert_activations(description, top_k=20)
    categories, meta = classify_arc_problem(description, experts)
    
    return {
        'description': description,
        'categories': categories,
        'meta': meta,
        'top_experts': experts[:5],
    }


# === プロンプト強化用のカテゴリ別ガイダンス ===

CATEGORY_GUIDANCE = {
    'spatial_transform': "Focus on geometric operations: rotation (90/180/270°), reflection (horizontal/vertical), translation, scaling. Check if objects move, rotate, or flip between input and output.",
    'color_operation': "Focus on color changes: systematic color replacement, flood fill, recoloring based on position or neighbors. Check color mapping consistency across examples.",
    'object_manipulation': "Focus on individual objects: extraction, movement, merging, splitting, sorting by size/color. Identify objects first, then determine what happens to each.",
    'pattern_replication': "Focus on repetition: tiling, stamping a template, periodic patterns. Check if the output is a repeated/tiled version of something from the input.",
    'structural_reasoning': "Focus on grid structure: panels divided by separators, boolean operations between panels (AND/OR/XOR), hierarchical relationships between grid sections.",
    'neighborhood_rule': "Focus on local rules: each cell's output depends on its neighbors (cellular automaton style). Check for growth, spreading, boundary detection, or neighborhood-based coloring.",
}


def get_svd_prompt_enhancement(svd_result):
    """SVD分析結果からプロンプト強化テキストを生成"""
    if not svd_result:
        return ""
    
    cats = svd_result.get('categories', [])
    meta = svd_result.get('meta', {})
    
    parts = ["\n## 600B Expert Analysis (DeepSeek V3 MoE SVD)"]
    parts.append(f"Layer distribution: {meta.get('layer_dist', {})}")
    
    parts.append("Cognitive categories:")
    for cat, score in cats[:3]:
        parts.append(f"  - **{cat}** ({score:.0%}): {CATEGORY_GUIDANCE.get(cat, '')}")
    
    return '\n'.join(parts)
