"""
mine_trace_from_shard.py
=========================
全15シャードから Math 知識を抽出

Strategy:
  1. HLE問題テキスト → 簡易埋め込み → expert routing
  2. ExpertLoaderで対応するExpertを特定
  3. Expertの知識方向 → piece_db のドメインマッチング
  4. 有望なpiece + expert の組み合わせを抽出

Agent F - Part 2: GGUF Knowledge Extraction
"""
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# workspace パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

# ExpertLoaderをインポート
from knowledge.expert_loader import build_expert_loader, ExpertLoader

# パス設定
SHARD_DIR = Path.home() / "avh_math/avh_math/downloads/v3_q8_0/Q8_0"
SVD_DIR = Path.home() / "avh_math/avh_math/db/moe_sparse_cross_600b_real"
INDEX_CACHE = Path.home() / "avh_math/avh_math/db/v3_tensor_index.jsonl"

CONCEPT_DIRS = SVD_DIR / "concept_dirs.npy"
EMBED_TOKENS = SVD_DIR / "embed_tokens.npy"
TOKENIZER_PATH = SVD_DIR / "tokenizer.json"

PIECE_DB_PATH = Path(__file__).parent.parent / "pieces/piece_db.jsonl"
HLE_2500_PATH = Path(__file__).parent.parent / "hle_2500_phase5h_final.json"


# ============================
# 簡易テキスト埋め込み
# ============================

def load_tokenizer_vocab():
    """簡易版: tokenizer.jsonから語彙を取得"""
    if not TOKENIZER_PATH.exists():
        return None

    with open(TOKENIZER_PATH) as f:
        data = json.load(f)

    vocab = data.get("model", {}).get("vocab", {})
    return vocab


def simple_text_to_vec(text: str, embed_tokens: np.ndarray, vocab: dict = None) -> np.ndarray:
    """
    簡易テキスト埋め込み

    Strategy:
      - 文字n-gramでハッシュしてトークンIDに対応
      - embed_tokensから平均して正規化
    """
    if vocab is None or embed_tokens is None:
        # フォールバック: 文字ハッシュで直接埋め込み生成
        vec = np.zeros(7168, dtype=np.float32)
        for i, ch in enumerate(text[:200]):
            idx = (ord(ch) * (i+1)) % 7168
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec

    # tokenizer語彙を使用
    tokens = text.lower().split()[:50]  # 最初の50単語
    token_vecs = []

    for tok in tokens:
        if tok in vocab:
            tok_id = vocab[tok]
            if tok_id < len(embed_tokens):
                token_vecs.append(embed_tokens[tok_id])

    if not token_vecs:
        # フォールバック
        return simple_text_to_vec(text, None, None)

    # 平均して正規化
    vec = np.mean(token_vecs, axis=0)
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm

    return vec.astype(np.float32)


# ============================
# Piece DB ロード
# ============================

def load_piece_db():
    """piece_db.jsonlを読み込み"""
    pieces = []
    with open(PIECE_DB_PATH) as f:
        for line in f:
            pieces.append(json.loads(line))
    return pieces


# ============================
# HLE問題ロード
# ============================

def load_hle_problems():
    """HLE問題サンプルを読み込み"""
    with open(HLE_2500_PATH) as f:
        data = json.load(f)

    problems = data.get("problems", [])
    return problems[:100]  # 最初の100問でテスト


# ============================
# Expert → Piece マッチング
# ============================

def match_expert_to_pieces(
    expert_layer: int,
    expert_id: int,
    expert_score: float,
    pieces: List[Dict],
    loader: ExpertLoader,
) -> List[Tuple[str, float]]:
    """
    Expertの知識方向からpiece候補を特定

    Returns:
        [(piece_id, match_score), ...]
    """
    # Expert の concept_dirs を取得
    flat_id = (expert_layer - 3) * 256 + expert_id
    if flat_id >= len(loader.concept_dirs):
        return []

    expert_dirs = loader.concept_dirs[flat_id]  # (4, 7168)
    primary_dir = expert_dirs[0]  # 第1主方向

    # 各pieceのキーワードと照合
    matches = []

    for piece in pieces:
        piece_id = piece.get("piece_id", "")
        tags = piece.get("tags", [])
        desc = piece.get("description", "")

        # piece のキーワードから埋め込み生成
        piece_text = f"{piece_id} {' '.join(tags)} {desc}"
        piece_vec = simple_text_to_vec(piece_text, loader.embed_tokens)

        # cosine similarity
        similarity = float(primary_dir @ piece_vec)

        if similarity > 0.1:  # 閾値
            matches.append((piece_id, similarity * expert_score))

    # スコア順にソート
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:5]  # 上位5件


# ============================
# メイン処理
# ============================

def main():
    print("=" * 70)
    print("Agent F - Part 2: GGUF Knowledge Extraction")
    print("=" * 70)

    # ExpertLoader構築
    print("\n[1] Building ExpertLoader...")
    loader = build_expert_loader(
        shard_dir=str(SHARD_DIR),
        concept_dirs_path=str(CONCEPT_DIRS),
        embed_tokens_path=str(EMBED_TOKENS),
        index_cache_path=str(INDEX_CACHE),
    )
    print("✓ ExpertLoader ready")

    # tokenizer読み込み
    print("\n[2] Loading tokenizer...")
    vocab = load_tokenizer_vocab()
    if vocab:
        print(f"✓ Loaded {len(vocab)} tokens")
    else:
        print("⚠ Tokenizer not available, using fallback")

    # embed_tokens 読み込み
    print("\n[3] Loading embed_tokens...")
    if loader.embed_tokens is not None:
        print(f"✓ embed_tokens shape: {loader.embed_tokens.shape}")
    else:
        print("⚠ embed_tokens not available, using fallback")

    # Piece DB 読み込み
    print("\n[4] Loading piece database...")
    pieces = load_piece_db()
    print(f"✓ Loaded {len(pieces)} pieces")

    # HLE問題読み込み
    print("\n[5] Loading HLE problems...")
    problems = load_hle_problems()
    print(f"✓ Loaded {len(problems)} problems")

    # 知識抽出
    print("\n[6] Extracting knowledge from HLE problems...")
    print("-" * 70)

    expert_piece_map = defaultdict(list)  # expert → [(piece_id, count)]
    piece_expert_map = defaultdict(list)  # piece_id → [(layer, expert_id, score)]

    for i, prob in enumerate(problems[:20]):  # 最初の20問でテスト
        question = prob.get("question", "")
        answer_type = prob.get("answer", {}).get("type", "")

        if not question:
            continue

        # テキスト → 埋め込み
        query_vec = simple_text_to_vec(question, loader.embed_tokens, vocab)

        # Expert選択
        experts = loader.query_to_experts(query_vec, top_k=3)

        print(f"\n[Q{i+1}] {question[:60]}...")
        print(f"  Answer type: {answer_type}")
        print(f"  Top experts:")

        for layer, expert_id, score in experts:
            print(f"    Layer {layer}, Expert {expert_id}: {score:.4f}")

            # Expert → Piece マッチング
            matches = match_expert_to_pieces(layer, expert_id, score, pieces, loader)

            if matches:
                print(f"      → Matched pieces:")
                for piece_id, match_score in matches[:3]:
                    print(f"         {piece_id}: {match_score:.4f}")
                    expert_key = f"L{layer}_E{expert_id}"
                    expert_piece_map[expert_key].append(piece_id)
                    piece_expert_map[piece_id].append((layer, expert_id, match_score))

    # 統計
    print("\n" + "=" * 70)
    print("Knowledge Extraction Summary")
    print("=" * 70)

    print(f"\nTotal experts activated: {len(expert_piece_map)}")
    print(f"Total pieces matched: {len(piece_expert_map)}")

    print("\nTop 10 pieces by expert activation:")
    piece_counts = Counter()
    for piece_id, expert_list in piece_expert_map.items():
        piece_counts[piece_id] = len(expert_list)

    for piece_id, count in piece_counts.most_common(10):
        print(f"  {piece_id}: {count} expert activations")

    print("\nTop 5 experts by piece coverage:")
    expert_counts = Counter()
    for expert_key, piece_list in expert_piece_map.items():
        expert_counts[expert_key] = len(set(piece_list))

    for expert_key, count in expert_counts.most_common(5):
        print(f"  {expert_key}: covers {count} unique pieces")

    # 結果保存
    output_path = Path(__file__).parent.parent / "knowledge_extraction_results.json"
    results = {
        "expert_piece_map": {k: list(set(v)) for k, v in expert_piece_map.items()},
        "piece_expert_map": {k: v for k, v in piece_expert_map.items()},
        "piece_activation_counts": dict(piece_counts),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {output_path}")
    print("\n✅ Knowledge extraction complete!")


if __name__ == "__main__":
    main()
