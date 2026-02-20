#!/usr/bin/env python3
"""
H100資産（600B DeepSeek-R1）からの知識抽出 v2
問題点：全トークン平均 → ベクトルが「溶ける」
解決策：重み付き投影 + Expert方向への射影強度
"""
import numpy as np
import json
from typing import List, Tuple, Dict
from pathlib import Path

# H100資産のパス
ASSET_DIR = Path("/Users/motonishikoudai/.openclaw/workspace/verantyx_v6/h100_assets")

class ConceptExtractorV2:
    """600B資産からの知識抽出（重み付き投影版）"""

    def __init__(self):
        # H100資産をロード
        self.concept_dirs = np.load(ASSET_DIR / "concept_dirs.npy")  # (15104, 4, 7168)
        self.embed_tokens = np.load(ASSET_DIR / "embed_tokens.npy")  # (129280, 7168)

        with open(ASSET_DIR / "expert_vocab_domains.json") as f:
            self.expert_domains = json.load(f)

        # Expertのドメイン情報
        self.num_experts = self.concept_dirs.shape[0]
        print(f"[ConceptExtractor] Loaded {self.num_experts} experts, {self.embed_tokens.shape[0]} tokens")

    def extract_weighted_query(self, token_ids: List[int], important_tokens: List[int] = None) -> np.ndarray:
        """
        重み付きクエリベクトル生成

        Args:
            token_ids: 全トークンID
            important_tokens: 重要トークンのインデックス（TF-IDF/キーワード抽出で取得）

        Returns:
            重み付きクエリベクトル (7168,)
        """
        # トークン埋め込み取得
        vecs = self.embed_tokens[token_ids]  # (N, 7168)

        # 重み計算
        if important_tokens is None:
            # 均等重み（フォールバック）
            weights = np.ones(len(token_ids))
        else:
            # 重要トークンに高重み
            weights = np.ones(len(token_ids))
            weights[important_tokens] = 3.0  # 重要トークンは3倍

        # 正規化
        weights = weights / weights.sum()

        # 重み付き平均
        query = (vecs.T @ weights)  # (7168,)

        # L2正規化
        query = query / (np.linalg.norm(query) + 1e-8)

        return query

    def project_to_expert_directions(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Expert方向への射影強度を計算

        Args:
            query: クエリベクトル (7168,)
            top_k: 上位K個のExpertを返す

        Returns:
            [(expert_id, projection_strength, domain), ...]
        """
        # 各Expertの方向ベクトル（SVD top-4）への射影
        scores = []

        for expert_id in range(self.num_experts):
            # SVD top-4方向ベクトル (4, 7168)
            dirs = self.concept_dirs[expert_id]  # (4, 7168)

            # クエリをExpert部分空間に射影
            # projection = dirs @ query → (4,)
            projection = dirs @ query  # (4,)

            # 射影強度（L2ノルム）
            strength = np.linalg.norm(projection)

            # ドメイン取得
            domain = self.expert_domains.get(str(expert_id), "Unknown")

            scores.append((expert_id, strength, domain))

        # 射影強度でソート
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def extract_knowledge(self, question: str, tokenizer_encode_fn, top_k: int = 5) -> Dict:
        """
        問題文から知識を抽出

        Args:
            question: 問題文
            tokenizer_encode_fn: トークナイザー関数 question -> List[int]
            top_k: 上位K個のExpert

        Returns:
            {
                "relevant_experts": [(expert_id, strength, domain), ...],
                "primary_domain": "...",
                "knowledge_confidence": float
            }
        """
        # トークン化
        token_ids = tokenizer_encode_fn(question)

        # 重要トークン抽出（簡易版：疑問詞/固有名詞/数式トークンを重視）
        # TODO: TF-IDF/キーワード抽出を実装
        important_tokens = self._extract_important_tokens(token_ids, question)

        # 重み付きクエリ生成
        query = self.extract_weighted_query(token_ids, important_tokens)

        # Expert射影
        experts = self.project_to_expert_directions(query, top_k)

        # 主要ドメイン判定
        domain_votes = {}
        for _, strength, domain in experts:
            domain_votes[domain] = domain_votes.get(domain, 0) + strength

        primary_domain = max(domain_votes.items(), key=lambda x: x[1])[0]

        # 知識信頼度（top-1とtop-2の射影強度差）
        if len(experts) >= 2:
            confidence = (experts[0][1] - experts[1][1]) / (experts[0][1] + 1e-8)
        else:
            confidence = 0.0

        return {
            "relevant_experts": experts,
            "primary_domain": primary_domain,
            "knowledge_confidence": confidence
        }

    def _extract_important_tokens(self, token_ids: List[int], question: str) -> List[int]:
        """
        重要トークンのインデックスを抽出（簡易版）

        TODO: TF-IDF/キーワード抽出を実装
        """
        # 簡易実装：数式記号・固有名詞っぽいトークン
        important = []

        # 数式記号トークンの判定（ASCII範囲の記号）
        # 本来はトークナイザー依存だが、簡易的に判定
        for i, tid in enumerate(token_ids):
            # 例：トークンIDが特定範囲（数式記号）なら重要
            # これは仮の実装
            if tid < 1000:  # 特殊トークン（例）
                important.append(i)

        return important


def demo():
    """デモ実行"""
    extractor = ConceptExtractorV2()

    # ダミートークナイザー（実際はDeepSeekのtokenizerを使用）
    def dummy_tokenizer(text: str) -> List[int]:
        # 簡易：文字列を適当にトークンIDに変換
        return [ord(c) % 129280 for c in text[:100]]

    # テスト問題
    questions = [
        "What is C(10,3)?",
        "A patient presents with GERD and dyspnea. What is the diagnosis?",
        "Prove that the chromatic number of the Petersen graph is 3."
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print(f"{'='*60}")

        result = extractor.extract_knowledge(q, dummy_tokenizer, top_k=3)

        print(f"Primary Domain: {result['primary_domain']}")
        print(f"Knowledge Confidence: {result['knowledge_confidence']:.3f}")
        print(f"\nTop-3 Relevant Experts:")
        for expert_id, strength, domain in result['relevant_experts']:
            print(f"  Expert {expert_id:5d} | {domain:20s} | Strength: {strength:.4f}")


if __name__ == "__main__":
    demo()
