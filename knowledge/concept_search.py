"""
concept_search.py
=================
HLEクエリ → embed_tokens × concept_dirs → ドメインブースト信号

パイプライン:
  1. テキスト → BPEトークン → embed_tokens の平均 → query_vec (7168-dim, L2正規化)
  2. query_vec × concept_dirs (15104 × 4 × 7168) → コサイン類似度
  3. expert毎に最大類似度を取る → Top-k experts
  4. domain 多数決（類似度重み付き） → ドメインスコア dict を返す

設計:
  - embed_tokens: mmap（3.5GB、常時ロードしない）
  - concept_dirs: 起動時に一度だけ全ロード（1.6GB）
  - tokenizer: tokenizers ライブラリ (BPE)

使い方:
  from knowledge.concept_search import ConceptSearcher
  cs = ConceptSearcher()
  scores = cs.search("What is the derivative of sin(x)?", top_k=20)
  # → {"calculus": 0.82, "algebra": 0.11, ...}
"""

import os
import json
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RoutingTrace:
    """
    立体十字ルーティングのトレース情報。
    AuditBundle に埋め込んで透明性を確保する。
    """
    mode: str                           # "keyword_maxpool" | "fulltext_mean"
    anchor_kws: List[str]               # 使用したアンカーキーワード
    top_domains: List[Tuple[str, float]]  # (domain, score) top3
    elapsed_ms: float                   # 検索時間
    cache_hit: bool = False             # キャッシュヒットしたか

DB_DIR = Path("/Users/motonishikoudai/avh_math/avh_math/db/moe_sparse_cross_600b_real")

CONCEPT_DIRS_PATH  = DB_DIR / "concept_dirs.npy"
EMBED_TOKENS_PATH  = DB_DIR / "embed_tokens.npy"
TOKENIZER_PATH     = DB_DIR / "tokenizer.json"
EXPERT_DOMAINS_PATH = DB_DIR / "expert_vocab_domains.json"


class ConceptSearcher:
    """
    embed_tokens × concept_dirs コサイン類似度で
    HLEクエリのドメイン信号を取得する。
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._tokenizer = None
        self._embed_tokens = None   # mmap (129280, 7168)
        self._concept_dirs = None   # (15104, 4, 7168) 全ロード
        self._expert_domains = None # {key: {domain, confidence, top_tokens}}
        self._expert_keys: List[str] = []
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        t0 = time.time()

        # 1. Tokenizer
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

        # 2. embed_tokens: mmap
        self._embed_tokens = np.load(str(EMBED_TOKENS_PATH), mmap_mode='r')
        # shape: (129280, 7168)

        # 3. concept_dirs: 全ロード → 正規化済みキャッシュ（60416, 7168）
        raw = np.load(str(CONCEPT_DIRS_PATH))  # (15104, 4, 7168)
        flat = raw.reshape(-1, 7168).astype(np.float32)
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self._concept_dirs = flat / norms  # (60416, 7168) 正規化済み

        # 4. expert domains
        with open(EXPERT_DOMAINS_PATH) as f:
            dom_data = json.load(f)
        self._expert_domains = dom_data.get("experts", {})

        # expert key の順序（concept_dirs のインデックスと対応）
        # expert_vocab_domains のキー順序がconcept_dirsと一致すると仮定
        # ただし安全のため routing_patterns.json からも確認
        routing_path = DB_DIR / "routing_patterns.json"
        with open(routing_path) as f:
            routing = json.load(f)
        routing_experts = routing.get("experts", {})
        # L{layer}E{expert_id} 形式でソート
        def sort_key(k: str):
            parts = k[1:].split('E')
            return (int(parts[0]), int(parts[1]))
        self._expert_keys = sorted(routing_experts.keys(), key=sort_key)

        elapsed = time.time() - t0
        if self.verbose:
            print(f"[ConceptSearcher] Loaded in {elapsed:.2f}s | "
                  f"experts={len(self._expert_keys)}, "
                  f"concept_dirs={self._concept_dirs.shape}")
        self._loaded = True

    def _encode_query(self, text: str) -> np.ndarray:
        """
        テキスト → query vector (7168-dim, L2正規化)
        BPEトークン → embed_tokens の平均
        """
        encoding = self._tokenizer.encode(text)
        ids = encoding.ids  # List[int]
        # 有効範囲にクリップ（特殊トークンなど）
        vocab_size = self._embed_tokens.shape[0]
        valid_ids = [i for i in ids if 0 <= i < vocab_size]
        if not valid_ids:
            return np.zeros(7168, dtype=np.float32)
        # mmap からバッチ取得
        vecs = self._embed_tokens[valid_ids]  # (n_tokens, 7168)
        query = vecs.mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query /= norm
        return query

    def _build_idf_weights(self) -> Dict[str, float]:
        """
        ドメイン別の inverse frequency weight を計算。
        少ないドメイン（logic: 61) を大きくウェイト、
        多いドメイン（general: 4420）を小さくウェイト。
        IDF = log(total / count)
        """
        from math import log
        counts: Dict[str, int] = {}
        for key in self._expert_keys:
            dom_info = self._expert_domains.get(key, {})
            d = dom_info.get("domain", "general")
            counts[d] = counts.get(d, 0) + 1
        total = sum(counts.values())
        return {d: log(total / max(c, 1)) for d, c in counts.items()}

    def search(
        self,
        text: str,
        top_k: int = 50,
        min_sim: float = 0.0,
        exclude_domains: Optional[List[str]] = None,
        use_idf: bool = True,
    ) -> Dict[str, float]:
        """
        テキストを受け取り、ドメインスコア dict を返す。

        Returns:
            {"calculus": 0.82, "algebra": 0.11, ...}  (スコア合計=1.0)
        """
        self._load()

        if exclude_domains is None:
            # general と multilingual はノイズが多いため除外
            exclude_domains = ["general", "multilingual"]

        # 1. クエリベクトル
        query = self._encode_query(text)

        # 2. concept_dirs との cosine 類似度（正規化済みキャッシュを直接使用）
        sims_flat = self._concept_dirs @ query  # (60416,)
        sims = sims_flat.reshape(15104, 4)  # (15104, 4)

        # 各 expert の最大類似度
        max_sims = sims.max(axis=1)  # (15104,)

        # 3. Top-k experts
        k = min(top_k, len(max_sims))
        top_indices = np.argpartition(max_sims, -k)[-k:]
        top_indices = top_indices[np.argsort(max_sims[top_indices])[::-1]]

        # IDF重み
        idf = self._build_idf_weights() if use_idf else {}

        # 4. ドメイン多数決（類似度 × IDF 重み付き）
        domain_scores: Dict[str, float] = {}
        total_weight = 0.0

        for idx in top_indices:
            sim = float(max_sims[idx])
            if sim < min_sim:
                continue
            if idx >= len(self._expert_keys):
                continue
            key = self._expert_keys[idx]
            dom_info = self._expert_domains.get(key, {})
            domain = dom_info.get("domain", "general")
            if domain in exclude_domains:
                continue
            idf_w = idf.get(domain, 1.0) if use_idf else 1.0
            weight = max(sim, 0.0) * idf_w
            domain_scores[domain] = domain_scores.get(domain, 0.0) + weight
            total_weight += weight

        if total_weight < 1e-8:
            return {}

        # 正規化
        for k in domain_scores:
            domain_scores[k] /= total_weight

        # ソート（降順）
        return dict(sorted(domain_scores.items(), key=lambda x: -x[1]))

    def search_keywords(
        self,
        keywords: List[str],
        top_k: int = 50,
        use_idf: bool = True,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Decomposerのキーワードリストを直接クエリとして使う高精度検索。

        全テキスト平均ではなく、各キーワードを個別にベクトル化して
        max-pooling で統合することで discriminative power を最大化。

        例: ["factorial", "permutation"] → 組合せ論Expertが上位に来る
        vs "How many ways to arrange..." → ノイズで溶ける
        """
        self._load()

        if not keywords:
            return {}

        if exclude_domains is None:
            exclude_domains = {"general", "multilingual"}

        # 各キーワードを個別にエンコード
        per_kw_sims: List[np.ndarray] = []
        for kw in keywords:
            kw_vec = self._encode_query(kw)  # (7168,)
            if np.linalg.norm(kw_vec) < 1e-8:
                continue
            sims_flat = self._concept_dirs @ kw_vec  # (60416,)
            sims = sims_flat.reshape(15104, 4)
            max_sims = sims.max(axis=1)  # (15104,)
            per_kw_sims.append(max_sims)

        if not per_kw_sims:
            return {}

        # Max-pooling: 各expertについて最もマッチするキーワードのスコアを採用
        combined_sims = np.stack(per_kw_sims).max(axis=0)  # (15104,)

        # Top-k experts を取得
        k = min(top_k, len(combined_sims))
        top_indices = np.argpartition(combined_sims, -k)[-k:]
        top_indices = top_indices[np.argsort(combined_sims[top_indices])[::-1]]

        # IDF重み
        idf = self._build_idf_weights() if use_idf else {}

        # ドメイン集計（similarities × IDF 重み付き投票）
        domain_scores: Dict[str, float] = {}
        total_weight = 0.0

        for idx in top_indices:
            sim = float(combined_sims[idx])
            if sim <= 0.0:
                continue
            if idx >= len(self._expert_keys):
                continue
            key = self._expert_keys[idx]
            dom_info = self._expert_domains.get(key, {})
            domain = dom_info.get("domain", "general")
            if domain in exclude_domains:
                continue
            idf_w = idf.get(domain, 1.0) if use_idf else 1.0
            weight = sim * idf_w
            domain_scores[domain] = domain_scores.get(domain, 0.0) + weight
            total_weight += weight

        if total_weight < 1e-8:
            return {}

        for d in domain_scores:
            domain_scores[d] /= total_weight

        return dict(sorted(domain_scores.items(), key=lambda x: -x[1]))

    def search_top_experts(
        self,
        text: str,
        top_k: int = 30,
        min_sim: float = 0.05,
    ) -> List[Tuple[str, float]]:
        """
        テキストから top-K experts を (expert_key, score) のリストで返す。

        A/B: Expert→Piece / Expert→WorldgenProfile 直結ルーティング用。

        Returns:
            [("L3E0", 0.82), ("L7E15", 0.71), ...]  (降順)
        """
        self._load()
        query = self._encode_query(text)
        sims_flat = self._concept_dirs @ query
        sims = sims_flat.reshape(15104, 4)
        max_sims = sims.max(axis=1)

        k = min(top_k, len(max_sims))
        top_indices = np.argpartition(max_sims, -k)[-k:]
        top_indices = top_indices[np.argsort(max_sims[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            sim = float(max_sims[idx])
            if sim < min_sim:
                break
            if idx < len(self._expert_keys):
                results.append((self._expert_keys[idx], sim))
        return results

    def search_with_trace(
        self,
        anchor_kws: List[str],
        full_text: Optional[str] = None,
        top_k: int = 50,
        use_idf: bool = True,
    ) -> Tuple[Dict[str, float], "RoutingTrace"]:
        """
        アンカーキーワードで max-pooling 検索し、RoutingTrace も返す。

        anchor_kws が空の場合は full_text での全文 mean 検索にフォールバック。

        Returns:
            (domain_scores, RoutingTrace)
        """
        t0 = time.time()
        if anchor_kws:
            scores = self.search_keywords(anchor_kws, top_k=top_k, use_idf=use_idf)
            mode = "keyword_maxpool"
            used_kws = anchor_kws
        elif full_text:
            scores = self.search(full_text, top_k=top_k, use_idf=use_idf)
            mode = "fulltext_mean"
            used_kws = []
        else:
            return {}, RoutingTrace(
                mode="empty", anchor_kws=[], top_domains=[], elapsed_ms=0.0
            )
        elapsed_ms = (time.time() - t0) * 1000
        top_domains = list(scores.items())[:3]
        trace = RoutingTrace(
            mode=mode,
            anchor_kws=used_kws,
            top_domains=top_domains,
            elapsed_ms=round(elapsed_ms, 2),
        )
        return scores, trace

    def get_domain_boost(
        self,
        text: str,
        top_k: int = 20,
        threshold: float = 0.15,
    ) -> List[str]:
        """
        Verantyx Decomposer 用のドメインブースト信号を返す。
        threshold 以上のドメインを「優先ドメイン候補」として返す。

        Returns:
            ["calculus", "algebra"]  など
        """
        scores = self.search(text, top_k=top_k)
        return [d for d, s in scores.items() if s >= threshold]


# ─── グローバルシングルトン ───────────────────────────────────────
_searcher: Optional[ConceptSearcher] = None

def get_searcher(verbose: bool = False) -> ConceptSearcher:
    global _searcher
    if _searcher is None:
        _searcher = ConceptSearcher(verbose=verbose)
    return _searcher


# ─── 簡易テスト ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    queries = [
        "What is the derivative of sin(x)?",
        "Find the eigenvalues of a 2x2 matrix",
        "How many ways to choose 3 items from 10?",
        "What is the probability of rolling a 6 twice?",
        "Solve x^2 - 5x + 6 = 0",
        "What is the chemical formula for water?",
        "Prove that sqrt(2) is irrational",
    ]

    cs = ConceptSearcher(verbose=True)
    for q in queries:
        t0 = time.time()
        scores = cs.search(q, top_k=20)
        elapsed = (time.time() - t0) * 1000
        top3 = list(scores.items())[:3]
        boost = cs.get_domain_boost(q)
        print(f"\nQ: {q[:60]}")
        print(f"  Top domains: {top3}")
        print(f"  Boost signal: {boost}")
        print(f"  ({elapsed:.1f}ms)")
