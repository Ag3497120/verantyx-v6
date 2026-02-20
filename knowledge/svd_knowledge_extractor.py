"""
svd_knowledge_extractor.py
==========================
非発火（Non-inference）知識抽出システム

600B DeepSeek V3.2の重みから抽出したSVD資産を使って、
数学的知識を抽出しCross Simulationに注入する。

アルゴリズム:
  1. 事前計算（一回だけ）:
     - 数学関連トークン (~914個) のembeddingを抽出
     - 各Expert (15104個) の concept_dirs × math_embed_tokens.T
       → Expert毎の「活性化数学トークン上位N個」を計算
     - expert_math_tokens.json に保存
  
  2. クエリ時:
     - 問題文 → query_vec (7168-dim)
     - query_vec × concept_dirs → top-k Expert活性化スコア
     - 活性化Expert → 事前計算済み数学トークンをLookup
     - 問題に関連する数学用語リストを返す

  3. Cross Simulation連携:
     - 返された数学用語でPieceスコアをブースト
     - Decomposerのドメイン検出を強化
"""

import os
import json
import time
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DB_DIR = Path("/Users/motonishikoudai/avh_math/avh_math/db/moe_sparse_cross_600b_real")
CACHE_DIR = Path(__file__).parent  # knowledge/

CONCEPT_DIRS_PATH   = DB_DIR / "concept_dirs.npy"
EMBED_TOKENS_PATH   = DB_DIR / "embed_tokens.npy"
TOKENIZER_PATH      = DB_DIR / "tokenizer.json"
EXPERT_DOMAINS_PATH = DB_DIR / "expert_vocab_domains.json"
ROUTING_PATH        = DB_DIR / "routing_patterns.json"
CACHE_PATH          = CACHE_DIR / "expert_math_tokens.json"

# ─── 数学関連キーワード ──────────────────────────────────────────────
MATH_KEYWORDS = [
    'deriv', 'integr', 'eigen', 'matri', 'vector', 'polynom', 'prime', 'modulo',
    'factori', 'binom', 'permut', 'probab', 'entropy', 'theorem', 'lemma',
    'formula', 'equat', 'inequal', 'function', 'sequenc', 'series', 'limit',
    'convex', 'linear', 'algebr', 'calcul', 'geometr', 'number', 'trig',
    'differen', 'partial', 'gradient', 'laplace', 'fourier', 'taylor', 'newton',
    'euler', 'gaussian', 'norm', 'orthog', 'span', 'basis', 'rank', 'kernel',
    'cyclic', 'group', 'ring', 'field', 'module', 'ideal', 'homom', 'isomor',
    'complex', 'real', 'rational', 'integer', 'natural', 'finite', 'infinite',
    'continu', 'smooth', 'analyt', 'holomorp', 'topolog', 'manifold', 'metric',
]

MATH_EXACT = {
    'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'sum', 'prod',
    'lim', 'max', 'min', 'sup', 'inf', 'gcd', 'lcm', 'mod',
    'det', 'tr', 'div', 'grad', 'dx', 'dy', 'dz', 'dt',
    'pi', 'sigma', 'lambda', 'theta', 'omega', 'alpha', 'beta',
    'gamma', 'delta', 'epsilon', 'mu', 'nu', 'rho', 'xi', 'zeta',
    'phi', 'chi', 'psi', 'eta', 'kappa', 'tau',
}

MATH_DOMAINS = {
    'calculus', 'algebra', 'number_theory', 'geometry', 'linear_algebra',
    'probability', 'combinatorics', 'statistics', 'logic', 'latex_math',
    'physics', 'chemistry', 'code',
}


def _is_math_token(token: str) -> bool:
    """数学関連トークンかどうか判定"""
    clean = token.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ').strip()
    if not clean:
        return False
    # LaTeX コマンド
    if re.search(r'\\[a-zA-Z]', clean):
        return True
    # 完全一致
    if clean.lower() in MATH_EXACT:
        return True
    # キーワード含有
    clean_lower = clean.lower()
    if len(clean) >= 3 and any(kw in clean_lower for kw in MATH_KEYWORDS):
        return True
    return False


class SVDKnowledgeExtractor:
    """
    非発火SVD知識抽出器
    
    事前計算: build_cache() で一度だけ実行（~2-5分）
    クエリ時: get_knowledge_terms() で高速Lookup（~200ms）
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._loaded = False
        self._concept_dirs_flat = None  # (60416, 7168) normalized
        self._expert_keys = None        # list of "L{l}E{e}"
        self._expert_domains = None     # {key: {domain, confidence, top_tokens}}
        self._math_knowledge = None     # {expert_key: [token_str, ...]}
        self._tokenizer = None

    def _load_base(self):
        """concept_dirs と expert domains を読み込む"""
        if self._loaded:
            return
        t0 = time.time()

        # concept_dirs: 正規化して flatten
        raw = np.load(str(CONCEPT_DIRS_PATH))          # (15104, 4, 7168)
        flat = raw.reshape(-1, 7168).astype(np.float32)
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self._concept_dirs_flat = flat / norms          # (60416, 7168)

        # expert keys
        with open(ROUTING_PATH) as f:
            routing = json.load(f)
        def _sort_key(k):
            parts = k[1:].split('E')
            return (int(parts[0]), int(parts[1]))
        self._expert_keys = sorted(routing.get("experts", {}).keys(), key=_sort_key)

        # expert domains
        with open(EXPERT_DOMAINS_PATH) as f:
            dom_data = json.load(f)
        self._expert_domains = dom_data.get("experts", {})

        if self.verbose:
            print(f"[SVDKnowledgeExtractor] Base loaded in {time.time()-t0:.1f}s")
        self._loaded = True

    def build_cache(self, save_path: Optional[str] = None, top_tokens: int = 30) -> int:
        """
        【一回だけ実行】各Expert の数学トークン上位Nを計算して保存。
        
        所要時間: ~5分（M1 Max CPU）
        """
        self._load_base()
        out_path = Path(save_path or str(CACHE_PATH))

        t0 = time.time()
        print(f"Building SVD knowledge cache...")
        print(f"  concept_dirs: {self._concept_dirs_flat.shape}")

        # 1. トークナイザーから数学トークンIDを抽出
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(TOKENIZER_PATH))
        vocab = tok.get_vocab()
        vocab_size = max(vocab.values()) + 1

        math_ids = []
        math_strs = []
        for token_str, tid in vocab.items():
            if tid < 129280 and _is_math_token(token_str):
                clean = token_str.replace('Ġ', ' ').replace('▁', ' ').strip()
                math_ids.append(tid)
                math_strs.append(clean)

        print(f"  Math tokens: {len(math_ids)}")
        if len(math_ids) == 0:
            raise RuntimeError("No math tokens found!")

        # 2. math token embeddings を抽出
        embed = np.load(str(EMBED_TOKENS_PATH), mmap_mode='r')  # (129280, 7168)
        math_embed = embed[math_ids].astype(np.float32)  # (N_math, 7168)
        # 正規化
        norms = np.linalg.norm(math_embed, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        math_embed_norm = math_embed / norms             # (N_math, 7168)

        print(f"  Math embeddings: {math_embed_norm.shape}")

        # 3. Expert毎にトップ数学トークンを計算（バッチ処理）
        # concept_dirs_flat: (60416, 7168)
        # math_embed_norm.T: (7168, N_math)
        # result per batch: (batch_size, N_math)
        n_experts_flat = self._concept_dirs_flat.shape[0]  # 60416
        n_math = len(math_ids)
        batch_size = 1000

        knowledge = {}  # expert_key → [token_str, ...]

        for batch_start in range(0, n_experts_flat, batch_size):
            batch_end = min(batch_start + batch_size, n_experts_flat)
            batch_dirs = self._concept_dirs_flat[batch_start:batch_end]  # (bs, 7168)

            # コサイン類似度
            scores = batch_dirs @ math_embed_norm.T  # (bs, N_math)

            for i in range(batch_end - batch_start):
                dir_idx = batch_start + i
                expert_idx = dir_idx // 4   # expert = dir_idx / 4
                dir_within = dir_idx % 4    # 0-3 の方向

                if expert_idx >= len(self._expert_keys):
                    continue
                expert_key = self._expert_keys[expert_idx]

                # このExpertのドメインが数学系か確認
                dom_info = self._expert_domains.get(expert_key, {})
                domain = dom_info.get("domain", "general")
                if domain not in MATH_DOMAINS:
                    continue

                top_k = min(top_tokens, n_math)
                top_token_indices = np.argpartition(scores[i], -top_k)[-top_k:]
                top_token_indices = top_token_indices[
                    np.argsort(scores[i][top_token_indices])[::-1]
                ]

                tokens_for_dir = [math_strs[j] for j in top_token_indices
                                  if scores[i][j] > 0.01]

                if expert_key not in knowledge:
                    knowledge[expert_key] = set()
                knowledge[expert_key].update(tokens_for_dir)

            if (batch_start // batch_size) % 10 == 0:
                elapsed = time.time() - t0
                progress = batch_end / n_experts_flat
                eta = elapsed / max(progress, 1e-6) * (1 - progress)
                print(f"  [{batch_end}/{n_experts_flat}] {elapsed:.0f}s | ETA {eta:.0f}s | experts_with_knowledge={len(knowledge)}", flush=True)

        # set → list に変換
        knowledge_list = {k: list(v) for k, v in knowledge.items()}

        # 保存
        with open(out_path, 'w') as f:
            json.dump(knowledge_list, f, ensure_ascii=False)

        elapsed = time.time() - t0
        print(f"SVD knowledge cache built: {len(knowledge_list)} experts → {out_path} ({elapsed:.1f}s)")
        self._math_knowledge = knowledge_list
        return len(knowledge_list)

    def load_cache(self, cache_path: Optional[str] = None) -> int:
        """事前計算済みキャッシュを読み込む"""
        path = Path(cache_path or str(CACHE_PATH))
        if not path.exists():
            return 0
        with open(path) as f:
            self._math_knowledge = json.load(f)
        return len(self._math_knowledge)

    def get_knowledge_terms(
        self,
        text: str,
        top_k_experts: int = 30,
        max_terms: int = 50,
    ) -> List[str]:
        """
        クエリ時: 問題テキスト → 関連数学用語リスト（高速）

        Args:
            text: HLE問題文
            top_k_experts: 活性化Expert上位k個
            max_terms: 返す数学用語の最大数

        Returns:
            ["derivative", "sin", "cos", ...] のような数学用語リスト
        """
        self._load_base()
        if self._math_knowledge is None:
            loaded = self.load_cache()
            if loaded == 0:
                return []  # キャッシュなし

        # query_vec: トークン平均
        from tokenizers import Tokenizer
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

        encoding = self._tokenizer.encode(text)
        ids = encoding.ids
        vocab_size = 129280

        embed = np.load(str(EMBED_TOKENS_PATH), mmap_mode='r')
        valid_ids = [i for i in ids if 0 <= i < vocab_size]
        if not valid_ids:
            return []

        vecs = embed[valid_ids].astype(np.float32)
        query = vecs.mean(axis=0)
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query /= norm

        # top-k expert を検索
        sims = self._concept_dirs_flat @ query        # (60416,)
        k = min(top_k_experts * 4, len(sims))        # 4 dirs per expert
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        # Expert → 数学トークン収集（スコア重み付き）
        term_scores: Dict[str, float] = {}
        seen_experts = set()

        for dir_idx in top_idx:
            expert_idx = dir_idx // 4
            if expert_idx >= len(self._expert_keys):
                continue
            if expert_idx in seen_experts:
                continue
            seen_experts.add(expert_idx)

            expert_key = self._expert_keys[expert_idx]
            terms = self._math_knowledge.get(expert_key, [])
            if not terms:
                continue

            sim_score = float(sims[dir_idx])
            for term in terms:
                if term and len(term) >= 2:
                    term_scores[term] = term_scores.get(term, 0.0) + sim_score

            if len(seen_experts) >= top_k_experts:
                break

        # スコア降順でソート
        sorted_terms = sorted(term_scores.items(), key=lambda x: -x[1])
        return [t for t, _ in sorted_terms[:max_terms]]


# ─── グローバルシングルトン ────────────────────────────────────────
_extractor: Optional[SVDKnowledgeExtractor] = None

def get_extractor(verbose: bool = False) -> SVDKnowledgeExtractor:
    global _extractor
    if _extractor is None:
        _extractor = SVDKnowledgeExtractor(verbose=verbose)
    return _extractor


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build cache")
    parser.add_argument("--test", action="store_true", help="Test query")
    args = parser.parse_args()

    ext = SVDKnowledgeExtractor(verbose=True)

    if args.build:
        ext.build_cache()

    if args.test or not args.build:
        n = ext.load_cache()
        print(f"Loaded {n} experts")
        queries = [
            "What is the derivative of sin(x)?",
            "Find the eigenvalues of [[1,2],[3,4]]",
            "How many ways to arrange 5 objects?",
            "What is the probability of rolling two 6s?",
            "Solve x^2 - 5x + 6 = 0",
        ]
        for q in queries:
            terms = ext.get_knowledge_terms(q, top_k_experts=20)
            print(f"\nQ: {q[:60]}")
            print(f"  Knowledge: {terms[:15]}")
