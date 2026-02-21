"""
concept_boost.py
================
ConceptSearcher → Decomposer Domain スコアブースト統合レイヤー

2つの使い方:
  A) リアルタイム: decompose() ごとに search() を呼ぶ（60ms/query）
  B) バッチ事前計算: build_cache(questions) → jsonl に保存 → ゼロコスト参照

HLE 2500問の場合は B) を推奨（150秒一回 → 以降0ms）

Usage:
  from knowledge.concept_boost import ConceptBooster
  booster = ConceptBooster()

  # (A) リアルタイム
  boost_scores = booster.get_scores("What is d/dx sin(x)?")

  # (B) バッチキャッシュ
  booster.build_cache(questions, cache_path="knowledge/concept_cache.jsonl")
  booster.load_cache("knowledge/concept_cache.jsonl")
  boost_scores = booster.get_scores(question_text)   # 0ms
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# concept_search domain → Verantyx Domain enum value のマッピング
# Key insight: latex_math should be WEAK and distributed, not equal strength
DOMAIN_MAP: Dict[str, List[Tuple[str, float]]] = {
    "calculus":      [("calculus", 1.0)],
    "algebra":       [("algebra", 1.0)],
    "number_theory": [("number_theory", 0.8), ("advanced_number_theory", 0.6), ("modular_arithmetic", 0.5)],
    "geometry":      [("geometry", 1.0)],
    "linear_algebra":[("linear_algebra", 1.0)],
    "probability":   [("probability", 0.8), ("advanced_probability", 0.6)],
    "combinatorics": [("combinatorics", 0.8), ("advanced_combinatorics", 0.6)],
    "logic":         [("logic_propositional", 0.7), ("logic_modal", 0.5), ("logic_first_order", 0.6)],
    "physics":       [("physics", 1.0)],
    "chemistry":     [("chemistry", 1.0)],
    "code":          [("computer_science", 1.0)],
    "statistics":    [("statistics", 1.0)],
    # latex_math → weak generic math signal (10% of specific domain strength)
    "latex_math":    [("arithmetic", 0.1), ("algebra", 0.1), ("calculus", 0.1),
                      ("number_theory", 0.1), ("geometry", 0.1)],
}

# ブースト強度係数
BOOST_FACTOR = 8.0  # Decomposer の keyword スコアと釣り合う強度


class ConceptBooster:
    """
    ConceptSearcher の出力を Decomposer ドメインスコアに変換する。
    """

    def __init__(self, use_cache: bool = True):
        self._searcher = None
        self._cache: Dict[str, Dict[str, float]] = {}  # text → domain → score
        self._use_cache = use_cache

    def _get_searcher(self):
        if self._searcher is None:
            from knowledge.concept_search import ConceptSearcher
            self._searcher = ConceptSearcher(verbose=False)
        return self._searcher

    def get_scores_by_keywords(self, keywords: List[str]) -> Dict[str, float]:
        """
        Decomposerが抽出したkeywordsを直接クエリとして使う高精度ブースト。

        全テキスト平均（get_scores）よりも discriminative power が高い。
        空リストの場合は {} を返す。
        """
        if not keywords:
            return {}

        # キャッシュキー（keywords のソート済みタプル）
        cache_key = "kw:" + "|".join(sorted(keywords))
        if self._use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        cs = self._get_searcher()
        raw_scores = cs.search_keywords(keywords, top_k=50)

        result: Dict[str, float] = {}
        for cs_domain, weight in raw_scores.items():
            mapped = DOMAIN_MAP.get(cs_domain, [])
            for dval, multiplier in mapped:
                score = weight * multiplier * BOOST_FACTOR
                result[dval] = max(result.get(dval, 0.0), score)

        if self._use_cache:
            self._cache[cache_key] = result

        return result

    def get_scores(self, text: str) -> Dict[str, float]:
        """
        テキスト → {domain_value: boost_score} を返す。
        domain_value は Domain enum の .value と一致する文字列。
        例: {"calculus": 5.2, "algebra": 3.1}

        キャッシュがあればそちらを使用（0ms）。
        なければ ConceptSearcher を呼ぶ（60ms）。
        """
        # キャッシュ参照（先頭100文字で照合）
        # lowercase正規化（decomposerはtext.lower()を渡すため）
        key = text.lower()[:200]
        if self._use_cache and key in self._cache:
            return self._cache[key]

        # ConceptSearcher で検索
        cs = self._get_searcher()
        raw_scores = cs.search(text, top_k=50)  # {domain_str: weight}

        # Domain enum value にマッピング (weighted)
        result: Dict[str, float] = {}
        for cs_domain, weight in raw_scores.items():
            mapped = DOMAIN_MAP.get(cs_domain, [])
            for dval, multiplier in mapped:
                score = weight * multiplier * BOOST_FACTOR
                if dval in result:
                    result[dval] = max(result[dval], score)
                else:
                    result[dval] = score

        # キャッシュに保存
        if self._use_cache:
            self._cache[key] = result

        return result

    def build_cache(
        self,
        questions: List[str],
        cache_path: str = "knowledge/concept_cache.jsonl",
        batch_size: int = 100,
        verbose: bool = True,
    ) -> None:
        """
        questions のリストを全て検索してキャッシュファイルに保存。
        HLE 2500問: 約150秒（一回のみ）
        """
        cs = self._get_searcher()
        out_path = Path(cache_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        written = 0

        with open(out_path, "w") as f:
            for i, text in enumerate(questions):
                key = text.lower()[:200]  # get_scoresと一致させるためlowercase
                raw_scores = cs.search(text, top_k=50)

                # マッピング (weighted)
                result: Dict[str, float] = {}
                for cs_domain, weight in raw_scores.items():
                    for dval, multiplier in DOMAIN_MAP.get(cs_domain, []):
                        score = weight * multiplier * BOOST_FACTOR
                        result[dval] = max(result.get(dval, 0.0), score)

                entry = {"key": key, "scores": result}
                f.write(json.dumps(entry) + "\n")
                self._cache[key] = result
                written += 1

                if verbose and (i + 1) % batch_size == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(questions) - i - 1) / rate
                    print(f"  [{i+1}/{len(questions)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s",
                          flush=True)

        elapsed = time.time() - t0
        if verbose:
            print(f"Cache built: {written} entries → {out_path} ({elapsed:.1f}s)")

    def _map_raw_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """raw ConceptSearcher スコア → Verantyx domain スコアに変換（内部ヘルパー）"""
        result: Dict[str, float] = {}
        for cs_domain, weight in raw_scores.items():
            for dval, multiplier in DOMAIN_MAP.get(cs_domain, []):
                score = weight * multiplier * BOOST_FACTOR
                result[dval] = max(result.get(dval, 0.0), score)
        return result

    def load_cache(self, cache_path: str = "knowledge/concept_cache.jsonl") -> int:
        """
        事前計算済みキャッシュを読み込む。
        Returns: ロードした件数
        """
        path = Path(cache_path)
        if not path.exists():
            return 0
        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._cache[entry["key"]] = entry["scores"]
                count += 1
        return count


# ─── アンカーキーワード抽出 ───────────────────────────────────────
# 数学・科学の discriminative term リスト（全文 mean よりも高精度）
_ANCHOR_TERMS = frozenset([
    # calculus
    "derivative", "integral", "limit", "differentiate", "integrate",
    "gradient", "partial", "taylor", "maclaurin", "fourier",
    "convergence", "divergence", "series",
    # algebra
    "polynomial", "equation", "solve", "factor", "eigenvalue",
    "eigenvector", "matrix", "determinant", "linear", "quadratic",
    # number theory
    "prime", "factorial", "divisor", "modulo", "gcd", "lcm",
    "congruent", "fermat", "euler", "totient",
    # combinatorics
    "permutation", "combination", "binomial", "stirling", "derangement",
    "pigeonhole", "counting",
    # probability
    "probability", "bayes", "conditional", "distribution", "variance",
    "expected", "random", "stochastic",
    # geometry
    "triangle", "circle", "radius", "area", "perimeter",
    "pythagorean", "angle", "hypotenuse", "ellipse",
    # linear algebra
    "vector", "dot product", "cross product", "orthogonal", "projection",
    "rank", "null space", "trace",
    # logic
    "implies", "satisfiable", "tautology", "modal", "kripke",
    "theorem", "proof", "corollary",
    # physics/chemistry
    "velocity", "acceleration", "force", "momentum", "entropy",
    "electron", "photon", "wavelength",
    # CS/AI
    "algorithm", "complexity", "automaton", "turing", "graph",
    "vertex", "edge", "cycle", "path",
])

_STOP_WORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "used",
    "which", "that", "this", "these", "those", "what", "how",
    "when", "where", "why", "who", "whom", "whose",
    "and", "or", "but", "if", "then", "than", "so",
    "of", "in", "to", "for", "on", "at", "by", "from", "with",
    "as", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "each", "every", "all", "both",
    "few", "more", "most", "other", "some", "such", "no", "not",
    "only", "same", "too", "very", "just", "also", "following",
    "given", "let", "find", "compute", "calculate", "determine",
    "show", "prove", "evaluate", "consider",
])


def extract_anchor_kws(text: str, max_kws: int = 12) -> List[str]:
    """
    問題文から discriminative なアンカーキーワードを抽出する。

    戦略:
    1. 既知の数学・科学用語リストに一致する単語を優先
    2. 英数字トークンで stop word 以外 + 長さ≥4 の単語も候補
    3. 最大 max_kws 個を返す
    """
    import re
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())
    anchor: List[str] = []
    generic: List[str] = []
    seen: set = set()
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        if w in _ANCHOR_TERMS:
            anchor.append(w)
        elif len(w) >= 4 and w not in _STOP_WORDS:
            generic.append(w)
    combined = anchor + generic
    return combined[:max_kws]


class ConceptBoosterWithTrace:
    """
    RoutingTrace 付きでブーストスコアを返す薄いラッパー。
    """
    def __init__(self, booster: "ConceptBooster"):
        self._b = booster

    def get_scores_with_trace(
        self,
        text: str,
        anchor_kws: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, float], "RoutingTrace"]:
        """
        anchor_kws が与えられればキーワード max-pooling 検索、
        なければ全文 mean 検索（キャッシュ優先）。
        RoutingTrace も返す。
        """
        from knowledge.concept_search import RoutingTrace as CSRoutingTrace
        import time

        t0 = time.time()
        kws = anchor_kws if anchor_kws else []
        cache_hit = False

        if kws:
            # keyword max-pooling 検索
            cs = self._b._get_searcher()
            raw_scores, rt = cs.search_with_trace(kws)
            scores = self._b._map_raw_scores(raw_scores)
            mode = "keyword_maxpool"
        else:
            # 全文キャッシュ優先
            key = text.lower()[:200]
            if self._b._use_cache and key in self._b._cache:
                scores = self._b._cache[key]
                cache_hit = True
                mode = "fulltext_cache"
            else:
                scores = self._b.get_scores(text)
                mode = "fulltext_mean"

        elapsed_ms = round((time.time() - t0) * 1000, 2)
        # RoutingTrace（AuditBundle 用の同型）
        from audit.audit_bundle import RoutingTrace as ABRoutingTrace
        rt_ab = ABRoutingTrace(
            mode=mode,
            anchor_kws=kws,
            top_domains=sorted(scores.items(), key=lambda x: -x[1])[:3],
            elapsed_ms=elapsed_ms,
            cache_hit=cache_hit,
        )
        return scores, rt_ab


# ─── グローバルシングルトン ───────────────────────────────────────
_booster: Optional[ConceptBooster] = None

def get_booster() -> ConceptBooster:
    global _booster
    if _booster is None:
        _booster = ConceptBooster()
        # キャッシュがあれば自動ロード
        cache_path = Path(__file__).parent.parent / "knowledge" / "concept_cache.jsonl"
        if cache_path.exists():
            n = _booster.load_cache(str(cache_path))
            # silent load
    return _booster


# ─── テスト ──────────────────────────────────────────────────────
if __name__ == "__main__":
    booster = ConceptBooster()
    tests = [
        "What is the derivative of sin(x)?",
        "Find the eigenvalues of a 2x2 matrix [[1,2],[3,4]]",
        "How many ways to choose 3 items from 10?",
        "Solve x^2 - 5x + 6 = 0",
    ]
    for q in tests:
        scores = booster.get_scores(q)
        top = sorted(scores.items(), key=lambda x: -x[1])[:5]
        print(f"Q: {q[:60]}")
        print(f"   Boost: {top}")
        print()
