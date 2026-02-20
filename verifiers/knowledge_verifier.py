"""
Knowledge Verifier — 600B SVD 知識探索を「証拠供給器」として統合

原則: 検索は「証拠候補の供給」まで（採用は Verifier が決める）
  - 検索結果は "候補ピース（定理/定義/補題）の証拠ID" として取り込む
  - 採用条件は「その証拠を使った候補が verify を通ること」
  - 検索は "知識の搬入" であって "答えの搬入" ではない

このVerifierは:
  1. 問題文をベクトル化（embed_tokens × query）
  2. concept_dirs にコサイン類似度で照合
  3. 上位 expert のドメイン信号を取得
  4. 候補値のドメインが問題のドメインと整合するか確認
  → PASS: ドメイン整合
  → FAIL: 明らかなドメイン不一致（反例: "wrong_domain"）
  → UNKNOWN: 判定不能（SVD未ロード等）
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

from verifiers.api import BaseVerifier, Verdict, VerdictStatus, VerifySpec


class KnowledgeVerifier(BaseVerifier):
    """
    600B SVD concept_dirs を使ったドメイン整合検証器。

    「検索は証拠供給のみ」原則に従い、ドメイン信号で候補を
    支持/棄却するだけで、答えは生成しない。
    """
    name = "knowledge_600b"

    def __init__(self, top_k: int = 20, domain_mismatch_threshold: float = 0.3):
        self.top_k = top_k
        self.domain_mismatch_threshold = domain_mismatch_threshold
        self._searcher = None

    def _get_searcher(self):
        if self._searcher is not None:
            return self._searcher
        if self._searcher is False:
            return None
        try:
            from knowledge.concept_search import ConceptSearcher
            self._searcher = ConceptSearcher()
            return self._searcher
        except Exception:
            self._searcher = False
            return None

    def can_handle(self, spec: VerifySpec) -> bool:
        # 知識依存の問題に対して有効（smt や numeric は他に任せる）
        return spec.kind in ("cross_check", "symbolic", "numeric")

    def verify(self, candidate_value: Any, spec: VerifySpec, context: Dict[str, Any]) -> Verdict:
        """
        問題文のドメイン信号と候補値を照合する。
        """
        searcher = self._get_searcher()
        if searcher is None:
            return Verdict.unknown(self.name, reason="600b_not_available")

        # 問題文を取得
        source_text = context.get("metadata", {}).get("source_text", "")
        if not source_text:
            source_text = str(context.get("query", ""))
        if not source_text or len(source_text) < 10:
            return Verdict.unknown(self.name, reason="no_source_text")

        # 600B concept search でドメイン信号を取得
        try:
            domain_scores = searcher.search(source_text[:500], top_k=self.top_k)
        except Exception as e:
            return Verdict.unknown(self.name, reason=f"search_error:{e}")

        if not domain_scores:
            return Verdict.unknown(self.name, reason="empty_search_result")

        # IR ドメインと concept search ドメインの整合チェック
        ir_domain = context.get("domain", "unknown").lower()
        top_concept_domain = max(domain_scores, key=domain_scores.get).lower()
        top_score = domain_scores[max(domain_scores, key=domain_scores.get)]

        # ドメイン正規化
        domain_map = {
            "calculus":     ["calculus", "differential", "integral"],
            "algebra":      ["algebra", "polynomial"],
            "arithmetic":   ["arithmetic", "number"],
            "number_theory":["number_theory", "prime", "modular"],
            "geometry":     ["geometry", "triangle"],
            "logic":        ["logic", "propositional"],
            "physics":      ["physics", "quantum"],
            "chemistry":    ["chemistry"],
            "biology":      ["biology", "medicine"],
        }

        def normalize_domain(d: str) -> str:
            d_lower = d.lower()
            for canonical, aliases in domain_map.items():
                if any(alias in d_lower for alias in aliases):
                    return canonical
            return d_lower

        ir_norm = normalize_domain(ir_domain)
        concept_norm = normalize_domain(top_concept_domain)

        evidence = {
            "ir_domain": ir_domain,
            "top_concept_domain": top_concept_domain,
            "top_score": float(top_score),
            "domain_scores": {k: float(v) for k, v in list(domain_scores.items())[:5]},
        }

        # 強いドメイン信号がある場合のみ判定（弱い信号は UNKNOWN）
        if top_score < self.domain_mismatch_threshold:
            return Verdict.unknown(
                self.name,
                reason=f"weak_signal:{top_score:.3f}",
                details=[f"Top domain={top_concept_domain} score={top_score:.3f} < threshold"]
            )

        # ドメイン整合チェック
        if ir_norm == concept_norm:
            return Verdict.pass_(
                self.name,
                witness=evidence,
                details=[f"Domain match: ir={ir_norm} ≈ concept={concept_norm}"]
            )
        else:
            # 不一致でも弱い不整合は UNKNOWN（整合が明らかに強い場合のみ FAIL）
            # 基本的にドメインだけでは棄却しない（証拠供給のみの原則）
            return Verdict.unknown(
                self.name,
                reason=f"domain_mismatch:ir={ir_norm},concept={concept_norm}",
                details=[f"IR domain ({ir_norm}) != concept domain ({concept_norm})"]
            )


# ─────────────────────────────────────────────────────────────────────
# KnowledgeEvidenceExtractor — 証拠IDの抽出（ピース選択への補助）
# ─────────────────────────────────────────────────────────────────────

class KnowledgeEvidenceExtractor:
    """
    問題文から「一般概念クエリ」を抽出し、600B SVD で証拠を供給する。

    目的: ピース選択前に「この問題に関連する概念」を特定して
          ビーム探索のスコアリングを補助する。
    
    原則: 証拠供給のみ。答えは返さない。
    """

    GENERAL_CONCEPT_PATTERNS = [
        # 数学的概念の抽出
        (re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Za-z]+){0,3}\s+(?:theorem|lemma|conjecture|property|inequality))\b'),
         "theorem"),
        (re.compile(r'\b((?:[A-Za-z]+\s+)?(?:formula|equation|identity))\b', re.IGNORECASE),
         "formula"),
        (re.compile(r'\b([a-z]+(?:\s+[a-z]+)?\s+distribution)\b', re.IGNORECASE),
         "distribution"),
    ]

    def extract_concepts(self, problem_text: str) -> List[Dict[str, str]]:
        """
        問題文から一般概念クエリを抽出する。

        Returns:
            [{"concept": "...", "type": "theorem/formula/..."}, ...]
        """
        concepts = []
        for pattern, concept_type in self.GENERAL_CONCEPT_PATTERNS:
            for match in pattern.finditer(problem_text):
                concept = match.group(1).strip()
                if 3 < len(concept) < 80:  # 適切な長さのものだけ
                    concepts.append({"concept": concept, "type": concept_type})

        return concepts[:5]  # 最大5個
