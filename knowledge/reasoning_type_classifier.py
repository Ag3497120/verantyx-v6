"""
reasoning_type_classifier.py
==============================
600B Expert活性化パターン → MCQ の推論型を特定する分類器

設計:
  問題文 → embed_tokens → concept_dirs projection →
  Expert活性化ベクトル → 推論型プローブとの相関 → 推論型

推論型一覧:
  verify_property    : 各選択肢に性質チェックを適用 (is_prime, is_palindrome等)
  compute_and_match  : 幹から値を計算し、一致する選択肢を返す
  elimination        : 計算で誤答を否定、残りを正答とする
  formal_proof       : 形式的証明が必要 → INCONCLUSIVE
  factual_lookup     : 事実確認が必要 → INCONCLUSIVE

使い方:
  from knowledge.reasoning_type_classifier import ReasoningTypeClassifier
  clf = ReasoningTypeClassifier()
  rtype, confidence = clf.classify(stem)
  # rtype: "verify_property" | "compute_and_match" | ...
  # confidence: 0.0-1.0 (低いほど判定困難)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# 推論型プローブ文（600B Expert活性化の方向を定義）
REASONING_TYPE_PROBES: Dict[str, list] = {
    "verify_property": [
        "is this a prime number",
        "check if palindrome",
        "is this divisible",
        "which satisfies the property",
        "which of these is true",
        "test each option",
    ],
    "compute_and_match": [
        "calculate the value",
        "what is the result of",
        "evaluate the expression",
        "compute factorial",
        "find the derivative",
        "solve for x",
        "what is the numerical answer",
    ],
    "elimination": [
        "which is NOT correct",
        "which cannot be",
        "which is impossible",
        "eliminate wrong answers",
        "which violates the rule",
        "which of the following is false",
    ],
    "formal_proof": [
        "prove that for all",
        "theorem in algebraic topology",
        "cohomology group",
        "manifold structure",
        "homotopy equivalence",
        "bordism class",
    ],
    "factual_lookup": [
        "who discovered",
        "what year was invented",
        "according to the theory named after",
        "which historical theorem states",
        "named impossibility theorem condition",
    ],
}

# 計算可能な推論型（これ以外はINCONCLUSIVE）
COMPUTABLE_TYPES = {"verify_property", "compute_and_match", "elimination"}
INCONCLUSIVE_TYPES = {"formal_proof", "factual_lookup"}

# 信頼度しきい値
CONFIDENCE_THRESHOLD = 0.12  # これ未満はINCONCLUSIVE


class ReasoningTypeClassifier:
    """
    600B concept_dirs ベースの MCQ 推論型分類器。
    Expert活性化パターンと推論型プローブの相関で推論型を特定する。
    """

    def __init__(self):
        self._loaded = False
        self._cs = None
        self._concept_norm = None
        self._type_activations: Dict[str, np.ndarray] = {}

    def _load(self):
        if self._loaded:
            return

        # ConceptSearcher (embed_tokens + tokenizer)
        from knowledge.concept_search import ConceptSearcher
        self._cs = ConceptSearcher()
        self._cs._load()

        # concept_dirs max-pool (15104, 7168) → 正規化
        from pathlib import Path
        concept_raw = self._cs._concept_dirs  # (60416, 7168) すでに正規化済み
        # expert単位でmax-pool: (60416,7168) → (15104,7168)
        concept_4d = concept_raw.reshape(15104, 4, 7168)
        concept_1d = concept_4d.max(axis=1).astype(np.float32)
        norms = np.linalg.norm(concept_1d, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self._concept_norm = concept_1d / norms  # (15104, 7168)

        # 各推論型のプローブ活性化ベクトルを事前計算
        for rtype, probes in REASONING_TYPE_PROBES.items():
            probe_vecs = [self._cs._encode_query(p) for p in probes]
            probe_vecs = [v for v in probe_vecs if np.linalg.norm(v) > 1e-8]
            if not probe_vecs:
                self._type_activations[rtype] = np.zeros(15104, dtype=np.float32)
                continue
            # max-pooling
            probe_max = np.max(np.stack(probe_vecs), axis=0).astype(np.float32)
            probe_max /= (np.linalg.norm(probe_max) + 1e-8)
            # Expert活性化
            sims = self._concept_norm @ probe_max  # (15104,)
            act = np.zeros(15104, dtype=np.float32)
            top_idx = np.argpartition(sims, -50)[-50:]
            act[top_idx] = np.maximum(sims[top_idx], 0)
            self._type_activations[rtype] = act

        self._loaded = True

    def _stem_activation(self, stem: str) -> np.ndarray:
        """問題文 → Expert活性化ベクトル (15104-dim, sparse)"""
        vec = self._cs._encode_query(stem)
        if np.linalg.norm(vec) < 1e-8:
            return np.zeros(15104, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        sims = self._concept_norm @ vec  # (15104,)
        act = np.zeros(15104, dtype=np.float32)
        top_idx = np.argpartition(sims, -50)[-50:]
        act[top_idx] = np.maximum(sims[top_idx], 0)
        return act

    def classify(self, stem: str) -> Tuple[str, float]:
        """
        問題文の推論型を特定する。

        Returns:
            (reasoning_type, confidence)
            reasoning_type: "verify_property" | "compute_and_match" |
                            "elimination" | "formal_proof" | "factual_lookup"
            confidence: 0.0-1.0 (CONFIDENCE_THRESHOLD 未満はINCONCLUSIVE推奨)
        """
        self._load()

        stem_act = self._stem_activation(stem)
        norm_s = np.linalg.norm(stem_act)
        if norm_s < 1e-8:
            return "factual_lookup", 0.0

        # 各推論型との cosine（活性化空間）
        scores = {}
        for rtype, type_act in self._type_activations.items():
            norm_t = np.linalg.norm(type_act)
            if norm_t < 1e-8:
                scores[rtype] = 0.0
            else:
                scores[rtype] = float(stem_act @ type_act) / (norm_s * norm_t)

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # 2位との差を信頼度とする
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        confidence = min(1.0, margin * 10.0)  # margin を 0-1 に正規化

        return best_type, confidence

    def is_computable(self, stem: str) -> Tuple[bool, str, float]:
        """
        この問題は計算で解けるか？

        Returns:
            (is_computable, reasoning_type, confidence)
        """
        rtype, conf = self.classify(stem)
        computable = rtype in COMPUTABLE_TYPES and conf >= CONFIDENCE_THRESHOLD
        return computable, rtype, conf

    def all_scores(self, stem: str) -> Dict[str, float]:
        """デバッグ用: 全推論型のスコアを返す"""
        self._load()
        stem_act = self._stem_activation(stem)
        norm_s = np.linalg.norm(stem_act)
        if norm_s < 1e-8:
            return {k: 0.0 for k in self._type_activations}
        return {
            rtype: float(stem_act @ act) / (norm_s * np.linalg.norm(act) + 1e-8)
            for rtype, act in self._type_activations.items()
        }


# ── グローバルシングルトン ─────────────────────────────────────────────
_classifier: Optional[ReasoningTypeClassifier] = None

def get_classifier() -> ReasoningTypeClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ReasoningTypeClassifier()
    return _classifier


# ── 簡易テスト ────────────────────────────────────────────────────────
if __name__ == "__main__":
    clf = ReasoningTypeClassifier()

    tests = [
        ("Which of the following is a prime number?", True),
        ("What is 5! (5 factorial)?", True),
        ("What is the derivative of sin(x)?", True),
        ("Which condition of Arrhenius's sixth impossibility theorem do critical-level views violate?", False),
        ("Compute the reduced 12-th dimensional Spin bordism of G2.", False),
        ("Which number is divisible by both 3 and 4?", True),
    ]

    print("=== ReasoningTypeClassifier テスト ===\n")
    correct = 0
    for stem, expected_computable in tests:
        computable, rtype, conf = clf.is_computable(stem)
        ok = "✅" if computable == expected_computable else "❌"
        print(f"{ok} [{rtype}] conf={conf:.3f} | {stem[:60]}")
        if computable == expected_computable:
            correct += 1

    print(f"\nScore: {correct}/{len(tests)}")
