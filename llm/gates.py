"""
Gate A/B/C — LLM出力の多段検証

Gate A: スキーマ/型/禁止フィールド検出
Gate B: "答えっぽい出力"検出 → 即リジェクト
Gate C: 制約整合チェック（units / 整数条件 / 境界）

すべての Gate は通過か失敗かを GateResult で返す。
失敗時は reason を返し、pipeline は re-prompt または skip を決定する。
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from llm.contract import LLMContract, LLMCandidate


# ─────────────────────────────────────────────────────────────────────
# GateResult
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    passed: bool
    gate: str               # "A" | "B" | "C"
    reason: Optional[str] = None
    details: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


# ─────────────────────────────────────────────────────────────────────
# Gate A — スキーマ / 型 / 禁止フィールド
# ─────────────────────────────────────────────────────────────────────

# LLM が絶対に出してはいけないフィールド名
_FORBIDDEN_FIELDS = frozenset([
    "answer", "result", "solution", "correct_answer", "final_answer",
    "the_answer", "output", "response", "correct", "label",
    "answer_choice", "selected_option", "my_answer",
])

# 有効なタスクタイプ
_VALID_TASKS = frozenset([
    "compute", "count", "find", "decide", "prove",
    "classify", "construct", "optimize",
])

# 有効なAnswer Schema
_VALID_SCHEMAS = frozenset([
    "integer", "decimal", "boolean", "option_label",
    "expression", "sequence", "proof_sketch", "string",
])


class GateA:
    """
    Gate A: スキーマ/型/禁止フィールド検出

    1. ir フィールドが存在する
    2. ir.task が有効値
    3. ir.answer_schema が有効値
    4. 禁止フィールドが含まれない（再帰チェック）
    5. candidates が list
    """

    @staticmethod
    def _find_forbidden(d: Any, path: str = "") -> List[str]:
        """再帰的に禁止フィールドを探す"""
        hits = []
        if isinstance(d, dict):
            for k, v in d.items():
                full_path = f"{path}.{k}" if path else k
                if k.lower() in _FORBIDDEN_FIELDS:
                    hits.append(full_path)
                hits.extend(GateA._find_forbidden(v, full_path))
        elif isinstance(d, list):
            for i, item in enumerate(d):
                hits.extend(GateA._find_forbidden(item, f"{path}[{i}]"))
        return hits

    @classmethod
    def check(cls, contract: LLMContract, raw_dict: Optional[Dict[str, Any]] = None) -> GateResult:
        """
        Args:
            contract: パース済み LLMContract
            raw_dict: LLM から受け取った生の辞書（禁止フィールド検出に使用）
                      None の場合は contract.to_dict() を使用（フォールバック）
        """
        details = []

        # 1. ir.task
        if contract.ir.task not in _VALID_TASKS:
            return GateResult(
                passed=False, gate="A",
                reason=f"invalid_task:{contract.ir.task}",
                details=[f"Expected one of {_VALID_TASKS}"]
            )

        # 2. ir.answer_schema
        if contract.ir.answer_schema not in _VALID_SCHEMAS:
            return GateResult(
                passed=False, gate="A",
                reason=f"invalid_schema:{contract.ir.answer_schema}",
                details=[f"Expected one of {_VALID_SCHEMAS}"]
            )

        # 3. 禁止フィールド（再帰チェック）
        # raw_dict を優先する（LLMが出したが from_dict でドロップされた危険フィールドも検出）
        check_dict = raw_dict if raw_dict is not None else contract.to_dict()
        forbidden_hits = cls._find_forbidden(check_dict)
        if forbidden_hits:
            return GateResult(
                passed=False, gate="A",
                reason="forbidden_fields_detected",
                details=forbidden_hits
            )

        # 4. candidates は list
        if not isinstance(contract.candidates, list):
            return GateResult(
                passed=False, gate="A",
                reason="candidates_not_list",
            )

        # 5. candidates 上限チェック（K=5）
        if len(contract.candidates) > 5:
            details.append(f"Too many candidates: {len(contract.candidates)} > 5 (truncated)")
            contract.candidates = contract.candidates[:5]

        # 6. missing スロット監査
        # is_verifiable=True なのに critical な missing がある場合は警告
        # （リジェクトはしない — slot_retry のヒントとして使う）
        if contract.ir.is_verifiable and contract.ir.missing:
            critical_missing = [m for m in contract.ir.missing
                                if any(kw in m.lower() for kw in
                                       ["target", "variable", "quantity", "binding", "unit", "domain"])]
            if critical_missing:
                details.append(f"WARNING: is_verifiable=True but critical missing slots: {critical_missing}")
                # slot_retry が必要なことをマーク（リジェクトではない）
                contract.gate_rejection_reason = f"slot_retry_needed:{critical_missing}"

        return GateResult(passed=True, gate="A", details=details)


# ─────────────────────────────────────────────────────────────────────
# Gate B — "答えっぽい出力" 検出
# ─────────────────────────────────────────────────────────────────────

# 答えを断言するフレーズパターン
_ANSWER_ASSERTION_PATTERNS = [
    # "the answer is X", "answer: X", "答えは X"
    re.compile(r'\bthe\s+answer\s+is\b', re.IGNORECASE),
    re.compile(r'\banswer\s*:\s*\S', re.IGNORECASE),
    re.compile(r'\bsolution\s*:\s*\S', re.IGNORECASE),
    re.compile(r'\bresult\s*:\s*\S', re.IGNORECASE),
    re.compile(r'\btherefore\s*(?:,\s*)?\S+\s*(?:is|=)\s*\d', re.IGNORECASE),
    # " = 42" で終わる断言（スペースつきの数学的等号、パラメータ代入 base=2 は除外）
    re.compile(r'(?<!\w)\s+=\s+-?\d+(?:\.\d+)?\s*$'),
    # MCQ直接断言: "The correct option is A"
    re.compile(r'\bcorrect\s+(?:option|choice|answer)\s+is\s+[A-E]\b', re.IGNORECASE),
    re.compile(r'\bselect\s+(?:option\s+)?[A-E]\b', re.IGNORECASE),
    # "so X = value", "thus X = value" のような計算結論
    re.compile(r'\b(?:so|thus|hence|therefore)\b.{0,50}=\s*-?\d+', re.IGNORECASE),
    # 「定理Xより明らか」のような知識断言
    re.compile(r'\bby\s+(?:theorem|lemma|corollary)\s+\w+\s*,?\s*(?:we\s+(?:get|have|obtain)|it\s+follows)',
               re.IGNORECASE),
]

# MCQ 直接ラベル（ステップの最後が "A" / "B" / "C" / "D" / "E" だけの場合）
_MCQ_DIRECT_LABEL = re.compile(r'^\s*[A-E]\s*$')


class GateB:
    """
    Gate B: "答えっぽい出力" 検出

    candidates.steps の各ステップに対して：
    1. 断言フレーズパターンチェック
    2. MCQ直接ラベルチェック
    3. "= 数値" で終わる断言チェック

    approachフィールドにも軽微チェックを行う。
    """

    @staticmethod
    def _check_text(text: str) -> Optional[str]:
        """テキストが答えっぽいなら理由を返す"""
        for pat in _ANSWER_ASSERTION_PATTERNS:
            if pat.search(text):
                return f"answer_assertion_pattern:{pat.pattern[:40]}"
        if _MCQ_DIRECT_LABEL.match(text):
            return "mcq_direct_label"
        return None

    @classmethod
    def check(cls, contract: LLMContract) -> GateResult:
        details = []

        for i, candidate in enumerate(contract.candidates):
            # approach チェック
            reason = cls._check_text(candidate.approach)
            if reason:
                return GateResult(
                    passed=False, gate="B",
                    reason=f"candidate[{i}].approach:{reason}",
                    details=[candidate.approach[:100]]
                )

            # steps チェック
            for j, step in enumerate(candidate.steps):
                reason = cls._check_text(step)
                if reason:
                    return GateResult(
                        passed=False, gate="B",
                        reason=f"candidate[{i}].steps[{j}]:{reason}",
                        details=[step[:100]]
                    )

            # method_name に "answer" が含まれていたら警告のみ
            if "answer" in candidate.method_name.lower():
                details.append(f"Suspicious method_name: {candidate.method_name}")

        return GateResult(passed=True, gate="B", details=details)


# ─────────────────────────────────────────────────────────────────────
# Gate C — 制約整合チェック（軽量版）
# ─────────────────────────────────────────────────────────────────────

class GateC:
    """
    Gate C: 制約整合チェック

    LLM が提示した ir.constraints に明らかな矛盾がないかを確認する。
    例: "n > 0" かつ entities に n=-5 が含まれる場合

    現状は基本的なチェックのみ。
    """

    @classmethod
    def check(cls, contract: LLMContract) -> GateResult:
        details = []

        # 検証不能フラグが立っている場合は早期パス
        if not contract.ir.is_verifiable:
            details.append(f"is_verifiable=False: {contract.ir.unverifiable_reason}")
            return GateResult(passed=True, gate="C", details=details)

        # answer_schema が "proof_sketch" の場合は計算チェックをスキップ
        if contract.ir.answer_schema == "proof_sketch":
            details.append("proof_sketch: skipping numeric constraint check")
            return GateResult(passed=True, gate="C", details=details)

        # 基本的な整合チェック（entities の型が answer_schema と矛盾しないか）
        entity_types = {e.type for e in contract.ir.entities}
        schema = contract.ir.answer_schema
        if schema == "boolean" and "numeric" in entity_types:
            details.append("Warning: boolean schema with numeric entities (may be OK)")

        return GateResult(passed=True, gate="C", details=details)


# ─────────────────────────────────────────────────────────────────────
# run_gates — 全 Gate を順番に実行
# ─────────────────────────────────────────────────────────────────────

def run_gates(contract: LLMContract) -> GateResult:
    """
    Gate A → B → C の順に実行。
    最初に失敗した Gate の GateResult を返す。
    全部通ったら passed=True の GateResult を返す。
    """
    for gate_cls, gate_name in [(GateA, "A"), (GateB, "B"), (GateC, "C")]:
        result = gate_cls.check(contract)
        if not result.passed:
            return result

    return GateResult(passed=True, gate="ALL", details=["All gates passed"])
