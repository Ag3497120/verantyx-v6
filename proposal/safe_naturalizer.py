"""
safe_naturalizer.py
最終答えの自然文化（ノイズ最小版）

3段階:
  Stage F1: Deterministic Formatter（テンプレート、LLMなし）
  Stage F2: LLM Naturalizer（answer_payloadのみ、問題文なし、推論禁止）
  Stage F3: Faithfulness Checker（LLM出力を再パースして意味一致を検証）

設計原則:
  - 問題文をLLMに渡さない
  - 答え候補を渡さない（確定答えだけ）
  - LLMに推論させない（表現整形のみ）
  - 失敗時はテンプレートにフォールバック
"""

import json
import re
import hashlib
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 1. Answer Payload（Verantyxが確定した答え）
# ---------------------------------------------------------------------------

@dataclass
class AnswerPayload:
    """Phase 4（パズル推論）で確定した答え。"""
    answer_type: str          # "mcq" | "integer" | "float" | "expression" | "name" |
                              # "boolean" | "list" | "move_sequence" | "free_text"
    content: list[str]        # 確定した答えの構成要素 e.g. ["Rxf3", "Rf1#"]
    raw_value: Any = None     # 計算結果（int/floatなど）
    unit: str = ""            # 単位（あれば）
    language: str = "en"      # 出力言語


@dataclass
class NaturalizationResult:
    surface_text: str                    # 最終出力文
    method: str                          # "template" | "llm_naturalized" | "fallback"
    faithful: bool                       # faithfulness check 通過
    faithfulness_map: list[dict] = field(default_factory=list)  # payload → surface 対応
    payload_hash: str = ""               # 改ざん検知用
    audit_log: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Stage F1: Deterministic Formatter（テンプレート）
# ---------------------------------------------------------------------------

class DeterministicFormatter:
    """
    テンプレートベースの答え整形。LLM不使用。
    HLEの答え型の大半はこれで十分。
    """

    TEMPLATES = {
        "mcq":            lambda p: p.content[0] if p.content else "INCONCLUSIVE",
        "integer":        lambda p: str(p.content[0]) if p.content else "INCONCLUSIVE",
        "float":          lambda p: str(p.content[0]) if p.content else "INCONCLUSIVE",
        "expression":     lambda p: p.content[0] if p.content else "INCONCLUSIVE",
        "name":           lambda p: p.content[0] if p.content else "INCONCLUSIVE",
        "boolean":        lambda p: p.content[0] if p.content else "INCONCLUSIVE",
        "list":           lambda p: ", ".join(p.content),
        "move_sequence":  lambda p: ", ".join(p.content),
    }

    def format(self, payload: AnswerPayload) -> NaturalizationResult | None:
        """
        テンプレートで整形できれば NaturalizationResult を返す。
        free_text など整形不能なら None（Stage F2 へ）。
        """
        template = self.TEMPLATES.get(payload.answer_type)
        if template is None:
            return None  # テンプレートなし → F2 へ

        text = template(payload)
        if payload.unit:
            text = f"{text} {payload.unit}"

        return NaturalizationResult(
            surface_text=text,
            method="template",
            faithful=True,  # テンプレートは改変しないので常に faithful
            faithfulness_map=[
                {"from_payload_index": i, "verbatim_or_equivalent": c}
                for i, c in enumerate(payload.content)
            ],
            payload_hash=_hash_payload(payload),
            audit_log=[f"template:{payload.answer_type}"],
        )


# ---------------------------------------------------------------------------
# 3. Stage F2: LLM Naturalizer（表現器、推論禁止）
# ---------------------------------------------------------------------------

# LLM に渡す system prompt（表現整形専用）
NATURALIZER_SYSTEM_PROMPT = """You are a surface text formatter. You convert structured data into natural language.

STRICT RULES:
- Do NOT infer, reason, explain, or add any information.
- Do NOT reference any original problem or question.
- Output ONLY the formatted text in the specified schema.
- Do NOT add facts, opinions, interpretations, or examples.
- Do NOT use words like "therefore", "because", "since", "so".
- Your ONLY job is grammatical joining and punctuation.
- If you cannot format, return: {"surface_text": "", "error": "cannot_format"}"""


def build_naturalizer_prompt(payload: AnswerPayload) -> str:
    """
    LLM に渡すユーザーメッセージ。
    問題文は含まない。answer_payload のみ。
    """
    request = {
        "task": "surface_realization_only",
        "instruction": "Do not infer, do not add facts, do not change content.",
        "answer_payload": {
            "type": payload.answer_type,
            "content": payload.content,
            "unit": payload.unit,
            "language": payload.language,
        },
        "format_constraints": {
            "language": payload.language,
            "length": "very_short",
            "allowed_transformations": [
                "punctuation", "joining", "light_grammatical_glue",
            ],
            "forbidden_transformations": [
                "new_facts", "reordering_meaning", "interpretation",
                "explanation", "reasoning", "opinion",
            ],
        },
        "output_schema": {
            "surface_text": "string",
            "faithfulness_map": [
                {
                    "from_payload_index": "int",
                    "verbatim_or_equivalent": "string",
                }
            ],
        },
    }
    return json.dumps(request, ensure_ascii=False, indent=2)


class LLMNaturalizer:
    """
    Stage F2: LLM を表現器として使う（推論禁止）。
    問題文は渡さない。answer_payload のみ。
    """

    def naturalize(self, payload: AnswerPayload) -> NaturalizationResult | None:
        """
        LLM に表現整形を依頼する。
        call_knowledge_llm を再利用。
        失敗時は None（F1 フォールバックへ）。
        """
        try:
            from knowledge.llm_knowledge_fetcher import call_knowledge_llm
        except ImportError:
            return None

        user_prompt = build_naturalizer_prompt(payload)

        try:
            raw = call_knowledge_llm(
                system=NATURALIZER_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=256,
            )
        except Exception:
            return None

        # JSON parse
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        surface = data.get("surface_text", "")
        if not surface:
            return None

        fmap = data.get("faithfulness_map", [])

        return NaturalizationResult(
            surface_text=surface,
            method="llm_naturalized",
            faithful=False,  # Stage F3 で検証するまで False
            faithfulness_map=fmap,
            payload_hash=_hash_payload(payload),
            audit_log=["llm_naturalizer"],
        )


# ---------------------------------------------------------------------------
# 4. Stage F3: Faithfulness Checker（再検証）
# ---------------------------------------------------------------------------

class FaithfulnessChecker:
    """
    LLM が生成した自然文が answer_payload と意味一致するか検証する。

    チェック内容:
      1. payload.content の各要素が surface_text に含まれているか
      2. 新規数値・新規固有名詞が追加されていないか
      3. 長さが妥当か（payload の2倍以上なら怪しい）
    """

    def check(
        self,
        payload: AnswerPayload,
        result: NaturalizationResult,
    ) -> NaturalizationResult:
        """faithful フラグを更新して返す。"""
        surface = result.surface_text.lower()
        issues: list[str] = []

        # Check 1: payload の各要素が surface に含まれるか
        for i, item in enumerate(payload.content):
            item_lower = item.lower().strip()
            if item_lower not in surface:
                # 完全一致でなくても、数値の場合は部分一致も許可
                if not any(part in surface for part in item_lower.split()):
                    issues.append(f"missing_content[{i}]='{item}'")

        # Check 2: surface に payload にない数値が追加されていないか
        payload_numbers = set()
        for item in payload.content:
            payload_numbers.update(re.findall(r"\d+\.?\d*", item))
        surface_numbers = set(re.findall(r"\d+\.?\d*", result.surface_text))
        new_numbers = surface_numbers - payload_numbers
        if new_numbers:
            issues.append(f"new_numbers_added={new_numbers}")

        # Check 3: 長さの妥当性
        payload_len = sum(len(c) for c in payload.content)
        if len(result.surface_text) > max(payload_len * 3, 100):
            issues.append(f"too_long: surface={len(result.surface_text)}, payload={payload_len}")

        # Check 4: 推論語彙の検出
        reasoning_words = ["therefore", "because", "since", "thus", "hence",
                           "so the answer", "which means", "this implies"]
        for word in reasoning_words:
            if word in surface:
                issues.append(f"reasoning_word='{word}'")

        result.faithful = len(issues) == 0
        result.audit_log.extend(issues)
        return result


# ---------------------------------------------------------------------------
# 5. 統合エントリポイント: SafeNaturalizer
# ---------------------------------------------------------------------------

class SafeNaturalizer:
    """
    答えの自然文化を安全に行う統合クラス。

    フロー:
      1. F1（テンプレート）で試みる → 成功ならそのまま返す
      2. F2（LLM表現器）で試みる → F3 で faithfulness check
      3. F3 通過なら返す、失敗なら F1 フォールバック
    """

    def __init__(self):
        self.f1 = DeterministicFormatter()
        self.f2 = LLMNaturalizer()
        self.f3 = FaithfulnessChecker()

    def naturalize(self, payload: AnswerPayload) -> NaturalizationResult:
        # --- Stage F1: テンプレート（最優先）---
        f1_result = self.f1.format(payload)
        if f1_result is not None:
            return f1_result

        # --- Stage F2: LLM 表現器 ---
        f2_result = self.f2.naturalize(payload)
        if f2_result is not None:
            # --- Stage F3: Faithfulness Check ---
            f2_result = self.f3.check(payload, f2_result)
            if f2_result.faithful:
                return f2_result
            else:
                # F3 失敗 → フォールバック
                pass

        # --- フォールバック: 素のjoin（LLMなし）---
        text = ", ".join(payload.content) if payload.content else "INCONCLUSIVE"
        if payload.unit:
            text = f"{text} {payload.unit}"
        return NaturalizationResult(
            surface_text=text,
            method="fallback",
            faithful=True,
            faithfulness_map=[
                {"from_payload_index": i, "verbatim_or_equivalent": c}
                for i, c in enumerate(payload.content)
            ],
            payload_hash=_hash_payload(payload),
            audit_log=["fallback:f1_none_f2_unfaithful"],
        )


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _hash_payload(payload: AnswerPayload) -> str:
    """改ざん検知用ハッシュ。"""
    data = json.dumps({
        "type": payload.answer_type,
        "content": payload.content,
        "unit": payload.unit,
    }, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]
