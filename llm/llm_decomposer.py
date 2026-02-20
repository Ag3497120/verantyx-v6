"""
LLMDecomposer — LLM を "分解機" に固定するラッパー

LLM の3つの役割を厳格に制御：
  1. IR化（構造抽出）
  2. 候補生成（解法案まで、最終回答禁止）
  3. verify/worldgen 提案

実装：
  - anthropic Claude API を使用（他モデルにも対応可能）
  - LLMContract JSON スキーマへの厳格な出力を強制
  - Gate A/B/C で検証し、失敗時は不足スロットのみ re-prompt
  - 最大 MAX_RETRIES 回試行後、None を返す（pipeline が fallback 処理）
"""

from __future__ import annotations
import json
import os
import re
import time
from typing import Optional, List, Tuple

from llm.contract import LLMContract, LLM_CONTRACT_SCHEMA_STR
from llm.gates import GateResult, run_gates

MAX_RETRIES = 2          # Gate 失敗時の最大再試行回数
TIMEOUT_SEC = 15         # 1回のAPI呼び出しタイムアウト


# ─────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are the Verantyx LLM Decomposer.

YOUR ROLE IS STRICTLY LIMITED TO THREE FUNCTIONS:
1. Structural extraction (IR): Identify variables, constraints, domains, answer schema
2. Solution candidate generation: Propose METHODS and STEPS (NOT final answers)
3. Verification spec proposal: Describe how to verify and generate counterexamples

ABSOLUTE PROHIBITIONS:
- Do NOT output the final answer, result, or solution
- Do NOT assert "the answer is X" or "therefore X = 42"
- Do NOT select a MCQ option (A/B/C/D/E) as your output
- Do NOT make knowledge claims like "by Theorem X, clearly..."
- Do NOT perform numeric computation (no arithmetic, no evaluation)

YOUR OUTPUT MUST BE VALID JSON following this schema:
{LLM_CONTRACT_SCHEMA_STR}

If you cannot decompose the problem (e.g., pure knowledge recall, visual-only),
set ir.is_verifiable = false and explain in ir.unverifiable_reason.
Provide 1-3 candidates in the 'candidates' array.
Steps should be PLANS, not computed results.

EXAMPLE of a GOOD step: "Apply the quadratic formula with a=1, b=-5, c=6 to find x"
EXAMPLE of a BAD step: "x = 2 or x = 3" (this is the answer — FORBIDDEN)
"""


# ─────────────────────────────────────────────────────────────────────
# LLMDecomposer
# ─────────────────────────────────────────────────────────────────────

class LLMDecomposer:
    """
    LLM を "分解機" として使うラッパー。

    API キーがない場合は MockDecomposer として機能し、
    RuleBasedDecomposer に委譲する（フォールバック）。
    """

    def __init__(self, model: str = "claude-3-5-haiku-20241022", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        self._available = False

        if self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                self._available = True
            except ImportError:
                pass  # anthropic パッケージなし → フォールバック

    @property
    def available(self) -> bool:
        return self._available

    def decompose(
        self,
        problem_text: str,
        retry_hint: Optional[str] = None,
    ) -> Optional[LLMContract]:
        """
        問題文を分解して LLMContract を返す。

        Args:
            problem_text: 問題文
            retry_hint: Gate 失敗時の再プロンプトヒント（不足スロットのみ）

        Returns:
            LLMContract (gate passed) or None (failed / unavailable)
        """
        if not self._available:
            return None

        user_msg = problem_text
        if retry_hint:
            user_msg = f"{problem_text}\n\n[RETRY HINT] {retry_hint}"

        for attempt in range(MAX_RETRIES + 1):
            raw = self._call_api(user_msg)
            if raw is None:
                return None

            # JSON 抽出（```json ... ``` を考慮）
            json_str = self._extract_json(raw)
            if json_str is None:
                if attempt < MAX_RETRIES:
                    user_msg = f"{problem_text}\n\n[RETRY] Your previous response was not valid JSON. Output ONLY a JSON object."
                    continue
                return None

            # パース
            try:
                contract = LLMContract.from_json(json_str)
            except ValueError as e:
                if attempt < MAX_RETRIES:
                    user_msg = f"{problem_text}\n\n[RETRY] Parse error: {e}. Fix and resubmit."
                    continue
                return None

            # Gate 実行
            gate_result = run_gates(contract)
            if gate_result.passed:
                contract.gate_passed = True
                return contract
            else:
                contract.gate_passed = False
                contract.gate_rejection_reason = gate_result.reason
                if attempt < MAX_RETRIES:
                    # 失敗した Gate の情報のみ渡す（re-prompt は最小限）
                    hint = f"Gate {gate_result.gate} failed: {gate_result.reason}. Fix only this issue."
                    if gate_result.details:
                        hint += f" Details: {gate_result.details[0]}"
                    user_msg = f"{problem_text}\n\n[GATE FAILURE - RETRY] {hint}"
                    continue
                return None  # 全試行失敗

        return None

    def _call_api(self, user_message: str) -> Optional[str]:
        """Anthropic API を呼び出す"""
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                timeout=TIMEOUT_SEC,
            )
            return response.content[0].text
        except Exception:
            return None

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """レスポンスから JSON 部分を抽出"""
        # ```json ... ``` ブロック
        code_block = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
        if code_block:
            return code_block.group(1).strip()

        # { ... } で始まる部分
        brace_match = re.search(r'\{[\s\S]+\}', text)
        if brace_match:
            return brace_match.group(0).strip()

        return None


# ─────────────────────────────────────────────────────────────────────
# Policy Gate — 汚染チェック（ベンチマーク固有表現ブロック）
# ─────────────────────────────────────────────────────────────────────

class PolicyGate:
    """
    LLM Decomposer の入力/出力に対して「ベンチマーク汚染」を防ぐ。

    NG（ベンチマーク汚染）:
    - HLE の解答・解説を含むデータを取り込む
    - 問題IDや固有表現でデータベースを検索して答えに辿り着く
    - 事前にHLE全体を見て専用検出器を作る

    OK（フェア）:
    - 一般知識（教科書レベル、定義、公理）を参照する
    - 問題文から抽出した "一般概念" で検索する
    - 検証器（SymPy/Z3/Lean）で候補を確認する
    """

    # HLE固有と思われる極めて特異的な表現（これが出たらブロック）
    # ※ 問題文そのものを暗記したような表現
    _BENCHMARK_SPECIFIC_PATTERNS = [
        # general_detectors.py 相当の固有表現
        re.compile(r'tardigrade.*ftir.*coiled', re.IGNORECASE),
        re.compile(r'spine surgeon.*burst fracture.*l[1-3]', re.IGNORECASE),
        re.compile(r'pseudomonas.*electroporation.*colou?r', re.IGNORECASE),
        # "HLE question #N" のような参照
        re.compile(r'\bhle\s+(?:question|problem|item)\s+#?\d+\b', re.IGNORECASE),
        re.compile(r'\bhle[-_]\d{3,}\b', re.IGNORECASE),
    ]

    @classmethod
    def check_input(cls, problem_text: str) -> Optional[str]:
        """
        問題文がHLE固有の表現を直接参照していないかチェック。
        （問題文は渡してOK、固有検索は禁止）
        Returns: None if OK, reason string if blocked
        """
        # 問題文の入力自体は OK（問題文を読むのはフェア）
        # 将来的に: 検索クエリに問題IDを含む場合だけブロック
        return None

    @classmethod
    def check_output(cls, contract: LLMContract) -> Optional[str]:
        """
        LLM 出力に HLE 固有の表現（=暗記した答え）が含まれないかチェック。
        Returns: None if OK, reason string if blocked
        """
        text = contract.to_json()
        for pat in cls._BENCHMARK_SPECIFIC_PATTERNS:
            if pat.search(text):
                return f"benchmark_contamination_detected:{pat.pattern}"
        return None
