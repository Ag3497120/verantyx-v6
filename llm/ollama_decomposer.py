"""
OllamaDecomposer — ローカル LLM（ollama）を分解機として使うラッパー

推奨モデル:
  - qwen2.5:7b-instruct  (安定度・数式処理のバランス最良)
  - qwen2.5:14b-instruct (高精度、CPU要件高)
  - llama3.1:8b-instruct (英語中心・取り回し良し)
  - gemma2:2b-it         (高速ゲート/一次抽出用)

設計:
  - temperature=0（決定論的出力）
  - JSON Schema 制約強制（format="json"）
  - 不足スロット宣言（missing: [...]）必須化
  - slot_retry は不足分のみ・最大1回
  - Gate A/B/C で "答え" を即リジェクト
"""

from __future__ import annotations
import json
import re
import time
from typing import Optional, Dict, Any, List

from llm.contract import LLMContract, LLM_CONTRACT_SCHEMA_STR
from llm.gates import run_gates

MAX_RETRIES = 1          # slot_retry は不足分のみ・最大1回（kofdai原則）
OLLAMA_BASE_URL = "http://localhost:11434"
TIMEOUT_SEC = 30


# ─────────────────────────────────────────────────────────────────────
# System Prompt（Qwen2.5 / Llama 3.1 最適化版）
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Verantyx IR Extractor. Your ONLY job is structural extraction.

OUTPUT: A single JSON object. Nothing else. No prose, no explanation outside JSON.

MANDATORY RULES:
1. Extract structure only. Never compute, never solve, never answer.
2. If you cannot determine something, declare it in ir.missing — do NOT guess.
3. Forbidden output fields: answer, result, solution, correct_answer, final_answer
4. Steps in candidates must be PLANS, not results. "Apply formula X" is OK. "x = 42" is FORBIDDEN.
5. ir.missing is REQUIRED (use [] if nothing is missing, but always include the field).

JSON SCHEMA:
{
  "ir": {
    "task": "compute|count|find|decide|prove|classify|construct",
    "domain": "<domain string>",
    "answer_schema": "integer|decimal|boolean|option_label|expression|sequence|proof_sketch|string",
    "entities": [{"name": "<name>", "type": "<type>", "description": "<desc>", "constraints": []}],
    "constraints": ["<constraint string>"],
    "query_description": "<what is being asked>",
    "is_verifiable": true|false,
    "unverifiable_reason": "<if not verifiable, why>",
    "missing": ["<slot name if unclear>"]
  },
  "candidates": [
    {
      "method_name": "<method>",
      "approach": "<how to solve — NO ANSWER>",
      "required_tools": ["sympy", "z3", ...],
      "steps": ["Step 1: ...", "Step 2: ..."],
      "verify_spec": {
        "method": "numeric_check|symbolic_check|z3_smt|worldgen_sample",
        "description": "<how to verify>"
      }
    }
  ],
  "decomp_notes": "<optional>"
}

EXAMPLE (good):
  Problem: "What is 6!?"
  {
    "ir": {"task":"compute","domain":"number_theory","answer_schema":"integer",
           "entities":[{"name":"n","type":"constant","description":"6","constraints":["n=6"]}],
           "constraints":["n>=0"],"query_description":"Compute factorial of 6",
           "is_verifiable":true,"missing":[]},
    "candidates":[{"method_name":"factorial_definition",
                   "approach":"Apply factorial definition n! = n*(n-1)*...*1",
                   "required_tools":["sympy"],
                   "steps":["Step 1: Identify n=6","Step 2: Pass n to factorial executor"],
                   "verify_spec":{"method":"numeric_check","description":"Verify via math.factorial(6)"}}],
    "decomp_notes":""
  }

EXAMPLE (bad — NEVER do this):
  "steps": ["x = 720"]  ← FORBIDDEN (this is the answer)
  "answer": "720"       ← FORBIDDEN (forbidden field)
"""


# ─────────────────────────────────────────────────────────────────────
# OllamaDecomposer
# ─────────────────────────────────────────────────────────────────────

class OllamaDecomposer:
    """
    ローカル ollama モデルを分解機として使うラッパー。

    ollama API: POST http://localhost:11434/api/generate
    JSON mode: {"format": "json"} で JSON を強制。
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = TIMEOUT_SEC,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._check_availability()
        return self._available or False

    def _check_availability(self) -> None:
        """ollama が起動しているか + モデルが存在するか確認"""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            # モデル名の前方一致チェック（"qwen2.5:7b-instruct" → "qwen2.5"）
            model_base = self.model.split(":")[0]
            self._available = any(m.startswith(model_base) for m in models)
            if not self._available:
                print(f"[OllamaDecomposer] Model {self.model!r} not found in ollama. Available: {models}")
        except Exception as e:
            self._available = False

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
        if not self.available:
            return None

        prompt = self._build_prompt(problem_text, retry_hint)

        for attempt in range(MAX_RETRIES + 1):
            raw = self._call_ollama(prompt)
            if raw is None:
                return None

            # JSON 抽出
            json_str = self._extract_json(raw)
            if json_str is None:
                if attempt < MAX_RETRIES:
                    prompt = self._build_retry_prompt(problem_text, "Invalid JSON. Output ONLY the JSON object.")
                    continue
                return None

            # パース
            try:
                raw_dict = json.loads(json_str)
                contract = LLMContract.from_dict(raw_dict)
            except (ValueError, json.JSONDecodeError) as e:
                if attempt < MAX_RETRIES:
                    prompt = self._build_retry_prompt(problem_text, f"Parse error: {e}. Fix the JSON.")
                    continue
                return None

            # Gate 実行（raw_dict も渡して禁止フィールドを正確に検出）
            from llm.gates import GateA, GateB, GateC
            gate_a = GateA.check(contract, raw_dict=raw_dict)
            if not gate_a.passed:
                if attempt < MAX_RETRIES:
                    # 不足スロットのみ再プロンプト（コスト節約）
                    hint = f"Gate A failed: {gate_a.reason}. Fix only this: {gate_a.details}"
                    prompt = self._build_retry_prompt(problem_text, hint)
                    continue
                return None

            gate_b = GateB.check(contract)
            if not gate_b.passed:
                if attempt < MAX_RETRIES:
                    hint = f"Gate B: {gate_b.reason}. Your steps contain an answer. Remove it."
                    prompt = self._build_retry_prompt(problem_text, hint)
                    continue
                return None

            # missing スロットがある場合: slot_retry（最大1回）
            if contract.ir.missing and attempt < MAX_RETRIES:
                # missing スロットのみ再要求（問題文全体は再送しない）
                missing_str = ", ".join(contract.ir.missing)
                hint = (f"SLOT_RETRY: These slots are unclear: [{missing_str}]. "
                        f"Try harder to extract them from the problem. "
                        f"If truly unknown, keep them in missing[].")
                prompt = self._build_retry_prompt(problem_text, hint)
                continue

            contract.gate_passed = True
            return contract

        return None

    def _build_prompt(self, problem_text: str, hint: Optional[str] = None) -> str:
        prompt = f"PROBLEM:\n{problem_text}"
        if hint:
            prompt += f"\n\n[HINT] {hint}"
        return prompt

    def _build_retry_prompt(self, problem_text: str, hint: str) -> str:
        return f"PROBLEM:\n{problem_text}\n\n[RETRY] {hint}"

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """ollama /api/generate を呼び出す"""
        try:
            import urllib.request
            payload = json.dumps({
                "model": self.model,
                "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                "stream": False,
                "format": "json",   # JSON mode 強制
                "options": {
                    "temperature": 0,    # 決定論的出力（kofdai原則）
                    "num_predict": 1024, # 最大トークン
                }
            }).encode()

            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
            return data.get("response", "")
        except Exception as e:
            return None

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """レスポンスから JSON 部分を抽出"""
        text = text.strip()
        # JSON mode の場合は全体が JSON
        if text.startswith("{"):
            return text
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
# 2段階構成: Gemma2-2B（一次） + Qwen2.5-7B（補完/監査）
# ─────────────────────────────────────────────────────────────────────

class TieredDecomposer:
    """
    2段構成のDecomposer（kofdaiの推奨構成）:
      Stage 1 (fast gate): Gemma 2 2B — 一次IR抽出
      Stage 2 (audit):     Qwen2.5 7B — missing スロット補完/監査

    Stage 1 で missing=[] かつ Gate 通過 → Stage 2 をスキップ（節約）
    Stage 1 で missing あり → Stage 2 に委譲（不足スロットのみ再要求）
    """

    def __init__(
        self,
        fast_model: str = "gemma2:2b-it",
        audit_model: str = "qwen2.5:7b-instruct",
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.fast = OllamaDecomposer(model=fast_model, base_url=base_url)
        self.audit = OllamaDecomposer(model=audit_model, base_url=base_url)

    def decompose(self, problem_text: str) -> Optional[LLMContract]:
        # Stage 1: 高速一次抽出
        contract = self.fast.decompose(problem_text)

        if contract is not None:
            # missing なし & gate 通過 → そのまま返す
            if not contract.ir.missing:
                contract.decomp_notes += " [stage1_only]"
                return contract
            # missing あり → Stage 2 で補完（不足スロットのみ）
            missing_hint = f"Stage1 result has missing slots: {contract.ir.missing}. Fill them."

        else:
            missing_hint = "Stage1 failed to parse. Re-extract."

        # Stage 2: 監査・補完
        if self.audit.available:
            contract2 = self.audit.decompose(problem_text, retry_hint=missing_hint)
            if contract2 is not None:
                contract2.decomp_notes += " [stage2_audit]"
                return contract2

        return contract  # Stage 2 も失敗 → Stage 1 の結果を（missing あっても）返す
