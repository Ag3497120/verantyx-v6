#!/usr/bin/env python3
"""
LLM Decomposer (Gemma2のみ - 軽量化設計)
役割: IR抽出・候補生成のみ（推論・回答禁止）

構成:
- Gemma2-2B IT: 一次抽出（高速ゲート）
- Gemma2-9B IT: 補完・監査（不足検出時のみ）
"""
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Gemma2の軽量ロード（transformersを使用）
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARNING] transformers not available. LLMDecomposer will run in stub mode.")


class GateViolation(Enum):
    """ゲート違反の種類"""
    FORBIDDEN_WORD = "forbidden_word"
    ANSWER_CLAIM = "answer_claim"
    MISSING_SLOTS = "missing_slots"
    INVALID_JSON = "invalid_json"
    COMPUTATION = "computation"


@dataclass
class IRResult:
    """IR抽出結果"""
    variables: List[str]
    constraints: List[str]
    target: str
    missing: List[str]
    format_type: str  # "MCQ" | "free-form" | "proof"
    visual_needed: bool


@dataclass
class Candidate:
    """解法候補（答えなし）"""
    method: str
    steps: List[str]
    verify_tool: str  # "sympy" | "z3" | "vision" | "search"


@dataclass
class DecomposeResult:
    """分解結果"""
    ir: IRResult
    candidates: List[Candidate]
    verify_spec: Dict
    gate_violations: List[GateViolation]


class LLMDecomposer:
    """
    LLMを分解機に限定（Gemma2のみ）

    禁止事項:
    - 数値計算
    - 最終回答の生成
    - 知識断言（「定理Xより明らか」など）
    - 自己採点
    """

    # 禁止語リスト（Gate A）
    FORBIDDEN_WORDS = [
        "answer is", "correct answer", "therefore the answer",
        "明らか", "自明", "よって答えは",
        "= ", "==", "equals",  # 計算結果っぽい
    ]

    # 必須スロット
    REQUIRED_SLOTS = ["variables", "constraints", "target", "missing"]

    def __init__(self,
                 primary_model: str = "google/gemma-2-2b-it",
                 fallback_model: str = "google/gemma-2-9b-it",
                 device: str = "cpu"):
        """
        Args:
            primary_model: 一次抽出用モデル（Gemma2-2B）
            fallback_model: 補完用モデル（Gemma2-9B）
            device: "cpu" or "cuda"
        """
        self.device = device

        if not TRANSFORMERS_AVAILABLE:
            print("[LLMDecomposer] Running in STUB mode (no transformers)")
            self.primary_model = None
            self.fallback_model = None
            self.tokenizer = None
            return

        print(f"[LLMDecomposer] Loading {primary_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(primary_model)
        self.primary_model = AutoModelForCausalLM.from_pretrained(
            primary_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )

        print(f"[LLMDecomposer] Loading {fallback_model}...")
        self.fallback_model = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )

        print("[LLMDecomposer] Models loaded successfully")

    def decompose(self, question: str, choices: Optional[List[tuple]] = None) -> DecomposeResult:
        """
        問題を分解（IR抽出 + 候補生成）

        Args:
            question: 問題文
            choices: 選択肢 [(label, text), ...] or None

        Returns:
            DecomposeResult
        """
        # 一次抽出（Gemma2-2B）
        raw_output = self._extract_with_primary(question, choices)

        # Gate A: 禁止語チェック
        violations = self._check_gate_a(raw_output)

        # JSON解析
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            violations.append(GateViolation.INVALID_JSON)
            # フォールバック（Gemma2-9B）で再試行
            raw_output = self._extract_with_fallback(question, choices)
            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError:
                # 完全に失敗
                return self._create_error_result(violations)

        # Gate B: 必須スロット確認
        missing_slots = self._check_missing_slots(parsed)
        if missing_slots:
            violations.append(GateViolation.MISSING_SLOTS)
            # 不足スロットのみ補完（Gemma2-9B）
            parsed = self._補完_missing_slots(question, parsed, missing_slots)

        # 構造化
        ir = IRResult(
            variables=parsed.get("variables", []),
            constraints=parsed.get("constraints", []),
            target=parsed.get("target", ""),
            missing=parsed.get("missing", []),
            format_type="MCQ" if choices else "free-form",
            visual_needed=parsed.get("visual_needed", False)
        )

        candidates = [
            Candidate(
                method=c.get("method", ""),
                steps=c.get("steps", []),
                verify_tool=c.get("verify_tool", "sympy")
            )
            for c in parsed.get("candidates", [])
        ]

        verify_spec = parsed.get("verify_spec", {})

        return DecomposeResult(
            ir=ir,
            candidates=candidates,
            verify_spec=verify_spec,
            gate_violations=violations
        )

    def _extract_with_primary(self, question: str, choices: Optional[List[tuple]]) -> str:
        """一次抽出（Gemma2-2B）"""
        if not TRANSFORMERS_AVAILABLE or self.primary_model is None:
            # STUB
            return self._stub_output(question, choices)

        prompt = self._build_decompose_prompt(question, choices)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.primary_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,  # 決定的
                do_sample=False
            )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # プロンプト部分を除去
        json_start = output_text.find("{")
        if json_start != -1:
            return output_text[json_start:]

        return output_text

    def _extract_with_fallback(self, question: str, choices: Optional[List[tuple]]) -> str:
        """補完抽出（Gemma2-9B）"""
        if not TRANSFORMERS_AVAILABLE or self.fallback_model is None:
            return self._stub_output(question, choices)

        prompt = self._build_decompose_prompt(question, choices)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.fallback_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False
            )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        json_start = output_text.find("{")
        if json_start != -1:
            return output_text[json_start:]

        return output_text

    def _build_decompose_prompt(self, question: str, choices: Optional[List[tuple]]) -> str:
        """分解用プロンプト生成"""
        prompt = f"""You are a problem decomposer. Extract the structure ONLY. DO NOT solve or compute.

Question: {question}
"""

        if choices:
            prompt += "\nChoices:\n"
            for label, text in choices:
                prompt += f"  {label}. {text}\n"

        prompt += """
Output JSON (REQUIRED format):
{
  "variables": ["list of variables/entities"],
  "constraints": ["list of constraints/conditions"],
  "target": "what to find/prove",
  "missing": ["list of unclear/missing information"],
  "visual_needed": true/false,
  "candidates": [
    {
      "method": "approach description (NO answer)",
      "steps": ["step1", "step2", ...],
      "verify_tool": "sympy|z3|vision|search"
    }
  ],
  "verify_spec": {
    "tool": "...",
    "check": "..."
  }
}

FORBIDDEN:
- Do NOT compute numbers
- Do NOT claim answers
- Do NOT say "therefore" or "明らか"

JSON output:
"""
        return prompt

    def _check_gate_a(self, output: str) -> List[GateViolation]:
        """Gate A: 禁止語チェック"""
        violations = []

        output_lower = output.lower()

        for word in self.FORBIDDEN_WORDS:
            if word in output_lower:
                violations.append(GateViolation.FORBIDDEN_WORD)
                break

        # 数値計算っぽいパターン
        if re.search(r'\d+\s*[+\-*/=]\s*\d+', output):
            violations.append(GateViolation.COMPUTATION)

        return violations

    def _check_missing_slots(self, parsed: dict) -> List[str]:
        """必須スロット確認"""
        missing = []
        for slot in self.REQUIRED_SLOTS:
            if slot not in parsed:
                missing.append(slot)
            # Allow empty list for "missing" field (it's valid if nothing is missing)
            elif slot != "missing" and not parsed[slot]:
                missing.append(slot)
        return missing

    def _補完_missing_slots(self, question: str, parsed: dict, missing_slots: List[str]) -> dict:
        """不足スロットのみ補完（Gemma2-9B）"""
        # TODO: 不足スロットのみを再抽出
        # 現時点では元のparsedをそのまま返す
        return parsed

    def _stub_output(self, question: str, choices: Optional[List[tuple]]) -> str:
        """STUB出力（transformers未導入時）"""
        return json.dumps({
            "variables": ["x", "y"],
            "constraints": ["x > 0"],
            "target": "find x",
            "missing": [],
            "visual_needed": False,
            "candidates": [
                {
                    "method": "algebraic",
                    "steps": ["step1", "step2"],
                    "verify_tool": "sympy"
                }
            ],
            "verify_spec": {
                "tool": "sympy",
                "check": "verify_solution"
            }
        })

    def _create_error_result(self, violations: List[GateViolation]) -> DecomposeResult:
        """エラー結果生成"""
        return DecomposeResult(
            ir=IRResult(
                variables=[],
                constraints=[],
                target="",
                missing=["extraction_failed"],
                format_type="unknown",
                visual_needed=False
            ),
            candidates=[],
            verify_spec={},
            gate_violations=violations
        )


def demo():
    """デモ実行"""
    decomposer = LLMDecomposer(device="cpu")

    # テスト問題
    questions = [
        ("What is C(10,3)?", [("A", "100"), ("B", "120"), ("C", "150"), ("D", "200")]),
        ("Prove that the chromatic number of the Petersen graph is 3.", None),
    ]

    for question, choices in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        if choices:
            print(f"Choices: {choices}")
        print(f"{'='*60}")

        result = decomposer.decompose(question, choices)

        print(f"\n[IR]")
        print(f"  Variables: {result.ir.variables}")
        print(f"  Constraints: {result.ir.constraints}")
        print(f"  Target: {result.ir.target}")
        print(f"  Missing: {result.ir.missing}")

        print(f"\n[Candidates] ({len(result.candidates)})")
        for i, c in enumerate(result.candidates):
            print(f"  {i+1}. {c.method} (tool: {c.verify_tool})")

        if result.gate_violations:
            print(f"\n[Gate Violations] {result.gate_violations}")


if __name__ == "__main__":
    demo()
