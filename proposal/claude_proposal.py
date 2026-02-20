"""
ClaudeProposalGenerator — LLM を "proposal専用" に封じ込める層

原則:
  - LLM は候補プログラム設計図（piece列 + slots）だけを出す
  - 最終答え・計算・選択肢の決定は禁止
  - 出力は JSON のみ（自然文ゼロ）
  - GateA を通らなければ即 reject
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# anthropic SDK（pip install anthropic）
try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


# ─── プロンプト定義 ──────────────────────────────────────────────────────────

_SYSTEM = """You are a proposal-only generator for Verantyx verification engine.

RULES (strictly enforced):
- Return ONLY valid JSON. No prose, no explanation, no markdown.
- Do NOT compute the final answer or decide the correct choice.
- Do NOT output fields named: answer, result, solution, correct, final.
- Produce 3-8 candidate_programs (piece pipelines) + slots + verification_plan.

Your role: generate CANDIDATE DESIGNS that Verantyx will verify independently."""

_PROMPT_TEMPLATE = """Return JSON matching schema "verantyx.ir.v1".

Problem: {problem}

Executor hint (may be empty): {hint}
MCQ choices (may be empty): {choices}

Required JSON structure:
{{
  "schema": "verantyx.ir.v1",
  "intent": {{
    "task": "select_choice|compute_value|prove_or_refute|construct_object",
    "query": "<what is being asked, <=80 chars>",
    "answer_type": "choice|int|rational|real|string|set|expression|bool"
  }},
  "objects": [
    {{"name": "<var>", "type": "int|real|str|graph|matrix|set|poly", "role": "given|parameter|unknown", "constraints": ["<constraint>"]}}
  ],
  "signals": {{
    "domain": "<primary domain>",
    "needs_tool": ["sympy","z3","python_eval","search"]
  }},
  "candidate_programs": [
    {{
      "program_id": "p0",
      "summary": "<<=120 chars>",
      "pipeline": [
        {{"piece": "<piece_id>", "args": {{"<slot>": "$<var>"}}}}
      ],
      "slots": {{
        "<slot>": {{"from": "text", "value": null}}
      }}
    }}
  ],
  "verification_plan": {{
    "worldgen": [{{"mode": "small_random", "count": 15}}],
    "verify": [{{"type": "cross_check", "methods": ["python_eval","sympy"]}}]
  }}
}}

Return ONLY the JSON object above."""


# ─── 設定 ────────────────────────────────────────────────────────────────────

@dataclass
class ProposalConfig:
    model: str = "claude-3-5-haiku-latest"   # コスト最小（proposal専用）
    max_tokens: int = 1500
    temperature: float = 0.0
    timeout_ms: float = 8000.0               # 8秒タイムアウト
    min_candidates: int = 1                  # 最低この数の candidate_programs が必要


# ─── GateA: IR schema validator ──────────────────────────────────────────────

def gateA_validate_ir(ir: Any) -> bool:
    """
    GateA: LLM 出力の IR を型・スキーマで検証
    失敗したら即 reject（LLM 暴走の安全弁）
    """
    if not isinstance(ir, dict):
        return False

    # 禁止フィールド（答えっぽい出力を即排除）
    BANNED = {"answer", "result", "solution", "correct", "final", "choice"}
    if BANNED & set(ir.keys()):
        return False

    # 必須フィールド
    if "candidate_programs" not in ir:
        return False
    progs = ir.get("candidate_programs", [])
    if not isinstance(progs, list) or len(progs) == 0:
        return False

    # 各 candidate_program に pipeline が必要
    for p in progs:
        if not isinstance(p, dict):
            return False
        if not p.get("pipeline") and not p.get("slots"):
            return False  # 完全に空の候補は reject

    return True


# ─── IRAdapter ────────────────────────────────────────────────────────────────

def make_candidates_from_ir(ir: dict) -> List[Dict[str, Any]]:
    """
    VerantyxIR.v1 → 既存 CEGISLoop が扱える candidate リストへ変換

    既存の make_candidates_from_executor_result() と同じ形式で返す。
    """
    candidates = []
    for prog in ir.get("candidate_programs", []):
        # slots から value を抽出
        slots = prog.get("slots", {})
        slot_values = {k: v.get("value") for k, v in slots.items() if isinstance(v, dict)}

        cand = {
            "source": "llm_proposal",
            "program_id": prog.get("program_id", "p?"),
            "summary": prog.get("summary", ""),
            "pipeline": prog.get("pipeline", []),
            "slots": slot_values,
            "metadata": {
                "from_llm": True,
                "ir_domain": ir.get("signals", {}).get("domain", "unknown"),
                "needs_tool": ir.get("signals", {}).get("needs_tool", []),
            }
        }
        candidates.append(cand)
    return candidates


# ─── メインクラス ─────────────────────────────────────────────────────────────

class ClaudeProposalGenerator:
    """
    Claude API を使った proposal 専用ジェネレータ

    呼び出し条件（should_call_proposal() で制御）:
      - Executor が候補ゼロ / 低確信 / slot 欠落
      - domain が ambiguous/unknown
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cfg: Optional[ProposalConfig] = None,
    ):
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic SDK が必要: pip install anthropic")

        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY が未設定")

        self.client = anthropic.Anthropic(api_key=key)
        self.cfg = cfg or ProposalConfig()
        self._call_count = 0
        self._total_tokens = 0

    def propose_ir(
        self,
        problem_text: str,
        executor_hint: str = "",
        choices: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        問題文から VerantyxIR.v1 を生成する

        Returns:
            IR dict (GateA 通過済み) | None (失敗 or GateA rejected)
        """
        choices = choices or []
        prompt = _PROMPT_TEMPLATE.format(
            problem=problem_text[:800],   # 長すぎる問題文を切り詰め
            hint=executor_hint[:200] if executor_hint else "(none)",
            choices=json.dumps(choices) if choices else "[]",
        )

        t0 = time.perf_counter()
        try:
            msg = self.client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._call_count += 1
            usage = getattr(msg, "usage", None)
            if usage:
                self._total_tokens += getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)

            # テキスト抽出
            text = ""
            for block in msg.content:
                if getattr(block, "type", None) == "text":
                    text += block.text

            # JSON パース
            # コードブロックがある場合は除去
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip().rstrip("```").strip()

            ir = json.loads(text)

            # GateA 検証
            if not gateA_validate_ir(ir):
                return None  # reject

            ir["_meta"] = {
                "elapsed_ms": elapsed_ms,
                "model": self.cfg.model,
                "source": "claude_proposal",
            }
            return ir

        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def should_call(self, executor_result: Any) -> bool:
        """
        LLM を呼ぶべきか判定（無駄撃ち防止）

        呼ぶ条件:
          A) candidates が空
          B) confidence < 0.4
          C) missing_slots がある
          D) domain が unknown/ambiguous
        """
        if executor_result is None:
            return True
        if not getattr(executor_result, "candidates", [True]):
            return True
        if getattr(executor_result, "confidence", 1.0) < 0.4:
            return True
        if getattr(executor_result, "missing_slots", []):
            return True
        domain = getattr(executor_result, "domain", "")
        if domain in ("unknown", "ambiguous", ""):
            return True
        return False

    @property
    def stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
        }
