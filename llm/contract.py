"""
LLMContract — LLM出力を厳格に制御するスキーマ定義

LLM の役割を3つに限定する：
  1. IR化（構造抽出）
  2. 候補生成（"解法案"まで、最終回答禁止）
  3. verify/worldgen 提案

これ以外の出力は Gate A/B で即リジェクト。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


# ─────────────────────────────────────────────────────────────────────
# Sub-schemas
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LLMEntitySpec:
    """IRエンティティ（変数・対象）"""
    name: str           # 変数名 / 対象名
    type: str           # "variable" | "constant" | "set" | "relation" | "function"
    description: str    # 自然言語での説明
    constraints: List[str] = field(default_factory=list)  # 型・範囲など

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMEntitySpec":
        return cls(
            name=str(d.get("name", "")),
            type=str(d.get("type", "unknown")),
            description=str(d.get("description", "")),
            constraints=[str(c) for c in d.get("constraints", [])],
        )


@dataclass
class LLMVerifySpec:
    """検証仕様（どう検証するか、どう反例を作るか）"""
    method: str             # "numeric_check" | "symbolic_check" | "z3_smt" | "worldgen_sample"
    description: str        # 検証の説明
    worldgen: Optional[Dict[str, Any]] = None   # worldgen パラメータ
    check_code: Optional[str] = None            # 検証コード（Pythonスニペット）

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "method": self.method,
            "description": self.description,
        }
        if self.worldgen:
            d["worldgen"] = self.worldgen
        if self.check_code:
            d["check_code"] = self.check_code
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMVerifySpec":
        return cls(
            method=str(d.get("method", "unknown")),
            description=str(d.get("description", "")),
            worldgen=d.get("worldgen"),
            check_code=d.get("check_code"),
        )


@dataclass
class LLMCandidate:
    """
    解法候補（解答禁止 — 解法の手順案まで）

    LLM は最終回答を出してはいけない。
    代わりに「この方法で解ける」「この式を使う」「この手順を踏む」を出す。
    実際の計算・答えの確定は Verantyx (Executor + CEGIS) が行う。
    """
    method_name: str            # 解法名 ("quadratic_formula", "pigeonhole", "truth_table", etc.)
    approach: str               # 解法の説明（自然言語）
    required_tools: List[str]   # 必要なツール ("sympy", "z3", "networkx", "cas", etc.)
    steps: List[str]            # 解法ステップ（"step 1: define ..." — 計算結果禁止）
    verify_spec: Optional[LLMVerifySpec] = None
    confidence_hint: float = 0.5    # LLM自身の確度ヒント（Verantyxは使用しない — 検証で上書き）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "approach": self.approach,
            "required_tools": self.required_tools,
            "steps": self.steps,
            "verify_spec": self.verify_spec.to_dict() if self.verify_spec else None,
            "confidence_hint": self.confidence_hint,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMCandidate":
        vs = d.get("verify_spec")
        return cls(
            method_name=str(d.get("method_name", "unknown")),
            approach=str(d.get("approach", "")),
            required_tools=[str(t) for t in d.get("required_tools", [])],
            steps=[str(s) for s in d.get("steps", [])],
            verify_spec=LLMVerifySpec.from_dict(vs) if vs else None,
            confidence_hint=float(d.get("confidence_hint", 0.5)),
        )


@dataclass
class LLMIRSpec:
    """
    LLMが生成する IR（中間表現）
    RuleBasedDecomposer の IR と互換だが、LLM 特有のフィールドを持つ
    """
    task: str               # "compute" | "count" | "decide" | "find" | "prove" | "classify"
    domain: str             # "arithmetic" | "algebra" | "geometry" | "logic" | ...
    answer_schema: str      # "integer" | "decimal" | "boolean" | "option_label" | "expression" | "proof_sketch"
    entities: List[LLMEntitySpec] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    query_description: str = ""     # 何を求めるかの自然言語説明
    input_modalities: List[str] = field(default_factory=list)  # ["text"] | ["text", "image"]
    is_verifiable: bool = True      # 検証可能かどうかの LLM 判断
    unverifiable_reason: Optional[str] = None  # 検証不能なら理由
    # ── 不足宣言（必須）────────────────────────────────────────────────
    # 解釈できなかった・勝手に埋めたくないスロットを明示する。
    # 空リストは「すべて解釈できた」を意味する。
    # 例: ["target_quantity_definition", "variable_binding", "unit"]
    missing: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "domain": self.domain,
            "answer_schema": self.answer_schema,
            "entities": [e.to_dict() for e in self.entities],
            "constraints": self.constraints,
            "query_description": self.query_description,
            "input_modalities": self.input_modalities,
            "is_verifiable": self.is_verifiable,
            "unverifiable_reason": self.unverifiable_reason,
            "missing": self.missing,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMIRSpec":
        return cls(
            task=str(d.get("task", "compute")),
            domain=str(d.get("domain", "unknown")),
            answer_schema=str(d.get("answer_schema", "integer")),
            entities=[LLMEntitySpec.from_dict(e) for e in d.get("entities", [])],
            constraints=[str(c) for c in d.get("constraints", [])],
            query_description=str(d.get("query_description", "")),
            input_modalities=d.get("input_modalities", ["text"]),
            is_verifiable=bool(d.get("is_verifiable", True)),
            unverifiable_reason=d.get("unverifiable_reason"),
            missing=[str(m) for m in d.get("missing", [])],
        )


@dataclass
class LLMContract:
    """
    LLM に要求し、受け取る契約オブジェクト。

    LLM はこのスキーマに厳密に従って JSON を出力しなければならない。
    - 'answer', 'result', 'solution', 'correct_*', 'final_*' フィールドは禁止
    - steps の最後が "the answer is X" のような文は禁止
    - 数値の断言（「= 42」で終わるstep）は禁止
    """
    ir: LLMIRSpec
    candidates: List[LLMCandidate] = field(default_factory=list)
    decomp_notes: str = ""      # 構造抽出に関する補足（任意）

    # メタデータ（Gate 結果の記録用）
    gate_passed: bool = False
    gate_rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ir": self.ir.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "decomp_notes": self.decomp_notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMContract":
        ir_d = d.get("ir", {})
        return cls(
            ir=LLMIRSpec.from_dict(ir_d),
            candidates=[LLMCandidate.from_dict(c) for c in d.get("candidates", [])],
            decomp_notes=str(d.get("decomp_notes", "")),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "LLMContract":
        """JSON文字列からパース。パース失敗時は ValueError を送出。"""
        try:
            d = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse error: {e}")
        if not isinstance(d, dict):
            raise ValueError("LLMContract must be a JSON object")
        if "ir" not in d:
            raise ValueError("Missing required field: 'ir'")
        return cls.from_dict(d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────
# JSON Schema (for LLM system prompt)
# ─────────────────────────────────────────────────────────────────────

LLM_CONTRACT_SCHEMA_STR = """
{
  "$schema": "LLMContract/v1",
  "description": "Verantyx LLM Decomposer Output Schema. STRICT COMPLIANCE REQUIRED.",
  "required": ["ir", "candidates"],
  "forbidden_fields": ["answer", "result", "solution", "correct_answer", "final_answer",
                        "the_answer", "output", "response"],
  "properties": {
    "ir": {
      "required": ["task", "domain", "answer_schema"],
      "properties": {
        "task":          {"enum": ["compute","count","find","decide","prove","classify","construct"]},
        "domain":        {"type": "string"},
        "answer_schema": {"enum": ["integer","decimal","boolean","option_label",
                                   "expression","sequence","proof_sketch","string"]},
        "entities":      {"type": "array"},
        "constraints":   {"type": "array", "items": {"type": "string"}},
        "query_description": {"type": "string"},
        "is_verifiable": {"type": "boolean"},
        "unverifiable_reason": {"type": "string"}
      }
    },
    "candidates": {
      "type": "array",
      "maxItems": 5,
      "items": {
        "required": ["method_name", "approach", "steps"],
        "properties": {
          "method_name":    {"type": "string"},
          "approach":       {"type": "string"},
          "required_tools": {"type": "array", "items": {"type": "string"}},
          "steps": {
            "type": "array",
            "description": "EACH STEP IS A PLAN, NOT A COMPUTED RESULT. NO FINAL ANSWERS.",
            "items": {"type": "string"}
          },
          "verify_spec": {
            "properties": {
              "method": {"enum": ["numeric_check","symbolic_check","z3_smt",
                                  "worldgen_sample","lean_proof","coq_proof"]},
              "description": {"type": "string"},
              "worldgen":    {"type": "object"},
              "check_code":  {"type": "string"}
            }
          }
        }
      }
    },
    "decomp_notes": {"type": "string"}
  }
}
"""
