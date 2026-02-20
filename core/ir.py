"""
Internal Representation (IR) - 問題文から抽出した構造化指示

完全型安全、LLM不使用での分解を前提とした設計
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class TaskType(Enum):
    """タスクタイプ"""
    COMPUTE = "compute"
    DECIDE = "decide"
    CONSTRUCT = "construct"
    PROVE = "prove"
    CHOOSE = "choose"
    COUNT = "count"
    FIND = "find"
    OPTIMIZE = "optimize"


class Domain(Enum):
    """ドメイン"""
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    LINEAR_ALGEBRA = "linear_algebra"
    CALCULUS = "calculus"
    LOGIC_PROPOSITIONAL = "logic_propositional"
    LOGIC_MODAL = "logic_modal"
    LOGIC_FIRST_ORDER = "logic_first_order"
    GRAPH_THEORY = "graph_theory"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    PROBABILITY = "probability"
    STATISTICS = "statistics"
    MODULAR_ARITHMETIC = "modular_arithmetic"
    ADVANCED_PROBABILITY = "advanced_probability"
    ADVANCED_NUMBER_THEORY = "advanced_number_theory"
    ADVANCED_COMBINATORICS = "advanced_combinatorics"
    STRING = "string"
    CRYPTOGRAPHY = "cryptography"
    MULTIPLE_CHOICE = "multiple_choice"
    CHESS = "chess"
    PUZZLE = "puzzle"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    COMPUTER_SCIENCE = "computer_science"
    PHILOSOPHY = "philosophy"
    LAW = "law"
    UNKNOWN = "unknown"


class AnswerSchema(Enum):
    """答えのスキーマ"""
    INTEGER = "integer"
    RATIONAL = "rational"
    DECIMAL = "decimal"
    COMPLEX = "complex"
    EXPRESSION = "expression"
    FORMULA = "formula"
    BOOLEAN = "boolean"
    OPTION_LABEL = "option_label"
    MOVE_SEQUENCE = "move_sequence"
    SEQUENCE = "sequence"
    SET = "set"
    GRAPH = "graph"
    MATRIX = "matrix"
    PROOF = "proof"
    TEXT = "text"


@dataclass
class Entity:
    """エンティティ"""
    type: str  # "number", "symbol", "graph", "formula", etc.
    name: Optional[str] = None
    value: Any = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Constraint:
    """制約条件"""
    type: str  # "equals", "range", "membership", etc.
    lhs: Any = None
    rhs: Any = None
    var: Optional[str] = None
    min: Any = None
    max: Any = None
    expression: Optional[str] = None


@dataclass
class Query:
    """クエリ"""
    type: str  # "find", "compute", "count", etc.
    target: Optional[str] = None
    property: Optional[str] = None  # "minimal", "maximal", "existence", etc.
    constraints: List[str] = field(default_factory=list)


@dataclass
class IR:
    """
    Internal Representation (IR)
    
    問題文を構造化した内部表現。
    LLM不使用でルールベース分解を前提とする。
    """
    task: TaskType
    domain: Domain
    answer_schema: AnswerSchema
    
    entities: List[Entity] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    query: Optional[Query] = None
    options: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "task": self.task.value,
            "domain": self.domain.value,
            "answer_schema": self.answer_schema.value,
            "entities": [
                {
                    "type": e.type,
                    "name": e.name,
                    "value": e.value,
                    "params": e.params
                }
                for e in self.entities
            ],
            "constraints": [
                {
                    "type": c.type,
                    "lhs": c.lhs,
                    "rhs": c.rhs,
                    "var": c.var,
                    "min": c.min,
                    "max": c.max,
                    "expression": c.expression
                }
                for c in self.constraints
            ],
            "query": {
                "type": self.query.type,
                "target": self.query.target,
                "property": self.query.property,
                "constraints": self.query.constraints
            } if self.query else None,
            "options": self.options,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IR':
        """辞書から復元"""
        return cls(
            task=TaskType(data["task"]),
            domain=Domain(data["domain"]),
            answer_schema=AnswerSchema(data["answer_schema"]),
            entities=[
                Entity(**e) for e in data.get("entities", [])
            ],
            constraints=[
                Constraint(**c) for c in data.get("constraints", [])
            ],
            query=Query(**data["query"]) if data.get("query") else None,
            options=data.get("options", []),
            metadata=data.get("metadata", {})
        )
