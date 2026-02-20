"""
Verantyx V6 - 構想準拠実装

完全ルールベース、LLM不使用の推論システム
"""

from .pipeline import VerantyxV6
from .core.ir import IR, TaskType, Domain, AnswerSchema
from .pieces.piece import Piece, PieceDB
from .grammar.composer import GrammarPiece, GrammarDB, AnswerComposer

__version__ = "6.0.0"
__all__ = [
    "VerantyxV6",
    "IR",
    "TaskType",
    "Domain",
    "AnswerSchema",
    "Piece",
    "PieceDB",
    "GrammarPiece",
    "GrammarDB",
    "AnswerComposer"
]
