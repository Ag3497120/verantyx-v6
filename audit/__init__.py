"""
Audit module for Verantyx V6

Provides transparent verification traces via AuditBundle.v1
"""

from .audit_bundle import (
    AuditBundle,
    TriggerInfo,
    GateResult,
    LLMProposalInfo,
    RefutationRecord,
    CEGISInfo,
    VerifyInfo,
    AnswerInfo,
    IntegrityInfo,
    ExpertTrace,
)

__all__ = [
    "AuditBundle",
    "TriggerInfo",
    "GateResult",
    "LLMProposalInfo",
    "RefutationRecord",
    "CEGISInfo",
    "VerifyInfo",
    "AnswerInfo",
    "IntegrityInfo",
    "ExpertTrace",
]
