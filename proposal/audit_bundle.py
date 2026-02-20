"""
AuditBundle.v1 — 検証ログの統一形式

ユーザーに見せるのは説明文ではなく「検証ログの束」。
status が VERIFIED 以外なら answer は null（断言禁止）。
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional


def _map_status(cegis_status: str) -> str:
    """CEGIS の内部 status → AuditBundle の公開 status"""
    mapping = {
        "proved":           "VERIFIED",
        "high_confidence":  "INCONCLUSIVE",  # 証明不完全
        "disproved":        "REFUTED",
        "timeout":          "INCONCLUSIVE",
        "unknown":          "INCONCLUSIVE",
        "cegis_disabled":   "INCONCLUSIVE",
        "failed":           "INCONCLUSIVE",
    }
    return mapping.get(cegis_status, "INCONCLUSIVE")


def build_audit_bundle(
    problem_id: str,
    cegis_result: Any,
    proposal_ir: Optional[Dict] = None,
    executor_summary: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    AuditBundle.v1 を構築して返す

    Args:
        problem_id:       問題 ID
        cegis_result:     CEGISResult オブジェクト
        proposal_ir:      Claude が生成した IR（使わなかった場合は None）
        executor_summary: Executor の概要（文字列）
        extra:            その他メタ情報

    Returns:
        AuditBundle dict
    """
    status = _map_status(getattr(cegis_result, "status", "unknown"))

    # VERIFIED 以外は answer を null（断言禁止ゲート）
    answer = None
    if status == "VERIFIED":
        answer = getattr(cegis_result, "answer", None)

    cert = getattr(cegis_result, "certificate", None)
    cert_info = None
    if cert:
        cert_info = {
            "kind": cert.kind.value if hasattr(cert.kind, "value") else str(cert.kind),
            "confidence": cert.confidence,
            "verified": cert.verified,
        }

    bundle = {
        "schema": "verantyx.audit.v1",
        "meta": {
            "problem_id": problem_id,
            "run_id": str(uuid.uuid4())[:8],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            "engine": "verantyx_v6",
            "llm_used": proposal_ir is not None,
            "llm_role": "proposal_only" if proposal_ir else "none",
            "llm_rejected": proposal_ir is not None and proposal_ir.get("rejected", False),
        },
        "summary": {
            "status": status,
            "final_answer": str(answer) if answer is not None else None,
            "confidence": getattr(cegis_result, "confidence", 0.0),
            "reason_codes": _extract_reason_codes(cegis_result, proposal_ir),
        },
        "trace": {
            "stages": _build_stages(cegis_result, proposal_ir, executor_summary),
            "cegis_iterations": getattr(cegis_result, "iterations", 0),
            "elapsed_ms": getattr(cegis_result, "elapsed_ms", 0.0),
            "cegis_trace": getattr(cegis_result, "trace", [])[-5:],  # 最後の5行
        },
        "certificate": cert_info,
        "counterexamples": getattr(cegis_result, "counterexamples", [])[:3],
        "proposal": {
            "used": proposal_ir is not None and not proposal_ir.get("rejected"),
            "candidates_proposed": len(proposal_ir.get("candidate_programs", [])) if proposal_ir and not proposal_ir.get("rejected") else 0,
            "ir_domain": proposal_ir.get("signals", {}).get("domain") if proposal_ir else None,
            "needs_tool": proposal_ir.get("signals", {}).get("needs_tool", []) if proposal_ir else [],
        },
    }

    if extra:
        bundle["extra"] = extra

    return bundle


def _extract_reason_codes(cegis_result: Any, proposal_ir: Optional[Dict]) -> List[str]:
    codes = []
    status = getattr(cegis_result, "status", "unknown")
    if status == "proved":
        codes.append("cegis_proved")
    elif status == "high_confidence":
        codes.append("high_confidence_no_proof")
    elif status == "timeout":
        codes.append("cegis_timeout")
    elif status == "unknown":
        codes.append("no_candidates")

    if proposal_ir:
        if proposal_ir.get("rejected"):
            codes.append("proposal_gate_rejected")
        else:
            codes.append("proposal_used")
        needs = proposal_ir.get("signals", {}).get("needs_tool", [])
        if "vision" in needs:
            codes.append("diagram_required")
        if "search" in needs:
            codes.append("external_reference")

    return codes


def _build_stages(cegis_result: Any, proposal_ir: Optional[Dict], executor_summary: Optional[str]) -> List[Dict]:
    stages = []

    # Executor
    stages.append({
        "name": "executor",
        "ok": True,
        "notes": executor_summary or "ran",
    })

    # Proposal
    if proposal_ir:
        rejected = proposal_ir.get("rejected", False)
        stages.append({
            "name": "proposal",
            "ok": not rejected,
            "candidates": len(proposal_ir.get("candidate_programs", [])) if not rejected else 0,
            "gate": "passed" if not rejected else "rejected",
        })

    # CEGIS
    stages.append({
        "name": "cegis",
        "ok": getattr(cegis_result, "status", "") in ("proved", "high_confidence"),
        "iters": getattr(cegis_result, "iterations", 0),
        "status": getattr(cegis_result, "status", "unknown"),
    })

    return stages


def format_final_output(
    cegis_result: Any,
    audit_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """
    solve() の最終戻り値フォーマット

    Returns:
        {"answer": ..., "status": "verified|inconclusive|refuted", "audit": {...}}
    """
    status_map = {
        "VERIFIED": "verified",
        "REFUTED": "refuted",
        "INCONCLUSIVE": "inconclusive",
    }
    public_status = status_map.get(audit_bundle["summary"]["status"], "inconclusive")

    # VERIFIED 以外は断言しない
    answer = None
    if public_status == "verified":
        answer = audit_bundle["summary"]["final_answer"]

    return {
        "answer": answer,
        "status": public_status,
        "confidence": audit_bundle["summary"]["confidence"],
        "reason": audit_bundle["summary"]["reason_codes"],
        "audit": audit_bundle,
    }
