"""
ExactMatch Detector Registry

NOTE: All hardcoded answer detectors have been removed.
Answers are now derived through the Verantyx reasoning pipeline
(CrossParamEngine, KB lookup, symbolic solvers).

Rules:
- Do NOT add detectors that hardcode specific answers
- Pattern matching is allowed only if backed by a general computation
"""

from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────
ALL_EXACT_DETECTORS = []


def run_exact_detectors(
    question: str,
    confidence_threshold: float = 0.85
) -> Optional[Tuple[str, float]]:
    """
    Run all registered exact detectors against a question.

    Args:
        question: Question text
        confidence_threshold: Minimum confidence threshold

    Returns:
        (answer, confidence) or None
    """
    for detector in ALL_EXACT_DETECTORS:
        try:
            result = detector(question)
            if result is not None:
                answer, confidence = result
                if confidence >= confidence_threshold:
                    return (str(answer), confidence)
        except Exception:
            continue
    return None
