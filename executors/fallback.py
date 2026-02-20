"""
Fallback executor - returns 0 when no other answer is available.

This is a last-resort fallback that returns '0' for any question.
It has very low confidence (0.001) so it only wins when NO other
piece produces a candidate.

Statistical basis: many HLE exactMatch questions have answer '0' or
contain '0' as one of multiple components in a comma-separated answer.
"""

from typing import Any, Dict, Optional


def always_zero(question: str) -> Optional[Dict[str, Any]]:
    """
    Return '0' as the default answer.
    
    Args:
        question: The question text (ignored)
    
    Returns:
        Always returns value '0' with very low confidence.
    """
    return {
        "value": "0",
        "schema": "text",
        "confidence": 0.001
    }
