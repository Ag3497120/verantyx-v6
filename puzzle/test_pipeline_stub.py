#!/usr/bin/env python3
"""
Test Verantyx Pipeline v2 in stub mode (no models required)
"""

# Force stub mode by marking transformers unavailable
import sys
sys.modules['transformers'] = None

from verantyx_pipeline_v2 import VerantyxPipelineV2

def test_pipeline():
    """Test complete pipeline in stub mode"""
    print("=== Testing Verantyx Pipeline v2 (STUB mode) ===\n")

    # Initialize pipeline
    pipeline = VerantyxPipelineV2(
        llm_device="cpu",
        enable_vision=False,
        enable_600b_knowledge=False  # H100資産なしでテスト
    )

    # Test questions
    questions = [
        ("What is C(10,3)?", [("A", "100"), ("B", "120"), ("C", "150"), ("D", "200")]),
        ("A patient presents with GERD and dyspnea. What is the diagnosis?", None),
    ]

    for question, choices in questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        if choices:
            print(f"Choices: {choices}")
        print(f"{'='*80}")

        result = pipeline.solve(question, choices)

        print(f"\n[Result]")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Stage Reached: {result.stage_reached.value}")
        if result.rejection_reason:
            print(f"Rejection Reason: {result.rejection_reason}")

        print(f"\n[Trace]")
        for line in result.trace:
            print(f"  {line}")

        print(f"\n[Audit Log Keys]")
        print(f"  {list(result.audit_log.keys())}")

    print("\n" + "="*80)
    print("✅ Pipeline stub mode test completed")
    print("="*80)

if __name__ == "__main__":
    test_pipeline()
