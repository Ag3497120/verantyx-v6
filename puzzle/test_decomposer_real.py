#!/usr/bin/env python3
"""
Test LLM Decomposer with real Gemma2 models (once loaded)
"""
from llm_decomposer import LLMDecomposer
import json

print("=== Testing LLM Decomposer with Gemma2-2B ===\n")

# Initialize decomposer
print("[1] Initializing LLM Decomposer...")
decomposer = LLMDecomposer(device="cpu")
print("✓ Decomposer initialized\n")

# Test questions
questions = [
    ("What is C(10,3)?", [("A", "100"), ("B", "120"), ("C", "150"), ("D", "200")]),
    ("Prove that the chromatic number of the Petersen graph is 3.", None),
    ("A patient presents with GERD and dyspnea. What is the diagnosis?", None),
]

for i, (question, choices) in enumerate(questions):
    print(f"\n{'='*60}")
    print(f"Question {i+1}: {question}")
    if choices:
        print(f"Choices: {choices}")
    print(f"{'='*60}")

    # Decompose
    print("\n[2] Decomposing with Gemma2-2B...")
    result = decomposer.decompose(question, choices)

    # Check for gate violations
    if result.gate_violations:
        print(f"\n⚠️  GATE VIOLATIONS: {result.gate_violations}")
        print("   LLM tried to compute/answer (prohibited)")
    else:
        print(f"\n✓ Gate A PASSED (no violations)")

    # Show IR
    print(f"\n[3] IR Extracted:")
    print(f"  Variables: {result.ir.variables}")
    print(f"  Constraints: {result.ir.constraints}")
    print(f"  Target: {result.ir.target}")
    print(f"  Missing: {result.ir.missing}")
    print(f"  Format Type: {result.ir.format_type}")
    print(f"  Visual Needed: {result.ir.visual_needed}")

    # Show candidates
    print(f"\n[4] Candidates ({len(result.candidates)}):")
    for j, c in enumerate(result.candidates):
        print(f"  {j+1}. Method: {c.method}")
        print(f"     Steps: {c.steps}")
        print(f"     Verify Tool: {c.verify_tool}")

    # Show verify spec
    print(f"\n[5] Verify Spec:")
    print(f"  {json.dumps(result.verify_spec, indent=2)}")

print("\n" + "="*60)
print("✅ All tests completed")
print("="*60)
