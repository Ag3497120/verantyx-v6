#!/usr/bin/env python3
"""
Test LLM Decomposer in stub mode (no model download required)
"""
import json
import sys

# Temporarily disable transformers to test stub mode
sys.modules['transformers'] = None

from llm_decomposer import LLMDecomposer, GateViolation

def test_stub_mode():
    """Test decomposer in stub mode"""
    print("=== Testing LLM Decomposer in STUB mode ===\n")

    # Initialize decomposer (should use stub mode)
    decomposer = LLMDecomposer(device="cpu")

    # Test questions
    questions = [
        ("What is C(10,3)?", [("A", "100"), ("B", "120"), ("C", "150"), ("D", "200")]),
        ("Prove that the chromatic number of the Petersen graph is 3.", None),
    ]

    for question, choices in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        if choices:
            print(f"Choices: {choices}")
        print(f"{'='*60}")

        # Decompose
        result = decomposer.decompose(question, choices)

        # Validate output
        print(f"\n[IR]")
        print(f"  Variables: {result.ir.variables}")
        print(f"  Constraints: {result.ir.constraints}")
        print(f"  Target: {result.ir.target}")
        print(f"  Missing: {result.ir.missing}")
        print(f"  Format Type: {result.ir.format_type}")
        print(f"  Visual Needed: {result.ir.visual_needed}")

        print(f"\n[Candidates] ({len(result.candidates)})")
        for i, c in enumerate(result.candidates):
            print(f"  {i+1}. Method: {c.method}")
            print(f"     Steps: {c.steps}")
            print(f"     Verify Tool: {c.verify_tool}")

        print(f"\n[Verify Spec]")
        print(f"  {result.verify_spec}")

        if result.gate_violations:
            print(f"\n[Gate Violations] {result.gate_violations}")
        else:
            print(f"\n[Gate Check] ✓ No violations")

def test_gate_a():
    """Test Gate A forbidden word detection"""
    print("\n\n=== Testing Gate A (Forbidden Word Detection) ===\n")

    decomposer = LLMDecomposer(device="cpu")

    # Test cases with forbidden words
    test_outputs = [
        ('{"variables": ["x"], "answer is 120"}', True, "Should detect 'answer is'"),
        ('{"variables": ["x"], "明らか"}', True, "Should detect '明らか'"),
        ('{"variables": ["x"], "x = 5"}', True, "Should detect computation '='"),
        ('{"variables": ["x"], "10 + 5 = 15"}', True, "Should detect computation"),
        ('{"variables": ["x"], "constraints": ["x > 0"]}', False, "Should NOT detect (no violations)"),
    ]

    for output, should_violate, description in test_outputs:
        violations = decomposer._check_gate_a(output)
        has_violations = len(violations) > 0

        status = "✓" if has_violations == should_violate else "✗"
        print(f"{status} {description}")
        print(f"   Output: {output[:50]}...")
        print(f"   Violations: {violations}")
        print()

def test_json_schema():
    """Test JSON schema validation"""
    print("\n=== Testing JSON Schema Validation ===\n")

    decomposer = LLMDecomposer(device="cpu")

    # Valid JSON
    valid_json = json.dumps({
        "variables": ["n", "k"],
        "constraints": ["n >= k", "k >= 0"],
        "target": "C(n,k)",
        "missing": [],
        "visual_needed": False,
        "candidates": [
            {
                "method": "combinatorial formula",
                "steps": ["apply formula", "verify result"],
                "verify_tool": "sympy"
            }
        ],
        "verify_spec": {"tool": "sympy", "check": "verify_combinatorial"}
    })

    print("Valid JSON test:")
    try:
        parsed = json.loads(valid_json)
        missing = decomposer._check_missing_slots(parsed)
        if not missing:
            print("  ✓ All required slots present")
        else:
            print(f"  ✗ Missing slots: {missing}")
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON decode error: {e}")

    # Invalid JSON (missing required slots)
    invalid_json = json.dumps({
        "variables": ["x"],
        # Missing: constraints, target, missing
    })

    print("\nInvalid JSON test (missing slots):")
    try:
        parsed = json.loads(invalid_json)
        missing = decomposer._check_missing_slots(parsed)
        if missing:
            print(f"  ✓ Correctly detected missing slots: {missing}")
        else:
            print(f"  ✗ Should have detected missing slots")
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON decode error: {e}")

if __name__ == "__main__":
    # Re-enable transformers import for stub mode detection
    import importlib
    import llm_decomposer
    importlib.reload(llm_decomposer)
    from llm_decomposer import LLMDecomposer

    test_stub_mode()
    test_gate_a()
    test_json_schema()

    print("\n" + "="*60)
    print("✅ All stub mode tests completed")
    print("="*60)
