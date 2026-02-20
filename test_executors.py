"""
Executor動作テスト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from executors import arithmetic, logic, enumerate


def test_arithmetic():
    """Arithmetic Executorテスト"""
    print("=" * 80)
    print("Arithmetic Executor Test")
    print("=" * 80)
    print()
    
    tests = [
        ("1+1", 2),
        ("5*6", 30),
        ("10-3", 7),
        ("100/4", 25.0),
        ("2**3", 8),
    ]
    
    for expr, expected in tests:
        result = arithmetic.evaluate(expression=expr)
        status = "✅" if result.get("value") == expected else "❌"
        print(f"{status} {expr} = {result.get('value')} (expected: {expected})")
    
    print()


def test_logic():
    """Logic Executorテスト"""
    print("=" * 80)
    print("Logic Executor Test")
    print("=" * 80)
    print()
    
    tests = [
        ("p -> p", True, "Tautology"),
        ("p & ~p", False, "Contradiction"),
        ("p | ~p", True, "Excluded middle"),
        ("(p -> q) & p", None, "Not a tautology"),
    ]
    
    for formula, expected, desc in tests:
        result = logic.prop_truth_table(formula=formula)
        value = result.get("value")
        status = "✅" if value == expected else ("⚠️" if expected is None else "❌")
        print(f"{status} {formula}: {value} - {desc}")
    
    print()


def test_enumerate():
    """Enumerate Executorテスト"""
    print("=" * 80)
    print("Enumerate Executor Test")
    print("=" * 80)
    print()
    
    # Integer range
    result = enumerate.integer_range(min=0, max=5)
    print(f"Integer range [0, 5]: {result.get('value')}")
    
    # Options
    ir = {
        "options": [
            "A. Answer 1",
            "B. Answer 2",
            "C. Answer 3"
        ]
    }
    result = enumerate.options(ir=ir)
    print(f"Options: {result.get('value')}")
    
    print()


if __name__ == "__main__":
    test_arithmetic()
    test_logic()
    test_enumerate()
