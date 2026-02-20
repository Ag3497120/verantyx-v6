#!/usr/bin/env python3
"""
Answer Matcher Test - 拡張機能のテスト
"""
from core.answer_matcher import flexible_match, normalize_number, normalize_string


def test_normalize_number():
    """数値正規化のテスト"""
    print("=" * 80)
    print("Test: normalize_number()")
    print("=" * 80)
    
    test_cases = [
        # (input, expected)
        ("42", 42.0),
        ("3.14", 3.14),
        ("1/2", 0.5),
        ("3/4", 0.75),
        ("50%", 0.5),
        ("12.5%", 0.125),
        ("1e-6", 1e-6),
        ("2.5E3", 2500.0),
        ("1,000", 1000.0),
        ("3+4i", complex(3, 4)),
        ("3+4j", complex(3, 4)),
    ]
    
    passed = 0
    for input_val, expected in test_cases:
        result = normalize_number(input_val)
        status = "✅" if result == expected else "❌"
        print(f"{status} normalize_number({repr(input_val)}) = {result} (expected: {expected})")
        if result == expected:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    print()


def test_flexible_match():
    """柔軟な正解判定のテスト"""
    print("=" * 80)
    print("Test: flexible_match()")
    print("=" * 80)
    
    test_cases = [
        # (predicted, expected, should_match)
        # 数値比較
        (42, "42", True),
        ("3.14", 3.14, True),
        ("1/2", 0.5, True),
        ("0.5", "1/2", True),
        ("50%", 0.5, True),
        
        # LaTeX正規化
        ("\\mathbb{Z}", "z", True),
        ("\\mathbb{R}", "R", True),
        ("\\pi", "pi", True),
        
        # 多肢選択
        ("A", "A", True),
        ("Option A", "A", True),
        ("B", "b", True),
        
        # Boolean値
        ("Yes", "yes", True),
        ("YES", "No", False),
        ("True", "true", True),
        ("1", "0", False),
        
        # 科学的記法
        ("1e-6", 0.000001, True),
        ("2.5E3", 2500, True),
        
        # 複素数
        ("3+4i", complex(3, 4), True),
        
        # 許容誤差
        (3.14159, 3.14160, True),  # tolerance=1e-4
        (3.14, 3.15, False),
        
        # 文字列正規化
        ("  hello  ", "Hello", True),
        ("WORLD", "world", True),
    ]
    
    passed = 0
    for predicted, expected, should_match in test_cases:
        result = flexible_match(predicted, expected, tolerance=1e-4)
        status = "✅" if result == should_match else "❌"
        print(f"{status} flexible_match({repr(predicted)}, {repr(expected)}) = {result} (expected: {should_match})")
        if result == should_match:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    print()


def main():
    """メインテスト実行"""
    print("\n" + "=" * 80)
    print("Answer Matcher Test Suite")
    print("=" * 80)
    print()
    
    test_normalize_number()
    test_flexible_match()
    
    print("=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
