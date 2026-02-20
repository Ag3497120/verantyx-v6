"""
Test string_length bug

HEARTBEAT.md reports: "hello world" → 47 (correct: 11)
"""

from executors.string_operations import string_length

# Direct test
test_string = "hello world"
result = string_length(test_string)
print(f"Direct call: string_length('{test_string}') = {result}")
print(f"Expected: {len(test_string)}")
print(f"Bug reproduced: {result == 47}")

# Test with various strings
test_cases = [
    "hello world",
    "test",
    "",
    "a",
    "Hello, World!",
]

print("\nTest cases:")
for s in test_cases:
    result = string_length(s)
    expected = len(s)
    status = "✓" if result == expected else "✗"
    print(f"  {status} '{s}' → {result} (expected {expected})")
