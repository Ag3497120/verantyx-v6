"""
Parse Matrix Debug
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from executors.linear_algebra import _parse_matrix

# Test cases
test_cases = [
    "Find the determinant of matrix [[2, 3], [1, 4]].",
    "[[2, 3], [1, 4]]",
    "matrix [[2, 3], [1, 4]]",
]

for text in test_cases:
    result = _parse_matrix(text)
    print(f"Text: {text}")
    print(f"Result: {result}")
    print()
