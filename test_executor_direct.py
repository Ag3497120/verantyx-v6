"""
Executor Direct Test
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from executors.linear_algebra import matrix_determinant

# テスト
source_text = "Find the determinant of matrix [[2, 3], [1, 4]]."
ir = {"source_text": source_text}

result = matrix_determinant(ir=ir)
print("Result:", result)

# 期待値: 2*4 - 3*1 = 8 - 3 = 5
