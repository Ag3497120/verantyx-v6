"""
ピース選択テスト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pieces.piece import PieceDB
from core.ir import IR, TaskType, Domain, AnswerSchema

# Test 1のIRを作成
print("=== Test 1: What is 1 + 1? ===")
ir = IR(
    task=TaskType.COMPUTE,
    domain=Domain.ARITHMETIC,
    answer_schema=AnswerSchema.INTEGER,
    metadata={'source_text': 'What is 1 + 1?'}
)
ir_dict = ir.to_dict()

print(f"IR: task={ir_dict['task']}, domain={ir_dict['domain']}, answer_schema={ir_dict['answer_schema']}")

# ピース検索
db = PieceDB('pieces/piece_db.jsonl')
results = db.search(ir_dict, top_k=5)

print('\nTop 5 pieces:')
for piece, score in results:
    print(f'{score:.3f} - {piece.piece_id}')
    print(f'       requires={piece.in_spec.requires}')
    print(f'       out_schema={piece.out_spec.schema}')
