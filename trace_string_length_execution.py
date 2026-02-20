"""
Complete trace of string_length execution
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from decomposer.decomposer import RuleBasedDecomposer
from pieces.piece import PieceDB
from assembler.beam_search import BeamSearch
from assembler.executor import Executor

# Simple test question
question = "What is the length of 'hello'?"
print(f"Question: {question}")
print("=" * 80)

# Step 1: Decompose
print("\n[Step 1: Decompose]")
decomposer = RuleBasedDecomposer()
ir = decomposer.decompose(question)
ir_dict = ir.to_dict()
print(f"  domain: {ir_dict['domain']}")
print(f"  task: {ir_dict['task']}")
print(f"  source_text: {ir_dict['metadata']['source_text']}")

# Step 2: Find piece
print("\n[Step 2: Find Piece]")
piece_db = PieceDB("pieces/piece_db.jsonl")
assembler = BeamSearch(piece_db)

# Try to find string_length piece directly
string_len_piece = piece_db.find_by_id("string_len")
if string_len_piece:
    print(f"  Found: {string_len_piece.piece_id}")
    print(f"  Executor: {string_len_piece.executor}")
    print(f"  Requires: {string_len_piece.in_spec.requires}")
    print(f"  Slots: {string_len_piece.in_spec.slots}")
    
    # Step 3: Execute manually with detailed tracing
    print("\n[Step 3: Execute with Tracing]")
    executor = Executor()
    
    # Manually prepare params to see what's happening
    print("\n  Manual parameter preparation:")
    context = {
        "ir": ir_dict,
        "artifacts": {},
        "confidence": 1.0
    }
    
    # Call internal method with trace
    params = executor._prepare_params(string_len_piece, context)
    print(f"  Prepared params: {params}")
    
    # Try direct execution
    print("\n  Direct function call:")
    try:
        from executors.string_operations import string_length
        
        # Test 1: Direct call with correct param
        result1 = string_length(text="hello")
        print(f"    string_length(text='hello') = {result1}")
        
        # Test 2: Call with prepared params
        if params:
            result2 = string_length(**params)
            print(f"    string_length(**prepared_params) = {result2}")
        else:
            print(f"    No params prepared!")
        
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 4: Full executor path
    print("\n[Step 4: Full Executor Path]")
    result = executor.execute_path([string_len_piece], ir_dict)
    if result:
        print(f"  Result: {result.to_dict()}")
    else:
        print(f"  Result: None (execution failed)")

else:
    print("  string_len piece NOT FOUND")

# Additional debug: check what source_text contains
print("\n[Additional Debug]")
print(f"  source_text length: {len(ir_dict['metadata']['source_text'])}")
print(f"  source_text repr: {repr(ir_dict['metadata']['source_text'])}")
