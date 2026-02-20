"""
Analyze Math failures in HLE

Extract Math problems and analyze failure patterns
"""
import json
import sys
from typing import List, Dict, Any

sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match


def load_hle_math_sample(file_path: str = "hle_2500_eval.jsonl", sample_size: int = 20) -> List[Dict[str, Any]]:
    """Load Math category questions"""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            if q.get('category') == 'Math':
                questions.append(q)
                if len(questions) >= sample_size:
                    break
    return questions


def analyze_failure(pipeline: VerantyxV6Enhanced, question: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze single question failure"""
    q_id = question['id']
    q_text = question['question']
    expected = question['answer']
    
    print(f"\n{'='*80}")
    print(f"Question {q_id}")
    print(f"{'='*80}")
    print(f"Q: {q_text[:200]}")
    print(f"Expected: {expected}")
    
    # Solve
    result = pipeline.solve(q_text)
    
    answer = result.get('answer')
    status = result.get('status')
    
    # Check correctness
    is_correct = flexible_match(answer, expected, tolerance=1e-4) if answer and expected else False
    
    print(f"\nAnswer: {answer}")
    print(f"Status: {status}")
    print(f"Correct: {'✓' if is_correct else '✗'}")
    
    # Extract failure info
    ir = result.get('ir', {})
    pieces_found = result.get('pieces_found', [])
    execution = result.get('execution', {})
    
    print(f"\nIR Domain: {ir.get('domain')}")
    print(f"IR Task: {ir.get('task')}")
    print(f"IR Answer Schema: {ir.get('answer_schema')}")
    print(f"Pieces Found: {len(pieces_found)}")
    if pieces_found:
        print(f"Top Piece: {pieces_found[0] if pieces_found else 'None'}")
    print(f"Execution: {execution.get('executor', 'N/A')}")
    
    return {
        'id': q_id,
        'correct': is_correct,
        'answer': answer,
        'expected': expected,
        'status': status,
        'domain': ir.get('domain'),
        'task': ir.get('task'),
        'pieces_count': len(pieces_found),
        'top_piece': pieces_found[0]['piece_id'] if pieces_found else None,
        'executor': execution.get('executor')
    }


def main():
    print("Phase 5H-1: Math Failure Analysis")
    print("="*80)
    
    # Initialize pipeline
    print("Initializing Verantyx V6...")
    pipeline = VerantyxV6Enhanced(
        piece_db_path="pieces/piece_db.jsonl"
    )
    
    # Load Math sample
    print("Loading Math sample (20 questions)...")
    questions = load_hle_math_sample(sample_size=20)
    print(f"Loaded {len(questions)} Math questions")
    
    # Analyze each
    results = []
    for q in questions:
        result = analyze_failure(pipeline, q)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    correct = sum(1 for r in results if r['correct'])
    print(f"Correct: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    
    # Domain distribution
    domains = {}
    for r in results:
        d = r['domain']
        domains[d] = domains.get(d, 0) + 1
    print(f"\nDomain Distribution:")
    for d, count in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {d}: {count}")
    
    # Top pieces used
    pieces = {}
    for r in results:
        p = r['top_piece']
        if p:
            pieces[p] = pieces.get(p, 0) + 1
    print(f"\nTop Pieces Used:")
    for p, count in sorted(pieces.items(), key=lambda x: -x[1])[:10]:
        print(f"  {p}: {count}")
    
    # Save results
    with open('math_failure_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to math_failure_analysis.json")


if __name__ == '__main__':
    main()
