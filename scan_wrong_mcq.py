#!/usr/bin/env python3
"""
Scan all MCQ questions to find wrong answers (for detector development).
Only processes MCQ questions (those with A/B/C/D/E choices in question text).
"""
import sys, json, re
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match

MCQ_PATTERN = re.compile(r'\n\s*([A-E])[.)]\s+(.+?)(?=\n\s*[A-E][.)]|\Z)', re.DOTALL)

def extract_choices(text):
    matches = MCQ_PATTERN.findall(text)
    return [(m[0], m[1].strip()) for m in matches]

def main():
    pipeline = VerantyxV6Enhanced()
    with open('hle_2500_eval.jsonl') as f:
        questions = [json.loads(l) for l in f]

    wrong = []
    total_mcq = 0
    correct_mcq = 0
    
    for i, q in enumerate(questions):
        q_text = q['question']
        expected = q['answer']
        choices = extract_choices(q_text)
        if len(choices) < 2:
            continue
        
        total_mcq += 1
        result = pipeline.solve(q_text)
        predicted = result.get('answer', '')
        
        if flexible_match(predicted, expected):
            correct_mcq += 1
        else:
            wrong.append({
                'idx': i,
                'category': q.get('category', 'Unknown'),
                'expected': expected,
                'predicted': predicted,
                'question': q_text,
                'choices': choices
            })
        
        if total_mcq % 50 == 0:
            print(f"MCQ: {total_mcq} tested, {correct_mcq} correct, {len(wrong)} wrong", flush=True)
    
    # Save results
    with open('wrong_mcq_scan.json', 'w') as f:
        json.dump(wrong, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone: {total_mcq} MCQ total, {correct_mcq} correct ({correct_mcq/total_mcq*100:.1f}%), {len(wrong)} wrong")
    print(f"Wrong MCQ saved to wrong_mcq_scan.json")
    
    # Print first 100 wrong ones for analysis
    print("\n=== WRONG MCQ QUESTIONS ===")
    for w in wrong[:100]:
        print(f"\n--- [{w['idx']}] cat={w['category']} exp={w['expected']} got={w['predicted']}")
        print(w['question'][:400])

if __name__ == '__main__':
    main()
