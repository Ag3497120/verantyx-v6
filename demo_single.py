"""
demo_single.py — AuditBundle.v1 + 1問デモランナー

3パターンの問題で AuditBundle の動作を確認する：
1. cegis_proved が期待できる問題（既存ピースが刺さる）
2. inconclusive が期待できる問題（証明できないが暗記もしない）
3. MCQ で CEGIS が選択肢を絞る問題
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_enhanced import VerantyxV6Enhanced
from audit import AuditBundle


# デモ問題
DEMO_QUESTIONS = [
    # Pattern 1: cegis_proved が期待できる（既存ピースが刺さる）
    {
        "id": "demo_01_factorial",
        "question": "What is 5! (5 factorial)?",
        "answer": "120",
        "category": "Math",
    },
    # Pattern 2: inconclusive が期待できる（証明できないが暗記もしない）
    {
        "id": "demo_02_knowledge",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "category": "Geography",
    },
    # Pattern 3: MCQ で CEGIS が選択肢を絞る
    {
        "id": "demo_03_mcq",
        "question": "Which of the following is a prime number? (A) 4 (B) 6 (C) 7 (D) 9",
        "answer": "C",
        "category": "Math",
    },
]


def run_demo():
    """Run demo with 3 test patterns"""
    print("=" * 80)
    print("AuditBundle.v1 Demo Runner")
    print("=" * 80)
    print()

    # Initialize pipeline
    print("Initializing Verantyx V6 Enhanced pipeline...")
    pipeline = VerantyxV6Enhanced(
        use_llm_decomposer=False,
        use_claude_proposal=False,
    )
    print("✓ Pipeline ready")
    print()

    results = []

    for i, problem in enumerate(DEMO_QUESTIONS, 1):
        print(f"{'─' * 80}")
        print(f"Problem {i}/{len(DEMO_QUESTIONS)}: {problem['id']}")
        print(f"{'─' * 80}")
        print(f"Question: {problem['question']}")
        print(f"Expected: {problem['answer']}")
        print()

        try:
            # Run pipeline
            result = pipeline.solve(
                problem_text=problem["question"],
                expected_answer=problem["answer"],
            )

            # Extract info
            answer = result.get("answer")
            status = result.get("status", "UNKNOWN")
            confidence = result.get("confidence", 0.0)
            method = result.get("method", "unknown")

            print(f"Status:     {status}")
            print(f"Answer:     {answer}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Method:     {method}")

            # Parse AuditBundle
            bundle_json = result.get("audit_bundle")
            if bundle_json:
                try:
                    bundle_dict = json.loads(bundle_json)
                    print()
                    print("AuditBundle Summary:")
                    print(f"  Bundle version: {bundle_dict.get('bundle_version')}")

                    # CEGIS info
                    cegis = bundle_dict.get("cegis")
                    if cegis and cegis.get("ran"):
                        print(f"  CEGIS ran:      yes (iters={cegis.get('iters', 0)})")
                        print(f"  CEGIS proved:   {cegis.get('proved', False)}")
                        if cegis.get("inconclusive_reason"):
                            print(f"  Inconclusive:   {cegis['inconclusive_reason']}")
                    else:
                        print(f"  CEGIS ran:      no")

                    # Verify info
                    verify = bundle_dict.get("verify")
                    if verify:
                        print(f"  Verify tool:    {verify.get('tool')}")
                        print(f"  Certificate:    {verify.get('certificate_type', 'none')}")

                    # Answer info
                    ans_info = bundle_dict.get("answer")
                    if ans_info:
                        print(f"  Answer status:  {ans_info.get('status')}")
                        print(f"  Answer value:   {ans_info.get('value')}")

                    # Integrity
                    integrity = bundle_dict.get("integrity")
                    if integrity:
                        print(f"  Bundle digest:  {integrity.get('bundle_digest')}")
                        print(f"  Pipeline ver:   {integrity.get('pipeline_version')}")

                except Exception as e:
                    print(f"⚠ Failed to parse AuditBundle: {e}")
            else:
                print()
                print("⚠ No AuditBundle in result")

            results.append({
                "id": problem["id"],
                "status": status,
                "answer": answer,
                "expected": problem["answer"],
                "confidence": confidence,
                "method": method,
                "bundle_available": bundle_json is not None,
            })

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "id": problem["id"],
                "status": "ERROR",
                "error": str(e),
            })

        print()

    # Summary table
    print(f"{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print()
    print(f"{'ID':<25} {'Status':<12} {'Answer':<15} {'Method':<20}")
    print(f"{'-' * 80}")
    for r in results:
        status = r.get("status", "?")
        answer = r.get("answer", "—")
        method = r.get("method", r.get("error", "?"))
        print(f"{r['id']:<25} {status:<12} {str(answer):<15} {method:<20}")
    print()

    # Check completion
    all_have_bundles = all(r.get("bundle_available", False) for r in results if r.get("status") != "ERROR")
    if all_have_bundles:
        print("✓ All problems have AuditBundle")
    else:
        print("⚠ Some problems missing AuditBundle")

    print()
    print("Demo complete!")


if __name__ == "__main__":
    run_demo()
