"""
AVH Math Integration Test

統合された論理ソルバーとCross Simulatorのテスト
"""
import sys
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

from puzzle.propositional_logic_solver import is_tautology, is_satisfiable
from puzzle.avh_adapters import CrossDB, ILSlots


def test_logic_solvers():
    """論理ソルバーのテスト"""
    print("="*80)
    print("AVH Math Integration Test - Logic Solvers")
    print("="*80)
    print()
    
    # Test 1: Tautology check
    print("Test 1: Tautology Check")
    test_cases = [
        ("(A | ~A)", True, "Law of Excluded Middle"),
        ("((A -> B) & A) -> B", True, "Modus Ponens"),
        ("A & ~A", False, "Contradiction"),
        ("A -> A", True, "Identity"),
    ]
    
    passed = 0
    for formula, expected, name in test_cases:
        try:
            result = is_tautology(formula)
            # 戻り値が(bool, dict)のタプルの場合、最初の要素を取得
            if isinstance(result, tuple):
                result = result[0]
            status = "✅" if result == expected else "❌"
            print(f"  {status} {name}: '{formula}' -> {result} (expected {expected})")
            if result == expected:
                passed += 1
        except Exception as e:
            print(f"  ❌ {name}: Error - {e}")
    
    print(f"\n  Result: {passed}/{len(test_cases)} passed")
    print()
    
    # Test 2: Satisfiability check
    print("Test 2: Satisfiability Check")
    sat_cases = [
        ("A & B", True, "Simple conjunction"),
        ("A & ~A", False, "Contradiction"),
        ("(A | B) & ~A", True, "Satisfiable disjunction"),
    ]
    
    sat_passed = 0
    for formula, expected, name in sat_cases:
        try:
            result = is_satisfiable(formula)
            # 戻り値が(bool, dict)のタプルの場合、最初の要素を取得
            if isinstance(result, tuple):
                result = result[0]
            status = "✅" if result == expected else "❌"
            print(f"  {status} {name}: '{formula}' -> {result} (expected {expected})")
            if result == expected:
                sat_passed += 1
        except Exception as e:
            print(f"  ❌ {name}: Error - {e}")
    
    print(f"\n  Result: {sat_passed}/{len(sat_cases)} passed")
    print()
    
    return passed + sat_passed, len(test_cases) + len(sat_cases)


def test_cross_db():
    """CrossDBのテスト"""
    print("="*80)
    print("AVH Math Integration Test - Cross DB")
    print("="*80)
    print()
    
    try:
        # CrossDBロード
        print("Loading CrossDB...")
        db = CrossDB("pieces/axioms_unified.json")
        
        print(f"✅ Loaded {len(db.assets)} axioms")
        print()
        
        # 600B抽出公理を確認
        print("600B Extracted Axioms:")
        axiom_600b = [a for a in db.assets if "600b" in a.asset_id]
        print(f"  Found {len(axiom_600b)} axioms extracted from 600B")
        for axiom in axiom_600b[:3]:
            print(f"    - {axiom.asset_id}")
        print()
        
        # 論理公理を検索
        print("Logic Axioms:")
        logic_axioms = db.search(domain="logic")
        print(f"  Found {len(logic_axioms)} logic axioms")
        for axiom in logic_axioms[:5]:
            print(f"    - {axiom.asset_id}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ CrossDB test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_il_slots_conversion():
    """ILSlots変換のテスト"""
    print("="*80)
    print("AVH Math Integration Test - IL Slots Conversion")
    print("="*80)
    print()
    
    # テスト用IR
    test_ir = {
        "task": "decide",
        "domain": "logic_propositional",
        "entities": [
            {"type": "formula", "value": "(A -> B) & A -> B"}
        ],
        "constraints": [],
        "query": {"target": "tautology"},
        "metadata": {"source_text": "Is '(A -> B) & A -> B' a tautology?"}
    }
    
    try:
        il_slots = ILSlots.from_ir(test_ir)
        
        print(f"✅ IR → ILSlots conversion successful")
        print(f"  Problem type: {il_slots.problem_type}")
        print(f"  Domain: {il_slots.domain}")
        print(f"  Formula: {il_slots.formula}")
        print(f"  Target: {il_slots.target}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ ILSlots conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AVH MATH INTEGRATION TEST SUITE")
    print("="*80)
    print()
    
    # Test 1: Logic Solvers
    logic_passed, logic_total = test_logic_solvers()
    
    # Test 2: CrossDB
    crossdb_ok = test_cross_db()
    
    # Test 3: ILSlots
    ilslots_ok = test_il_slots_conversion()
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Logic Solvers: {logic_passed}/{logic_total} tests passed")
    print(f"CrossDB: {'✅ PASS' if crossdb_ok else '❌ FAIL'}")
    print(f"ILSlots: {'✅ PASS' if ilslots_ok else '❌ FAIL'}")
    print()
    
    total_passed = logic_passed + (1 if crossdb_ok else 0) + (1 if ilslots_ok else 0)
    total_tests = logic_total + 2
    
    print(f"Overall: {total_passed}/{total_tests} components working")
    print("="*80)
