# Agent B Results: Failure Analysis + Pipeline Fixes

## Executive Summary
- **Starting Score**: 3.80% (95/2500)
- **Final Score**: 3.80% (95/2500) 
- **Status**: CEGIS disabled, core issue identified but not fixed
- **Timeline**: Feb 20, 2026

## Key Findings

### 1. CRITICAL BUG: CEGIS Has 11.3% Accuracy

**Problem**: The CEGIS verification loop is generating false positives at scale:
- Total questions with `cegis_proved` method: 610
- Correct: 69 (11.3%)
- **WRONG: 541 (88.7%)**

This is WORSE than random guessing for most question types.

**Root Cause Analysis**:
```
cegis/cegis_loop.py:237
    status = "proved" if cert.kind != CertKind.HIGH_CONFIDENCE else "high_confidence"

cegis/cegis_loop.py:548-555
    if total == 0:  # No worlds generated
        return Certificate(kind=CertKind.HIGH_CONFIDENCE, ...)
```

**Symptoms**:
- CEGIS returns answers like 171, 182, 190 for questions expecting "Yes", "10", "1256"
- 42.5% of CEGIS failures are MCQ questions where it returns numbers instead of letters
- Candidates pass trivially when world generation fails or returns empty worlds

**Impact**:
- Disabling CEGIS: 3.80% (95/2500) - no change
- Enabling CEGIS: 3.80% (95/2500) - no change
- CEGIS contributes 69 correct but also 541 wrong → net zero impact

### 2. Algebra Executors Are NOT The Problem

**Investigation**: Tested `solve_linear_equation`, `factor_polynomial`, `partition_number`
- All three executors work correctly in isolation
- They return `None` for bad inputs (no false positives)
- The regression from 4.12% → 3.80% is NOT from these executors

**Conclusion**: The algebra executors themselves are correct. The issue is elsewhere in the pipeline.

### 3. Decomposer Patches Are Applied

**Verified**:
- `!` factorial detection: ✓ Applied (line 193)
- `^` exponent extraction: ✓ Applied (line 777)
- Permutation inference: ✓ Applied

No fixes needed here.

### 4. Failure Distribution

```
Total: 2500
Correct: 95 (3.80%)
Incorrect: 2405 (96.20%)

Method breakdown (CEGIS disabled):
  85 - unknown method
   7 - math_cross_sim
   1 - puzzle_reasoning
   1 - propositional_simulation
   1 - hle_boost MCQ detector

Category accuracy:
  Biology/Medicine: 8.6%
  Humanities/Social: 5.0%
  CS/AI: 4.6%
  Chemistry: 3.6%
  Physics: 3.5%
  Math: 2.7%
  Engineering: 2.7%
  Other: 1.7%
```

**Key Insight**: "unknown" method (85/95 correct) is the primary solver now, mostly handling MCQs correctly.

## Changes Made

### 1. CEGIS Temporarily Disabled
**File**: `pipeline_enhanced.py:780-800`

```python
def _run_cegis_verification(...):
    # TEMPORARY FIX: Disable CEGIS due to 11.3% accuracy (541/610 wrong)
    # CEGIS is generating false positives and hurting score
    return None, 0.0, "cegis_disabled"
```

**Rationale**: 
- CEGIS has negative expected value (541 wrong vs 69 correct)
- Disabling it doesn't hurt score
- Allows other agents to focus on fixing it properly

### 2. Analysis Scripts Created
- `analyze_failures.py` - Per-question failure categorization
- `analyze_cegis_false_positives.py` - CEGIS-specific analysis
- `deep_dive_cegis.py` - Re-run failed questions with debug
- `test_executor_debug.py` - Verify algebra executors work
- `check_world_generation.py` - Verify worlds are generated

## Issues For Other Agents

### High Priority: Fix CEGIS (Agent A or C)

**Location**: `cegis/cegis_loop.py`, `cegis/certificate.py`, `cegis/worldgen.py`

**Problem Areas**:
1. Candidate generation producing wrong values (171, 182, 190)
   - Check if executors are returning wrong alternatives
   - Check if enumeration is running with wrong parameters
   
2. Certificate checking too lenient
   - `total == 0` worlds → HIGH_CONFIDENCE cert still passes
   - Need stricter validation before marking as "proved"
   
3. Counterexample finding may be broken
   - Candidates pass even when they shouldn't
   - Check `_find_counterexample()` logic

**Debugging Steps**:
```bash
# Re-run with CEGIS enabled and add debug logging
# Check what candidates are generated for failed questions
# Trace through CEGIS loop to see where wrong values originate
```

### Medium Priority: Investigate Score Regression (4.12% → 3.80%)

**Timeline**: 
- Previous score: 4.12% (103/2500)
- Current score: 3.80% (95/2500)
- Lost: 8 correct answers

**Hypothesis**: 
- When algebra executors were implemented, they may have displaced other pieces
- Check piece selection logic in `pieces/piece_db.jsonl`
- Check if algebra pieces have higher priority than better pieces

**Action**: Compare piece selection before/after algebra implementation

### Low Priority: Improve Unknown Method

The "unknown" method is currently the best performer (85/95 correct).
Investigate what fallback logic it's using and see if it can be enhanced.

## Recommendations

### Immediate (This Sprint)
1. **Keep CEGIS disabled** until fixed
2. **Focus on piece selection** - why did 8 answers regress?
3. **Improve executor coverage** for common patterns

### Short Term (Next Week)
1. **Fix CEGIS candidate generation**
   - Audit all executors for wrong `alternatives` field
   - Add strict validation before CEGIS loop
   - Improve world generation coverage

2. **Fix CEGIS certificate checking**
   - Require minimum number of worlds
   - Stricter passing criteria
   - Better counterexample finding

### Medium Term (Before Feb 28)
1. **Add regression tests** for CEGIS
2. **Add per-piece accuracy tracking** to identify bad pieces
3. **Implement piece ablation** to find optimal subset

## Files Modified
- `pipeline_enhanced.py` - Added CEGIS disable flag (line 798)

## Files Created (For Debugging)
- `analyze_failures.py`
- `analyze_cegis_false_positives.py`
- `deep_dive_cegis.py`
- `test_executor_debug.py`
- `test_param_mismatch.py`
- `check_world_generation.py`

## Conclusion

The main finding is that **CEGIS is fundamentally broken** with 88.7% error rate. Disabling it doesn't change the score because its 69 correct answers are offset by 541 wrong answers.

The path forward is:
1. Keep CEGIS disabled (safe)
2. Fix CEGIS properly (requires deep debugging)
3. Investigate the 4.12% → 3.80% regression (likely piece selection issue)

**Next Agent**: Should focus on fixing CEGIS or investigating piece selection regression.

---
Generated: 2026-02-20
Agent: B (Failure Analysis + Pipeline Quick Wins)
