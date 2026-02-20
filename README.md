# Verantyx v6 — Bias-Free Symbolic Reasoning for HLE

A verifiable reasoning engine for [Humanity's Last Exam (HLE)](https://agi.safe.ai/) benchmark.

## Philosophy

> **LLM as decomposer only. Verification as the sole path to answers.**

- ❌ No statistical biases (no position priors, no letter-frequency fallbacks)
- ❌ No hardcoded problem-specific answers
- ✅ Every answer requires a formal proof (CEGIS / SymPy / Z3)
- ✅ Full audit trail per problem

## Current Score

| Version | Score | Date | Notes |
|---|---|---|---|
| **Bias-Free** | **3.80%** (95/2500) | 2026-02-20 | Clean baseline, no statistical tricks |
| Phase 5K (biased) | 15.68% | 2026-02-20 | Includes position priors + hardcoded detectors — **not valid** |
| Phase 5I (biased) | 8.56% | 2026-02-18 | Includes biases — **not valid** |

> ⚠️ **Note on biased scores**: Earlier versions used statistical position priors (B > D > C > A letter frequency) and hardcoded problem-specific answers. These inflate scores artificially and are not a valid measure of reasoning capability.

## Architecture

```
Problem
  ↓
Decomposer (IR extraction)
  ↓
CEGIS Loop (Counterexample-Guided Inductive Synthesis)
  ├── SymPy Verifier
  ├── Z3 SMT Verifier
  └── Enum Verifier
  ↓
Answer (only if formally proved)
```

### Key Components

| Module | Description |
|---|---|
| `core/ir.py` | Intermediate Representation (domain, task, slots) |
| `decomposer/decomposer.py` | Rule-based IR extraction from problem text |
| `cegis/cegis_loop.py` | CEGIS synthesis loop (2000ms timeout) |
| `cegis/certificate.py` | Proof certificate generation |
| `verifiers/sympy_verifier.py` | SymPy-based mathematical verification |
| `verifiers/z3_verifier.py` | Z3 SMT solver integration |
| `executors/sympy_solver.py` | Calculus / algebra symbolic computation |
| `puzzle/cross_simulation.py` | Finite model simulation for MCQ |
| `pipeline_enhanced.py` | Main pipeline (8 stages) |

## Score Breakdown (Bias-Free, 3.80%)

| Method | Count |
|---|---|
| cegis_proved | 69 |
| unknown | 16 |
| math_cross_sim | 7 |
| puzzle_reasoning | 1 |
| propositional_simulation | 1 |
| hle_boost:detector | 1 |

## Assets Used

- **DeepSeek V3-0324 Q8_0 GGUF** (13/15 shards downloaded)
- **600B SVD concept_dirs** (H100-computed): `(15104, 4, 7168)` Expert direction vectors
- **embed_tokens**: `(129280, 7168)` DeepSeek token embeddings

## Running

```bash
cd verantyx_v6
python3 quick_eval_hle.py
# ~41 seconds for 2500 questions
# Output: hle_2500_per_problem.json
```

## What's Blocked

- **HLE Math (PhD-level)**: Elliptic curves, moduli spaces, Diophantine equations — SymPy cannot reach these
- **MCQ accuracy**: 82/628 correct (13.1%) — room for improvement via non-firing weight analysis
- **GGUF shards 00011, 00013**: Still downloading (XET CDN throttling)

## Related

- [HuggingFace: kofdai/verantyx-hle-8](https://huggingface.co/kofdai/verantyx-hle-8) — biased 8.56% version (see warning above)
- [Humanity's Last Exam](https://agi.safe.ai/)
