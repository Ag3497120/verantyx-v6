# ⚡ Verantyx v6 — LLM-Free Symbolic Reasoning Engine for HLE

> **No LLM solves the problems. Every answer is structurally derived.**

[![HLE Score](https://img.shields.io/badge/HLE_Score-3.80%25_(bias--free)-blue)](https://agi.safe.ai/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-yellow)](https://python.org)

A verifiable, white-box reasoning engine that tackles [Humanity's Last Exam (HLE)](https://lastexam.ai/) — a PhD-level benchmark of 3,000 questions across all academic disciplines — **without using an LLM to answer questions directly**.

## 🤖 AI Context Guide

Verantyx is a **symbolic/structural reasoning system** that decomposes problems into intermediate representations (IR), retrieves knowledge from Wikipedia, atomizes facts into subject-predicate-object triples, and resolves answers through formal verification (CEGIS/SymPy/Z3) and cross-structure matching. The LLM (Qwen 7B, local) is used **only** as an auxiliary tool for IR extraction and MCQ tiebreaking — never for direct problem solving. This makes Verantyx fundamentally different from "prompt an LLM and hope" approaches.

## Why This Matters

Most HLE submissions are thin wrappers around frontier LLMs (GPT-4o, Claude, Gemini). Verantyx takes the opposite approach:

- 🔍 **Transparent**: Every answer comes with an audit trail showing exactly *why*
- 🧩 **Structural**: Problems decomposed into IR → pieces → Cross-structure verification
- ⚡ **Fast**: ~5 seconds per question (no heavy LLM inference)
- 🛡️ **Honest**: Returns INCONCLUSIVE rather than guessing — no statistical tricks

## Current Scores

| Version | Score | Date | Method |
|---|---|---|---|
| **Bias-Free v6** | **3.80%** (95/2500) | 2026-02-20 | Pure structural reasoning |
| No-cheat v2 | 12.5% (5/40)* | 2026-02-22 | + Wikipedia atom matching + MCQ cross-decompose |
| Atom v3 (in progress) | ~4-6% | 2026-02-23 | + LLM→Atom classifier replacement (1000x faster) |

*\*40-question sample evaluation*

> **Why the low score matters**: This is achieved with *zero* LLM problem-solving. The score represents genuine structural reasoning capability, not memorized knowledge from a 70B+ parameter model.

## Architecture

```
Problem Text
  ↓
┌──────────────────────────────────────┐
│ Decomposer (Rule-based IR extraction)│
│  → domain, entities, constraints,    │
│    missing knowledge, query type     │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ Knowledge Pipeline v2                │
│  → Wikipedia search + retrieval      │
│  → Fact Atomizer (200+ regex → SPO)  │
│  → Sentence splitter (clause-level)  │
│  → Exact Answer Assembler            │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ Cross-Structure Verification         │
│  ├── CEGIS Loop (SymPy/Z3/Enum)     │
│  ├── Atom Relation Classifier        │
│  │   (supports/contradicts/unknown)  │
│  ├── MCQ Cross-Decompose Solver      │
│  ├── MCQ Knowledge Matcher v2        │
│  └── MCQ Elimination Solver          │
└──────────────┬───────────────────────┘
               ↓
Answer (only if structurally justified)
  or INCONCLUSIVE (honest uncertainty)
```

### Key Innovation: Atom-Based Relation Classification

Instead of asking an LLM "which answer is correct?", Verantyx:

1. **Atomizes** Wikipedia facts into `(subject, predicate, object)` triples
2. **Atomizes** each MCQ choice the same way
3. **Cross-matches** atoms structurally (subject overlap → predicate match → object agreement)
4. **Detects contradictions** via antonym pairs (60+), negation flips, numeric mismatches
5. Returns `supports` / `contradicts` / `unknown` with full evidence trail

This runs in **3ms** vs 3-5 seconds for LLM classification — a **1000x speedup**.

### Key Components

| Module | Description |
|---|---|
| `pipeline_enhanced.py` | Main pipeline (8+ stages) |
| `knowledge/fact_atomizer.py` | 200+ regex patterns → FactAtom(S,P,O) extraction |
| `knowledge/sentence_splitter.py` | Rule-based clause splitting for complex sentences |
| `knowledge/exact_answer_assembler.py` | Query→Atom matching for direct answers |
| `executors/atom_relation_classifier.py` | **NEW** Atom-based supports/contradicts (replaces LLM) |
| `executors/mcq_knowledge_matcher_v2.py` | MCQ scoring via Atom relations + lexical cross-validation |
| `executors/mcq_cross_decompose_solver.py` | Per-choice decomposition → Wikipedia → cross-match |
| `cegis/cegis_loop.py` | Counterexample-Guided Inductive Synthesis |
| `verifiers/sympy_verifier.py` | SymPy mathematical verification |
| `verifiers/z3_verifier.py` | Z3 SMT solver integration |
| `puzzle/cross_simulation.py` | Finite model simulation for MCQ |
| `knowledge/concept_search.py` | 600B SVD concept search (DeepSeek V3 Expert directions) |

## Assets

- **DeepSeek V3-0324 Q8_0 GGUF** — 15 shards (713GB) for weight-based reasoning
- **600B SVD concept_dirs** — `(15104, 4, 7168)` Expert direction vectors (H100-computed)
- **embed_tokens** — `(129280, 7168)` DeepSeek token embeddings

## Quick Start

```bash
git clone https://github.com/Ag3497120/verantyx-v6.git
cd verantyx-v6

# Install dependencies
pip install sympy z3-solver wikipedia-api requests

# Run evaluation (bias-free, ~5s/question)
DISABLE_PATTERN_DETECTORS=1 python3 eval_2500_v2.py

# Single question demo
python3 demo_single.py
```

## Design Principles

1. **INCONCLUSIVE is correct behavior** — guessing is worse than silence
2. **LLM is auxiliary, not primary** — decomposition and tiebreaking only
3. **Every answer needs evidence** — audit bundles track the full reasoning chain
4. **Cross-structure verification** — multiple independent signals must agree
5. **Speed matters** — 5s/question enables rapid iteration

## Roadmap

- [x] Atom-based relation classifier (LLM-free, 3ms/call)
- [x] Fact Atomizer with 200+ regex patterns (20 categories)
- [x] Sentence splitter for complex Wikipedia text
- [ ] Parallel evaluation (4-8 workers, ~35min for 2500 questions)
- [ ] Atom classifier threshold tuning (supports precision improvement)
- [ ] MCQ elimination solver → Atom contradiction detection
- [ ] SymPy executor revival for integer-answer questions
- [ ] Cross-simulator answer generation

## Related

- [HuggingFace: kofdai/verantyx-hle-8](https://huggingface.co/kofdai/verantyx-hle-8) — 8.56% version
- [HuggingFace: kofdai/verantyx-hle-5](https://huggingface.co/kofdai/verantyx-hle-5) — 6.84% version
- [Humanity's Last Exam](https://lastexam.ai/)
- [HLE Paper](https://arxiv.org/abs/2501.14249)

## License

MIT

---

*Built by [@kofdai](https://github.com/kofdai) — structural reasoning over statistical guessing.*
