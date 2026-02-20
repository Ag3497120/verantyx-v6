---
license: mit
tags:
  - symbolic-reasoning
  - rule-based
  - math
  - hle
  - reasoning
  - verantyx
  - no-gpu
  - deterministic
  - 600b-svd
pipeline_tag: question-answering
language:
  - en
metrics:
  - accuracy
model-index:
  - name: Verantyx V6 (HLE 8.56%)
    results:
      - task:
          type: text2text-generation
          name: Symbolic Reasoning
        dataset:
          name: Humanity's Last Exam (HLE)
          type: hle-2500
        metrics:
          - type: accuracy
            value: 8.56
            name: HLE 2500-question accuracy
            verified: false
---

# Verantyx V6 — HLE 8.56% (verantyx-hle-8)

---

## Model Overview

| Item | Details |
|------|---------|
| **Name** | Verantyx V6 |
| **Version** | 8 (Phase 5I — 600B SVD Integration) |
| **Type** | Rule-based symbolic reasoning system (non-LLM) |
| **Developer** | kofdai |
| **Language** | Python 3.8+ |
| **License** | MIT |
| **HLE Score** | **8.56%** (214 / 2500 questions) |
| **Previous best** | 6.84% (verantyx-hle-5) |
| **Improvement** | **+1.72pt (+25% relative)** |

---

## What is Verantyx?

Verantyx is a **purely rule-based, symbolic reasoning pipeline** — no neural network inference, no language model API calls. Every inference is deterministic and explainable.

---

## Architecture

```
Question (text)
    ↓ Decomposer (domain/task classification)
        ↑ [NEW] 600B SVD concept_dirs boost signal
Intermediate Representation (IR)
    ↓ Beam Search (piece retrieval from 108-piece DB)
Execution Path
    ↓ Executor (24 domain executors)
Structured Candidate
    ↓ Grammar Composer + Answer Matcher (LaTeX/fraction/percent/sci-notation)
Final Answer (string)
```

---

## What's New in v8 (vs v5 / 6.84%)

### 🔬 600B SVD Knowledge Integration (Major)
- Analyzed **DeepSeek V3 671B MoE** model weights **without inference** (static SVD)
- Extracted concept direction vectors from all **15,104 MoE expert weight matrices**
- Shape: `(15104, 4, 7168)` — 4 SVD directions × 7168-dim hidden space per expert
- Each expert classified into domain: calculus, algebra, number_theory, geometry, physics, etc.
- At inference time: query → BPE tokenize → embed_tokens average → cosine similarity against concept_dirs → Top-50 expert majority vote → domain boost signal
- Result: **more accurate domain detection** → correct executor selection

### ✅ Other improvements (from v5)
- Flexible answer matching (LaTeX normalization, fractions, percentages, scientific notation)
- Problem type detector (13 types)
- Equation solver (linear, quadratic, simultaneous)
- Specificity bias fix (`_score_specificity` weight: 0.3 → 0.05)
- 108 knowledge pieces across 24 domains

---

## HLE Results

### v8 (this version) — 8.56%
| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| Biology/Medicine | 38 | 280 | **13.6%** |
| Physics | 23 | 230 | **10.0%** |
| Humanities/Social Science | 19 | 219 | 8.7% |
| Engineering | 9 | 111 | 8.1% |
| Math | 82 | 1021 | **8.0%** |
| Computer Science/AI | 18 | 241 | 7.5% |
| Other | 16 | 233 | 6.9% |
| Chemistry | 9 | 165 | 5.5% |
| **Total** | **214** | **2500** | **8.56%** |

### Score history
| Version | Score | Notes |
|---------|-------|-------|
| v3 (Phase 5A) | 3.50% | Baseline |
| v5 (Phase 5G) | 5.36% | Flexible matching + equation solver |
| v5 (Phase 5H) | 6.84% | Specificity bias fix |
| **v8 (Phase 5I)** | **8.56%** | **+600B SVD concept_dirs domain boost** |

---

## Key Technical Detail: Non-Inference Weight Analysis

The 600B knowledge extraction was performed entirely **statically** — the model weights were loaded as safetensors files and SVD was applied to each expert's `W_gate`/`W_up` matrices. No inference (forward pass) was needed.

- Input space directions: top-4 left singular vectors of W_gate (shape `[7168, ffn_dim]`)
- These are **7168-dimensional vectors in the same space as token embeddings**
- At query time: average token embeddings of the question → cosine similarity against all 60,416 direction vectors → domain classification boost

This approach extracts "what each expert specializes in" purely from weight geometry.

---

## Limitations

- Rule-based system: cannot generalize beyond implemented executors
- Many HLE questions require open-ended reasoning not covered by current pieces
- Chess problems (stockfish) not yet implemented
- Calculus symbolic computation (derivative/integral) still stub

---

## Files

| File | Description |
|------|-------------|
| `pipeline_enhanced.py` | Main pipeline |
| `decomposer/decomposer.py` | Domain/task classification + 600B boost |
| `knowledge/concept_search.py` | 600B SVD cosine similarity search |
| `knowledge/concept_boost.py` | Domain boost integration layer |
| `knowledge/concept_cache.jsonl` | Pre-computed query→domain cache (2500 entries) |
| `pieces/piece_db.jsonl` | 108 knowledge pieces |
| `executors/` | 24 domain executors |

---

*verantyx-hle-8 | kofdai | 2026-02-18*
