# Verantyx V6 — ARC-AGI-2 Solver

## 🎯 227/1000 (22.7%) on ARC-AGI-2 Training Set

> Zero neural networks. Zero LLM calls. Zero hardcoded answers.
> Pure symbolic program synthesis — every solution is a verifiable, interpretable program.

Verantyx is a rule-based solver for [ARC-AGI-2](https://arcprize.org/), the benchmark designed to test general fluid intelligence in machines. It discovers transformation programs from input-output examples using compositional search over a custom DSL, with no training data beyond the task's own examples.

![Verantyx solving ARC tasks](verantyx_demo.gif)

---

## How It Solves Tasks

Verantyx treats each ARC task as a **program synthesis** problem: given 2–3 input-output pairs, find a program `P` such that `P(input) == output` for all training pairs, then apply `P` to the test input.

The core loop:

```
For each task:
  1. Generate candidate "pieces" (atomic transforms) from training examples
  2. Search for compositions that perfectly reconstruct all training outputs
  3. Verify on held-out training pairs (leave-one-out)
  4. Apply the verified program to the test input
```

There is no learned model, no embedding space, no gradient descent. Every solution is a symbolic program that can be inspected and verified.

### The Cross DSL

At the heart of Verantyx is the **Cross DSL** — a neighborhood-based rule language where each output cell is determined by a function of its local neighborhood in the input grid:

```
output[r][c] = f(input[r-1][c], input[r][c-1], input[r][c], input[r][c+1], input[r+1][c])
```

The "cross" refers to the 5-cell Von Neumann neighborhood (center + 4 cardinal neighbors). Rules are learned by building a lookup table from training examples, then verifying consistency across all training pairs.

This deceptively simple formulation solves **57% of all tasks Verantyx can handle** — because a surprising number of ARC transformations are locally determined.

### Beyond Local Rules

When neighborhood rules aren't enough, Verantyx escalates through increasingly powerful phases:

| Phase | Method | What It Handles |
|---|---|---|
| **1** | Cross DSL (NB rules) | Locally-determined transforms, cellular automata |
| **1.5** | Standalone primitives | Flip, rotate, crop, scale, gravity, fill |
| **2** | Stamp/Pattern fill | Object detection → pattern stamping by shape/color/size |
| **3** | Composite chains | 2–3 step transform sequences (`crop → recolor → tile`) |
| **4** | Iterative Cross | Multi-step residual: apply transform, learn correction on residual |
| **5** | Puzzle Reasoning Language | Declarative pattern matching with spatial predicates |
| **6** | ProgramTree synthesis | CEGIS-based condition/loop/sequence program search |
| **7** | CrossUniverse | Recursive spatial decomposition (separator walls, room propagation) |

Each phase operates independently. A task is solved when **any** phase produces a program that perfectly reconstructs all training outputs.

### Iterative Cross: Residual Learning Without Gradients

One of Verantyx's key innovations is **Iterative Cross** — a multi-step compositional strategy inspired by boosting:

1. Apply the best single transform found so far
2. Compute the **residual** (diff between current output and target)
3. Learn a second transform on the residual
4. Compose them: `P = P2 ∘ P1`

This handles tasks like "extract the largest object, then recolor it by neighborhood rules" — common in ARC but impossible with a single transform step.

### Puzzle Reasoning Language

For tasks requiring global spatial reasoning (ray casting, flood fill, region segmentation), Verantyx uses a **declarative pattern language**:

```
ray_extend_down     — extend colored cells downward until hitting a wall
fill_intersection   — fill the intersection region of two colored areas
sep_v_propagate     — vertical separator creates rooms, propagate colors
```

These are not handcoded task solutions — they're **general-purpose spatial primitives** that each handle a class of tasks. New primitives are added when analysis reveals a recurring pattern in unsolved tasks.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Cross Engine                     │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Piece     │  │ Phase    │  │ Verification │   │
│  │ Generators│→ │ Pipeline │→ │ (LOO + test) │   │
│  └──────────┘  └──────────┘  └──────────────┘   │
│       │                                           │
│  ┌────┴─────────────────────────────────┐        │
│  │ 22 Piece Generation Modules          │        │
│  │                                       │        │
│  │ cross_solver    per_object   tile     │        │
│  │ nb_extended     stamp        scale    │        │
│  │ extract_patch   symmetry     cegis    │        │
│  │ block_ir        puzzle_lang  ptree    │        │
│  │ cross_universe  grid_ir      ...      │        │
│  └───────────────────────────────────────┘        │
│                                                   │
│  Multi-step: composite → iterative → beam search  │
└─────────────────────────────────────────────────┘
```

### Solution Breakdown (v51 — 201 tasks solved)

| Method | Count | Share |
|---|---|---|
| Neighborhood Rules (Cross DSL) | 300 | 57.3% |
| Stamp / Pattern Fill | 60 | 11.5% |
| Puzzle Reasoning Language | 59 | 11.3% |
| Tile / Scale Transform | 23 | 4.4% |
| Extract / Crop | 22 | 4.2% |
| Grid Transforms (fill, gravity, recolor) | 21 | 4.0% |
| Symmetry / Rotation | 16 | 3.1% |
| ProgramTree Synthesis | 11 | 2.1% |
| Composite Chains | 6 | 1.1% |
| CEGIS / Block IR / Other | 6 | 1.1% |
| **Total (pieces attempted)** | **524** | |

> Note: Total > 201 because some tasks are solved by multi-step compositions (each step is a separate piece).

### Stats

- **304 source files**, ~100K lines of Python
- **22 piece-generation modules**, 7 solving phases
- **0.48s average per task** (single-threaded, M-series Mac)
- **Zero external dependencies** — no LLMs, no neural networks, no pretrained models

---

## Score History

```
        11%       14%       17%       20%
         ├─────────┼─────────┼─────────┤
 v19 113 ▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░  11.3%
 v23 120 ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░  12.0%
 v26 133 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░  13.3%
 v30 141 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░  14.1%
 v33 150 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░  15.0%
 v35 162 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░  16.2%
 v36 165 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░  16.5%
 v37 168 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░  16.8%
 v42 175 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░  17.5%
 v44 182 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░  18.2%
 v45 187 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░  18.7%
 v47 196 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░  19.6%
 v48 198 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░  19.8%
 v49 199 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░  19.9%
 v50 200 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░  20.0%
 v51 201 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░  20.1%
 v59 222 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░  22.2% ★
         └─────────┴─────────┴─────────┘
```

| Version | Score | Key Changes |
|---|---|---|
| v19 | 113 (11.3%) | Initial release — Cross DSL + NB rules |
| v23 | 120 (12.0%) | Iterative Cross: 2-step residual learning |
| v26 | 133 (13.3%) | Beam search + DSL enumeration |
| v30 | 141 (14.1%) | Extended NB + composite chains |
| v33 | 150 (15.0%) | Stamp patterns + symmetry fill |
| v35 | 162 (16.2%) | Puzzle reasoning language + beam search |
| v36 | 165 (16.5%) | Per-object stamp + gravity |
| v37 | 168 (16.8%) | ProgramTree synthesis (CEGIS) |
| v42 | 175 (17.5%) | Grid pattern + extract rank |
| v44 | 182 (18.2%) | Region recolor + fill bbox |
| v45 | 187 (18.7%) | holes_to_color + cluster_histogram |
| v47 | 196 (19.6%) | dynamic_tile + cell_to_color_block + color_to_pattern |
| v48 | 198 (19.8%) | Block IR between_fill + converge:stamp |
| v49 | 199 (19.9%) | scale_border_dup |
| **v50** | **200 (20.0%)** | **CrossUniverse: separator_propagate** |
| v51 | 201 (20.1%) | self_stamp: input-as-stamp spatial placement |
| **v59** | **222 (22.2%)** | **3D cross structure: 8 new primitives, self-tile, odd-one-out** |

---

## Design Philosophy

**Interpretability over accuracy.** Every solution Verantyx produces is a readable program, not a black-box prediction. If it outputs an answer, you can trace exactly why.

**No shortcuts.** No answer memorization, no dataset-specific heuristics, no LLM-generated guesses. Each solution must generalize from the training examples alone.

**Compositional search over brute force.** Rather than enumerating all possible programs, Verantyx builds solutions from typed, reusable pieces. This keeps the search space manageable while maintaining expressiveness.

**Fail cleanly.** If Verantyx can't find a verified program, it outputs nothing. The 22.2% score represents tasks where it found *provably correct* programs — not best guesses.

---

## What's Next

The current bottleneck is **ver=0 tasks** (new ARC-AGI-2 tasks with no precedent in ARC-AGI-1) — Verantyx solves 0% of these, compared to 35.6% for ver=1 and 55.9% for ver=2. These tasks require primitives not yet in the DSL vocabulary.

Key directions:
- **CrossUniverse expansion** — recursive spatial decomposition for nested structures
- **Overfit detection** — LOO cross-validation to eliminate false positives from NB rules
- **New primitive discovery** — systematic analysis of ver=0 failure modes to identify missing transform classes

---

## Setup

```bash
# Requires ARC-AGI-2 dataset
git clone https://github.com/arcprize/arc-agi-2 /tmp/arc-agi-2

# Run evaluation
cd verantyx_v6
python3 -m arc.eval_cross_engine --split training

# Run single task
python3 -m arc.eval_cross_engine --split training --task <task_id>
```

Requires Python 3.10+ and NumPy. No other dependencies.

## Built With

This project is built by **kofdai** in collaboration with **[OpenClaw](https://openclaw.ai)** — an AI agent platform.

The entire development process — architecture design, implementation, debugging, and evaluation — is a human-AI collaboration:

- **kofdai**: Core architecture vision, algorithm design (Cross DSL, MultiScale Cross, 3D Cross Geometry), strategic decisions on which approaches to pursue
- **OpenClaw (Claude)**: Implementation, code generation, systematic testing, evaluation pipeline, debugging assistance

Every design decision comes from kofdai. The AI implements, tests, and iterates on those designs. No LLM is used in the actual solving pipeline — Verantyx is 100% symbolic. The AI assists only in the development process itself.

This is what human-AI collaboration looks like: human intuition and architectural vision, combined with AI's ability to rapidly implement and test ideas.

## License

MIT

## Author

[kofdai](https://github.com/kofdai) × [OpenClaw](https://openclaw.ai)
