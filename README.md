# Verantyx V6 — ARC-AGI2 Results

## Score: 826/1000 (82.6%)

## Method

### Architecture: Program Synthesis + Verification

Verantyx uses a **hybrid approach** combining LLM-based program synthesis with deterministic verification:

1. **Program Synthesis** — Claude Sonnet 4.5 (`claude-sonnet-4-5`, Anthropic) generates a `transform(grid)` function in Python for each task
2. **Verification** — Verantyx's rule-based engine (`verify_transform.py`) executes the generated code against all training examples
3. **Adoption** — Only transforms that pass all training examples are adopted as solutions

The LLM does not output answers directly — it writes code, and the system verifies correctness.

### Execution
- **Parallel sub-agents**: 5-6 Claude Sonnet 4.5 instances processing 50-task batches concurrently via OpenClaw
- **Orchestration**: Claude Opus 4 (`claude-opus-4-6`, Anthropic) — task distribution and sub-agent management
- **Batched pipeline**: 1000 tasks distributed across agents, results aggregated upon completion
- **Fallback**: Verantyx's hand-crafted solvers handle tasks where program synthesis is unnecessary

### Stage Breakdown
| Stage | Method | Solved | Score |
|---|---|---|---|
| Stage 1 | Hand-crafted solvers (cross_engine v82) | 244 | 24.4% |
| Stage 2 | Claude Sonnet 4.5 program synthesis | 582 | — |
| **Combined** | **Stage 1 + Stage 2 (no overlap)** | **826** | **82.6%** |

### Hand-crafted Solvers (Stage 1)
- `cross_probe_fill` — Cross-structure expansion strategies
- `object_mover` — 7 movement strategies (gravity, slide, wall absorb, etc.)
- `cross_multiscale` — 6-axis cross descriptors + probe-based hole detection
- `cross3d` — 3D panel operations (invert_recolor, sym4fold, panel_compact, etc.)
- `iterative_cross_2` — 2-step residual learning
- And 30+ additional pattern matchers

## Models Used
- **Program synthesis**: `claude-sonnet-4-5` (Anthropic Claude Sonnet 4.5) — zero-shot, no fine-tuning
- **Orchestration**: `claude-opus-4-6` (Anthropic Claude Opus 4) — task distribution and sub-agent management
- All models used via API, no training or fine-tuning involved

## Key Insight
Traditional hand-crafted pattern matching plateaus around 24%. Letting an LLM **write and verify** transformation programs unlocks compositional generalization through code — reaching 82.6% on the ARC-AGI2 training set.

## Score Progression
| Version | Score | Method |
|---|---|---|
| v19 | 11.3% | Hand-crafted solvers |
| v50 | 20.0% | + CrossUniverse, separator_propagate |
| v60 | 22.4% | + cross3d 12 strategies |
| v72 | 23.4% | + object_mover, cross_probe_fill |
| v82 | 24.4% | Hand-crafted plateau |
| **v82 + Synth** | **82.6%** | **+ Claude Sonnet 4.5 program synthesis** |

## Repository
- GitHub: https://github.com/Ag3497120/verantyx-v6
- Author: kofdai
