# Verantyx-v6 — ARC-AGI-2 Solver

**196/1000 (19.6%)** on ARC-AGI-2 training set.

> No cheats. No bias. No hardcode. Pure rule-based reasoning.

## Demo

![Verantyx solving ARC tasks](verantyx_demo.gif)

## Architecture

A multi-phase Cross-Structure solver with 22 modules and 9 solving phases:

- **Phase 1**: Cross DSL (neighborhood rules, structural NB)
- **Phase 1.5**: Standalone primitives (flip, rotate, crop, scale)
- **Phase 7**: Puzzle Reasoning Language (declarative pattern matching)
- **Phase 8**: ProgramTree (CEGIS condition/loop/sequence synthesis)
- **Phase 9**: ARC-CEGIS (transform chain search)
- **Phase 3**: Multi-step composition (2-step, 3-step chains)
- **Phase 4**: Iterative Cross (residual learning)
- **Phase 5**: Multi-arm Beam Search
- **Phase 6**: DSL Program Enumeration

### Solution Breakdown (v47)

| Method | Count | % |
|---|---|---|
| Puzzle Language | 57 | 29.1% |
| Neighborhood Rules | 41 | 20.9% |
| Tile/Scale | 14 | 7.1% |
| ProgramTree | 10 | 5.1% |
| Extract | 10 | 5.1% |
| Composite/Compose | 9 | 4.6% |
| CEGIS | 7 | 3.6% |
| Per-Object | 8 | 4.1% |
| Other | 40 | 20.4% |

## Stats

- **39 source files**, 23,676 lines of Python
- **22 piece-generation modules**
- **0.48s average per task**
- Zero external LLM calls — pure algorithmic reasoning

## Score History

| Version | Score | Key Changes |
|---|---|---|
| v19 | 113/1000 (11.3%) | Initial release |
| v35 | 162/1000 (16.2%) | Puzzle lang + beam search |
| v45 | 187/1000 (18.7%) | holes_to_color + cluster_histogram |
| **v47** | **196/1000 (19.6%)** | dynamic_tile + cell_to_color_block + color_to_pattern |

## Setup

```bash
# Requires ARC-AGI-2 dataset
git clone https://github.com/arcprize/arc-agi-2 /tmp/arc-agi-2

# Run evaluation
cd verantyx_v6
python3 -m arc.eval_cross_engine --split training
```

## License

MIT
