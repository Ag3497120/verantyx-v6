# Sponsors-Only Content

This directory contains research data available to [GitHub Sponsors](https://github.com/sponsors/Ag3497120).

## Contents

### 📊 Inference Logs
- `inference_log_v62.jsonl` — Per-task results for all 1,000 ARC-AGI-2 training tasks
  - Task ID, solve status, rule name, time taken, verification score
  - For failed tasks: which phase was reached, partial match score

### 📈 Failure Analysis
- `failure_analysis_v62.md` — Breakdown of unsolved tasks by category
  - LLM-classified categories (gravity, pattern_stamp, neighborhood_rule, etc.)
  - ver= distribution (how many train examples matched)
  - Actionable insights for each failure mode

### 🗺️ Development Roadmap
- `roadmap_202602.md` — Monthly roadmap with priority targets
  - Which task categories we're attacking next
  - Estimated impact per new primitive
  - DSL design sketches for upcoming features

### 🔬 DSL Design Drafts
- `dsl_drafts/` — Experimental DSL extensions before they hit main
  - Cross3D Probe specification
  - Corner stacking algebra
  - Gravity simulation formalization

---

*Updated with each version bump. Current: v62 (228/1000, 22.8%)*
