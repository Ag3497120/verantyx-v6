# Sponsors-Only Research Data

This directory contains detailed research data for [GitHub Sponsors](https://github.com/sponsors/Ag3497120).

## Structure

```
sponsors/
├── eval_and_save.py           ← Auto-generates all data below after each eval
├── data/
│   ├── v65/                   ← Latest: 228/1000 (22.8%)
│   │   ├── summary.json       ← Top-level stats
│   │   ├── results.jsonl      ← Per-task: solved, rule, time, ver
│   │   ├── failure_details.jsonl ← Per-task: why it failed, category, description
│   │   ├── failure_analysis.md   ← Human-readable breakdown
│   │   ├── rule_distribution.json ← Which rules solved what
│   │   └── raw_log.txt        ← Full eval stdout
│   └── (previous versions archived here)
```

## What's In Each Version

| File | Description | Who Wants This |
|------|-------------|----------------|
| `results.jsonl` | 1,000 task results with solve status, rule name, timing | Everyone |
| `failure_details.jsonl` | Failure reason, LLM-classified category, description for each unsolved task | Researchers building ARC solvers |
| `failure_analysis.md` | Category breakdown, ver distribution, top rules, unsolved task IDs | Quick reference |
| `rule_distribution.json` | Which symbolic rules solved which tasks | DSL designers |
| `raw_log.txt` | Full evaluation output | Reproducibility |
| `summary.json` | Score, timing, ver distribution | Dashboards |

## How Data Is Generated

After each eval run:
```bash
cd ~/verantyx_v6
PYTHONPATH=. python3 sponsors/eval_and_save.py --version <version> --log <log_file>
```

All data is auto-generated from the eval log + LLM classification file.

---

*Updated each version. Current: v65 (228/1000, 22.8%)*
