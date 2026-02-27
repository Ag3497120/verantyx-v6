"""
sponsors/eval_and_save.py — Run eval and save detailed sponsor-tier data

Usage:
  cd ~/verantyx_v6
  PYTHONPATH=. python3 sponsors/eval_and_save.py --version v63 [--split training]

Saves to sponsors/data/<version>/:
  - results.jsonl          — per-task: solved, rule, time, ver
  - failure_details.jsonl  — per-task failure: phase reached, partial scores, why
  - failure_analysis.md    — human-readable breakdown
  - rule_distribution.json — which rules solved what
  - summary.json           — top-level stats
  - raw_log.txt            — full eval stdout
"""

import json
import re
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter

def parse_eval_log(log_path: str):
    """Parse eval log into structured results"""
    results = []
    with open(log_path) as f:
        for line in f:
            m = re.match(
                r'\s*\[(\d+)/1000\]\s+(✓|✗)\s+(\d+\.\d+)s/t\s+(\w+)\s+ver=(\d+)\s*(rule=(.+))?',
                line.strip()
            )
            if m:
                results.append({
                    'rank': int(m.group(1)),
                    'solved': m.group(2) == '✓',
                    'time_s': float(m.group(3)),
                    'task_id': m.group(4),
                    'ver': int(m.group(5)),
                    'rule': m.group(7).strip() if m.group(7) else None,
                })
    return results


def generate_failure_details(results, classifications_path=None):
    """Generate detailed failure analysis per task"""
    cls = {}
    if classifications_path and os.path.exists(classifications_path):
        with open(classifications_path) as f:
            cls = json.load(f)
    
    failures = []
    for r in results:
        if r['solved']:
            continue
        
        tid = r['task_id']
        detail = {
            'task_id': tid,
            'ver': r['ver'],
            'time_s': r['time_s'],
            'category': cls.get(tid, {}).get('primary', 'unclassified'),
            'subcategory': cls.get(tid, {}).get('secondary', None),
            'description': cls.get(tid, {}).get('description', None),
        }
        
        # Failure reason heuristic based on ver
        if r['ver'] == 0:
            detail['failure_reason'] = 'no_rule_matched'
            detail['failure_detail'] = 'No piece generator produced a rule matching any training example'
        elif r['ver'] == 1:
            detail['failure_reason'] = 'partial_match'
            detail['failure_detail'] = 'Rule matched 1 training example but failed on remaining examples (overfitting)'
        elif r['ver'] == 2:
            detail['failure_reason'] = 'incomplete_generalization'
            detail['failure_detail'] = 'Rule matched 2 training examples but not all'
        else:
            detail['failure_reason'] = 'verification_gap'
            detail['failure_detail'] = f'Matched {r["ver"]} examples but not enough for full verification'
        
        failures.append(detail)
    
    return failures


def generate_analysis_report(results, failures, version):
    """Generate human-readable failure analysis markdown"""
    solved = [r for r in results if r['solved']]
    unsolved = [r for r in results if not r['solved']]
    
    ver_dist = Counter(r['ver'] for r in results)
    ver_solved = Counter(r['ver'] for r in solved)
    rule_dist = Counter(r['rule'] for r in solved if r['rule'])
    cat_dist = Counter(f['category'] for f in failures)
    reason_dist = Counter(f['failure_reason'] for f in failures)
    
    report = f"""# Failure Analysis — Verantyx {version} ({len(solved)}/1000)

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M JST')}

## Overview
- **Solved**: {len(solved)}/1000 ({len(solved)/10:.1f}%)
- **Unsolved**: {len(unsolved)}/1000
- **Avg time**: {sum(r['time_s'] for r in results) / len(results):.2f}s/task
- **Total eval time**: {sum(r['time_s'] for r in results):.0f}s

## Score by Verification Level

| ver | Total | Solved | Rate |
|-----|-------|--------|------|
"""
    for v in sorted(ver_dist.keys()):
        total = ver_dist[v]
        s = ver_solved.get(v, 0)
        rate = f'{s/total*100:.1f}%' if total > 0 else '0%'
        report += f'| {v} | {total} | {s} | {rate} |\n'
    
    report += """
## Failure Reasons

| Reason | Count | % |
|--------|-------|----|
"""
    for reason, count in reason_dist.most_common():
        report += f'| {reason} | {count} | {count/len(unsolved)*100:.1f}% |\n'
    
    report += """
## Unsolved by Category (LLM-classified)

| Category | Count | % of Unsolved |
|----------|-------|--------------|
"""
    for cat, count in cat_dist.most_common(20):
        report += f'| {cat} | {count} | {count/len(unsolved)*100:.1f}% |\n'
    
    report += """
## Top 20 Solving Rules

| Rule | Count |
|------|-------|
"""
    for rule, count in rule_dist.most_common(20):
        report += f'| `{rule}` | {count} |\n'
    
    report += """
## Unsolved Task IDs by Category

"""
    cat_tasks = {}
    for f in failures:
        cat_tasks.setdefault(f['category'], []).append(f['task_id'])
    
    for cat in sorted(cat_tasks.keys()):
        tids = cat_tasks[cat]
        report += f"### {cat} ({len(tids)} tasks)\n"
        report += f"`{'`, `'.join(sorted(tids)[:20])}`"
        if len(tids) > 20:
            report += f" ... and {len(tids)-20} more"
        report += "\n\n"
    
    report += "\n---\n*Auto-generated. Sponsor-exclusive detailed per-task reasoning traces available.*\n"
    
    return report


def save_sponsor_data(version, log_path, classifications_path=None):
    """Main entry: parse log, generate analysis, save everything"""
    out_dir = Path(__file__).parent / 'data' / version
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse results
    results = parse_eval_log(log_path)
    if not results:
        print(f"ERROR: No results parsed from {log_path}")
        return
    
    solved_count = sum(1 for r in results if r['solved'])
    print(f"Parsed {len(results)} tasks, {solved_count} solved ({solved_count/10:.1f}%)")
    
    # Save results.jsonl
    with open(out_dir / 'results.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    # Generate and save failure details
    failures = generate_failure_details(results, classifications_path)
    with open(out_dir / 'failure_details.jsonl', 'w') as f:
        for fd in failures:
            f.write(json.dumps(fd) + '\n')
    
    # Rule distribution
    rule_dist = Counter(r['rule'] for r in results if r['solved'] and r['rule'])
    with open(out_dir / 'rule_distribution.json', 'w') as f:
        json.dump(dict(rule_dist.most_common()), f, indent=2)
    
    # Summary
    summary = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'total_tasks': len(results),
        'solved': solved_count,
        'score_pct': round(solved_count / 10, 1),
        'avg_time_s': round(sum(r['time_s'] for r in results) / len(results), 3),
        'total_time_s': round(sum(r['time_s'] for r in results), 1),
        'ver_distribution': dict(Counter(r['ver'] for r in results)),
        'ver_solved': dict(Counter(r['ver'] for r in results if r['solved'])),
        'top_rules': dict(rule_dist.most_common(10)),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Copy raw log
    import shutil
    shutil.copy2(log_path, out_dir / 'raw_log.txt')
    
    # Failure analysis report
    report = generate_analysis_report(results, failures, version)
    with open(out_dir / 'failure_analysis.md', 'w') as f:
        f.write(report)
    
    print(f"\nSaved to {out_dir}/:")
    for p in sorted(out_dir.iterdir()):
        size = p.stat().st_size
        print(f"  {p.name:30s} {size:>10,} bytes")
    
    return out_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True, help='Version label (e.g. v63)')
    parser.add_argument('--log', default=None, help='Path to eval log file')
    parser.add_argument('--classifications', default='llm_classifications.json')
    args = parser.parse_args()
    
    if args.log is None:
        # Find latest log
        import glob
        logs = sorted(glob.glob('arc_v*_*.log') + glob.glob('arc_v*.log'), key=os.path.getmtime, reverse=True)
        if logs:
            args.log = logs[0]
            print(f"Using latest log: {args.log}")
        else:
            print("ERROR: No log file found. Run eval first or specify --log")
            sys.exit(1)
    
    save_sponsor_data(args.version, args.log, args.classifications)
