"""
mcq_elimination_solver.py
MCQ消去法ソルバー — 知識ベースで矛盾する選択肢を除去

設計:
  1. Wikipedia/LLM facts から制約条件を抽出
  2. 各選択肢が制約に矛盾するかチェック
  3. 残った選択肢が1つなら確定、2つなら LLM で判定（レベル2）

消去戦略:
  - 数値範囲チェック（facts に "X is between 10 and 20" → 選択肢 "5" を除去）
  - カテゴリ不整合（facts に "X is a mammal" → 選択肢 "reptile" を除去）
  - 否定チェック（facts に "X is NOT Y" → 選択肢 "Y" を除去）
  - 定義矛盾（facts の定義と選択肢の内容が矛盾）
"""

from __future__ import annotations
import re
import json
from typing import Dict, List, Optional, Tuple


def eliminate_choices(
    ir_dict: dict,
    choices: Dict[str, str],
    facts: List[dict],
    use_llm: bool = True,
) -> Optional[Tuple[str, float, str]]:
    """
    知識ベースの消去法で MCQ を解く。

    Returns:
        (label, confidence, method) or None
    """
    if not choices or len(choices) < 2:
        return None

    # facts からテキストを統合
    facts_text = _facts_to_text(facts)
    if not facts_text:
        return None

    # 制約抽出
    constraints = _extract_constraints(facts_text)
    if not constraints:
        # 制約抽出できなくても LLM 消去法を試す
        if use_llm:
            return _llm_elimination(ir_dict, choices, facts)
        return None

    # 各選択肢を制約でチェック
    eliminated = {}  # label -> reason
    for label, text in choices.items():
        for constraint in constraints:
            violation = _check_violation(text, constraint)
            if violation:
                eliminated[label] = violation
                break

    remaining = {k: v for k, v in choices.items() if k not in eliminated}

    if len(remaining) == 1:
        winner = list(remaining.keys())[0]
        n_eliminated = len(eliminated)
        reasons = "; ".join(f"{k}:{v}" for k, v in eliminated.items())
        return winner, 0.8, f"mcq_elimination(removed={n_eliminated}: {reasons})"

    if len(remaining) == 2 and use_llm:
        # 2択に絞れたら LLM で最終判定
        result = _llm_final_pick(ir_dict, remaining, facts)
        if result:
            return result[0], result[1] * 0.7, f"mcq_elimination+llm(removed={len(eliminated)})"

    if len(remaining) < len(choices) and use_llm:
        # 一部消去できたら LLM で残りを判定
        result = _llm_final_pick(ir_dict, remaining, facts)
        if result:
            return result[0], result[1] * 0.6, f"mcq_partial_elimination(removed={len(eliminated)})"

    return None


def _facts_to_text(facts: List[dict]) -> str:
    """facts リストをテキストに変換"""
    parts = []
    for f in facts:
        if isinstance(f, dict):
            s = f.get("summary", "") or f.get("plain", "")
            if s:
                parts.append(s)
            for p in f.get("properties", []):
                parts.append(str(p))
        elif hasattr(f, 'summary'):
            if f.summary:
                parts.append(f.summary)
    return " ".join(parts)


def _extract_constraints(facts_text: str) -> list[dict]:
    """facts テキストから制約条件を抽出"""
    constraints = []
    text = facts_text

    # 数値範囲: "X is between A and B", "X ranges from A to B"
    for m in re.finditer(
        r'(?:is\s+(?:between|approximately|about|around))\s+(\d+[\d.,]*)\s+(?:and|to)\s+(\d+[\d.,]*)',
        text, re.IGNORECASE
    ):
        try:
            lo = float(m.group(1).replace(',', ''))
            hi = float(m.group(2).replace(',', ''))
            constraints.append({"type": "range", "min": lo, "max": hi})
        except ValueError:
            pass

    # 否定: "X is not Y", "X does not Y"
    for m in re.finditer(
        r'(?:is\s+not|does\s+not|cannot|never|unlike)\s+(?:a\s+|an\s+)?(\w+(?:\s+\w+){0,2})',
        text, re.IGNORECASE
    ):
        constraints.append({"type": "negation", "value": m.group(1).lower().strip()})

    # 等値: "X is exactly Y", "X equals Y"
    for m in re.finditer(
        r'(?:is\s+exactly|equals?|is\s+equal\s+to)\s+(\d+[\d.,]*)',
        text, re.IGNORECASE
    ):
        try:
            val = float(m.group(1).replace(',', ''))
            constraints.append({"type": "exact", "value": val})
        except ValueError:
            pass

    # カテゴリ: "X is a Y"
    for m in re.finditer(
        r'(?:is\s+(?:a|an)\s+)(\w+(?:\s+\w+){0,2})',
        text, re.IGNORECASE
    ):
        constraints.append({"type": "category", "value": m.group(1).lower().strip()})

    # 比較: "X is greater/less than Y"
    for m in re.finditer(
        r'(?:is\s+(?:greater|more|larger|higher)\s+than)\s+(\d+[\d.,]*)',
        text, re.IGNORECASE
    ):
        try:
            val = float(m.group(1).replace(',', ''))
            constraints.append({"type": "min", "value": val})
        except ValueError:
            pass

    for m in re.finditer(
        r'(?:is\s+(?:less|fewer|smaller|lower)\s+than)\s+(\d+[\d.,]*)',
        text, re.IGNORECASE
    ):
        try:
            val = float(m.group(1).replace(',', ''))
            constraints.append({"type": "max", "value": val})
        except ValueError:
            pass

    return constraints


def _check_violation(choice_text: str, constraint: dict) -> Optional[str]:
    """選択肢が制約に矛盾するかチェック"""
    ct = constraint["type"]

    # 選択肢から数値を抽出
    numbers = re.findall(r'[-+]?\d+[\d.,]*', choice_text)

    if ct == "range" and numbers:
        for num_str in numbers:
            try:
                val = float(num_str.replace(',', ''))
                if val < constraint["min"] or val > constraint["max"]:
                    return f"out_of_range({val} not in [{constraint['min']}, {constraint['max']}])"
            except ValueError:
                pass

    elif ct == "exact" and numbers:
        for num_str in numbers:
            try:
                val = float(num_str.replace(',', ''))
                if abs(val - constraint["value"]) > 0.001 * max(abs(val), abs(constraint["value"]), 1):
                    return f"not_exact({val} != {constraint['value']})"
            except ValueError:
                pass

    elif ct == "negation":
        neg_val = constraint["value"]
        if neg_val in choice_text.lower():
            return f"negated({neg_val})"

    elif ct == "min" and numbers:
        for num_str in numbers:
            try:
                val = float(num_str.replace(',', ''))
                if val <= constraint["value"]:
                    return f"below_min({val} <= {constraint['value']})"
            except ValueError:
                pass

    elif ct == "max" and numbers:
        for num_str in numbers:
            try:
                val = float(num_str.replace(',', ''))
                if val >= constraint["value"]:
                    return f"above_max({val} >= {constraint['value']})"
            except ValueError:
                pass

    return None


def _llm_elimination(
    ir_dict: dict, choices: Dict[str, str], facts: List[dict]
) -> Optional[Tuple[str, float, str]]:
    """LLM に直接消去法を依頼（レベル2）"""
    return _llm_final_pick(ir_dict, choices, facts)


def _llm_final_pick(
    ir_dict: dict, choices: Dict[str, str], facts: List[dict]
) -> Optional[Tuple[str, float]]:
    """
    LLM にIR + 選択肢 + facts を渡して最終判定。

    鉄の壁レベル2: 問題文本体は渡さない。
    """
    try:
        import urllib.request

        domain = ir_dict.get("domain", ir_dict.get("domain_hint", ["unknown"]))
        if isinstance(domain, list):
            domain = domain[0] if domain else "unknown"
        task = ir_dict.get("task", "unknown")
        entities = ir_dict.get("entities", [])
        entity_str = ", ".join(
            f"{e.get('name', '')}: {e.get('value', '')}"
            for e in entities[:5] if e.get('name')
        )

        facts_str = "\n".join(
            f"- {(_get_summary(f))[:200]}"
            for f in facts[:5]
        )

        choices_str = "\n".join(f"{k}: {v}" for k, v in sorted(choices.items()))

        prompt = f"""You are eliminating wrong answers in a multiple-choice question.

Domain: {domain}
Task: {task}
Entities: {entity_str}

Known facts:
{facts_str}

Remaining choices:
{choices_str}

Based ONLY on the known facts, which choices can be eliminated and which is correct?
Reply with ONLY: {{"answer": "X", "confidence": 0.0-1.0, "eliminated": ["Y", "Z"], "reason": "brief"}}"""

        from config import LLM_CONFIG
        api_url = LLM_CONFIG.get("base_url", "http://localhost:11434/v1") + "/chat/completions"
        model = LLM_CONFIG.get("model", "qwen2.5:7b-instruct")

        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 150,
        }).encode()

        req = urllib.request.Request(
            api_url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            content = data["choices"][0]["message"]["content"]

            # JSON パース
            json_match = re.search(r'\{[^}]*"answer"[^}]*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                answer = parsed.get("answer", "").strip().upper()
                conf = float(parsed.get("confidence", 0.5))
                if answer in choices:
                    return (answer, conf)

    except Exception:
        pass

    return None


def _get_summary(f) -> str:
    if isinstance(f, dict):
        return f.get("summary", "") or f.get("plain", "")
    return getattr(f, 'summary', '') or ''
