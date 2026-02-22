"""
mcq_knowledge_matcher_v2.py
MCQ知識マッチング v2 — LLM confidence依存を排除

設計原則:
  - LLM = 構造化変換器（スコアラーではない）
  - LLMに「どれが正しい？」を聞かない
  - LLMに「各選択肢とfactsの関係を分類せよ」を聞く
  - 最終判定はルールベース

3段階:
  KM-1: Lexical match（ルール）
  KM-2: LLM relation classification（supports/contradicts/unknown）
  KM-3: Rule-based decision（survivors判定）

鉄の壁レベル2: IR + 選択肢 + facts のみ（問題文は渡さない）
"""

from __future__ import annotations
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class ChoiceAnalysis:
    label: str
    text: str
    relation: str = "unknown"   # supports | contradicts | unknown
    fact_ids: list = field(default_factory=list)
    lexical_score: float = 0.0
    negation_detected: bool = False


@dataclass
class KMResult:
    """知識マッチング結果（デバッグ情報付き）"""
    answer: Optional[str] = None
    confidence: float = 0.0
    method: str = ""
    analyses: list = field(default_factory=list)
    survivors: list = field(default_factory=list)
    reject_reason: str = ""
    raw_llm_output: str = ""
    facts_count: int = 0
    choice_count: int = 0


# ── 否定語 ──
NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nor', 'none',
    'cannot', "can't", "doesn't", "don't", "isn't", "aren't",
    'without', 'lack', 'absence', 'except', 'exclude',
    'false', 'incorrect', 'wrong', 'invalid',
    'unlike', 'contrary', 'opposite',
}

# ── 同義語辞書（小規模） ──
SYNONYMS = {
    'increase': {'rise', 'grow', 'elevate', 'higher', 'more', 'greater', 'up'},
    'decrease': {'fall', 'drop', 'reduce', 'lower', 'less', 'fewer', 'down', 'decline'},
    'cause': {'lead', 'result', 'produce', 'induce', 'trigger'},
    'prevent': {'inhibit', 'block', 'stop', 'suppress', 'hinder'},
    'true': {'correct', 'valid', 'accurate', 'right'},
    'false': {'incorrect', 'invalid', 'inaccurate', 'wrong'},
}


def match_choices_v2(
    ir_dict: dict,
    choices: Dict[str, str],
    facts: List[dict],
) -> KMResult:
    """
    v2: LLM confidence不要のMCQ知識マッチング。

    Returns:
        KMResult（常に返す。answer=None = INCONCLUSIVE）
    """
    result = KMResult(
        facts_count=len(facts),
        choice_count=len(choices),
    )

    if not choices or len(choices) < 2 or not facts:
        result.reject_reason = f"insufficient_input(choices={len(choices)},facts={len(facts)})"
        return result

    # ── KM-1: Lexical match ──
    fact_texts = _collect_fact_texts(facts)
    analyses = []

    for label, text in choices.items():
        ca = ChoiceAnalysis(label=label, text=text)

        # 語彙一致スコア
        ca.lexical_score = _lexical_similarity(text, fact_texts)

        # 否定語検出
        ca.negation_detected = _has_negation(text, fact_texts)

        analyses.append(ca)

    # ── KM-2: Atom-based relation classification (LLM replaced) ──
    from executors.atom_relation_classifier import classify_relations_by_atoms
    llm_result = classify_relations_by_atoms(ir_dict, choices, facts)
    if llm_result:
        result.raw_llm_output = llm_result.get("_raw", "")
        for ca in analyses:
            llm_analysis = llm_result.get(ca.label, {})
            if llm_analysis:
                ca.relation = llm_analysis.get("relation", "unknown")
                ca.fact_ids = llm_analysis.get("fact_ids", [])

    # ── KM-3: Rule-based decision ──
    result.analyses = analyses

    # 1. contradicts が多いものを除去
    contradicted = [ca for ca in analyses if ca.relation == "contradicts"]
    supported = [ca for ca in analyses if ca.relation == "supports"]
    unknown = [ca for ca in analyses if ca.relation == "unknown"]

    survivors = [ca for ca in analyses if ca.relation != "contradicts"]
    result.survivors = [ca.label for ca in survivors]

    # 2. supports が1つだけ → 採用（fact_idsありかつ lexical cross-validation）
    if len(supported) == 1:
        winner = supported[0]
        n_contradicted = len(contradicted)
        has_evidence = bool(winner.fact_ids)
        # fact_ids 必須（evidence がないsupportsは信頼できない）
        if has_evidence:
            # lexical cross-validation: winner の lexical_score が最低ではないことを確認
            all_scores = [ca.lexical_score for ca in analyses]
            min_score = min(all_scores) if all_scores else 0
            if winner.lexical_score > min_score or n_contradicted >= 2:
                result.answer = winner.label
                result.confidence = 0.6 + 0.1 * n_contradicted
                result.method = f"km_v2:sole_support(supports=1,contradicts={n_contradicted},facts={len(facts)},evidence={len(winner.fact_ids)},lex={winner.lexical_score:.2f})"
                return result

    # 3. supports が0で、contradicts で1つだけ残った場合 → 採用
    if len(supported) == 0 and len(survivors) == 1:
        winner = survivors[0]
        result.answer = winner.label
        result.confidence = 0.6
        result.method = f"km_v2:last_standing(contradicts={len(contradicted)},facts={len(facts)})"
        return result

    # 4. supports が複数 → lexical_score で決着
    if len(supported) >= 2:
        supported.sort(key=lambda ca: ca.lexical_score, reverse=True)
        best = supported[0]
        second = supported[1]
        gap = best.lexical_score - second.lexical_score
        if gap >= 0.05:
            result.answer = best.label
            result.confidence = min(0.5 + gap, 0.8)
            result.method = f"km_v2:multi_support_lexical(gap={gap:.3f},supports={len(supported)})"
            return result

    # 5. LLM tiebreak 廃止（position bias で常に A を返す、0/3 正解）
    # → INCONCLUSIVE に落とす

    # INCONCLUSIVE
    result.reject_reason = f"no_clear_winner(supports={len(supported)},contradicts={len(contradicted)},unknown={len(unknown)})"
    return result


def _collect_fact_texts(facts: List[dict]) -> list[str]:
    """facts からテキストを収集"""
    texts = []
    for f in facts:
        if isinstance(f, dict):
            s = f.get("summary", "") or f.get("plain", "")
            if s:
                texts.append(s)
            for p in f.get("properties", []):
                texts.append(str(p))
        elif hasattr(f, 'summary') and f.summary:
            texts.append(f.summary)
    return texts


def _lexical_similarity(choice_text: str, fact_texts: list[str]) -> float:
    """選択肢と facts の語彙的類似度"""
    choice_words = _tokenize(choice_text)
    if not choice_words:
        return 0.0

    best = 0.0
    for ft in fact_texts:
        fact_words = _tokenize(ft)
        if not fact_words:
            continue

        # 共通語数 / 選択肢語数
        common = choice_words & fact_words
        score = len(common) / len(choice_words)

        # 同義語マッチ
        for cw in choice_words:
            for syn_group in SYNONYMS.values():
                if cw in syn_group:
                    if syn_group & fact_words:
                        score += 0.05

        best = max(best, score)

    return min(best, 1.0)


def _has_negation(choice_text: str, fact_texts: list[str]) -> bool:
    """選択肢に否定語が含まれ、facts と矛盾する可能性があるか"""
    choice_lower = choice_text.lower()
    return any(neg in choice_lower for neg in NEGATION_WORDS)


def _tokenize(text: str) -> set:
    """テキストをトークン化（ストップワード除去）"""
    stops = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'and', 'or', 'but', 'for', 'with', 'from', 'to', 'in', 'on',
        'at', 'by', 'of', 'it', 'its', 'this', 'that', 'which',
    }
    words = re.findall(r'[a-z]{3,}', text.lower())
    return {w for w in words if w not in stops}


def _llm_classify_relations(
    ir_dict: dict,
    choices: Dict[str, str],
    facts: List[dict],
) -> Optional[dict]:
    """
    LLM にfacts×選択肢の関係分類を依頼。

    鉄の壁レベル2: 問題文は渡さない。IR構造 + 選択肢 + facts のみ。
    LLMには「答え」を選ばせない。「関係」を分類させる。
    """
    try:
        import urllib.request
        import random

        domain = ir_dict.get("domain", ir_dict.get("domain_hint", ["unknown"]))
        if isinstance(domain, list):
            domain = domain[0] if domain else "unknown"
        task = ir_dict.get("task", "unknown")

        # IR からコンテキストを抽出（鉄の壁レベル2: IRは許可）
        entities = ir_dict.get("entities", [])
        constraints = ir_dict.get("constraints", [])
        query = ir_dict.get("query", "")
        missing = ir_dict.get("missing", [])
        # metadata からキーワードも取得
        metadata = ir_dict.get("metadata", {})
        keywords = metadata.get("keywords", [])

        context_parts = [f"Domain: {domain}", f"Task: {task}"]
        if entities:
            ent_str = ", ".join(str(e) for e in entities[:10])
            context_parts.append(f"Entities: {ent_str}")
        if constraints:
            con_str = "; ".join(str(c) for c in constraints[:5])
            context_parts.append(f"Constraints: {con_str}")
        if query:
            context_parts.append(f"Query: {query}")
        if missing:
            miss_str = ", ".join(str(m) for m in missing[:5])
            context_parts.append(f"Missing knowledge: {miss_str}")
        if keywords:
            context_parts.append(f"Keywords: {', '.join(keywords[:10])}")

        context_str = "\n".join(context_parts)

        # facts を fact_id 付きで整形（もっとたくさん含める）
        facts_lines = []
        for idx, f in enumerate(facts[:12]):
            if isinstance(f, dict):
                s = (f.get("summary", "") or f.get("plain", ""))[:300]
                # properties と formulas も追加
                props = f.get("properties", [])
                if props:
                    s += " | Props: " + "; ".join(str(p) for p in props[:3])
                formulas = f.get("formulas", [])
                if formulas:
                    s += " | Formulas: " + "; ".join(str(fl) for fl in formulas[:2])
            elif hasattr(f, 'summary'):
                s = (f.summary or "")[:300]
            else:
                continue
            if s:
                facts_lines.append(f"  fact_{idx}: {s}")

        if not facts_lines:
            return None

        facts_str = "\n".join(facts_lines)

        # 選択肢をシャッフルして position bias を除去
        choice_items = list(choices.items())
        random.shuffle(choice_items)
        # シャッフル後の位置マッピングを保持
        shuffle_map = {item[0]: item[0] for item in choice_items}  # label→label
        choices_str = "\n".join(f"  {k}: {v}" for k, v in choice_items)

        prompt = f"""Classify the relationship between each choice and the known facts.

{context_str}

Known facts:
{facts_str}

Choices:
{choices_str}

Rules:
- For each choice, check if the facts DIRECTLY support or contradict it.
- "supports" = facts provide clear evidence that this choice is correct.
- "contradicts" = facts provide clear evidence that this choice is wrong.
- "unknown" = facts don't clearly support or contradict.
- Use ONLY the facts above. Do NOT use external knowledge.
- Be strict: only mark "supports" if there is SPECIFIC, DIRECT evidence in the facts.
- If unsure, use "unknown".

Reply with ONLY this JSON (no other text):
{{
  "choice_analysis": [
    {{"choice": "A", "relation": "supports|contradicts|unknown", "reasoning": "brief reason", "fact_ids": ["fact_0"]}},
    {{"choice": "B", "relation": "supports|contradicts|unknown", "reasoning": "brief reason", "fact_ids": []}},
    ...
  ]
}}"""

        from config import VLLM_BASE_URL, VLLM_MODEL
        api_url = VLLM_BASE_URL + "/chat/completions"
        model = VLLM_MODEL

        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 400,
        }).encode()

        req = urllib.request.Request(
            api_url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            content = data["choices"][0]["message"]["content"]

            # JSON 抽出
            parsed = _parse_llm_response(content)
            if parsed:
                parsed["_raw"] = content[:300]
            return parsed

    except Exception as e:
        log.debug(f"LLM classify error: {e}")
        return None


def _parse_llm_response(content: str) -> Optional[dict]:
    """LLM応答からJSON構造を抽出"""
    # まず直接JSONパース
    try:
        data = json.loads(content)
        return _normalize_llm_data(data)
    except json.JSONDecodeError:
        pass

    # コードブロック内のJSON
    m = re.search(r'```(?:json)?\s*(\{.+?\})\s*```', content, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return _normalize_llm_data(data)
        except json.JSONDecodeError:
            pass

    # { ... } を探す
    m = re.search(r'\{[^{}]*"choice_analysis"[^{}]*\[.*?\].*?\}', content, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            return _normalize_llm_data(data)
        except json.JSONDecodeError:
            pass

    # 最後の手段: choice_analysis配列を探す
    m = re.search(r'"choice_analysis"\s*:\s*\[(.+?)\]', content, re.DOTALL)
    if m:
        try:
            arr = json.loads(f"[{m.group(1)}]")
            return _normalize_llm_data({"choice_analysis": arr})
        except json.JSONDecodeError:
            pass

    return None


def _normalize_llm_data(data: dict) -> dict:
    """LLMデータを正規化して label → {relation, fact_ids} のdictに変換"""
    result = {}
    analysis = data.get("choice_analysis", [])
    for item in analysis:
        label = item.get("choice", "").strip().upper()
        relation = item.get("relation", "unknown").strip().lower()
        if relation not in ("supports", "contradicts", "unknown"):
            relation = "unknown"
        fact_ids = item.get("fact_ids", [])
        if label:
            result[label] = {"relation": relation, "fact_ids": fact_ids}

    # decision/survivors は廃止（position bias の原因）
    result["decision"] = ""
    result["survivors"] = []
    return result


# ── パイプライン統合用のラッパー ──

def score_choices_v2(
    ir_dict: dict,
    choices: Dict[str, str],
    facts: List[dict],
    use_llm: bool = True,
) -> Optional[Tuple[str, float, str]]:
    """
    pipeline_enhanced.py から呼ばれるインターフェース。

    Returns:
        (label, confidence, method) or None
    """
    result = match_choices_v2(ir_dict, choices, facts)

    # デバッグログ
    log.info(
        f"KM_v2: facts={result.facts_count} choices={result.choice_count} "
        f"answer={result.answer} conf={result.confidence:.2f} "
        f"survivors={result.survivors} reject={result.reject_reason} "
        f"method={result.method}"
    )

    if result.answer:
        return result.answer, result.confidence, result.method
    return None
