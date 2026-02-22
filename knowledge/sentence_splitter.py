"""
sentence_splitter.py — Split complex English sentences into simple clauses for atomization.

LLM不使用。ルールベースで以下を処理:
1. 関係代名詞節 (which/that/who/whom/whose/where/when) の分離
2. 分詞構文 (V-ing, V-ed) の分離
3. 同格節 (, also known as ...) の分離
4. 接続詞 (and/but/or/while/although) での分割
5. セミコロン/コロンでの分割
6. 括弧内の補足情報の抽出
"""

import re
from typing import List


def split_to_clauses(sentence: str) -> List[str]:
    """Split a complex sentence into simple clauses suitable for atomization."""
    sentence = sentence.strip()
    if not sentence:
        return []

    clauses = []

    # Step 0: Remove citation markers like [1], [2], etc.
    sentence = re.sub(r'\[\d+\]', '', sentence)

    # Step 1: Split on semicolons
    parts = re.split(r';\s+', sentence)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Step 2: Extract parenthetical info
        parens = re.findall(r'\(([^)]{5,})\)', part)
        clean = re.sub(r'\s*\([^)]*\)\s*', ' ', part)
        clean = re.sub(r'\s+', ' ', clean).strip()

        # Step 3: Split on coordinating conjunctions at clause level
        # Only split "X, and Y" or "X, but Y" (comma + conjunction)
        sub_parts = re.split(r',\s+(?:and|but|while|whereas|although|however)\s+', clean)

        for sp in sub_parts:
            sp = sp.strip().rstrip('.')
            if not sp or len(sp) < 10:
                continue

            # Step 4: Extract relative clauses (which/that/who)
            # "X, which is Y" → "X" + "X is Y"
            rel_match = re.search(
                r'^(.+?),?\s+which\s+(?:is|was|are|were|has|had|have)\s+(.+?)$', sp, re.I)
            if rel_match:
                main_subj = rel_match.group(1).strip()
                rel_pred = rel_match.group(2).strip()
                clauses.append(main_subj)
                clauses.append(f"{_get_head_np(main_subj)} is {rel_pred}")
                continue

            rel_match2 = re.search(
                r'^(.+?),?\s+which\s+(\w+(?:s|ed|es)?)\s+(.+?)$', sp, re.I)
            if rel_match2:
                main_subj = rel_match2.group(1).strip()
                verb = rel_match2.group(2)
                obj = rel_match2.group(3).strip()
                clauses.append(main_subj)
                clauses.append(f"{_get_head_np(main_subj)} {verb} {obj}")
                continue

            # "X who/that VERBed Y" → "X" + "X VERBed Y"
            rel_match3 = re.search(
                r'^(.+?),?\s+(?:who|that)\s+(\w+(?:s|ed|es)?)\s+(.+?)$', sp, re.I)
            if rel_match3:
                main_subj = rel_match3.group(1).strip()
                verb = rel_match3.group(2)
                obj = rel_match3.group(3).strip()
                clauses.append(main_subj)
                clauses.append(f"{_get_head_np(main_subj)} {verb} {obj}")
                continue

            # "X where Y" → "X" + "Y is in X" (location)
            rel_match4 = re.search(
                r'^(.+?),?\s+where\s+(.+?)$', sp, re.I)
            if rel_match4:
                clauses.append(rel_match4.group(1).strip())
                # Don't try to restructure "where" clauses, just keep original
                clauses.append(sp)
                continue

            # Step 5: Handle participial phrases
            # "X, VERBing Y" → "X" + "X VERBs Y"
            part_match = re.search(
                r'^(.+?),\s+(\w+ing)\s+(.+?)$', sp, re.I)
            if part_match and len(part_match.group(1)) > 10:
                main = part_match.group(1).strip()
                verb_ing = part_match.group(2)
                obj = part_match.group(3).strip()
                clauses.append(main)
                # Convert "making" → "makes"
                verb_base = verb_ing[:-3] if verb_ing.endswith('ing') else verb_ing
                clauses.append(f"{_get_head_np(main)} {verb_base}s {obj}")
                continue

            # No splitting needed
            clauses.append(sp)

        # Add parenthetical info as separate clauses
        for paren in parens:
            paren = paren.strip()
            if len(paren) > 10:
                # Try to connect with the subject
                head = _get_head_np(clean)
                if head and not paren.startswith(('born', 'died', 'also', 'lit')):
                    clauses.append(f"{head} is {paren}")
                else:
                    clauses.append(paren)

    # Dedup while preserving order
    seen = set()
    unique = []
    for c in clauses:
        c = c.strip().rstrip('.')
        if c and len(c) >= 10 and c.lower() not in seen:
            seen.add(c.lower())
            unique.append(c)

    return unique if unique else [sentence]


def _get_head_np(text: str) -> str:
    """Extract the head noun phrase (first significant NP) from text."""
    text = text.strip()
    # Remove leading articles
    text = re.sub(r'^(?:The|A|An)\s+', '', text)
    # Take first N words that look like a noun phrase
    words = text.split()
    # Find the head: take words until we hit a verb or preposition
    stop_words = {'is', 'was', 'are', 'were', 'has', 'had', 'have',
                  'does', 'did', 'can', 'could', 'will', 'would',
                  'in', 'on', 'at', 'of', 'for', 'from', 'with', 'by',
                  'that', 'which', 'who', 'whom', 'where', 'when'}
    head_words = []
    for w in words[:8]:
        if w.lower() in stop_words and len(head_words) >= 1:
            break
        head_words.append(w)

    return ' '.join(head_words) if head_words else text[:30]
