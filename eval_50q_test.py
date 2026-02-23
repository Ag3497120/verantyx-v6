"""50問 A/Bテスト — Atom classifier v2 (緩和後)"""
import json, os, sys, types, time
os.environ['DISABLE_PATTERN_DETECTORS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))

import knowledge.knowledge_pipeline as kp_mod
from knowledge.knowledge_pipeline_v2 import KnowledgePipelineV2
kp_mod.KnowledgePipeline = KnowledgePipelineV2

from pipeline_enhanced import VerantyxV6Enhanced
from core.answer_matcher import flexible_match
from decomposer.concept_extractor_v2 import extract_concepts_v2

pipeline = VerantyxV6Enhanced(piece_db_path='pieces/piece_db_empty.jsonl', use_llm_decomposer=False)

questions = []
with open('hle_2500_eval.jsonl') as f:
    for line in f:
        questions.append(json.loads(line))

correct = 0; km_fires = 0; wiki_hits = 0; direct_fires = 0; xdec = 0
km_weak = 0; no_answer = 0
N = 50; t0 = time.time()

for i in range(N):
    q = questions[i]
    text = q['question']; expected = q['answer']
    try:
        extracted = extract_concepts_v2(text)
        high_conf = [ec for ec in extracted if ec.confidence >= 0.5]
        kp = pipeline._knowledge_pipeline
        if kp and high_conf:
            def _wrapped(self_kp, ir, extra_concepts=None, _hc=high_conf):
                merged = list(_hc)
                if extra_concepts: merged.extend(extra_concepts)
                return KnowledgePipelineV2.run(self_kp, ir, extra_concepts=merged)
            kp.run = types.MethodType(_wrapped, kp)
        result = pipeline.solve(text, expected_answer=expected)
        if kp and high_conf:
            kp.run = types.MethodType(KnowledgePipelineV2.run, kp)
        pred = result.get('answer')
        method = result.get('method', '')
        trace = result.get('trace', [])
        is_correct = flexible_match(pred, expected) if pred and expected else False
        if is_correct: correct += 1
        if pred is None: no_answer += 1
        if 'km_v2' in method:
            km_fires += 1
            if 'weak' in method: km_weak += 1
        if 'mcq_direct' in method: direct_fires += 1
        if 'cross_decompose' in method or 'exact_from_atoms' in method: xdec += 1
        if any('knowledge:accepted' in t for t in trace): wiki_hits += 1
        mark = "Y" if is_correct else "N"
        elapsed = time.time() - t0; spq = elapsed / (i+1)
        print(f"[{i+1}/{N}] {mark} {spq:.1f}s/q pred={pred} gold={str(expected)[:25]} m={method[:55]}", flush=True)
    except Exception as e:
        print(f"[{i+1}/{N}] ERR: {e}", flush=True)

elapsed = time.time() - t0
print(f"\n{'='*55}")
print(f"RESULT: {correct}/{N} ({correct/N*100:.1f}%)")
print(f"no_answer={no_answer}/{N}  wiki_hits={wiki_hits}")
print(f"km_v2={km_fires}(weak={km_weak})  direct={direct_fires}  xdec={xdec}")
print(f"Time: {elapsed:.0f}s ({elapsed/N:.1f}s/q)")
