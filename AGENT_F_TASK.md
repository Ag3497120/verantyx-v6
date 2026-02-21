# Agent F: Math ピース verify/worldgen 強化 + GGUF 知識抽出

## 目標
1. 既存 Math ピースに verify/worldgen spec を追加（CEGIS が使えるように）
2. 全15シャードが揃ったので知識抽出を試みる

## ワークスペース
`/Users/motonishikoudai/.openclaw/workspace/verantyx_v6/`

## Part 1: Math ピースに verify/worldgen を追加

### 現状確認
```bash
python3 -c "
import json
pieces = []
with open('pieces/piece_db.jsonl') as f:
    for line in f:
        pieces.append(json.loads(line))

# verify/worldgen がない Math ピースを探す
math_domains = ['arithmetic','algebra','number_theory','combinatorics','calculus','linear_algebra']
missing = [p for p in pieces if p.get('domain') in math_domains and not p.get('verify')]
print(f'verify なし Math ピース: {len(missing)}')
for p in missing[:10]:
    print(f'  {p[\"piece_id\"]}: {p.get(\"description\",\"\")}')
"
```

### verify spec を追加すべきピースの例

`nt_factorial`:
```json
{
  "verify": {
    "kind": "computation_log",
    "method": "sympy_factorial",
    "spec": {"func": "factorial", "param": "n"}
  },
  "worldgen": {
    "domain": "number",
    "params": {"lo": 0, "hi": 20}
  }
}
```

`arithmetic_power`:
```json
{
  "verify": {
    "kind": "substitution",
    "method": "eval",
    "spec": {"expr": "base**exponent"}
  },
  "worldgen": {
    "domain": "substitution",
    "params": {"vars": ["base", "exponent"]}
  }
}
```

`combinatorics_nCr`:
```json
{
  "verify": {
    "kind": "computation_log",
    "method": "sympy_binomial",
    "spec": {"func": "binomial", "params": ["n", "r"]}
  },
  "worldgen": {
    "domain": "number",
    "params": {"lo": 0, "hi": 30}
  }
}
```

### 実装方法
`pieces/piece_db.jsonl` を読み込み、各 Math ピースに verify/worldgen を追加してバックアップ付きで書き戻す:

```python
import json, copy

with open('pieces/piece_db.jsonl') as f:
    pieces = [json.loads(l) for l in f]

# バックアップ
with open('pieces/piece_db_pre_agent_f.jsonl.bak', 'w') as f:
    for p in pieces:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')

# Math ドメイン別の verify/worldgen テンプレート
VERIFY_TEMPLATES = {
    'nt_factorial': {
        'verify': {'kind': 'computation_log', 'method': 'sympy', 'expr': 'factorial(n)'},
        'worldgen': {'domain': 'number', 'params': {'lo': 0, 'hi': 15}}
    },
    'arithmetic_power': {
        'verify': {'kind': 'substitution', 'method': 'eval', 'expr': 'base**exponent'},
        'worldgen': {'domain': 'substitution', 'params': {'vars': ['base', 'exponent']}}
    },
    'nt_gcd_compute': {
        'verify': {'kind': 'computation_log', 'method': 'sympy', 'expr': 'gcd(a, b)'},
        'worldgen': {'domain': 'number', 'params': {'lo': 1, 'hi': 100}}
    },
    'combinatorics_permutation': {
        'verify': {'kind': 'computation_log', 'method': 'sympy', 'expr': 'factorial(n)/factorial(n-r)'},
        'worldgen': {'domain': 'number', 'params': {'lo': 1, 'hi': 15}}
    },
    'combinatorics_combination': {
        'verify': {'kind': 'computation_log', 'method': 'sympy', 'expr': 'binomial(n, r)'},
        'worldgen': {'domain': 'number', 'params': {'lo': 1, 'hi': 20}}
    },
}

# 更新
updated = 0
for p in pieces:
    pid = p.get('piece_id', '')
    if pid in VERIFY_TEMPLATES and not p.get('verify'):
        tmpl = VERIFY_TEMPLATES[pid]
        p['verify'] = tmpl['verify']
        p['worldgen'] = tmpl['worldgen']
        updated += 1

print(f'Updated: {updated} pieces')

with open('pieces/piece_db.jsonl', 'w') as f:
    for p in pieces:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')
```

すべての Math ドメインピースを確認して、verify のないものに適切な spec を追加する。

## Part 2: GGUF 知識抽出（全15シャード完了）

### GGUF ファイル確認
```bash
ls -lh ~/avh_math/avh_math/downloads/v3_q8_0/Q8_0/ | grep gguf
```

### ExpertLoader テスト
```python
import sys
sys.path.insert(0, '.')
from knowledge.expert_loader import ExpertLoader

# ExpertLoader の仕様を確認
import inspect
print(inspect.getsource(ExpertLoader.__init__))
```

### knowledge extraction の実装
`knowledge/mine_trace_from_shard.py` を作成:
```python
"""
全15シャードから Math 知識を抽出
Strategy: HLE question text → token embedding → expert routing → piece matching
"""
import sys, json, numpy as np, os
sys.path.insert(0, '/Users/motonishikoudai/.openclaw/workspace/verantyx_v6')

SHARD_DIR = os.path.expanduser('~/avh_math/avh_math/downloads/v3_q8_0/Q8_0/')
SVD_DIR = os.path.expanduser('~/avh_math/avh_math/db/moe_sparse_cross_600b_real/')

# concept_dirs: (15104, 4, 7168) — 全ExpertのSVD方向
# embed_tokens: (129280, 7168) — トークン埋め込み
try:
    concept_dirs = np.load(f'{SVD_DIR}/concept_dirs.npy')
    embed_tokens = np.load(f'{SVD_DIR}/embed_tokens.npy')
    print(f'concept_dirs: {concept_dirs.shape}')
    print(f'embed_tokens: {embed_tokens.shape}')
except Exception as e:
    print(f'SVD load failed: {e}')
    concept_dirs = None
    embed_tokens = None

def embed_text_simple(text: str, tokenizer_vocab=None) -> np.ndarray:
    """簡易テキスト埋め込み（文字n-gramベース）"""
    # BPE tokenizer がなければ文字単位でハッシュ
    vec = np.zeros(7168, dtype=np.float32)
    for i, ch in enumerate(text[:100]):
        idx = ord(ch) % 7168
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

def find_top_experts(text_vec: np.ndarray, concept_dirs: np.ndarray, top_k=5):
    """テキスト埋め込みと最も類似するエキスパートを探す"""
    # concept_dirs: (15104, 4, 7168) → 最初のSVD方向だけ使う
    dirs_0 = concept_dirs[:, 0, :]  # (15104, 7168)
    scores = dirs_0 @ text_vec  # (15104,)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]

# HLE問題からエキスパート分析
if concept_dirs is not None:
    test_questions = [
        "What is 5 factorial?",
        "Find all prime numbers p such that p^2 + 2 is also prime.",
        "How many ways can 4 people be arranged in a row?",
    ]
    for q in test_questions:
        vec = embed_text_simple(q)
        experts = find_top_experts(vec, concept_dirs)
        print(f'Q: {q[:50]}')
        print(f'  Top experts: {experts[:3]}')
```

## 完了条件
- Part 1: 最低10個の Math ピースに verify/worldgen 追加
- Part 2: mine_trace_from_shard.py 実行・概念実証
- 評価スコア確認

## 結果報告
`AGENT_F_RESULTS.md` に書いて:
```bash
openclaw system event --text "Agent F完了: Mathピース強化+知識抽出。結果: AGENT_F_RESULTS.md" --mode now
```
