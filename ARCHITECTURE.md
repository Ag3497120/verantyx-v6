# Verantyx V6 — アーキテクチャ & 設計ドキュメント

> **本ドキュメントについて**: このファイルは、Verantyx V6の設計思想・構造・ロジック・変遷を包括的に記述したものです。初めてこのプロジェクトに触れる方（教員・共同開発者）がプロジェクト全体を理解し、開発に参加できることを目的としています。

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [設計思想（コア哲学）](#2-設計思想コア哲学)
3. [禁止事項（重要）](#3-禁止事項重要)
4. [ファイル階層と構造](#4-ファイル階層と構造)
5. [パイプライン全体フロー](#5-パイプライン全体フロー)
6. [各モジュール詳細](#6-各モジュール詳細)
7. [スコアの変遷と歴史](#7-スコアの変遷と歴史)
8. [技術的知見とレッスン](#8-技術的知見とレッスン)
9. [現在の課題と今後の方向性](#9-現在の課題と今後の方向性)
10. [開発環境・実行方法](#10-開発環境実行方法)

---

## 1. プロジェクト概要

**Verantyx V6** は、[Humanity's Last Exam (HLE)](https://lastexam.ai/) というPhD級ベンチマーク（3,000問、全学術分野）に対して、**LLMを問題解決に使わずに**構造的推論で回答するシステムです。

### なぜ「LLMを使わない」のか

一般的なHLE提出物は、GPT-4oやClaudeに問題文をそのまま投げて答えさせます。これは「モデルの暗記力テスト」であり、**推論能力の実証にはなりません**。

Verantyxは逆のアプローチを取ります：
- 問題文から**構造化された中間表現（IR）**を抽出する
- Wikipediaから**知識ファクト**を取得する
- ファクトを**原子命題（Atom）**に分解する
- **形式検証（CEGIS/SymPy/Z3）**で答えを導出する
- 答えが証明できない場合は**INCONCLUSIVE**を返す（推測しない）

### 現在のスコア

| バージョン | スコア | 日付 | 手法 |
|---|---|---|---|
| Bias-Free v6 | 3.80% (95/2500) | 2026-02-20 | 純粋な構造的推論 |
| No-cheat v2 | 12.5% (5/40)* | 2026-02-22 | + Atom matching + MCQ cross-decompose |
| Atom v3 (進行中) | ~5.6% (20/360) | 2026-02-23 | + LLM→Atom classifier置換 |

*40問サンプル評価

---

## 2. 設計思想（コア哲学）

### 2.1 「立体十字構造」— Cross Structure

Verantyxの中核概念は**立体十字構造（Cross Structure）**です。

```
        [検証]
          |
[分解] ─── 問題 ─── [知識]
          |
        [合成]
```

問題を4方向から攻める：
- **分解（Decompose）**: 問題文 → IR（中間表現）
- **知識（Knowledge）**: Wikipedia → FactAtom → 証拠収集
- **検証（Verify）**: CEGIS / SymPy / Z3 による形式証明
- **合成（Compose）**: 証拠を組み合わせて答えを構築

### 2.2 「鉄の壁」— LLM隔離原則

LLMの使用には3段階の制限（「鉄の壁」）があります：

| レベル | 制限 | 説明 |
|---|---|---|
| レベル1（最厳格） | 問題文をLLMに渡さない | 問題文はLLMに一切見せない |
| レベル2（現行） | IR + 選択肢 + facts のみ | LLMは構造化データのみ受け取る |
| レベル3（禁止） | LLMに直接答えを選ばせない | LLMは分類・変換のみ |

**現在の運用**: レベル2。LLM（Qwen 2.5 7B、ローカル）は以下の用途のみ：
- MCQ選択肢の最終tiebreak（`mcq_direct_solver.py`）
- 将来的にはこれもAtom-basedに置換予定

### 2.3 「INCONCLUSIVE は正しい答え」

Verantyxは**答えが分からないとき、推測しません**。

```python
# ❌ 他のシステム
if no_answer:
    return random.choice(["A", "B", "C", "D"])  # 25%で当たる

# ✅ Verantyx
if no_answer:
    return None  # INCONCLUSIVE — 推測より誠実
```

これにより「スコアは低いが、出した答えは信頼できる」というシステムになります。

### 2.4 監査可能性（Audit Trail）

すべての答えには**なぜその答えに至ったか**の完全な記録が付きます：
- どのWikipedia記事を参照したか
- どのAtomがどの選択肢を支持/矛盾したか
- どの検証器が何を証明したか

---

## 3. 禁止事項（重要）

以下の手法は**明確に禁止**されています。過去にスコアを膨らませましたが、推論の実証にはならないため排除しました。

### 3.1 カンニング（Pattern Detection）の禁止

```python
# ❌ 禁止: 問題固有のハードコード答え
if "trefoil knot" in problem_text:
    return "A"  # 特定の問題の答えを暗記
```

環境変数 `DISABLE_PATTERN_DETECTORS=1` で無効化。`puzzle/general_detectors.py` と `puzzle/math_cross_sim.py` 内の問題固有ディテクターが該当。

### 3.2 ポジションバイアスの禁止

```python
# ❌ 禁止: 選択肢の位置に基づく推測
position_prior = {"B": 0.28, "D": 0.25, "C": 0.24, "A": 0.23}
```

HLEの訓練データでは選択肢Bが統計的に多いが、これを利用するのは推論ではありません。

### 3.3 デフォルト回答の禁止

```python
# ❌ 禁止: 答えが分からないとき固定値を返す
if answer is None:
    return "0"  # or "B" or any default
```

### 3.4 これらがなぜ重要か

Phase 5K（2026-02-20）では上記の「トリック」を使って**15.68%**を達成しました。しかしこれらを全て除去すると**3.80%**に下がりました。差分の11.88%は「推論」ではなく「統計的ハック」でした。

---

## 4. ファイル階層と構造

```
verantyx_v6/
├── pipeline_enhanced.py          ★ メインパイプライン（全8+ステージ統合）
├── config.py                     設定（Ollama URL, モデル名等）
├── README.md                     公開用README
├── ARCHITECTURE.md               本ドキュメント
│
├── core/                         ═══ コアデータ構造 ═══
│   ├── ir.py                     中間表現（IR）定義: Domain, Task, Entity, Constraint
│   └── answer_matcher.py         正解判定: LaTeX正規化, 分数, 科学的記数法の柔軟マッチング
│
├── decomposer/                   ═══ 問題分解層 ═══
│   ├── decomposer.py             ルールベースIR抽出（問題文→IR変換）
│   ├── concept_extractor_v2.py   概念抽出（問題文→高信頼度キーワード）
│   ├── problem_type_detector.py  問題タイプ検出（MCQ/exactMatch/proof等）
│   ├── knowledge_need_extractor.py  不足知識検出
│   └── latex_normalizer.py       LaTeX正規化
│
├── knowledge/                    ═══ 知識取得・構造化層 ═══ ★最重要
│   ├── knowledge_pipeline_v2.py  知識パイプラインv2: Gap検出→Wikipedia取得→構造化
│   ├── knowledge_gap_detector.py Gap検出: IRから不足知識を特定
│   ├── wiki_knowledge_fetcher_v2.py  Wikipedia取得（セクション単位）
│   ├── fact_atomizer.py          ★ FactAtom変換: 200+正規表現で英文→(S,P,O)トリプル
│   ├── sentence_splitter.py      節分割: 複文→単文に分割してAtom化精度向上
│   ├── exact_answer_assembler.py Atom→回答組み立て: query_type×predicate×answer_shape
│   ├── knowledge_crystallizer.py 結晶化: facts→FactAtom+RelationPiece+CrossPiece
│   ├── crystal_to_cross.py       Cross構造検証: Atom間の整合性チェック
│   ├── concept_search.py         600B SVD概念検索（DeepSeek V3 Expert方向ベクトル）
│   ├── concept_boost.py          概念ブースト（SVDスコアによるドメイン判定）
│   ├── expert_loader.py          GGUFリーダー（DeepSeek V3 Q8_0シャード読み込み）
│   ├── reasoning_type_classifier.py  推論型分類（verify/compute/elimination）
│   ├── knowledge_pipeline.py     知識パイプラインv1（旧版、v2に置換済み）
│   ├── llm_knowledge_fetcher.py  LLM知識取得（現在無効化）
│   └── knowledge_sanitizer.py    知識サニタイザー
│
├── executors/                    ═══ 実行器層 ═══
│   ├── atom_relation_classifier.py  ★ Atom関係分類: facts×choices→supports/contradicts/unknown
│   ├── mcq_knowledge_matcher_v2.py  ★ MCQ知識マッチングv2: Atom分類→ルールベース判定
│   ├── mcq_cross_decompose_solver.py  MCQ選択肢分解: 各選択肢→Wikipedia→cross-match
│   ├── mcq_direct_solver.py      MCQ直接回答: Qwen 7B（鉄の壁レベル2、最終fallback）
│   ├── mcq_elimination_solver.py MCQ消去法
│   ├── mcq_knowledge_matcher.py  MCQ知識マッチングv1（旧版）
│   ├── mcq_verifier.py           MCQ計算検証
│   ├── mcq_reasoning_executor.py 推論型ルーティング（600B分類→Executor委譲）
│   ├── multiple_choice.py        MCQ検出・選択肢分割
│   ├── sympy_solver.py           SymPy記号計算
│   ├── sympy_latex_executor.py   LaTeX→SymPy変換・計算
│   ├── equation_solver.py        方程式ソルバー
│   ├── algebra.py                代数Executor
│   ├── calculus.py               微積分Executor
│   ├── combinatorics.py          組み合わせ論Executor
│   ├── number_theory.py          数論Executor
│   ├── geometry.py               幾何Executor
│   ├── graph_theory.py           グラフ理論Executor
│   ├── linear_algebra.py         線形代数Executor
│   ├── logic.py                  論理学Executor
│   ├── probability.py            確率Executor
│   ├── statistics.py             統計Executor
│   ├── chess.py / chess_stockfish.py  チェスExecutor
│   └── string_operations.py      文字列操作Executor
│
├── cegis/                        ═══ CEGIS（反例誘導帰納合成）═══
│   ├── cegis_loop.py             CEGISメインループ（2000msタイムアウト）
│   ├── certificate.py            証明証書生成
│   ├── worldgen.py               世界生成（検証用テストケース）
│   └── worldgen_registry.py      世界生成レジストリ
│
├── verifiers/                    ═══ 検証器層 ═══
│   ├── sympy_verifier.py         SymPy数学検証
│   ├── z3_verifier.py            Z3 SMTソルバー検証
│   ├── enum_verifier.py          列挙検証
│   ├── knowledge_verifier.py     知識検証
│   └── api.py                    検証器API
│
├── puzzle/                       ═══ パズル推論エンジン ═══
│   ├── cross_simulation.py       有限モデルシミュレーション
│   ├── cross_simulator.py        Cross構造シミュレータ
│   ├── puzzle_reasoning_engine.py パズル推論エンジン
│   ├── math_cross_sim.py         数学Cross Simulator（計算ベースMCQソルバー群）
│   ├── hle_boost_engine.py       HLEブーストエンジン（パターンディテクター、現在無効化）
│   ├── general_detectors.py      一般ディテクター（問題固有、現在無効化）
│   ├── math_theorem_db.py        数学定理DB（43定理）
│   └── verantyx_pipeline_v2.py   パズルパイプラインv2
│
├── assembler/                    ═══ 組み立て層 ═══
│   ├── beam_search.py            ビームサーチ（ピース組み合わせ探索）
│   └── executor.py               ピース実行
│
├── grammar/                      ═══ 文法層 ═══
│   ├── composer.py               文法合成
│   └── glue_templates.py         接着テンプレート
│
├── audit/                        ═══ 監査層 ═══
│   └── audit_bundle.py           監査バンドル（推論過程の記録）
│
├── gates/                        ═══ ゲート層 ═══
│   ├── oracle_filter.py          オラクルフィルター
│   └── proposal_gate.py          提案ゲート
│
├── pieces/                       ═══ ピースDB ═══
│   └── piece.py                  ピースデータ構造（Piece, PieceInput, PieceOutput等）
│
├── llm/                          ═══ LLM連携（補助用途のみ）═══
│   ├── ollama_decomposer.py      Ollama分解器（現在無効化）
│   ├── llm_decomposer.py         LLM分解器
│   └── contract.py               LLM契約（入出力定義）
│
├── proposal/                     ═══ 提案生成 ═══
│   ├── claude_proposal.py        Claude提案生成（Path B）
│   └── safe_naturalizer.py       安全な自然言語化
│
├── arc/                          ═══ ARC（Abstract Reasoning Corpus）═══
│   ├── arc_cegis.py              ARC用CEGIS
│   ├── grid_ir.py                グリッドIR
│   └── transforms.py             変換
│
├── eval_2500_v2.py               ★ 2500問評価スクリプト（直列版）
├── eval_2500_parallel.py         ★ 2500問評価スクリプト（並列版、4ワーカー）
├── eval_50q_test.py              50問テストスクリプト
├── quick_eval_hle.py             クイック評価（旧版）
│
└── hle_2500_eval.jsonl           HLE 2500問データセット
```

---

## 5. パイプライン全体フロー

`pipeline_enhanced.py` の `solve()` メソッドが全体を制御します。

```
問題文(text)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Step 1: Decompose（分解）                        │
│   decomposer.py → IR(domain, task, entities,    │
│                      constraints, missing)       │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│ Step 1.2: Expert Routing（600B SVD）             │
│   concept_search.py → domain boost signal       │
│   expert_loader.py → DeepSeek V3 Expert分析     │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│ Step 1.3: Knowledge Acquisition（知識取得）       │
│   knowledge_pipeline_v2.py:                      │
│     1. Gap Detector → 不足知識を特定             │
│     2. concept_extractor_v2 → 追加概念           │
│     3. wiki_fetcher_v2 → Wikipedia取得           │
│   → _knowledge_facts[] に格納                    │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│ Step 1.4: Crystallization（結晶化）              │
│   knowledge_crystallizer.py:                     │
│     facts → FactAtom(S,P,O) + RelationPiece     │
│   exact_answer_assembler.py:                     │
│     query_type × atom → 直接回答（non-MCQ）      │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│ Step 1.5: MCQ Parallel Solvers（MCQ並列解法）    │
│                                                  │
│   1.5.8: cross_decompose                         │
│     各選択肢を個別分解→Wikipedia→cross-match     │
│                                                  │
│   1.5.9: km_v2 (Atom-based)                      │
│     atom_relation_classifier →                   │
│     supports/contradicts/unknown →               │
│     ルールベース判定                              │
│                                                  │
│   1.5.9.5: mcq_direct (Qwen 7B)                 │
│     IR + choices + facts → LLM回答               │
│     （鉄の壁レベル2: 問題文は渡さない）           │
│                                                  │
│   → 全候補からconfidence最高を採用                │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│ Step 2+: CEGIS / SymPy / Grammar Search          │
│   cegis_loop.py → 形式証明                       │
│   sympy_solver.py → 記号計算                     │
│   grammar/composer.py → 文法合成                 │
└──────────────────────┬──────────────────────────┘
                       ▼
              答え or INCONCLUSIVE
```

### MCQ並列候補選択の仕組み（2026-02-23実装）

```python
_mcq_candidates = []  # [(answer, confidence, method)]

# 1. cross_decompose（ルールベース、Wikipedia独自検索）
xd_result = solve_by_cross_decomposition(stem, choices, facts, ir)
if xd_result:
    _mcq_candidates.append(xd_result)

# 2. km_v2（Atom-based、LLM不使用、3ms）
km_result = score_choices_v2(ir, choices, facts)
if km_result:
    _mcq_candidates.append(km_result)

# 3. mcq_direct（Qwen 7B、最終fallback）
direct_result = solve_mcq_directly(ir, choices, facts)
if direct_result:
    _mcq_candidates.append(direct_result)

# 最高confidence候補を採用
best = max(_mcq_candidates, key=lambda x: x[1])
```

---

## 6. 各モジュール詳細

### 6.1 Decomposer（分解器）

**ファイル**: `decomposer/decomposer.py`

問題文を**中間表現（IR）**に変換します。IRは以下の要素を持ちます：

```python
@dataclass
class IR:
    domain: Domain          # 数学, 物理, 生物, CS, ...
    task: str              # "solve", "prove", "choose", ...
    entities: List[dict]   # 登場する概念・変数
    constraints: List[dict] # 制約条件
    missing: List[dict]    # 不足している知識
    query: dict            # 問いの構造
    metadata: dict         # キーワード等
```

**重要**: Decomposerはルールベースです。LLMは使いません（`use_llm_decomposer=False`）。

### 6.2 FactAtomizer（事実原子化器）

**ファイル**: `knowledge/fact_atomizer.py`

英文を `(Subject, Predicate, Object)` トリプルに変換する**200以上の正規表現パターン**を持つ中核モジュール。

```python
# 入力
"DNA polymerase III is the primary enzyme responsible for DNA replication in E. coli."

# 出力
FactAtom(
    subject="DNA polymerase III",
    predicate="is_a",
    object="primary enzyme responsible for DNA replication in E. coli",
    confidence=0.35
)
```

**パターンカテゴリ（20分類）**:
- A. Identity/Definition: `X is a/the Y`, `X refers to Y`
- B. Location/Geography: `X is located in Y`, `X is the capital of Y`
- C. Authorship/Creation: `X was written by Y`, `X invented Y`
- D. Temporal/Historical: `X occurred in Y`, `X was signed in Y`
- E. Possession/Quantity: `X has N Y`, `X contains Y`
- F. Measurement/Physical: `X has a boiling point of Y`
- G. Classification/Taxonomy: `X belongs to Y`, `X is a type of Y`
- H. Comparison/Ordering: `X is larger than Y`
- I. Causation/Effect: `X causes Y`, `X leads to Y`
- J. Awards/Achievement: `X won the Y prize`
- K. Composition/Structure: `X consists of Y`
- L. Language/Naming: `X is also known as Y`
- M. Science/Formula: `X is described by Y equation`
- N. Function/Purpose: `X is used for Y`
- O. Relationship/Association: `X is associated with Y`
- P. Process/Method: `X is produced by Y`
- Q. Legal/Political: `X was enacted in Y`
- R. Medical/Biological: `X is a symptom of Y`
- S. Music/Art/Culture: `X is a genre of Y`
- T. Mathematical/Logical: `X is a special case of Y`

### 6.3 Atom Relation Classifier（原子関係分類器）

**ファイル**: `executors/atom_relation_classifier.py`

LLMの `_llm_classify_relations` を置換するために2026-02-23に作成。

**仕組み**:
1. Wikipedia factsをAtomに分解
2. MCQ選択肢をAtomに分解
3. Atom×Atomのcross-match:
   - Subject overlap ≥ 0.3 → 候補ペア
   - Predicate + Object一致 → `supports`
   - 同Subject・同Predicate・異Object → `contradicts`（対義語60+ペア）
   - Negation flip → `contradicts`
4. Keyword fallback（Atomマッチ不発時）

**速度**: 3ms/call（LLMの3-5秒の1000倍速）

### 6.4 MCQ Knowledge Matcher v2

**ファイル**: `executors/mcq_knowledge_matcher_v2.py`

3段階の判定：
1. **KM-1: Lexical Match**（語彙一致スコア）
2. **KM-2: Atom Classification**（`atom_relation_classifier`による関係分類）
3. **KM-3: Rule-based Decision**
   - `supports` 1つだけ + evidence あり → 採用
   - `contradicts` で1つだけ残存 → 採用
   - `supports_weak` 1つだけ → 低confidence採用
   - それ以外 → INCONCLUSIVE

### 6.5 Cross Decompose Solver

**ファイル**: `executors/mcq_cross_decompose_solver.py`

各MCQ選択肢を**個別に**分解・Wikipedia検索し、stem（問題本文）のfactsとcross-matchする。

```
選択肢A → concepts → Wikipedia → facts_A
選択肢B → concepts → Wikipedia → facts_B
...

stem_facts ∩ facts_A → overlap score A
stem_facts ∩ facts_B → overlap score B
...

最高スコアの選択肢を回答（gap閾値あり）
```

**動的gap閾値**（2026-02-23実装）:
- 4択以下: min_gap = 0.03
- 5-6択: min_gap = 0.045
- 7択以上: min_gap = 0.06（ランダムノイズ排除）

### 6.6 CEGIS（反例誘導帰納合成）

**ファイル**: `cegis/cegis_loop.py`

1. **仮説生成**: 候補答えを生成
2. **検証**: SymPy/Z3で仮説を検証
3. **反例生成**: 検証失敗時に反例を作成
4. **修正**: 反例を満たす新仮説を生成
5. 2000msタイムアウトまでループ

Bias-free版の95正解中69問がCEGIS由来。

### 6.7 600B SVD Knowledge（DeepSeek V3 Expert分析）

**ファイル群**: `knowledge/concept_search.py`, `knowledge/expert_loader.py`

DeepSeek V3 MoEモデル（671Bパラメータ）の**重みを静的解析**して知識を抽出：

- 15,104 Expert × 4 SVD方向 × 7168次元 = `concept_dirs.npy` (15104, 4, 7168)
- 問題文→BPEトークン化→埋め込み平均→cosine類似度→Top-50 Expert
- Expert多数決→ドメイン推定→適切なExecutor選択

**重要**: LLM推論（forward pass）は一切行いません。重みの静的解析のみ。

---

## 7. スコアの変遷と歴史

### Phase 1-4: 基盤構築（〜2026-02-14）

- **Pipeline v1**: 基本的なDecomposer + CEGIS
- **Piece DB**: 108個のピース（数学定理、論理規則等）
- **初期スコア**: ~1-2%

### Phase 5A-5D: CEGIS最適化（2026-02-15〜16）

- CEGIS証明の精度向上
- **3.50% → 5.36%**

### Phase 5E-5H: 正解判定改善（2026-02-16〜17）

- LaTeX正規化、分数マッチング
- specificity bias修正（0.3→0.05）
- **5.36% → 6.84%** → HuggingFace `kofdai/verantyx-hle-5`

### Phase 5I: 600B SVD統合（2026-02-18）

- Thunder Compute H100で実行
- concept_dirs (15104, 4, 7168) 生成
- **6.84% → 8.56%** → HuggingFace `kofdai/verantyx-hle-8`

### Phase 5J-5K: ディテクター追加（2026-02-19〜20）

- general_detectors 6種、math_cross_sim計算ソルバー
- **8.56% → 15.68%**（ただしバイアス込み）

### Bias-Free リセット（2026-02-20）

**転換点**: すべての統計的バイアスを除去
- position prior 除去
- general_detectors 無効化
- デフォルト回答 除去
- **15.68% → 3.80%**（真のベースライン）

### No-cheat v1-v2（2026-02-22）

- cross_decompose + km_v2 + mcq_direct 実装
- LLM Decomposer無効化
- **1.0% → 12.5%**（40問サンプル）

### Atom v3（2026-02-23、現在）

- LLM relation classifierをAtom-basedに置換（1000倍速）
- NameErrorバグ修正（km_v2不発火→発火）
- MCQ並列候補選択
- **進行中: ~5.6%**（360問時点）

---

## 8. 技術的知見とレッスン

### 8.1 やってはいけないこと（学んだ教訓）

1. **LLMに答えを選ばせない**: LLMは文脈から「もっともらしい」答えを返すが、それは推論ではない
2. **position biasを利用しない**: B>D>C>Aの統計は一時的にスコアを上げるが、本質ではない
3. **`answer='0'` フォールバックしない**: HLEでは0が答えの問題が多いため、デフォルト0は不正行為
4. **CEGIS trivial passを許さない**: 空の検証で「証明された」とカウントしない

### 8.2 効果があったこと

1. **CEGIS + 形式検証**: 最も信頼性の高い手法（69/95正解）
2. **Wikipedia知識取得**: 94%の問題で関連知識を取得可能
3. **概念抽出v2**: 問題文からのキーワード抽出で知識検索精度向上
4. **Atom化**: facts→(S,P,O)変換で構造的マッチングが可能に
5. **MCQ並列候補**: 複数手法の結果を比較して最良を選択

### 8.3 効果がなかったこと

1. **600B rank_choices_by_transform**: Expert重み→SwiGLU→cosineでの選択肢ランキング（スコア差0.002未満）
2. **query_to_experts overlap**: Expert重複と正答は相関しない
3. **LLM tiebreak**: position biasで常にAを返す

---

## 9. 現在の課題と今後の方向性

### 9.1 現在の課題

| 課題 | 詳細 |
|---|---|
| no_answer 68% | 大半の問題で答えを出せない |
| integer答え 0% | 数値計算問題に全く対応できていない |
| LaTeX答え 0% | 数式答え問題に対応できていない |
| yes/no答え 0% | 真偽判定問題に対応できていない |
| km_v2精度 | HLEのPhD級問題ではAtomマッチが困難 |

### 9.2 今後の方向性

1. **SymPy Executor復活**: integer答え（22%のno_answer）に対応
2. **Yes/No判定器**: Atom支持/矛盾の集計でtrue/false判定
3. **Cross Simulator答え生成**: Cross構造から直接答えを生成
4. **Meta-Piece テンプレート**: 分野別の解法テンプレート
5. **Atom Classifier精度改善**: synonym辞書拡張、partial match強化

---

## 10. 開発環境・実行方法

### 10.1 環境

- **OS**: macOS (Apple Silicon M4)
- **Python**: 3.14+
- **LLM**: Qwen 2.5 7B via Ollama (localhost:11434)
- **依存**: sympy, z3-solver, wikipedia-api, requests

### 10.2 実行

```bash
# クローン
git clone https://github.com/Ag3497120/verantyx-v6.git
cd verantyx-v6

# 50問テスト（~7分）
DISABLE_PATTERN_DETECTORS=1 python3 eval_50q_test.py

# 2500問評価・直列（~5.5時間）
DISABLE_PATTERN_DETECTORS=1 python3 eval_2500_v2.py

# 2500問評価・並列（~1.5時間）
DISABLE_PATTERN_DETECTORS=1 EVAL_WORKERS=4 python3 eval_2500_parallel.py
```

### 10.3 環境変数

| 変数 | 説明 |
|---|---|
| `DISABLE_PATTERN_DETECTORS=1` | カンニング（問題固有ディテクター）を無効化 |
| `DISABLE_CONCEPT_BOOST=1` | 600B SVDブーストを無効化 |
| `EVAL_WORKERS=N` | 並列評価のワーカー数 |

### 10.4 重要なファイル

| ファイル | 説明 |
|---|---|
| `hle_2500_eval.jsonl` | HLE 2500問データ |
| `hle_2500_v2_checkpoint.json` | 直列評価チェックポイント |
| `hle_2500_parallel_checkpoint.json` | 並列評価チェックポイント |
| `pieces/piece_db_empty.jsonl` | 空のピースDB（no-cheat評価用） |

---

## 付録: 用語集

| 用語 | 説明 |
|---|---|
| **IR** | Intermediate Representation — 問題の構造化表現 |
| **FactAtom** | (Subject, Predicate, Object) の知識三つ組 |
| **Cross Structure** | 分解・知識・検証・合成の4方向構造 |
| **CEGIS** | Counterexample-Guided Inductive Synthesis |
| **Piece** | 推論の構成要素（定理、ルール等） |
| **INCONCLUSIVE** | 答えが証明できない → 推測せず返す |
| **鉄の壁** | LLMの使用制限レベル（1-3） |
| **600B SVD** | DeepSeek V3の重み静的解析による知識抽出 |
| **no_answer** | 答えを出せなかった問題（=INCONCLUSIVE） |
| **bias-free** | 統計的バイアスを全て除去した状態 |

---

*最終更新: 2026-02-23 | 作成: verantyx (OpenClaw AI Assistant)*
*プロジェクト: https://github.com/Ag3497120/verantyx-v6*
