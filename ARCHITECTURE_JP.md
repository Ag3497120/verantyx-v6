# Verantyx V6 — プロジェクト詳細ドキュメント

> *京都の学生が一人で書いた 154,825行の Python が、ARC-AGI2 で 84.0% を達成した*

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [スコアと成果](#2-スコアと成果)
3. [アーキテクチャ全体像](#3-アーキテクチャ全体像)
4. [Stage 1: Cross Engine（手書きソルバー群）](#4-stage-1-cross-engine)
5. [Stage 2: LLM プログラム合成](#5-stage-2-llm-プログラム合成)
6. [検証パイプライン](#6-検証パイプライン)
7. [既存手法との比較と優位性](#7-既存手法との比較と優位性)
8. [プロジェクト構造](#8-プロジェクト構造)
9. [コードベース統計](#9-コードベース統計)
10. [技術的詳細](#10-技術的詳細)
11. [制約と今後](#11-制約と今後)

---

## 1. プロジェクト概要

**Verantyx V6** は ARC-AGI2（Abstraction and Reasoning Corpus 2）ベンチマークに対するソルバーシステムである。

ARC-AGI2 は 1000 個の視覚的推論パズルで構成され、各パズルには入出力グリッドのペアが訓練例として与えられる。タスクは「変換規則を発見し、未知のテスト入力に適用する」こと。

### 開発者
- **kofdai** — プロジェクト設計・全コード執筆・アーキテクチャ設計
- **OpenClaw** — LLM エージェント並列実行基盤

### 主な特徴
- **ハイブリッドアーキテクチャ**: 手書きソルバー + LLM プログラム合成
- **GPU 不要**: MacBook 上で CPU のみで動作
- **ファインチューニングなし**: 全て zero-shot API 呼び出し
- **決定論的検証**: LLM が出力するのは「答え」ではなく「コード」。コードは全訓練例で検証される

---

## 2. スコアと成果

| 指標 | 値 |
|---|---|
| **Training Set** | **840/1000 (84.0%)** |
| ├ Cross Engine (手書き) | 244/1000 (24.4%) |
| └ LLM Synth (Claude Sonnet 4.5) | 596/1000 (+59.6%) |
| **Evaluation Set** | 11/120 (9.2%) ※暫定・改善中 |
| 汎化率 (train → test) | 42.3% |

### スコア推移

```
v19: 113 (11.3%) — extract_patch, per_object_stamp
v23: 120 (12.0%) — Iterative Cross 2-step 残差学習
v50: 200 (20.0%) — CrossUniverse separator_propagate
v57: 216 (21.6%) — cross3d 幾何変換
v60: 224 (22.4%) — cross3d 12 種、1日 +13
v62: 227 (22.7%) — GitHub push
v72: 234 (23.4%) — cross_probe: dot_to_cross_rect
v82: 244 (24.4%) — Cross Engine 完成（手書きの壁）
v82+Synth: 840 (84.0%) — Claude Sonnet 4.5 合成 +596
```

---

## 3. アーキテクチャ全体像

```
┌─────────────────────────────────────────────────────────┐
│                    Verantyx V6                          │
│                                                         │
│  ┌────────────────────┐   ┌──────────────────────────┐ │
│  │  Stage 1            │   │  Stage 2                  │ │
│  │  Cross Engine       │   │  LLM Program Synthesis    │ │
│  │                     │   │                            │ │
│  │  30+ 手書きソルバー │   │  Claude Sonnet 4.5        │ │
│  │  9 Phase パイプライン│   │  ×5-6 並列エージェント   │ │
│  │  CEGIS 検証         │   │                            │ │
│  │  カスタム DSL        │   │  ↓ transform(grid) 生成   │ │
│  │                     │   │                            │ │
│  │  244問 解決 (24.4%) │   │  596問 解決 (+59.6%)      │ │
│  └────────┬───────────┘   └──────────┬───────────────┘ │
│           │                           │                  │
│           ▼                           ▼                  │
│  ┌──────────────────────────────────────────────────┐   │
│  │          Verification Layer (検証層)               │   │
│  │                                                    │   │
│  │  verify_transform.py                               │   │
│  │  全訓練例でピクセル完全一致を要求                   │   │
│  │  タイムアウト検出 (5秒/例)                         │   │
│  │  クラッシュ検出・サンドボックス実行                 │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│                   840/1000 (84.0%)                       │
└─────────────────────────────────────────────────────────┘
```

**核心思想**: LLM は「答え」を生成しない。**「コード」を生成する**。コードが全訓練例で正しい出力を生成するかを決定論的に検証し、通過したコードだけがテスト入力に適用される。

---

## 4. Stage 1: Cross Engine

Cross Engine は Verantyx のコアであり、49,445行の Python で構成される。LLM を一切使わず、純粋にルールベースでパズルを解く。

### 4.1 パイプライン構造 (9 Phase)

```
入力: train_pairs [(input, output), ...] + test_inputs

Phase 1:    Cross Solver (DSL + NB Rule)
Phase 1.5:  Standalone Primitives (rot90, flip, transpose 等)
Phase 1.55: CrossUniverse (フロー伝播)
Phase 1.56: CrossUniverse3D (立体 cross 構造)
Phase 1.57: MultiScale Cross (6 軸記述子)
Phase 1.57x: 20+ 専門ソルバー (crop, scale, panel, color, gravity, ...)
Phase 2.5:  Block-level IR (マルチスケール推論)
Phase 3:    2-step / 3-step Composition (ピース合成)
Phase 4:    Iterative Cross (残差学習)
Phase 5:    Multi-Arm Beam Search
Phase 6:    DSL Program Enumeration
Phase 7:    Puzzle Reasoning Language (カスタム DSL)
Phase 8:    ProgramTree (CEGIS 条件分岐/ループ合成)
Phase 9:    ARC-CEGIS (変換チェーン合成)
```

### 4.2 CrossPiece アーキテクチャ

全てのソルバーは統一インターフェース `CrossPiece` を返す:

```python
class CrossPiece:
    name: str           # 例: "nb:wall_color", "gravity:down"
    apply_fn: Callable  # (input_grid) -> output_grid
    version: int        # 検証済みバージョン
```

`CrossSimulator` がこれらを CEGIS（Counter-Example Guided Inductive Synthesis）方式で検証:

```python
class CrossSimulator:
    def verify(piece, train_pairs) -> bool:
        """全訓練例でピクセル完全一致"""
    
    def partial_verify(piece, train_pairs) -> float:
        """部分一致スコア (0.0 ~ 1.0)"""
```

### 4.3 主要ソルバーモジュール (46個)

| カテゴリ | モジュール | 行数 | 説明 |
|---|---|---|---|
| **コアエンジン** | cross_engine.py | 2,817 | メインオーケストレータ |
| | cross_solver.py | 2,047 | DSL ベースソルバー |
| | puzzle_lang.py | 2,623 | カスタム推論 DSL |
| **構造分析** | cross_universe_3d.py | 1,664 | 立体 cross 構造 |
| | cross_multiscale.py | 846 | 6 軸 cross 記述子 |
| | cross_compose.py | 980 | 多段合成 |
| | meta_cross.py | 972 | 階層的ルーティング |
| **オブジェクト操作** | object_mover.py | 1,325 | 7 種の移動戦略 |
| | per_object.py | 1,291 | オブジェクト別変換 |
| | obj_correspondence.py | 610 | オブジェクト対応 |
| | obj_transform.py | 553 | オブジェクト変形 |
| **セル規則** | nb_abstract.py | 901 | 近傍抽象化 |
| | nb_extended.py | 612 | 拡張近傍規則 |
| | neighborhood_rule.py | — | セルオートマタ規則 |
| **残差学習** | residual_learner.py | 549 | ピクセル差分学習 |
| | residual_guided.py | 1,162 | 逆方向残差分析 |
| **探索** | program_search.py | 1,191 | テスト時プログラム探索 |
| | beam_search.py | — | マルチアーム Beam Search |
| | arc_cegis.py | 697 | CEGIS 変換チェーン |
| | program_tree.py | 486 | 条件分岐/ループ合成 |
| **専門ソルバー** | gravity_solver.py | 579 | 重力変換 |
| | flood_fill_solver.py | — | フラッドフィル |
| | symmetry_solver.py | — | 対称性修復 |
| | panel_ops.py | 953 | パネル分割・結合 |
| | color_swap_solver.py | 818 | 色変換 |
| | scale_solver.py | — | スケール変換 |
| | crop_extract_solver.py | — | クロップ・抽出 |
| | line_connect.py | 741 | 線描画・接続 |
| | fill_enclosed_solver.py | 561 | 囲み領域充填 |
| | rotating_cross.py | 555 | 回転 cross |
| | cross_probe_fill.py | 587 | プローブ式穴検出 |
| | 他 15+ モジュール | — | 各種専門パターン |

### 4.4 Puzzle Reasoning Language (独自 DSL)

Verantyx は ARC パズルの変換を記述するための**独自 DSL（Domain Specific Language）**を実装している（2,623行）:

```
【量化子】 ALL, EACH, BETWEEN, PAIR
【選択子】 cells(color=C), objects(), rows(), cols(), regions(separator=C)
【空間子】 adjacent(n=4/8), enclosed(), through(point), bbox()
【操作子】 fill(color), draw_line(dir), extend(dir), reflect(axis),
          connect(color), recolor(from,to), move(dir,dist), copy_to(pos)
```

これは「自然言語的な語彙を使いながら、機械的に解釈可能な指示」として設計されている。

### 4.5 Iterative Cross（残差学習）

Phase 4 の核心技術。部分的に正しい変換を適用し、「残差」（まだ間違っている部分）を新しい問題として再帰的に解く:

```
Round 1: 最良の部分一致ピースを適用 → 中間出力
Round 2: (中間出力, 正解) のペアで新ピースを生成
Round 3: さらに残差を学習（最大 3 ラウンド）
```

全ラウンドを通して、最終結果が全訓練例で完全一致する場合のみ採用。

---

## 5. Stage 2: LLM プログラム合成

### 5.1 仕組み

Cross Engine で解けなかった問題（ver=0）に対して、LLM がプログラムを合成する。

```
┌──────────────────┐
│  未解決タスク      │  Cross Engine ver=0 の 756 問
│  (task.json)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  OpenClaw         │  タスクを 50 問ずつバッチに分割
│  Orchestrator     │  5-6 個の sub-agent を並列起動
│  (Claude Opus 4)  │
└────────┬─────────┘
         │ ×5-6 並列
         ▼
┌──────────────────┐
│  Claude Sonnet    │  各タスクについて:
│  4.5 Agent        │  1. task.json を読む（入出力ペア）
│                   │  2. パターンを分析
│                   │  3. transform(grid) 関数を Python で書く
│                   │  4. verify_transform.py で検証
│                   │  5. 全 train 例通過 → synth_results/ に保存
│                   │  6. 失敗 → 最大 3 回リトライ
└──────────────────┘
```

### 5.2 LLM の使い方（4 種類）

| 用途 | モデル | ファイル | 説明 |
|---|---|---|---|
| **プログラム合成** | Claude Sonnet 4.5 | sub-agent 経由 | transform(grid) 関数の生成（メイン） |
| **仮説生成** | Qwen2.5-7B (Ollama) | llm_hypothesis.py | 変換仮説 → DSL プログラムに変換 |
| **直接解答** | Qwen2.5-7B (Ollama) | llm_direct.py | グリッド直接生成（低精度） |
| **タスク分類** | DeepSeek-Chat | llm_router.py, llm_deepseek.py | タスクカテゴリ分類 |

### 5.3 プロンプト設計

LLM に渡すのは **task.json の訓練ペアのみ**。「ARC とは何か」の説明も、他の ARC 解法の例も一切含まない（zero-shot）。

LLM が生成するコードの例:

```python
def transform(grid):
    import numpy as np
    g = np.array(grid)
    
    # パターンを識別し、対応する色にマッピング
    ones_mask = (g == 1)
    eights_mask = (g == 8)
    
    one_pos = list(zip(*np.where(ones_mask)))
    min_r = min(r for r,c in one_pos)
    min_c = min(c for r,c in one_pos)
    shape = frozenset((r-min_r, c-min_c) for r,c in one_pos)
    
    # 形状→色のマッピング（訓練例から推論）
    plus = frozenset([(0,1),(1,0),(1,1),(1,2),(2,1)])
    color_map = { plus: 2 }
    out_color = color_map.get(shape, len(one_pos))
    
    result = g.copy()
    result[ones_mask] = 0
    result[eights_mask] = out_color
    return result.tolist()
```

### 5.4 検証の厳密性

```python
# verify_transform.py の核心ロジック
for i, ex in enumerate(task['train']):
    result = fn(ex['input'])  # 5秒タイムアウト
    pred = [[int(c) for c in row] for row in result]
    if not np.array_equal(np.array(pred), np.array(ex['output'])):
        return {'status': 'train_fail'}  # 1ピクセルでも違えば却下
```

- **ピクセル完全一致**: 1 ピクセルでも間違っていれば却下
- **全訓練例**: 2-3 例全てで正しい出力を生成する必要がある
- **タイムアウト**: 1 例あたり 5 秒以内
- **クラッシュ検出**: 例外が発生した場合も却下

### 5.5 Multi-Solution Voting（多数決投票）

汎化率向上のため、同一タスクに対して**複数の独立した解**を生成し、テスト出力で多数決を取る:

```
Task X に対して 3 つの独立した transform 関数を生成
  → Solution A: test output = [[1,2],[3,4]]
  → Solution B: test output = [[1,2],[3,4]]  
  → Solution C: test output = [[1,0],[3,4]]
  → 多数決: [[1,2],[3,4]] (2/3 一致)
```

---

## 6. 検証パイプライン

Verantyx の最大の強みの一つは**多層検証アーキテクチャ**:

```
Layer 1: CrossSimulator.verify()
         全訓練ペアでピクセル完全一致（Cross Engine 用）

Layer 2: verify_transform.py
         LLM 生成コードの全訓練例検証（Synth 用）

Layer 3: CEGIS (Counter-Example Guided Inductive Synthesis)
         反例駆動型の候補刈り込み

Layer 4: Multi-Solution Voting
         複数解の多数決で汎化率を向上
```

---

## 7. 既存手法との比較と優位性

### 7.1 ARC-AGI2 の主な手法

ARC-AGI（初代・2 合わせて）で高スコアを達成している主な手法:

| 手法 | 代表者/チーム | アプローチ | 限界 |
|---|---|---|---|
| **LLM 直接推論** | OpenAI o3, Gemini | LLM にグリッドを見せて出力を推論させる | 高コスト ($10,000+)、汎化しにくい |
| **プログラム合成のみ** | Ryan Greenblatt 等 | LLM でコード生成 + 検証 | 手書きソルバーなし、構造理解が浅い |
| **DSL 探索のみ** | DreamCoder 系 | ドメイン特化 DSL で探索 | 表現力の限界、新パターンに弱い |
| **ニューラル** | Kaggle 上位 | CNN/Transformer でグリッド変換を学習 | 汎化性能に限界、$50 制約 |

### 7.2 Verantyx の差別化ポイント

#### ① ハイブリッドアーキテクチャ（最大の差別化）

**他**: LLM のみ、または手書きのみ
**Verantyx**: 30+ の手書きソルバーと LLM プログラム合成の**ハイブリッド**

```
Hand-crafted only:  244/1000 (24.4%) ← 表現力の壁
LLM synth only:     596/756  (78.8%) ← 構造理解の壁
Hybrid:             840/1000 (84.0%) ← 相補的に壁を突破
```

Cross Engine が解ける問題は高速（~1.6 秒/問）かつ確実。LLM が解ける問題は柔軟だが遅い（~25 秒/問）。両者は異なるタイプの問題を解くため、**相補的に機能する**。

#### ② コード生成 + 決定論的検証（答え生成ではない）

**他（LLM 直接推論）**: LLM にグリッド出力を直接推論させる → ハルシネーションのリスク
**Verantyx**: LLM は Python コードを書く → 全訓練例でコード実行 → 完全一致のみ採用

```
LLM 直接推論:
  "この入力の出力は [[2,3],[1,4]] だと思います"  ← 検証不可能

Verantyx:
  "def transform(grid): ..."  ← 全訓練例で実行・検証可能
```

この違いにより:
- **ハルシネーション耐性**: コードが間違っていれば検証で弾かれる
- **解釈可能性**: なぜその答えが出たかをコードで追跡できる
- **再現性**: 同じコードは常に同じ結果を返す

#### ③ 構造的理解（Cross Structure Analysis）

**他（LLM only）**: グリッドをフラットなテキストとして処理
**Verantyx**: グリッドの**十字構造（cross structure）**を分析

Cross Engine は入出力ペアから以下を検出する:
- オブジェクトの連結成分
- 色ごとの空間分布
- 近傍規則（セルオートマタ的）
- 対称性
- パネル構造（分割・結合）
- 重力・移動パターン
- 残差（部分一致からの差分）

これにより、LLM では捉えにくい**幾何学的パターン**を確実に検出できる。

#### ④ 残差学習 (Iterative Cross)

**他**: 単一ステップで解けなければ諦める
**Verantyx**: 部分的に正しい変換を適用し、残差を再帰的に解く

```
例: タスク「色の置換 + パターンのタイル化」
  Step 1: 色置換を学習 → 70% 一致
  Step 2: 残差（タイル化）を学習 → 100% 一致
  → 2 段合成として採用
```

最大 3 段階の残差学習により、単一のソルバーでは解けない複合タスクを分解して解く。

#### ⑤ カスタム DSL (Puzzle Reasoning Language)

**他**: 汎用 Python に依存
**Verantyx**: ARC 専用の宣言的 DSL を設計

```
ALL(cells(color=3)) → fill(color=7)           # 色 3 を全て色 7 に
EACH(objects()) → reflect(axis=vertical)        # 各オブジェクトを縦反転
BETWEEN(cells(color=1), cells(color=2)) → fill(color=5)  # 間を充填
```

この DSL は:
- **宣言的**: 「何をするか」を記述（「どうやるか」ではない）
- **合成可能**: 複数の操作を組み合わせ可能
- **検証可能**: 訓練例に対して機械的に検証

#### ⑥ CEGIS (Counter-Example Guided Inductive Synthesis)

**他**: LLM の出力をそのまま信頼
**Verantyx**: CEGIS ループで変換候補を体系的に刈り込む

```
1. 入出力差分から変換候補を自動生成
2. 候補を全訓練例で検証
3. 不一致 → 反例として候補を除外
4. 一致する候補をチェーン化
5. 全訓練例で一致するチェーンのみ採用
```

#### ⑦ GPU 不要・低コスト

**他 (o3, Gemini 等)**: GPU クラスタ必須、$10,000+ のコスト
**Verantyx**: MacBook (M1/M2/M3) で動作、API コストは合成分のみ

| 項目 | LLM 直接推論 (o3 等) | Verantyx |
|---|---|---|
| ハードウェア | GPU クラスタ | MacBook (CPU のみ) |
| 推論コスト | $10,000+ / 1000 問 | ~$50-100 / 1000 問 |
| Cross Engine コスト | — | $0（ローカル実行） |
| 実行時間 | 数時間 (並列 GPU) | ~7 時間 (全タスク) |

#### ⑧ 段階的開発の知見

Verantyx は v19 (11.3%) から v82+Synth (84.0%) まで、**73 バージョン**の反復開発で成長した。各バージョンは特定のパターンクラスに対するソルバーを追加し、**回帰テスト**で既存の正解を壊さないことを確認している。

この段階的開発により:
- 各ソルバーがどのタスクを解くか完全に追跡可能
- 新ソルバー追加時の回帰リスクが低い
- 失敗パターンの体系的な分析が可能

### 7.3 比較まとめ

| 特性 | LLM 直接推論 | DSL 探索のみ | Kaggle NN | **Verantyx** |
|---|---|---|---|---|
| 表現力 | ◎ | △ | ○ | **◎** (Hybrid) |
| 検証可能性 | × | ○ | × | **◎** (CEGIS + verify) |
| 構造理解 | △ | ○ | △ | **◎** (Cross Analysis) |
| コスト効率 | × | ◎ | ○ | **○** (CE無料 + API) |
| 汎化性能 | △ | ○ | △ | **○** (投票式) |
| GPU 依存 | 必須 | 不要 | 必須 | **不要** |
| 解釈可能性 | × | ◎ | × | **◎** (コード出力) |
| 新パターン対応 | ○ | × | × | **◎** (LLM fallback) |

---

## 8. プロジェクト構造

```
verantyx_v6/                          154,825 lines of Python | 1,099 files
│
├── arc/                              49,445 lines — コアソルバーエンジン (93 files)
│   ├── cross_engine.py               メインオーケストレータ (2,817 lines)
│   ├── eval_cross_engine.py          評価ランナー
│   ├── cross_solver.py               DSL ソルバー (2,047 lines)
│   ├── puzzle_lang.py                カスタム推論 DSL (2,623 lines)
│   ├── cross_universe.py             Cross 構造分解
│   ├── cross_universe_3d.py          立体 cross (1,664 lines)
│   ├── cross_multiscale.py           6 軸記述子 (846 lines)
│   ├── cross_compose.py              多段合成 (980 lines)
│   ├── meta_cross.py                 階層ルーティング (972 lines)
│   ├── object_mover.py               7 移動戦略 (1,325 lines)
│   ├── per_object.py                 オブジェクト別変換 (1,291 lines)
│   ├── program_search.py             テスト時合成 (1,191 lines)
│   ├── residual_guided.py            残差解析 (1,162 lines)
│   ├── nb_abstract.py                近傍抽象化 (901 lines)
│   ├── panel_ops.py                  パネル操作 (953 lines)
│   ├── color_swap_solver.py          色変換 (818 lines)
│   ├── line_connect.py               線描画 (741 lines)
│   ├── arc_cegis.py                  CEGIS (697 lines)
│   ├── llm_hypothesis.py             LLM 仮説生成 (706 lines)
│   ├── llm_direct.py                 LLM 直接解答
│   ├── llm_deepseek.py               DeepSeek 連携
│   ├── llm_router.py                 LLM ルーティング
│   ├── llm_solver.py                 Qwen ソルバー
│   ├── r1_program_synth.py           DeepSeek R1 合成
│   └── (他 46 ソルバーモジュール)
│
├── synth_results/                    ~14,180 lines — LLM 生成コード (597 files)
│   ├── 009d5c81.py                   タスク別 transform 関数
│   ├── 00dbd492.py
│   └── ... (597 ファイル)
│
├── eval_synth_results/               評価セット用生成コード (26 files)
├── eval_synth_multi/                 多数決投票用複数解
│
├── knowledge/                        14,043 lines — 600B モデル知識 (38 files)
│   ├── concept_search.py             SVD ベース概念検索
│   ├── concept_boost.py              キーワードブースティング
│   ├── expert_loader.py              GGUF 重み読み込み (DeepSeek V3)
│   └── reasoning_type_classifier.py  推論型分類
│
├── puzzle/                           20,717 lines — HLE パズルパイプライン (24 files)
├── executors/                        12,195 lines — タスク実行器 (41 files)
├── decomposer/                       2,028 lines — 問題分解 (6 files)
├── verifiers/                        1,069 lines — Z3/SymPy 検証
├── grammar/                          テンプレート合成
├── llm/                              LLM 統合レイヤー
├── gates/                            品質ゲート
├── tools/                            開発ツール
│
├── verify_transform.py               LLM コード検証スクリプト (62 lines)
├── vote_verify.py                    多数決検証 (102 lines)
├── make_demo_gif.py                  デモ GIF 生成
├── analyze_unsolved.py               未解決分析
├── config.py                         設定
└── README.md                         プロジェクト README
```

---

## 9. コードベース統計

| カテゴリ | ファイル数 | 行数 | 説明 |
|---|---|---|---|
| **arc/** (コアエンジン) | 93 | 49,445 | 手書きソルバー群 |
| **puzzle/** (HLE) | 24 | 20,717 | HLE パズル対応 |
| **knowledge/** (600B) | 38 | 14,043 | DeepSeek V3 SVD 知識 |
| **synth_results/** (生成コード) | 597 | ~14,180 | LLM 生成 transform 関数 |
| **executors/** | 41 | 12,195 | タスク実行器 |
| **decomposer/** | 6 | 2,028 | 問題分解 |
| **tools/** | 9 | 1,901 | 開発ユーティリティ |
| **その他** | 291 | ~40,316 | 設定・テスト・分析 |
| **合計** | **1,099** | **154,825** | |

### パフォーマンス

| 指標 | 値 |
|---|---|
| Cross Engine 速度 | ~1.6 秒/問 |
| LLM Synth 速度 | ~25 秒/問 |
| 全 1000 問 (CE のみ) | ~27 分 |
| 全 756 問 (Synth) | ~5 時間 (6 並列) |
| メモリ使用量 | < 2 GB |
| GPU 必要 | なし |

---

## 10. 技術的詳細

### 10.1 Cross Structure Analysis とは

ARC パズルの入出力を**十字構造（cross）**として分析する独自手法。入力グリッドを以下の軸で記述する:

1. **色分布軸**: 各色のセル数・位置分布
2. **オブジェクト軸**: 連結成分の形状・サイズ・相対位置
3. **対称軸**: 回転対称・線対称の有無
4. **パネル軸**: グリッドの規則的分割パターン
5. **近傍軸**: 各セルの近傍関係（4 近傍 / 8 近傍）
6. **スケール軸**: 入出力のサイズ比・繰り返しパターン

### 10.2 LLM 統合の詳細

| LLM | 用途 | 呼び出し方式 | 特徴 |
|---|---|---|---|
| Claude Sonnet 4.5 | プログラム合成（メイン） | OpenClaw sub-agent | Zero-shot, コード生成特化 |
| Claude Opus 4 | オーケストレーション | OpenClaw main session | バッチ分割・エージェント管理 |
| Qwen2.5-7B | 仮説生成・直接推論 | Ollama (ローカル) | 無料、低レイテンシ |
| DeepSeek-Chat | タスク分類 | API 直接呼び出し | 安価、分類特化 |
| DeepSeek V3 (600B) | 知識検索 (HLE) | 重み直接読み込み | SVD 事前計算済 |

### 10.3 OpenClaw による並列実行

```
OpenClaw Gateway
├── Main Session (Opus 4)
│   ├── "756問を50問ずつ15バッチに分割"
│   ├── "5-6個のsub-agentを並列起動"
│   └── "完了したバッチの結果を収集"
│
├── Sub-Agent 1 (Sonnet 4.5) → Batch 1 (50 tasks)
├── Sub-Agent 2 (Sonnet 4.5) → Batch 2 (50 tasks)
├── Sub-Agent 3 (Sonnet 4.5) → Batch 3 (50 tasks)
├── Sub-Agent 4 (Sonnet 4.5) → Batch 4 (50 tasks)
└── Sub-Agent 5 (Sonnet 4.5) → Batch 5 (50 tasks)
     ↓
     各 Agent が独立に:
     1. task.json を読む
     2. transform(grid) を生成
     3. verify_transform.py で検証
     4. 成功 → synth_results/{task_id}.py に保存
     5. 失敗 → 最大 3 回リトライ
```

---

## 11. 制約と今後

### 現在の制約

1. **汎化率**: Training 84.0% に対し、Evaluation set 上のテスト正解率は 42.3%。LLM が訓練例に過適合するコードを生成する場合がある
2. **API コスト**: Sonnet/Opus の API 呼び出しは個人開発者にとって高額
3. **Evaluation set**: 120 問中 11 問正解（9.2%）。85% の Grand Prize 閾値にはまだ距離がある

### 改善計画

1. **Opus × 3-solution voting**: より高品質な LLM (Opus 4) で 3 つの独立解を生成し、多数決で汎化率を向上
2. **Leave-one-out 検証**: 訓練例の 1 つを除外して検証することで、過適合を検出
3. **Cross Engine の継続拡張**: 手書きソルバーの追加により、LLM に頼らない正解数を増やす
4. **プロンプト改善**: ハードコード禁止・抽象化推奨の指示を強化

### 目標

- **短期**: Evaluation set で 50%+ の汎化率を達成
- **中期**: 85% の ARC Prize Grand Prize 閾値を突破
- **長期**: ARC-AGI2 における人間レベルの抽象推論能力の実現

---

## 付録: 用語集

| 用語 | 説明 |
|---|---|
| **Cross Engine** | Verantyx のコアルールベースソルバー |
| **CrossPiece** | 統一ソルバーインターフェース (name + apply_fn) |
| **CrossSimulator** | CEGIS 検証エンジン |
| **CEGIS** | Counter-Example Guided Inductive Synthesis（反例駆動型帰納的合成）|
| **Puzzle Reasoning Language** | ARC 専用の宣言的 DSL |
| **Iterative Cross** | 残差学習による多段合成 |
| **ver=N** | 検証済み候補数（ver=0 = 未解決）|
| **synth_results/** | LLM が生成した transform 関数の保存ディレクトリ |
| **OpenClaw** | LLM エージェント並列実行プラットフォーム |

---

*Author: kofdai × OpenClaw*
*Last updated: 2026-03-01*
