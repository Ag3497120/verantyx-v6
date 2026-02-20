# Phase 5F実装計画

**対象**: 高度な数論・確率（117問, 11.5%）  
**推定時間**: 16-32時間  
**難易度**: VERY_HARD

---

## 📊 対象ドメイン分析

### HLE Dataset内訳
- **number_theory_advanced**: 1問
- **probability_advanced**: 9問  
- **combinatorics（高度な部分）**: 約50問
- **algebra_advanced（数論的要素）**: 約20問
- **VERY_HARD全般**: 117問

**合計推定**: 約117問

---

## 🎯 必要な実装

### 1. 高度な数論 Executor

#### modular_arithmetic.py
- `mod_power`: べき乗の余り（a^b mod m）
- `mod_inverse`: 逆元（a^-1 mod m）
- `chinese_remainder`: 中国剰余定理
- `fermat_little`: フェルマーの小定理（a^(p-1) ≡ 1 mod p）
- `euler_phi`: オイラーのφ関数

#### advanced_number_theory.py
- `prime_factorization`: 素因数分解（大きい数）
- `is_primitive_root`: 原始根判定
- `quadratic_residue`: 平方剰余
- `legendre_symbol`: ルジャンドル記号
- `miller_rabin`: Miller-Rabin素数判定

### 2. 高度な確率・統計 Executor

#### advanced_probability.py
- `conditional_probability`: 条件付き確率 P(A|B)
- `bayes_theorem`: ベイズの定理
- `binomial_distribution`: 二項分布
- `normal_distribution`: 正規分布
- `poisson_distribution`: ポアソン分布

#### statistics.py
- `variance`: 分散
- `standard_deviation`: 標準偏差
- `covariance`: 共分散
- `correlation`: 相関係数
- `expected_value_advanced`: 期待値（複雑な分布）

### 3. 高度な組み合わせ論 Executor

#### advanced_combinatorics.py
- `stirling_first`: 第1種スターリング数
- `stirling_second`: 第2種スターリング数
- `catalan_number`: カタラン数
- `partition_number`: 分割数
- `derangement`: 完全順列（攪乱順列）

---

## 📝 実装の課題

### 技術的難易度

1. **大きな数の計算**
   - Pythonの`int`は任意精度だが、計算時間が問題
   - 高速化アルゴリズムが必要（例: 高速べき乗）

2. **数論アルゴリズムの正確性**
   - 中国剰余定理: 実装が複雑
   - Miller-Rabin: 確率的判定
   - 原始根: 計算量が大きい

3. **確率分布の精度**
   - 二項分布: 大きなnで計算困難
   - 正規分布: 近似が必要
   - scipyへの依存を検討

4. **テストケースの難しさ**
   - 正解の検証が困難
   - 数学的知識が必要

---

## 🗓️ 実装スケジュール（推定）

### Week 1: 基本的な高度数論（6-8時間）
- Day 1-2: modular_arithmetic.py実装
- Day 3: 中国剰余定理・フェルマー小定理
- Day 4: オイラーのφ関数

**成果物**: 
- modular_arithmetic.py完成
- ピース5-10個追加
- テスト5問（70%目標）

### Week 2: 高度な確率・統計（6-8時間）
- Day 1-2: conditional_probability, bayes_theorem
- Day 3: 分布（binomial, normal, poisson）
- Day 4: 統計量（variance, std, correlation）

**成果物**:
- advanced_probability.py完成
- statistics.py完成
- ピース8-12個追加
- テスト5問（70%目標）

### Week 3: 高度な組み合わせ論（4-8時間）
- Day 1: stirling numbers
- Day 2: catalan, partition, derangement

**成果物**:
- advanced_combinatorics.py完成
- ピース5-8個追加
- テスト5問（70%目標）

### Week 4: 統合・最適化（2-4時間）
- 全体テスト
- デバッグ・修正
- ドキュメント作成

---

## 🎯 段階的実装戦略

### オプション1: フル実装（16-32時間）
- 全ての高度Executorを実装
- HLE 117問中 70%（82問）達成目標

**利点**: 完全なカバレッジ  
**欠点**: 時間がかかる

### オプション2: 部分実装（8-16時間）
- 基本的な高度数論のみ（modular_arithmetic）
- 基本的な高度確率のみ（conditional, bayes）
- HLE 117問中 30-50%（35-60問）達成目標

**利点**: 短時間で主要機能を実装  
**欠点**: カバレッジが不完全

### オプション3: Skip & 次へ（0時間）
- Phase 5Fをスキップ
- Phase 5G（UNKNOWN分析）へ進む
- 既存の571問で目標714問の80%達成済み

**利点**: 時間節約  
**欠点**: HLE 70%目標未達の可能性

---

## 💡 推奨アプローチ

### 推奨: オプション2（部分実装）

**理由**:
1. 現在571/714問（80.0%）達成済み
2. 残り143問で70%達成には最低143問必要だが、117問では不足
3. 高度な内容は実装時間が長い
4. 部分実装で重要な機能を確保し、残りはPhase 5G以降で補完

**実装内容**:
- modular_arithmetic.py（mod演算、中国剰余定理）
- conditional_probability（条件付き確率、ベイズ）
- 推定時間: 8-12時間
- 推定達成: 40-60問（HLE 611-631問、85.6-88.4%）

---

## 📋 次のアクション

### すぐに開始する場合
1. modular_arithmetic.py実装（2-3時間）
2. テストケース作成
3. 動作確認

### 計画を見直す場合
1. Phase 5G（UNKNOWN分析）の内容確認
2. 残り143問の内訳分析
3. 最適な実装順序の再検討

---

**決定事項**: ユーザーに確認を求める  
**推奨**: オプション2（部分実装、8-12時間）  
**代替案**: オプション3（Skip、Phase 5Gへ）

---

*作成日: 2026-02-16 06:47 JST*
