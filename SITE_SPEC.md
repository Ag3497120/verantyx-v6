# verantyx.ai サイト改修 設計仕様書

## 概要

現在の verantyx.ai は Verantyx（シンボリック推論エンジン）の紹介サイトだが、以下の改修を行う：

1. **トップページをポータル/ハブ型に改修** — 現在のVerantyxエンジンコンテンツを「Verantyx」ボックスにまとめ、その上に「Apps」ボックスを配置。各ボックスをクリックすると該当ページに遷移できるカード型ナビゲーション。
2. **iOSアプリの紹介ページを追加** — パクパク釣り、MouthEatそれぞれの詳細ページ
3. **AdMob要件・App Store要件のページを追加** — app-ads.txt、プライバシーポリシー、サポート

---

## 現在の技術スタック（変更不要）

- **Framework**: Next.js 15.1.0 (TypeScript)
- **Rendering**: 静的エクスポート (`output: 'export'`)
- **Styling**: Tailwind CSS 4.0.0
- **Animations**: Framer Motion 11.11.17
- **React**: 19.0.0
- **デプロイ**: 静的HTML出力 (`/out`)

---

## ★ トップページ改修（最重要）

### 現在の状態
`/`（page.tsx）にVerantyxシンボリック推論エンジンの全コンテンツ（Hero, BenchmarkCounters, SolverAnimation, HowItWorks, ArchitecturePipeline, ScoreChart, LinkReveal, SupportSection, Stats, Footer）が1ページに展開されている。

### 改修内容: ポータル/ハブ型トップページ

トップページを **カード型ナビゲーションハブ** に変更する。

#### レイアウト構成

```
[Navbar（既存を改修）]

[ヒーロー: "Verantyx" ロゴ + "Explore Our Projects" サブタイトル]

[カードグリッド（2〜3列）]
  ┌─────────────────────┐  ┌─────────────────────┐
  │  📱 Apps             │  │  ⚡ Verantyx Engine  │
  │                      │  │                      │
  │  口で遊ぶiOSゲーム    │  │  シンボリック推論     │
  │  パクパク釣り          │  │  ARC-AGI-2: 20.7%   │
  │  MouthEat            │  │  Zero Neural Networks│
  │                      │  │                      │
  │  → アプリを見る       │  │  → 詳しく見る        │
  └─────────────────────┘  └─────────────────────┘

  ┌─────────────────────┐  ┌─────────────────────┐
  │  📦 Verantyx-CLI     │  │  📚 .jcross Language │
  │                      │  │                      │
  │  コマンドラインツール   │  │  クロスワードDSL      │
  │                      │  │                      │
  │  → ドキュメント       │  │  → ドキュメント       │
  └─────────────────────┘  └─────────────────────┘

[Footer（Legal リンク付き）]
```

#### 設計詳細

**ヒーローセクション:**
- 「Verantyx」の大きなロゴテキスト（既存のグラデーションスタイル）
- サブタイトル: "Projects & Apps" or "Explore Our Work"
- 背景: 控えめなパーティクルエフェクト or グラデーション（既存のCrossStructure3Dは使わない。シンプルに）

**カード仕様:**
- 各カードは `<a>` or `<Link>` でページ遷移
- ホバーでグロー＋スケールアップ（Framer Motion `whileHover`）
- ダークテーマ: `bg-gray-900/50` + `border: 1px solid rgba(14,165,233,0.2)`
- ホバー時: ボーダーが明るくなる + subtle glow
- アイコン（絵文字）+ タイトル + 説明文 + CTAリンク

**カード内容（4枚）:**

1. **📱 Apps — iOSゲーム**
   - リンク先: `/apps`
   - 説明: "口の動きで遊ぶ革新的なiOSゲーム"
   - サブ情報: "パクパク釣り / MouthEat"

2. **⚡ Verantyx Engine — シンボリック推論エンジン**
   - リンク先: `/verantyx` （← 現在のトップページコンテンツをここに移動）
   - 説明: "LLM-free symbolic reasoning"
   - サブ情報: "ARC-AGI-2: 20.7% — Zero Neural Networks"

3. **📦 Verantyx-CLI**
   - リンク先: `/verantyx-cli`（既存ページ）
   - 説明: "コマンドラインインターフェース"

4. **📚 .jcross Language**
   - リンク先: `/jcross-language`（既存ページ）
   - 説明: "クロスワードパズルDSL"

#### 現在のトップページコンテンツの移動先

現在の `/` (page.tsx) の全セクションを **`/verantyx`** に移動する:

- `src/app/page.tsx` → 新しいポータルページに書き換え
- `src/app/verantyx/page.tsx` → 新規作成。現在のpage.tsxの内容をそのまま移す

これにより:
- `verantyx.ai/` = ポータルハブ（カード型ナビゲーション）
- `verantyx.ai/verantyx` = 現在のVerantyxエンジン紹介ページ（既存コンテンツそのまま）

---

## 追加が必要なページ・ファイル

### 1. `/app-ads.txt`（ルート直下 — 最優先）

**目的**: Google AdMob の広告配信認証ファイル
**配置場所**: `public/app-ads.txt`（静的エクスポートでルート直下に出力される）
**内容**:
```
google.com, pub-1029897132990069, DIRECT, f08c47fec0942fa0
```

**重要**: これはHTMLページではなくプレーンテキストファイル。`https://verantyx.ai/app-ads.txt` でそのまま表示されること。

---

### 2. `/apps` — アプリ一覧ページ

**パス**: `src/app/apps/page.tsx`
**デザイン**: 現在のサイトのダーク＋ブルーアクセントテーマに合わせる

**内容**:
- ページタイトル: "Apps" / "アプリケーション"
- 2つのアプリカードを表示:
  1. **パクパク釣り（PakuPaku Fishing）** → `/apps/pakupaku-fishing` へリンク
  2. **MouthEat** → `/apps/mouth-eat` へリンク
- 各カードにはアプリアイコン（後述）、アプリ名、短い説明文、App Storeバッジ
- **追加カード**: 🏆 World Rankings → `/apps/ranking` へリンク

---

### 2.5. `/apps/ranking` — ワールドランキングページ（新規）

**パス**: `src/app/apps/ranking/page.tsx`

#### 概要
両アプリ（パクパク釣り・MouthEat）のワールドランキングをWebブラウザで閲覧できるページ。
Supabase REST APIをクライアントサイドから直接呼び出してリアルタイムのランキングデータを取得する。

#### ログイン機能: 不要
- ランキング閲覧は**ログインなし**で全員が可能
- Supabase の anon key（公開キー）でRPC関数を呼び出すため認証不要
- 「自分のランク検索」はニックネーム検索で実現（Apple IDは不要）

#### Supabase API 接続情報

```typescript
// lib/supabase.ts
const SUPABASE_URL = "https://zekypqjmvyxevwyujicn.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpla3lwcWptdnl4ZXZ3eXVqaWNuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM1NjI5NjcsImV4cCI6MjA4OTEzODk2N30.HzU2FjYp03xEqTISUkOvkkfzzKdMBZ2Bkw2hHE-6IH0";
```

#### 使用するRPC関数

**1. `get_top_ranking`** — トップランキング取得
```
POST /rest/v1/rpc/get_top_ranking
Body: { "p_mode": "timeattack", "p_limit": 100 }
Response: [{ user_id, nickname, score, world_rank }]
```

**2. `get_my_rank`** — 特定ユーザーのランク取得（将来的にWeb Sign In追加時に使用）
```
POST /rest/v1/rpc/get_my_rank
Body: { "p_user_id": "xxx", "p_mode": "timeattack" }
Response: { world_rank, total_players, best_score }
```

**3. `get_nearby_ranking`** — 周辺ランキング（将来用）
```
POST /rest/v1/rpc/get_nearby_ranking
Body: { "p_user_id": "xxx", "p_mode": "timeattack" }
Response: [{ user_id, nickname, score, world_rank }] (±5 positions)
```

#### レスポンス型定義

```typescript
interface RankEntry {
  user_id: string;
  nickname: string;
  score: number;
  world_rank: number;
}

interface RankResult {
  world_rank: number;
  total_players: number;
  best_score: number;
}
```

#### ゲームモード一覧（タブ切り替え）

| ゲーム | モードID | 表示名 |
|--------|----------|--------|
| パクパク釣り | `timeattack` | ⏱️ タイムアタック |
| MouthEat | `eat_timeattack` | ⏱️ タイムアタック |
| MouthEat | `eat_survival` | ❤️ サバイバル |
| MouthEat | `eat_combo` | 🔥 コンボチャレンジ |
| MouthEat | `eat_judgment` | 🧠 ジャッジメント |

#### UI レイアウト

```
[Navbar]

[ヒーロー: "🏆 World Rankings" + "リアルタイムのワールドランキング"]

[ゲーム選択タブ（横並び）]
  ┌──────────────┐ ┌──────────────┐
  │ 🎣 パクパク釣り │ │ 🍔 MouthEat  │
  └──────────────┘ └──────────────┘

[モード選択タブ（選択中のゲーム依存で表示切替）]
  パクパク釣り選択時:  [⏱️ タイムアタック]
  MouthEat選択時:    [⏱️ タイムアタック] [❤️ サバイバル] [🔥 コンボ] [🧠 ジャッジ]

[ニックネーム検索バー]
  ┌─────────────────────────────┐ ┌──────┐
  │ ニックネームで検索...           │ │ 🔍   │
  └─────────────────────────────┘ └──────┘

[ランキングテーブル]
  ┌──────┬──────────────┬─────────┐
  │ 順位  │ ニックネーム    │ スコア   │
  ├──────┼──────────────┼─────────┤
  │ 🥇 1  │ FishMaster   │ 12,500  │
  │ 🥈 2  │ パクパク王     │ 11,200  │
  │ 🥉 3  │ 釣りキング     │ 10,800  │
  │  4   │ player123    │  9,500  │
  │  5   │ gamer_xx     │  8,900  │
  │ ...  │              │         │
  └──────┴──────────────┴─────────┘

[ページネーション or 無限スクロール（Top 100）]

[Footer]
```

#### テーブル仕様
- Top 3 はメダルアイコン（🥇🥈🥉）表示
- ニックネーム検索でヒットした行はシアン色でハイライト
- ダークテーマ（黒背景 + シアンアクセント）
- モバイル: テーブルを横スクロール or カード形式に切り替え
- ローディング中はスケルトンUI表示
- データ取得失敗時はエラーメッセージ + リトライボタン

#### 自動リフレッシュ
- ページロード時に自動取得
- 30秒ごとに自動リフレッシュ（setInterval）
- 手動リフレッシュボタンあり（🔄）
- 最終更新時刻を表示

#### 必要なコンポーネント

```
src/components/ranking/
├── RankingHero.tsx         ← ヒーローセクション
├── GameSelector.tsx        ← ゲーム選択タブ
├── ModeSelector.tsx        ← モード選択タブ
├── NicknameSearch.tsx      ← ニックネーム検索バー
├── RankingTable.tsx        ← ランキングテーブル本体
├── RankingRow.tsx          ← テーブル行コンポーネント
└── RankingSkeleton.tsx     ← ローディングスケルトン

src/lib/
└── supabase.ts             ← Supabase API クライアント
```

#### 将来拡張: Web Sign In with Apple（現時点では不要）

もし「Web上で自分の詳細ランク（周辺±5プレイヤー含む）を見たい」需要が出た場合:

1. **Apple Developer Console**: Web用 Services ID を作成
2. **Supabase Auth**: Apple Sign In プロバイダを有効化
3. **Next.js**: OAuth フロー実装（サインインボタン → Apple認証 → コールバック）
4. **取得したappleUserIDで**: `get_my_rank` + `get_nearby_ranking` を呼び出し

→ 現時点ではニックネーム検索で十分なため**実装不要**。

---

### 3. `/apps/pakupaku-fishing` — パクパク釣り詳細ページ

**パス**: `src/app/apps/pakupaku-fishing/page.tsx`

#### アプリ基本情報
- **アプリ名**: パクパク釣り (PakuPaku Fishing)
- **Bundle ID**: `marimokodai.PakuPakuFishing`
- **対応OS**: iOS 17.0+
- **対応デバイス**: iPhone（TrueDepthカメラ搭載機種）
- **言語**: 日本語 / English
- **価格**: 無料（アプリ内課金あり）

#### ヒーローセクション
- アプリ名 + キャッチコピー: "口で釣る、新感覚フィッシングゲーム"
- 英語: "Catch fish using only your mouth — a revolutionary fishing experience"
- App Store ダウンロードバッジ（リンクは後から設定）

#### ゲーム概要セクション
顔の動きで釣りを楽しむ革新的なiOSゲーム。デバイスのカメラで口の動きと顔のジェスチャーをリアルタイム検出し、バーチャルな釣りを体験できます。

**ゲームの流れ（6フェーズ）:**
1. 🎯 **観察** — 池を見渡し、キャスティングポイントを決める
2. 🎣 **キャスティング** — タップして釣り糸を投げる
3. ⏳ **待機** — 口を閉じてじっと待つ
4. 🐟 **アタリ** — 魚が食いつく
5. ⚡ **フッキング** — タイミングよく口を閉じて針を合わせる
6. 🔄 **リーリング** — 口を素早く開閉して魚を巻き上げる

#### ゲームモードセクション（3カラム or カード形式）

**フリーモード 🎣**
- 無制限に釣りを楽しめる
- スコア蓄積 & コンボシステム
- 深度ゾーンによる魚の分布

**タイムアタック ⏱️**
- 60秒の時間制限チャレンジ
- 3-2-1 カウントダウンでスタート
- ランキング対応

**一本釣りチャレンジ 🏆**
- 巨大な一匹を狙う
- 持続的な集中力が必要
- 5〜120秒以上の長時間ファイト
- 伝説級の魚との真剣勝負

#### 魚図鑑セクション
**20種の魚 × 4段階の難易度:**

| 難易度 | 魚種 | スコア |
|--------|------|--------|
| **Easy** | メダカ、ハゼ、イワシ、フナ、アユ | 50〜150pt |
| **Medium** | タイ、サバ、イカ、ヒラメ、ブリ | 200〜400pt |
| **Hard** | マグロ、カツオ、タコ、スズキ、フグ | 450〜800pt |
| **Legendary** | クジラ、サメ、ダイオウイカ、シーラカンス、金の鯉 | 1,000〜5,000pt |

深さ4ゾーン（浅瀬・中層・深海・深淵）で出現分布が変化。

#### 顔ジェスチャーシステム セクション
**13種の顔ジェスチャーに対応:**

**基本（無料）:**
- 口を開ける / 口を閉じる

**PASS限定（11種）:**
- 頭を上/下/左/右に傾ける
- あごを前に突き出す
- 眉を上げる
- 左ウインク / 右ウインク
- ほっぺを膨らませる
- 舌を出す
- 笑顔

#### コンボシステム セクション
**5つのビルトインコンボ（全ユーザー利用可能）:**

1. **ラピッドマウスフラップ 💥** — 超高速口パクパクで1.5倍スコア
2. **パワーリフト 🔥** — 口閉じ+上を向いて2倍ダメージ
3. **ダイブキャスト 🎯** — 下を向いて口を開けると1.5倍射程
4. **ドッジカウンター 🌊** — 左右に首振りで1.5倍ダメージ+魚フリーズ
5. **フォーカスフィッシング 🧘** — 眉上げ3秒キープでレア魚70%出現

**カスタムコンボ作成（PASS限定）:**
- 最大5個のオリジナルコンボを作成可能
- ジェスチャーの組み合わせ・エフェクト・クールダウンを自由にカスタマイズ

#### カスタマイズ セクション
**カスタム魚パック:**
- オリジナルの魚を作成（写真 or 絵文字）
- 難易度・名前・著作権情報を設定
- ディープリンクで友達と共有可能

**池キャプチャ（PASS限定）:**
- カメラで実際の水辺を撮影
- 水の色・形状を自動抽出
- 自分だけの釣り場を作成

#### 料金プラン セクション

| 機能 | Free | PASS (¥250) |
|------|------|-------------|
| 基本ゲームプレイ | ✅ | ✅ |
| ビルトインコンボ | ✅ | ✅ |
| 広告なし | ❌ | ✅ |
| 感度調整 | ❌ | ✅ |
| カスタムコンボ作成 | ❌ | ✅ |
| 高度なジェスチャー（11種） | ❌ | ✅ |
| カスタム魚パック | ❌ | ✅ |
| 池キャプチャ | ❌ | ✅ |

**PASS** = ¥250の買い切り（サブスクリプションではない）

#### プライバシーセクション
- カメラ映像はデバイス上でのみ処理
- 外部サーバーへの映像送信なし
- 顔データの保存・収集なし
- 広告: Google AdMob（Free版のみ）

---

### 4. `/apps/mouth-eat` — MouthEat 詳細ページ

**パス**: `src/app/apps/mouth-eat/page.tsx`

#### アプリ基本情報
- **アプリ名**: MouthEat
- **Bundle ID**: `marimokodai.MouthEat`
- **対応OS**: iOS 17.0+
- **対応デバイス**: iPhone（TrueDepthカメラ搭載機種）
- **言語**: 日本語 / English
- **価格**: 無料（アプリ内課金あり）

#### ヒーローセクション
- アプリ名 + キャッチコピー: "口を開けて食べまくれ！リアルタイム食べゲーム"
- 英語: "Open wide and eat! A real-time mouth-controlled eating game"
- App Store ダウンロードバッジ

#### ゲーム概要セクション
口の動きをリアルタイム検出して食べ物をキャッチする新感覚ゲーム。画面を流れてくる100種類以上の食べ物を、実際に口を開けてパクパク食べよう！食べられないものには気をつけて。

#### ゲームモードセクション（4カラム or カード形式）

**タイムアタック ⏱️**
- 制限時間60秒
- 1レーン
- 時間とともにスピードアップ
- テーマカラー: オレンジ/レッド

**サバイバル ❤️**
- HP制（初期HP: 100）
- 3レーン（高難易度）
- 食べられない物でダメージ
- テーマカラー: レッド

**コンボチャレンジ 🔥**
- コンボ継続がカギ
- 1レーン
- コンボ段階: なし → Warming(1.2x) → Hot(1.5x) → Blazing(2x) → Legendary(3x)
- コンボが途切れたらゲームオーバー
- テーマカラー: パープル/ピンク

**ジャッジメント 🧠**
- ベルトコンベア方式
- 全35アイテム（75%食用, 25%非食用）
- ノッチ/Dynamic Islandからもアイテムが落下
- 正しく判断して食べよう
- テーマカラー: ティール/シアン

#### フード図鑑セクション
**100種類以上のアイテム:**

**食べ物（77種）:**
- 🍎 フルーツ（16種）: りんご、バナナ、ぶどう、みかん、すいか...
- 🥩 肉類（6種）: ステーキ、チキン、ハンバーガー...
- 🦐 海鮮（5種）: 寿司、エビ、たこ焼き...
- 🍰 スイーツ（12種）: ケーキ、ドーナツ、アイスクリーム...
- 🥕 野菜（11種）: にんじん、ブロッコリー、とうもろこし...
- 🍕 料理（24種）: ピザ、ラーメン、おにぎり、カレー...
- 🥤 ドリンク（8種）: ジュース、牛乳、お茶、コーヒー...

**スペシャルアイテム:**
- ✨ 金のりんご — レア出現、200pt
- 🌈 レインボー — 超レア、500pt
- 🌶️ 激辛唐辛子 — ダメージ
- 💣 爆弾 — マイナススコア

**食べられないもの（60種以上）:**
車、バス、ロケット、家、病院、ハンマー、パソコン、靴、サッカーボール...

#### アバターシステム セクション
- 複数のアバタースタイル
- 10種類の表情（アイドル、口開け、もぐもぐ、幸せ、満腹、気持ち悪い、辛い、超幸せ、瀕死、ハート目）
- 写真からカスタムアバター作成
- テーマカラーのカスタマイズ

#### ゲームプレイ録画 セクション
- ReplayKitで15秒クリップ録画
- フォトライブラリに保存
- ベストプレイを友達とシェア

#### 料金プラン セクション

| 機能 | Free | PASS |
|------|------|------|
| 全ゲームモード | ✅ | ✅ |
| 基本プレイ | ✅ | ✅ |
| 広告なし | ❌ | ✅ |
| スピード調整（0.5x〜2.0x） | ❌ | ✅ |
| 出現間隔調整 | ❌ | ✅ |

#### プライバシーセクション
- カメラ: 口の動き検出（デバイス上処理のみ、外部送信なし）
- フォトライブラリ: 録画クリップの保存
- ランキング: スコアデータのみ送信
- 顔データの保存・収集なし

---

### 5. `/privacy` — プライバシーポリシー

**パス**: `src/app/privacy/page.tsx`

**内容構成:**

#### 1. はじめに
Verantyx（以下「当社」）が提供するiOSアプリケーション（パクパク釣り、MouthEat）に関するプライバシーポリシーです。

#### 2. 収集するデータ

**カメラ映像:**
- 顔の動き・口の動きを検出するために使用
- 映像データはすべてデバイス上でリアルタイム処理
- サーバーへの送信・保存は一切行いません
- ARKit（ARFaceAnchor）を使用した顔トラッキング

**フォトライブラリ:**
- MouthEat: ゲームプレイ録画クリップの保存
- パクパク釣り: カスタム魚パックの画像追加、池キャプチャ
- ユーザーの明示的な許可を得た場合のみアクセス

**スコアデータ:**
- ランキング機能のためにスコア・ユーザー名をサーバー送信
- 個人を特定する情報は含まれません

#### 3. 広告について

**Google AdMob（パクパク釣りの無料版のみ）:**
- Google Mobile Adsを利用してバナー広告・インタースティシャル広告を表示
- AdMobは広告配信のために広告識別子（IDFA）を使用する場合があります
- PASS購入後は広告は表示されません
- 詳細: https://policies.google.com/privacy

#### 4. アプリ内課金
- Apple StoreKit 2 を使用
- 決済はAppleが処理し、当社はクレジットカード情報等を一切保持しません

#### 5. データの保存
- ゲーム設定・進行状況はデバイスローカル（UserDefaults）に保存
- iCloudやクラウドサービスへの自動同期はありません

#### 6. 第三者への提供
- Google AdMob以外の第三者へのデータ提供はありません
- Apple App Storeの規約に準拠

#### 7. 児童のプライバシー
- 13歳未満の児童から意図的に個人情報を収集することはありません

#### 8. お問い合わせ
- メール: (設定予定)
- ウェブサイト: https://verantyx.ai/support

#### 9. 改定
- 本ポリシーは予告なく変更される場合があります
- 最終更新日を表示

---

### 6. `/support` — サポートページ

**パス**: `src/app/support/page.tsx`

**内容:**

#### よくある質問 (FAQ)

**Q: どのiPhoneで使えますか？**
A: TrueDepthカメラ搭載のiPhone（iPhone X以降）、iOS 17.0以上が必要です。

**Q: カメラの映像は保存されますか？**
A: いいえ。カメラ映像はデバイス上でリアルタイム処理のみ行われ、保存・送信されることは一切ありません。

**Q: PASSを購入したら広告は消えますか？**
A: はい。PASS購入後は全ての広告が非表示になります。

**Q: PASSの購入はサブスクリプションですか？**
A: いいえ。¥250の買い切りです。一度購入すれば永久にご利用いただけます。

**Q: 購入を復元するには？**
A: アプリ内の設定画面から「購入を復元」をタップしてください。

**Q: 口の検出がうまくいきません**
A: 明るい場所で、カメラに顔全体が映るように端末を持ってください。キャリブレーション機能で感度を調整することもできます。

#### お問い合わせ
- メールアドレス表示（設定予定）
- 「お問い合わせはメールにてお願いいたします」

#### 動作環境
- iOS 17.0以上
- iPhone X 以降（TrueDepthカメラ搭載機種）
- 約XX MBのストレージ空間

---

### 7. `/terms` — 利用規約（任意だが推奨）

**パス**: `src/app/terms/page.tsx`

簡潔な利用規約：
- アプリの利用条件
- 禁止事項
- 免責事項
- 準拠法（日本法）

---

## ナビゲーション改修

### Navbar の変更

**現在のリンク:**
```
📦 Verantyx-CLI | 📚 .jcross Language | ⭐ Star on GitHub
```

**変更後:**
```
🏠 Home | 📱 Apps | ⚡ Engine | 📦 CLI | 📚 .jcross | ⭐ Star on GitHub
```

- 🏠 Home → `/` （ポータルトップ）
- 📱 Apps → `/apps`
- ⚡ Engine → `/verantyx`（旧トップページコンテンツ）
- 📦 CLI → `/verantyx-cli`（既存）
- 📚 .jcross → `/jcross-language`（既存）

### Footer の変更

Footer に以下のリンクを追加:
```
Projects: Verantyx Engine | Verantyx-CLI | .jcross Language
Apps: パクパク釣り | MouthEat
Legal: Privacy Policy | Support | Terms
```

---

## ページ設計の詳細

### 共通デザインルール

1. **テーマ**: 既存の黒背景 + 電気ブルーアクセント（#0EA5E9）を維持
2. **フォント**: Inter（既存のまま）
3. **アニメーション**: Framer Motion で scroll-triggered fade-in を使用（既存パターンに準拠）
4. **レスポンシブ**: モバイルファースト、768px ブレークポイント
5. **パーティクル背景**: アプリページにはParticleBackgroundは不要（テキスト中心のため）

### アプリページ共通レイアウト

```
[Navbar]
[ヒーロー: アプリ名 + キャッチコピー + App Store バッジ]
[ゲーム概要]
[ゲームモード紹介（カード形式）]
[特徴セクション（アイテム/魚/コンボなど）]
[料金プラン比較表]
[プライバシー概要]
[他のアプリへの導線]
[Footer]
```

### App Store バッジ

「Download on the App Store」SVGバッジを使用。
リンク先URLは後から設定するため、暫定で `#` をhrefに設定。

---

## ファイル配置まとめ

```
src/app/
├── page.tsx                        ← 改修: ポータルハブに書き換え（カード型ナビゲーション）
├── layout.tsx                      ← 改修: metadata更新
├── verantyx/
│   └── page.tsx                    ← 新規: 旧トップページのコンテンツをそのまま移動
├── apps/
│   ├── page.tsx                    ← 新規: アプリ一覧
│   ├── ranking/
│   │   └── page.tsx                ← 新規: ワールドランキング（Supabase API連携）
│   ├── pakupaku-fishing/
│   │   └── page.tsx                ← 新規: パクパク釣り詳細
│   └── mouth-eat/
│       └── page.tsx                ← 新規: MouthEat詳細
├── privacy/
│   └── page.tsx                    ← 新規: プライバシーポリシー
├── support/
│   └── page.tsx                    ← 新規: サポート
├── terms/
│   └── page.tsx                    ← 新規: 利用規約
├── verantyx-cli/                   ← 既存（変更なし）
└── jcross-language/                ← 既存（変更なし）

src/components/
├── Navbar.tsx                      ← 改修: 新ナビゲーションリンク構成
├── Footer.tsx                      ← 改修: Projects/Apps/Legal リンク追加
├── PortalCard.tsx                  ← 新規: トップページのカードコンポーネント
├── AppCard.tsx                     ← 新規: アプリ一覧用カードコンポーネント
├── ranking/                        ← 新規: ランキング用コンポーネント群
│   ├── RankingHero.tsx
│   ├── GameSelector.tsx
│   ├── ModeSelector.tsx
│   ├── NicknameSearch.tsx
│   ├── RankingTable.tsx
│   ├── RankingRow.tsx
│   └── RankingSkeleton.tsx
├── FeatureCard.tsx                 ← 新規: 機能紹介カード
├── PricingTable.tsx                ← 新規: 料金比較テーブル
├── AppStoreBadge.tsx               ← 新規: App Storeバッジ
└── (既存コンポーネント群)           ← 変更なし（/verantyx ページで引き続き使用）

src/lib/
└── supabase.ts                     ← 新規: Supabase API クライアント（ランキング用）

public/
├── app-ads.txt                     ← 新規: AdMob認証ファイル
├── images/
│   ├── pakupaku-icon.png           ← 新規: アプリアイコン（後から配置）
│   └── moutheat-icon.png           ← 新規: アプリアイコン（後から配置）
└── (既存ファイル群)                 ← 変更なし
```

---

## SEO / Metadata

### layout.tsx の更新

```typescript
export const metadata: Metadata = {
  title: 'Verantyx - Symbolic Reasoning & iOS Apps',
  description: 'Symbolic reasoning engine and innovative iOS games using facial recognition. PakuPaku Fishing, MouthEat.',
  keywords: ['symbolic reasoning', 'ARC-AGI', 'iOS game', 'face tracking', 'fishing game', 'mouth game'],
};
```

### 各アプリページのメタデータ

**パクパク釣り:**
```typescript
export const metadata: Metadata = {
  title: 'パクパク釣り - 口で釣る新感覚フィッシングゲーム | Verantyx',
  description: '顔の動きで魚を釣る革新的なiOSゲーム。20種の魚、13種のジェスチャー、カスタムコンボ作成。iOS 17対応。',
};
```

**MouthEat:**
```typescript
export const metadata: Metadata = {
  title: 'MouthEat - リアルタイム食べゲーム | Verantyx',
  description: '口を開けて100種以上の食べ物をキャッチ！4つのゲームモード、コンボシステム、アバターカスタマイズ。iOS 17対応。',
};
```

---

## 優先順位

1. 🔴 **最優先**: `public/app-ads.txt` — AdMob配信に必須（テキストファイル1行だけ）
2. 🔴 **最優先**: トップページ(`/`)ポータルハブ化 + `/verantyx` への旧コンテンツ移動
3. 🔴 **最優先**: `/privacy` — App Store審査に必須
4. 🟡 **高**: Navbar/Footer 改修（新ナビゲーション構成）
5. 🟡 **高**: `/apps` — アプリ一覧ページ
6. 🟡 **高**: `/apps/pakupaku-fishing` — パクパク釣り詳細
7. 🟡 **高**: `/apps/mouth-eat` — MouthEat詳細
8. 🟡 **高**: `/apps/ranking` — ワールドランキング（Supabase API連携）
9. 🟢 **中**: `/support` — サポート・FAQ
10. 🔵 **低**: `/terms` — 利用規約

---

## ルーティング変更まとめ

| URL | 変更前 | 変更後 |
|-----|--------|--------|
| `/` | Verantyxエンジン紹介（フル） | **ポータルハブ**（カード型ナビゲーション） |
| `/verantyx` | (存在しない) | **Verantyxエンジン紹介**（旧`/`のコンテンツをそのまま移動） |
| `/apps` | (存在しない) | **アプリ一覧** |
| `/apps/ranking` | (存在しない) | **ワールドランキング**（Supabase API連携） |
| `/apps/pakupaku-fishing` | (存在しない) | **パクパク釣り詳細** |
| `/apps/mouth-eat` | (存在しない) | **MouthEat詳細** |
| `/privacy` | (存在しない) | **プライバシーポリシー** |
| `/support` | (存在しない) | **サポート** |
| `/terms` | (存在しない) | **利用規約** |
| `/verantyx-cli` | 既存 | **変更なし** |
| `/jcross-language` | 既存 | **変更なし** |

---

## 注意事項

- 現在のトップページ (`/`) のコンテンツは **`/verantyx` に移動**する（削除ではない）
- 既存の `/verantyx-cli` と `/jcross-language` は**変更しない**
- `/verantyx` は現在の page.tsx のコンテンツをそのままコピー。既存コンポーネント（CrossStructure3D, BenchmarkCounters等）はすべてそのまま使用
- App Store のリンクURLはまだ未定のため `href="#"` をプレースホルダとして使用
- アプリアイコン画像はまだ未配置 → プレースホルダ絵文字で代用（🎣, 🍔）
- お問い合わせメールアドレスは後から設定
