/* GENERATED — do not edit by hand.
 *
 * Built by scripts/build_catalogue.py, which reads the READMEs of every
 * non-fork repository and runs them through Vera itself: the same loaders,
 * the same sentence splitter, the same Japanese segmentation the engine uses
 * on any other corpus.
 *
 * 22 repositories · 96,271 characters read.
 *
 * Every string below is verbatim from a README. Summaries are the first real
 * sentence, not a paraphrase — a paraphrase would be the tool marking its own
 * homework. Topics are the cores Vera extracted, ranked by mass within that
 * document, which is why they are also the bot's match keys: what a visitor
 * can ask about is exactly what the document is about.
 */

export type CatalogueEntry = {
  name: string;
  url: string;
  description: string;
  language: string;
  stars: number;
  updated: string;
  lang: string;
  summary: string;
  topics: string[];
  facts: string[];
  chars: number;
  sentences: number;
};

export const CATALOGUE_CHARS = 96271;

export const CATALOGUE: CatalogueEntry[] = [
  {
    "name": "MBP-AVP-VR",
    "url": "https://github.com/Ag3497120/MBP-AVP-VR",
    "description": "A native stereoscopic VR container for Mac and Apple Vision Pro. Bypasses D3DMetal IPC constraints to stream x86 PCVR engines (Source 2) via low-level C++ openvr_api.dll memory hooking and direct visionOS hardware compositing.",
    "language": "C++",
    "stars": 1,
    "updated": "2026-08-01",
    "lang": "en",
    "summary": "MBP-AVP-VR: The Ultimate Ultra-Low Latency VR Streaming Architecture",
    "topics": [
      "mbp",
      "openvr",
      "apple's",
      "xcode",
      "game",
      "system_architecture",
      "120fps",
      "analog",
      "capture",
      "chunks",
      "client",
      "color",
      "compilation",
      "consolidates"
    ],
    "facts": [
      "The result is a \"Mirage\"—an illusion so perfectly orchestrated that the game believes it's rendering to a tethered PCVR headset, while delivering a wireless 120Hz retinal experience to the Apple Vision Pro.",
      "To prevent segmentation faults caused by missing vtable entries, our class injects precisely 100 dummy padding functions ( to ).",
      "The Payload: The memory map consists of a 16-byte header, the raw pixel buffer, and a struct (236 bytes) containing full transform matrices, button states, and stick states from the Vision Pro.",
      "This bidirectional memory pool allows the Mac encoder and the Windows game process to communicate synchronously at 120Hz."
    ],
    "chars": 8010,
    "sentences": 96
  },
  {
    "name": "TALKIEPRESS-1930",
    "url": "https://github.com/Ag3497120/TALKIEPRESS-1930",
    "description": "",
    "language": "Python",
    "stars": 0,
    "updated": "2026-05-23",
    "lang": "en",
    "summary": "This application generates 1930s-style newspaper articles based on modern news.",
    "topics": [
      "talkiepress",
      "colorfrom",
      "colorto",
      "machine",
      "upload",
      "1930s",
      "access",
      "hardware_requirements",
      "please",
      "hugging_face_spaces",
      "title",
      "sdk"
    ],
    "facts": [
      "1930s Newspaper Generator (TalkiePress)",
      "This application generates 1930s-style newspaper articles based on modern news.",
      "It directly loads a specialized LLM ( ) that possesses knowledge exclusively from 1930 and earlier.",
      "It converts modern events into 1930s journalism, perfectly reflecting the historical background, unique phrasing, and ethical standards of the era."
    ],
    "chars": 2000,
    "sentences": 14
  },
  {
    "name": "Vera-qwen-0.5b-kit",
    "url": "https://github.com/Ag3497120/Vera-qwen-0.5b-kit",
    "description": "",
    "language": "Python",
    "stars": 0,
    "updated": "2026-07-22",
    "lang": "en",
    "summary": "This repository contains the inference engine and mathematical proofs for Generative Spatial Compression, a revolutionary technique that shrinks Large Language Models by transforming dense weight matrices into low-rank generative spatial coordinates.",
    "topics": [
      "vera",
      "absolute",
      "bandwidth",
      "bypass",
      "coordinates",
      "engine",
      "enjoy",
      "kit",
      "matrix",
      "part",
      "reduces",
      "relies",
      "breakthrough",
      "disadvantage_loss"
    ],
    "facts": [
      "Attempting to zero-shot compress weights like this breaks the Residual Stream and causes a noise cascade (gibberish output).",
      "Because we reduce the rank of the matrices significantly (e.g. , Rank 128), the model physically loses its capacity to store vast amounts of \"trivia\" or \"world knowledge\" (e.g. , niche historical dates, minor celebrity names, specific factual data).",
      "If you ask this compressed model a trivia question without context, it is highly likely to hallucinate because its dense memory has been structurally erased."
    ],
    "chars": 3349,
    "sentences": 31
  },
  {
    "name": "Vera",
    "url": "https://github.com/Ag3497120/Vera",
    "description": "立体十字構造体 (Stereo-Cross Container) — reorganizing LLM knowledge into a shared cross-layer coordinate system with Matryoshka granularity and puzzle-style inference. Pre-registered experiments, personal-scale compute.",
    "language": "Python",
    "stars": 0,
    "updated": "2026-07-22",
    "lang": "ja",
    "summary": "Vera — 立体十字構造体（Stereo-Cross Container）検証プロジェクト",
    "topics": [
      "事前登録フォーク",
      "立体十字構造体",
      "層固有性",
      "間劣化",
      "共有ブリッジ",
      "座標系",
      "座標軸",
      "ランク",
      "立体十字容",
      "訓練な",
      "共有低次元座標系",
      "再事前学習",
      "対照群",
      "格納座標"
    ],
    "facts": [
      "LLM の構造とは全く異なる構造のものをゼロから作るか、既存のモデルをうまく流用して作れないか。",
      "安い順: 解析的（訓練ゼロ）→ ソフト介入 → 蒸留、の順で費用をかける",
      "低ランクでは相対的な悪さは近い（＋8〜16%） → 圧縮の物語のみ。",
      "C_valve の非対角成分は自由にすると 99% 使われるが品質は救えない。"
    ],
    "chars": 8774,
    "sentences": 158
  },
  {
    "name": "Verantyx-Logic",
    "url": "https://github.com/Ag3497120/Verantyx-Logic",
    "description": "Verantyx-Logic",
    "language": "Python",
    "stars": 0,
    "updated": "2026-05-23",
    "lang": "en",
    "summary": "Verantyx (avh-math) is not a large language model (LLM).",
    "topics": [
      "verantyx",
      "pseudo",
      "hugging_face",
      "配布物",
      "本プロジェクト",
      "web_ui",
      "prove",
      "verification",
      "windows",
      "パッケージング",
      "様相論理",
      "疑似重",
      "cli",
      "counterexample"
    ],
    "facts": [
      "These pseudo-weights are treated as read-only snapshots.",
      "Verantyx’s core feature remains: the DB is editable and replaceable (swap rules, swap domains, change behavior without retraining).",
      "Pseudo-weights are treated as read-only snapshots.",
      "Treat the packaged snapshot as read-only"
    ],
    "chars": 9272,
    "sentences": 125
  },
  {
    "name": "Verantyx-Vera-alpha",
    "url": "https://github.com/Ag3497120/Verantyx-Vera-alpha",
    "description": "Verantyx Vera α — deterministic, LM-free knowledge & reasoning engine with typed refusal (no hallucination by construction). CLI + MCP server.",
    "language": "Python",
    "stars": 0,
    "updated": "2026-08-04",
    "lang": "en",
    "summary": "A deterministic, LM-free knowledge & reasoning engine that refuses to hallucinate.",
    "topics": [
      "vera",
      "facets",
      "traceback",
      "verantyx",
      "accumulates",
      "action",
      "agent",
      "areas",
      "bug",
      "builder",
      "chat",
      "consensus",
      "content",
      "contradiction"
    ],
    "facts": [
      "Vera is honest about what it cannot > do: it does not write fluent prose, it does not chat casually, and it does > not invent anything it was never taught.",
      "In hybrid mode Vera stays the controller; the local model is only the language surface.",
      "Native memory harness (no MCP, no triggers): in chat, every declarative utterance is remembered automatically; questions and imperatives are not (so \"tell me something\" never becomes a fake fact).",
      "Exact finishes with no LLM and no tools; web search is a stdlib"
    ],
    "chars": 6264,
    "sentences": 72
  },
  {
    "name": "Verantyx-Vera-beta",
    "url": "https://github.com/Ag3497120/Verantyx-Vera-beta",
    "description": "Verantyx Vera β — deterministic, LM-free knowledge & reasoning engine + fingerprint-derived leak-attribution watermarking. CLI + MCP server.",
    "language": "Python",
    "stars": 0,
    "updated": "2026-07-26",
    "lang": "en",
    "summary": "A deterministic, LM-free knowledge & reasoning engine that refuses to hallucinate.",
    "topics": [
      "vera",
      "facets",
      "traceback",
      "verantyx",
      "accumulates",
      "action",
      "agent",
      "areas",
      "bug",
      "builder",
      "chat",
      "consensus",
      "content",
      "contradiction"
    ],
    "facts": [
      "Vera is honest about what it cannot > do: it does not write fluent prose, it does not chat casually, and it does > not invent anything it was never taught.",
      "In hybrid mode Vera stays the controller; the local model is only the language surface.",
      "Native memory harness (no MCP, no triggers): in chat, every declarative utterance is remembered automatically; questions and imperatives are not (so \"tell me something\" never becomes a fake fact).",
      "Exact finishes with no LLM and no tools; web search is a stdlib"
    ],
    "chars": 6264,
    "sentences": 72
  },
  {
    "name": "Verantyx",
    "url": "https://github.com/Ag3497120/Verantyx",
    "description": "Zero-Trust Enterprise AI IDE for macOS. Gatekeeper Mode ensures 100% security by using Local SLMs to convert raw code into JCross IR, letting Cloud LLMs (Claude/DeepSeek) refactor logic without leaking any proprietary semantics.",
    "language": "Swift",
    "stars": 1,
    "updated": "2026-08-07",
    "lang": "en",
    "summary": "Verantyx is a macOS IDE carrying Vera-α, a deterministic knowledge engine that answers from stored facts or says — as a type — why it cannot.",
    "topics": [
      "verantyx",
      "engine",
      "recall",
      "hand",
      "growth",
      "bot's",
      "contribution",
      "decisions",
      "decomposer",
      "documents",
      "domain",
      "drop",
      "fit",
      "grammar"
    ],
    "facts": [
      "The switch sits in the chat header rather than in settings, because the two produce different kinds of answer, and which kind you are reading should never be something you go and check.",
      "A refusal is never rewritten by a model.",
      "Watching those numbers shrink is what learning honestly means here — and a reader who cannot see the unknowns in one place has no way to watch.",
      "Quarantined proposals never act on their own."
    ],
    "chars": 8722,
    "sentences": 93
  },
  {
    "name": "VisionSpatialTools",
    "url": "https://github.com/Ag3497120/VisionSpatialTools",
    "description": "Advanced visionOS app for Apple Vision Pro - Attach virtual tools to real-world objects with magnetic snap and tracking",
    "language": "Swift",
    "stars": 0,
    "updated": "2026-05-23",
    "lang": "en",
    "summary": "⚠️ Note: This project has not been fully tested yet due to time constraints.",
    "topics": [
      "attach",
      "xcode",
      "select",
      "object",
      "plist",
      "tap",
      "info",
      "arkit",
      "hand",
      "canvas",
      "capabilities",
      "click",
      "scan",
      "welcome"
    ],
    "facts": [
      "This project contains only Swift source files.",
      "Attach virtual keyboard (auto-sized to 80% of MacBook width)",
      "Attach trackpad (40% of width)",
      "✅ All Swift source code is implemented (~3,000 lines)"
    ],
    "chars": 5959,
    "sentences": 92
  },
  {
    "name": "cross-memory-space",
    "url": "https://github.com/Ag3497120/cross-memory-space",
    "description": "6次元記憶空間 - Claude AIの能動的記憶操作システム",
    "language": "Python",
    "stars": 0,
    "updated": "2026-05-23",
    "lang": "ja",
    "summary": "Cross Memory Spaceは、6次元空間で記憶を表現し、Claude AIが能動的に記憶を操作できるシステムです。",
    "topics": [
      "6次元記憶空間",
      "6次元空間",
      "claude",
      "crud",
      "json",
      "sqlite",
      "system",
      "tool",
      "マルチモーダル",
      "固定的な",
      "claude_aianthropicmodel_context_protocol",
      "claude_desktop",
      "github_issues",
      "mcp"
    ],
    "facts": [
      "✅ 動的な記憶管理: よく使う記憶は自動的に近づき、使わない記憶は遠ざかる- ✅ 能動的記憶操作: Claudeが自律的に記憶を保存・検索・強化- ✅ 物理法則による自律移動: 時間漂流・引力・斥力による記憶の自動管理- ✅ MCP統合: Claude Desktopで直接利用可能（APIキー不要）"
    ],
    "chars": 1971,
    "sentences": 11
  },
  {
    "name": "dendritic-memory-editor",
    "url": "https://github.com/Ag3497120/dendritic-memory-editor",
    "description": "樹木型空間記憶（Dendritic Memory Space）を編集するためのWebサイト",
    "language": "TypeScript",
    "stars": 0,
    "updated": "2026-05-23",
    "lang": "ja",
    "summary": "Dendritic Memory Editor は、複雑な情報を整理・管理・共有するための包括的なプラットフォームです。",
    "topics": [
      "リアルタイム",
      "cloudflare",
      "tailwindcss",
      "typescript",
      "cloudflare_workers",
      "複雑な",
      "axios",
      "docker",
      "graphql",
      "uuid",
      "エンタープライズレベル",
      "cloudflare_pages",
      "github_pages",
      "operational_transformation"
    ],
    "facts": [
      "合計実装: ~16,900行のプロダクションレディコード"
    ],
    "chars": 2483,
    "sentences": 19
  },
  {
    "name": "desktop-tutorial",
    "url": "https://github.com/Ag3497120/desktop-tutorial",
    "description": "GitHub Desktop tutorial repository",
    "language": "",
    "stars": 0,
    "updated": "2026-02-27",
    "lang": "en",
    "summary": "READMEs are where you can communicate what your project is and how to use it.",
    "topics": [
      "name",
      "readmes",
      "welcome"
    ],
    "facts": [],
    "chars": 202,
    "sentences": 3
  },
  {
    "name": "paku-paku-eating",
    "url": "https://github.com/Ag3497120/paku-paku-eating",
    "description": "Paku Paku Eating",
    "language": "",
    "stars": 0,
    "updated": "2026-05-23",
    "lang": "en",
    "summary": "A real-time mouth-controlled eating game",
    "topics": [
      "moutheat",
      "カメラ",
      "combo",
      "デバイス",
      "bug",
      "damage",
      "food",
      "github_issues",
      "multiple",
      "paku",
      "theme",
      "time",
      "unique",
      "テーマカラー"
    ],
    "facts": [
      "Over 100 types of food and items stream across the screen — open your mouth and eat them up!",
      "HP-based (starting HP: 100)",
      "35 items (75% edible, 25% inedible)",
      "PASS = ¥250 one-time purchase (not a subscription)"
    ],
    "chars": 3280,
    "sentences": 42
  },
  {
    "name": "paku-paku-fishing",
    "url": "https://github.com/Ag3497120/paku-paku-fishing",
    "description": "Paku Paku Fishing",
    "language": "",
    "stars": 0,
    "updated": "2026-05-23",
    "lang": "en",
    "summary": "Catch fish using only your mouth — a revolutionary fishing experience",
    "topics": [
      "デバイス",
      "mouth",
      "出現分布",
      "ジェスチャー",
      "bug",
      "distribution",
      "gesture",
      "github_issues",
      "ios",
      "paku",
      "paku_paku_fishing",
      "photograph",
      "photos",
      "requires"
    ],
    "facts": [
      "Catch fish using only your mouth — a revolutionary fishing experience",
      "Unlimited fishing with no time pressure - Score accumulation & combo system",
      "PASS = ¥250 one-time purchase (not a subscription)",
      "✓ Camera footage processed on-device only - ✓ No video transmitted to external servers - ✓ No facial data stored or collected"
    ],
    "chars": 3087,
    "sentences": 35
  },
  {
    "name": "tool-search-oss",
    "url": "https://github.com/Ag3497120/tool-search-oss",
    "description": "BM25 tool search for MCP — find the right tool from 50+ without context collapse. Works with any LLM.",
    "language": "Python",
    "stars": 1,
    "updated": "2026-05-13",
    "lang": "en",
    "summary": "Open-source, LLM-agnostic implementation of the \"Tool Search\" architectural pattern.",
    "topics": [
      "tool",
      "embeddings",
      "bm25",
      "mcp",
      "cascade",
      "defer",
      "democratizes",
      "eliminates",
      "layer",
      "lightweight",
      "llm",
      "origin",
      "regex",
      "renders"
    ],
    "facts": [
      "Anthropic recently demonstrated (Advanced Tool Use, Nov 2025) that dynamically routing tools — instead of loading all definitions upfront — improves routing accuracy from 79.5% to 88.1% on large tool catalogs.",
      "Find the right MCP tool from 2000+ without context collapse, working locally with any LLM (Ollama, GPT-4o, Claude API, Gemini).",
      "82% context reduction · 96% routing accuracy · 135x TTFT at 2000 tools · zero dependencies",
      "BM25 — default, handles tokenization, zero dependencies"
    ],
    "chars": 3461,
    "sentences": 43
  },
  {
    "name": "verantyx-arc-agi2",
    "url": "https://github.com/Ag3497120/verantyx-arc-agi2",
    "description": "Pure program synthesis solver for ARC-AGI-2 — 74/1000=7.4%, no neural networks, no LLMs",
    "language": "Python",
    "stars": 0,
    "updated": "2026-05-23",
    "lang": "en",
    "summary": "Pure program synthesis solver for ARC-AGI-2 — no neural networks, no LLMs, no hardcoded patterns.",
    "topics": [
      "verantyx",
      "color",
      "cell",
      "decompose",
      "deduplication",
      "gravity",
      "programs",
      "pure",
      "requirements",
      "single",
      "solver",
      "solves",
      "subgrid",
      "whole"
    ],
    "facts": [
      "Pure program synthesis solver for ARC-AGI-2 — no neural networks, no LLMs, no hardcoded patterns.",
      "The solver only uses training examples to discover patterns, then applies verified programs to test inputs."
    ],
    "chars": 1631,
    "sentences": 20
  },
  {
    "name": "verantyx-cli",
    "url": "https://github.com/Ag3497120/verantyx-cli",
    "description": "By incorporating a proprietary model into the routing process, we can overcome uncertainties such as those associated with harnesses. The proprietary model thinks in terms of vectors, and another AI translates those vectors.",
    "language": "Rust",
    "stars": 0,
    "updated": "2026-08-04",
    "lang": "en",
    "summary": "Verantyx is a local AI operations harness.",
    "topics": [
      "router",
      "quickstart",
      "python3",
      "verantyx",
      "benchmarks",
      "controls",
      "council",
      "honest",
      "japanese",
      "omni",
      "smoke",
      "source",
      "survives",
      "venv"
    ],
    "facts": [
      "Larger models only when needed.",
      "The router stays local; larger speaker models are called only when needed.",
      "60-second demo (no model weights required)",
      "python3 scripts/ --no-model"
    ],
    "chars": 985,
    "sentences": 12
  },
  {
    "name": "verantyx-cortex-ios",
    "url": "https://github.com/Ag3497120/verantyx-cortex-ios",
    "description": "On-device long-term memory benchmark - LongMemEval on iOS with llama3.2:1b via MLX",
    "language": "Swift",
    "stars": 0,
    "updated": "2026-05-23",
    "lang": "en",
    "summary": "On-device long-term memory benchmark system running entirely on iOS with local LLM inference — no cloud, no latency, no privacy compromise.",
    "topics": [
      "benchmark",
      "verantyx",
      "tap",
      "device",
      "remove",
      "term",
      "verantyx_cortex",
      "bottleneck",
      "breakdown",
      "context",
      "core",
      "dataset",
      "development",
      "facts"
    ],
    "facts": [
      "On-device long-term memory benchmark system running entirely on iOS with local LLM inference — no cloud, no latency, no privacy compromise.",
      "The core challenge: LongMemEval requires recalling specific facts from conversations held days or weeks ago, across 500 test questions.",
      "Instead of feeding 100,000+ characters of raw history into context (impossible at 8K window), we use a Verantyx-designed 4-layer JCross memory system to route each question directly to the relevant 2–3KB of context.",
      "Estimated target: 25–35% accuracy"
    ],
    "chars": 4462,
    "sentences": 51
  },
  {
    "name": "verantyx-cortex",
    "url": "https://github.com/Ag3497120/verantyx-cortex",
    "description": "Context-window problems solved with 35 files. MCP-native spatial memory for any LLM.",
    "language": "Python",
    "stars": 0,
    "updated": "2026-07-22",
    "lang": "en",
    "summary": "Verantyx Cortex — Tri-Layer JCross Spatial Memory Engine",
    "topics": [
      "nodes",
      "calibration",
      "zone",
      "node",
      "tasks",
      "tombstone",
      "kanji",
      "verantyx_cortex",
      "track",
      "compiles",
      "context",
      "determines",
      "fetches",
      "gear"
    ],
    "facts": [
      "Any AI (Claude, Gemini, GPT, Cursor) can use it via MCP with zero configuration overhead.",
      "The system's design goal: a new LLM session started after a model switch should reach project-expert-level context within 5–7 tool calls, without any human prompting.",
      "Routine tasks use only L1+L2 → zero context pollution, maximum speed",
      "Critical tasks fall back to L3 → full fidelity, no information loss"
    ],
    "chars": 5427,
    "sentences": 80
  },
  {
    "name": "verantyx-pure-through-ja",
    "url": "https://github.com/Ag3497120/verantyx-pure-through-ja",
    "description": "Verantyx Pure-Through (日本語版) - AI記憶リフレッシュシステム。コンテキスト汚染を防ぎ、クリーンな知能を継続的に使用。",
    "language": "Python",
    "stars": 0,
    "updated": "2026-05-23",
    "lang": "ja",
    "summary": "現代のエージェント型AI（Claude Code、ChatGPT、Geminiなど）には、根本的な問題があります",
    "topics": [
      "解決策",
      "エージェント",
      "コンテキスト",
      "verantyx",
      "モデル",
      "クリーン",
      "セッション",
      "継続性",
      "計算資源",
      "12kb",
      "16kb",
      "27kb",
      "cli",
      "gemini"
    ],
    "facts": [
      "\"file_001を確認して\" → 自動的にREADコマンド実行",
      "\"file_002を移動して\" → 自動的にMOVEコマンド実行（要パラメータ）"
    ],
    "chars": 2993,
    "sentences": 31
  },
  {
    "name": "verantyx-pure-through",
    "url": "https://github.com/Ag3497120/verantyx-pure-through",
    "description": "Verantyx Pure-Through - AI Memory Refresh System. Prevents context pollution and enables continuous use of clean intelligence.",
    "language": "Python",
    "stars": 1,
    "updated": "2026-05-23",
    "lang": "en",
    "summary": "An AI Memory Refresh System for Continuous Use of Clean Intelligence",
    "topics": [
      "verantyx",
      "research",
      "background",
      "safari",
      "tool",
      "context",
      "key_features",
      "system_architecture",
      "technical_details",
      "business",
      "community",
      "complete",
      "constraint",
      "contents"
    ],
    "facts": [
      "Ideal: Experiment with local LLMs in a fully controlled environment Reality: As a student without money, I lack computational resources Solution: Use free browser-based AI with externalized memory and periodic refresh",
      "AI never knows real file paths",
      "Mapping table ( ) never exposed to AI",
      "\"Check file_001\" → Auto-executes READ command"
    ],
    "chars": 5365,
    "sentences": 71
  },
  {
    "name": "verantyx-v6",
    "url": "https://github.com/Ag3497120/verantyx-v6",
    "description": "ARC-AGI2 solver — 84.0% on training (840/1000). Hybrid: 30+ hand-crafted solvers + Claude Sonnet 4.5 program synthesis with deterministic verification. No fine-tuning, no GPU.",
    "language": "HTML",
    "stars": 8,
    "updated": "2026-08-07",
    "lang": "en",
    "summary": "Cutting-edge website for Verantyx — an LLM-free symbolic reasoning engine.",
    "topics": [
      "deploy",
      "pages",
      "typescript",
      "canvas",
      "catalog",
      "comma",
      "edge",
      "framer",
      "functions",
      "prefer",
      "production",
      "purge",
      "retry",
      "secrets"
    ],
    "facts": [
      "Pure SVG charts — zero chart libraries",
      "Humanity's Last Exam: 4.6%",
      "Catalog-only edge proxy for friend apps",
      ", , (PEM — never commit )"
    ],
    "chars": 2310,
    "sentences": 34
  }
];
