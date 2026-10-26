'use client';

import { useEffect, useRef, useState, type ReactNode } from 'react';
import {
  animate,
  motion,
  useInView,
  useMotionValue,
  useReducedMotion,
  useScroll,
  useSpring,
  useTransform,
} from 'framer-motion';
import Navbar from '@/components/Navbar';
import VeraFooter from '@/components/vera-home/VeraFooter';
import { Body, H2, Section } from '@/components/vera-home/ui';
import { useHomeLang, type L } from '@/components/vera-home/useHomeLang';

/* The home of verantyx.ai: VeRantyx (PC VR from a Mac) and vxrt (its Windows runtime) first, then every other
 * project the site already had. Every number here is one the VeRantyx README states with its run or recording
 * (github.com/Ag3497120/VeRantyx, 2026-10-17) or a later recorded run named next to it (Elite 2026-10-20, No Man's Sky
 * 2026-10-26); do not round them or add new ones without a run behind them. */

const REPO = 'https://github.com/Ag3497120/VeRantyx';
// The Mac app, served from this site while the GitHub account is hidden (since 2026-10-24).
// Every released version, newest first; the first one is the stable version the buttons download.
// SHA256 values come from each release's .sha256 file (filled in by the release step).
type Release = { ver: string; date: string; sha256: string; notes: L; lgpl?: boolean };
const RELEASES: Release[] = [
  { ver: '0.3.1', date: '2026-10-26', sha256: '7a432988784f652e2a05d884a37bcbae199b9510da24f87f3046bc1d548a8e28', notes: {
    ja: 'No Man\u2019s Sky が速く（Mac の画面で 32.5 → 45.7 fps、M1 Max）。Pico などで投げた物が飛びすぎる問題、ゲーム中の ALVR の再接続、Half-Life: Alyx の読み込み後の黒い画面（「トリガーを長押しして開始」を表示）を修正。',
    en: 'No Man\u2019s Sky draws faster (32.5 \u2192 45.7 fps on the Mac\u2019s screen, M1 Max). Fixes for thrown objects flying too far on Pico and similar headsets, reconnecting ALVR mid-game, and Half-Life: Alyx\u2019s black screen after a load (now shows "Hold the trigger to start").' } },
  { ver: '0.3.0', date: '2026-10-25', sha256: '532479873ab1117f77ad77ad23b92dac745e2dedd3e97ad51f80b79d3050270a', notes: {
    ja: 'No Man\u2019s Sky（Mac の画面と VR、あなた自身の Windows 版 Steam で起動）、ゲームごとに ALVR アプリか VeRantyx アプリかを選べる、まだ入れていないゲームの「入手してプレイ」。',
    en: 'No Man\u2019s Sky (on the Mac\u2019s screen and in VR, started by your own Windows Steam), ALVR app or VeRantyx app for each game, "Get and play" for games you have not installed yet.' } },
  { ver: '0.2.2', date: '2026-10-25', sha256: 'a866d56c1959056758602a4a3c6081e98ce780485b6a7d17a6ae35d4f828dd8e', notes: {
    ja: 'Elite Dangerous のランチャーの更新が止まらないように。Quest（ALVR）の「高精細」で描く画素数が本当に増える、ヘッドセット側の fps と遅延の表示、通信に合わせたビットレート、一時停止したヘッドセットの再接続。',
    en: 'Elite Dangerous\u2019s launcher can update the game again. Quest (ALVR): "Sharper" really renders more pixels, the headset\u2019s own fps and latency are shown, the bitrate follows the link, a paused headset reconnects.' } },
  { ver: '0.2.1', date: '2026-10-23', sha256: '55a9d1106205a4224db164a671b718cbd6dae8d5a9812406594e95bb8af4e0ae', notes: {
    ja: 'Direct3D 11 のゲームの CPU 負荷を大きく削減（Alyx の重い場面 42〜47 → 60 fps）、ALVR の再接続後の「os error 11」を修正、プレイ後のワンクリック報告。',
    en: 'Much less CPU per Direct3D 11 draw (a busy Alyx scene 42-47 \u2192 60 fps), ALVR no longer stuck at "os error 11" after a reconnect, one-click reports after a play.' } },
  { ver: '0.2.0', date: '2026-10-22', sha256: '011acca340f34fe629bde517c8e618cebc87b4b87b90daa86fdfca40e7bc6403', lgpl: true, notes: {
    ja: 'Elite Dangerous（Mac の画面と VR）、The Lab のコントローラー、ALVR の Wi-Fi の遅延の修正。',
    en: 'Elite Dangerous (on the Mac\u2019s screen and in VR), The Lab\u2019s controllers, ALVR Wi-Fi delay fixes.' } },
  { ver: '0.1.0', date: '2026-10-18', sha256: 'bd035007e383617b5964f6528e32a729714f1136db507232252a747842bd2079', lgpl: true, notes: {
    ja: '最初の Mac アプリ。ターミナルなしでセットアップと起動。',
    en: 'The first Mac app: setup and launch without Terminal.' } },
];
const DMG = `/download/VeRantyx-${RELEASES[0].ver}.dmg`;
const DMG_SHA256 = RELEASES[0].sha256;
const VIDEO_FULL = 'qF09ETv1TpU'; // vxrt version in a Vision Pro, 4 min unedited, 2026-10-14
const VIDEO_MACS = 'YwznYiAXqBg'; // MacBook Pro M1 Max and MacBook Air M5 side by side, 2026-10-17

const HERO_FACTS: { value: L; label: L }[] = [
  { value: { ja: '60 fps', en: '60 fps' }, label: { ja: 'Alyx・3K 設定・M1 Max', en: 'Alyx · 3K preset · M1 Max' } },
  { value: { ja: 'Wine 不要', en: 'No Wine' }, label: { ja: '自前の Windows ランタイム', en: 'its own Windows runtime' } },
  { value: { ja: 'MIT', en: 'MIT' }, label: { ja: 'すべてのソースを公開', en: 'all of the source, open' } },
];

/* ── How a frame travels ─────────────────────────────────── */

type Stage = { key: string; name: string; where: L; body: L };

const STAGES: Stage[] = [
  {
    key: 'game',
    name: 'Windows game',
    where: { ja: 'Mac・あなたの Steam から', en: 'Mac · from your own Steam' },
    body: {
      ja: 'Half-Life: Alyx などの Windows 版そのもの。ゲームのファイルは自分の Steam アカウントから入れ、リポジトリには一切含めません。',
      en: 'The Windows build itself — Half-Life: Alyx and others. The game comes from your own Steam account; nothing of it is in the repository.',
    },
  },
  {
    key: 'vxrt',
    name: 'vxrt',
    where: { ja: 'Mac・ゼロから書いたランタイム', en: 'Mac · a runtime written from scratch' },
    body: {
      ja: 'Windows のプログラム（PE）を読み込み、呼ばれる Windows API を macOS の上に直接実装します。Direct3D 11 は自前の翻訳器でシェーダごと Metal に変換。Wine は使いません。',
      en: 'Loads Windows programs (PE files) and implements the Windows API they call directly on macOS. Direct3D 11 goes to Metal through our own shader translator. No Wine.',
    },
  },
  {
    key: 'steamvr',
    name: 'SteamVR',
    where: { ja: 'Mac・本物の Windows 版', en: 'Mac · the real Windows build' },
    body: {
      ja: '本物の Windows 版 SteamVR が vxrt の上で動き、VeRantyx のドライバがヘッドセットとコントローラになります。ゲーム独自のボタン割り当てもそのまま使えます。',
      en: 'The real Windows SteamVR runs on vxrt, and the VeRantyx driver is its headset and controllers — so each game’s own controller bindings keep working.',
    },
  },
  {
    key: 'encoder',
    name: 'HEVC encoder',
    where: { ja: 'Mac・VideoToolbox', en: 'Mac · VideoToolbox' },
    body: {
      ja: '左右の目の画像をハードウェアで HEVC（3K 設定では 10 bit）に圧縮。各フレームに、描いたときの頭の姿勢と視野を付けて送ります。',
      en: 'Both eyes are encoded in hardware to HEVC (10-bit with the 3K preset). Every frame carries the head pose and field of view it was rendered with.',
    },
  },
  {
    key: 'headset',
    name: 'Headset',
    where: { ja: 'Vision Pro・ALVR', en: 'Vision Pro · ALVR' },
    body: {
      ja: 'Vision Pro 用の VeRantyx アプリ、または ALVR のアプリ（Vision Pro・Quest・Pico）で表示。手と頭の動きは逆向きに Mac へ戻ります。',
      en: 'Shown by the VeRantyx app for Vision Pro, or by an ALVR app (Vision Pro, Quest, Pico). Head and hand poses travel back to the Mac.',
    },
  },
];

/* ── What vxrt implements ────────────────────────────────── */

type Layer = { name: L; detail: L; state: 'works' | 'growing' };

const LAYERS: Layer[] = [
  {
    name: { ja: 'Direct3D 11 / DXGI → Metal', en: 'Direct3D 11 / DXGI → Metal' },
    detail: {
      ja: 'DXBC から Metal Shading Language への翻訳器。境界外の読み取りも Direct3D と同じく 0 を返す',
      en: 'A DXBC → Metal Shading Language translator; out-of-bounds reads return 0, as in Direct3D',
    },
    state: 'works',
  },
  {
    name: { ja: 'Vulkan → Metal', en: 'Vulkan → Metal' },
    detail: {
      ja: 'SPIR-V から Metal Shading Language への翻訳器。テッセレーション、Direct3D 11 との共有テクスチャ（SteamVR へ）まで。No Man\u2019s Sky が動く',
      en: 'A SPIR-V → Metal Shading Language translator, with tessellation and textures shared with Direct3D 11 (for SteamVR). No Man\u2019s Sky runs on it',
    },
    state: 'works',
  },
  {
    name: { ja: 'Direct3D 12 → Metal', en: 'Direct3D 12 → Metal' },
    detail: {
      ja: 'LLVM を使わない DXIL の読み取りと Metal への翻訳。テストは通るが、動くゲームはまだない',
      en: 'DXIL read without LLVM and translated to Metal; its tests pass, but no game runs on it yet',
    },
    state: 'growing',
  },
  {
    name: { ja: 'PE ローダー', en: 'PE loader' },
    detail: { ja: 'Windows の実行ファイルと DLL を読み込み、つなぐ', en: 'Loads and links Windows executables and DLLs' },
    state: 'works',
  },
  {
    name: { ja: 'ファイル・レジストリ', en: 'Files · registry' },
    detail: { ja: 'macOS のファイルシステムの上に', en: 'On the macOS file system' },
    state: 'works',
  },
  {
    name: { ja: 'スレッド・同期・例外', en: 'Threads · sync · exceptions' },
    detail: { ja: 'Mono や Unity の GC が使う一時停止まで', en: 'Down to the thread suspension Mono’s and Unity’s GC rely on' },
    state: 'works',
  },
  {
    name: { ja: 'ウィンドウ・入力', en: 'Windows · input' },
    detail: { ja: 'AppKit の上に', en: 'On AppKit' },
    state: 'works',
  },
  {
    name: { ja: 'ソケット・COM', en: 'Sockets · COM' },
    detail: { ja: 'ゲームは Mac の Steam クライアントにつながる。DRM で包まれたゲーム（No Man\u2019s Sky）は、本物の Windows 版 Steam を vxrt で動かして起動する', en: 'Games reach the Steam client already running on the Mac; DRM-wrapped games (No Man\u2019s Sky) start from the real Windows Steam client, run on vxrt' },
    state: 'works',
  },
  {
    name: { ja: 'オーディオ', en: 'Audio' },
    detail: { ja: 'Core Audio の上に。ヘッドセットへはプロセスタップで', en: 'On Core Audio; to the headset through a process tap' },
    state: 'works',
  },
  {
    name: { ja: 'x86 → ARM64 翻訳器', en: 'x86 → ARM64 translator' },
    detail: {
      ja: 'Rosetta 2 の代わりの自前の翻訳器。The Lab が 52.7〜56.4 fps（Rosetta は 59.4〜59.6）。まだ安定していない',
      en: 'Our own, in place of Rosetta 2. The Lab at 52.7–56.4 fps (Rosetta 59.4–59.6); not reliable yet',
    },
    state: 'growing',
  },
];

/* ── Measured ────────────────────────────────────────────── */

const RUNS: { game: string; fps: number; note: L }[] = [
  {
    game: 'Half-Life: Alyx',
    fps: 60.0,
    note: { ja: '第 1 章・3K 設定（片目 1440×1620）', en: 'chapter 1 · 3K preset (1440×1620 per eye)' },
  },
  { game: 'Google Earth VR', fps: 57.4, note: { ja: '起動から', en: 'from start' } },
  { game: 'The Lab', fps: 54.8, note: { ja: 'タイトル画面', en: 'title scene' } },
  { game: 'No Man\u2019s Sky', fps: 45.7, note: { ja: 'Mac の画面・1512×982・Medium', en: 'Mac screen · 1512×982 · Medium' } },
  { game: 'Elite Dangerous', fps: 33, note: { ja: 'VR・3K 設定（片目 1700×1372）', en: 'VR · 3K preset (1700×1372 per eye)' } },
];

/* ── Headsets ────────────────────────────────────────────── */

const HEADSETS: { title: L; tag: L; points: L[]; checked: boolean }[] = [
  {
    title: { ja: 'Apple Vision Pro・VeRantyx アプリ', en: 'Apple Vision Pro · the VeRantyx app' },
    tag: { ja: '確認済み', en: 'checked' },
    checked: true,
    points: [
      { ja: '手は Vision Pro のハンドトラッキング、ボタンとスティックは Joy-Con', en: 'Hands from Vision Pro hand tracking; buttons and sticks from Joy-Con' },
      { ja: 'VR コントローラ不要。素手だけでも遊べる', en: 'No VR controllers needed — or play with bare hands' },
      { ja: '描画は Mac の GPU なので、初代（M2）の Vision Pro でも同じ', en: 'The Mac’s GPU renders, so a first-generation (M2) Vision Pro plays the same' },
    ],
  },
  {
    title: { ja: 'ALVR のアプリ・Quest / Pico / Vision Pro', en: 'ALVR apps · Quest / Pico / Vision Pro' },
    tag: { ja: 'Quest・Pico はプレイヤーが確認', en: 'Quest and Pico: checked by players' },
    checked: false,
    points: [
      { ja: 'ALVR の配信部分（v20.14.1）を変更なしで組み込み', en: 'ALVR’s streaming core (v20.14.1), built in unchanged' },
      { ja: 'ゲーム音声・IPD・振動・ハンドトラッキングに対応', en: 'Game audio, IPD, vibration and hand tracking' },
      { ja: 'Wi-Fi のほか USB ケーブルでも接続できる', en: 'Over Wi-Fi, or over a USB cable' },
    ],
  },
];

/* ── Journey ─────────────────────────────────────────────── */

const JOURNEY: { date: string; title: L; body: L }[] = [
  {
    date: '10.10',
    title: { ja: 'Wine 版', en: 'The Wine version' },
    body: { ja: 'Wine・DXVK・MoltenVK で Alyx が Vision Pro に。約 30 fps', en: 'Alyx in a Vision Pro through Wine, DXVK and MoltenVK — about 30 fps' },
  },
  {
    date: '10.11',
    title: { ja: 'ネイティブ ARM 版', en: 'The native ARM version' },
    body: { ja: 'Valve の Linux ARM64 版 Alyx を macOS で直接。メニュー 60 fps', en: 'Valve’s Linux ARM64 Alyx straight on macOS — 60 fps in the menu' },
  },
  {
    date: '10.12',
    title: { ja: 'vxrt を書き始める', en: 'vxrt begins' },
    body: { ja: 'Wine が限界だったので、Windows ランタイムをゼロから', en: 'Wine was the bottleneck, so a Windows runtime from scratch' },
  },
  {
    date: '10.14',
    title: { ja: 'vxrt で Alyx をプレイ', en: 'Alyx played on vxrt' },
    body: { ja: 'Vision Pro の中で、ステージの切り替えとセーブを越えて', en: 'In a Vision Pro, through level changes and saves' },
  },
  {
    date: '10.16',
    title: { ja: 'ALVR へ', en: 'Into ALVR' },
    body: { ja: 'Quest や Pico へ配信する道', en: 'A path to Quest and Pico' },
  },
  {
    date: '10.17',
    title: { ja: '音・IPD・振動・USB', en: 'Audio, IPD, vibration, USB' },
    body: { ja: '光る描画の不具合も修正。自前の翻訳器で The Lab が動く', en: 'The flashing-surface bug fixed; The Lab runs on our own translator' },
  },
  {
    date: '10.20',
    title: { ja: 'Elite Dangerous', en: 'Elite Dangerous' },
    body: { ja: '公式ランチャー（.NET）からヘッドセットまで、Wine なしで', en: 'From its official launcher (.NET) to the headset, no Wine' },
  },
  {
    date: '10.25',
    title: { ja: 'No Man\u2019s Sky', en: 'No Man\u2019s Sky' },
    body: { ja: '本物の Windows 版 Steam を vxrt で動かして起動。Vulkan とテッセレーションを Metal へ', en: 'Started from the real Windows Steam client on vxrt; Vulkan and tessellation to Metal' },
  },
  {
    date: '10.26',
    title: { ja: 'Direct3D 12 へ', en: 'On to Direct3D 12' },
    body: { ja: 'No Man\u2019s Sky を 32.5 → 45.7 fps に。次の目標は Cyberpunk 2077', en: 'No Man\u2019s Sky from 32.5 to 45.7 fps; next target, Cyberpunk 2077' },
  },
];

/* ── The rest of the site ────────────────────────────────── */

const PROJECTS: { href: string; name: string; kind: L; body: L; external?: boolean; glyph: string }[] = [
  {
    href: '/vera/about/',
    name: 'Vera',
    glyph: 'V',
    kind: { ja: '嘘をつけない枠をつくる AI', en: 'An AI that builds a frame it cannot lie in' },
    body: {
      ja: '答えには出所を付け、分からなければ型つきで黙る。重みを持たない、規則と記録の AI。',
      en: 'Answers with a source, or stays silent with a type. An AI of rules and records, with no weights.',
    },
  },
  {
    href: '/cleanroom/',
    name: 'Cleanroom',
    glyph: 'C',
    kind: { ja: '開発者であり続けるための作業場', en: 'A workspace for staying the developer' },
    body: {
      ja: 'AI と作りながら、決定・理解・根拠・経験を自分の手に残す。',
      en: 'Build with AI while keeping your decisions, understanding, evidence and experience.',
    },
  },
  {
    href: '/apps/',
    name: 'Apps',
    glyph: 'A',
    kind: { ja: '口で遊ぶ iOS ゲーム', en: 'iOS games you play with your mouth' },
    body: { ja: 'パクパク釣り、MouthEat など。', en: 'PakuPaku Fishing, MouthEat and more.' },
  },
  {
    href: 'https://github.com/Ag3497120/SpatialCue',
    name: 'SpatialCue',
    glyph: 'S',
    external: true,
    kind: { ja: 'Apple Watch + AirPods', en: 'Apple Watch + AirPods' },
    body: {
      ja: 'Vision Pro のような手のひらジェスチャーを Apple Watch で。手を返すと時刻が聞こえる。',
      en: 'Vision Pro–style palm gestures on Apple Watch: flip your hand to hear the time.',
    },
  },
  {
    href: 'https://github.com/Ag3497120/deskbridge-vision-pro',
    name: 'DeskBridge',
    glyph: 'D',
    external: true,
    kind: { ja: 'Vision Pro の机上キーボード', en: 'A desk keyboard for Vision Pro' },
    body: {
      ja: 'Vision Pro と Mac 仮想ディスプレイのための、机の上の仮想キーボードとトラックパッド。',
      en: 'A virtual desk keyboard and trackpad for Apple Vision Pro and Mac Virtual Display.',
    },
  },
];

const MORE: { href: string; label: L }[] = [
  { href: '/try/', label: { ja: 'Vera を体験', en: 'Try Vera' } },
  { href: '/vera/', label: { ja: 'Vera の測定', en: 'Vera measurements' } },
  { href: '/writing/metamorphic-parser/', label: { ja: '読み物', en: 'Writing' } },
  { href: '/author/', label: { ja: '作者', en: 'Author' } },
];

/* ────────────────────────────────────────────────────────── */

export default function PortalHome() {
  const view = useHomeLang();
  const t = (o: L) => o[view];
  const ja = view === 'ja';

  return (
    <main lang={view} className="relative min-h-screen" style={{ overflowX: 'clip', color: 'var(--ink)' }}>
      <Navbar />
      <Hero t={t} ja={ja} />
      <Pipeline t={t} ja={ja} />
      <Runtime t={t} ja={ja} />
      <Numbers t={t} ja={ja} />
      <Headsets t={t} ja={ja} />
      <Journey t={t} ja={ja} />
      <Watch t={t} ja={ja} />
      <Downloads t={t} ja={ja} />
      <Projects t={t} ja={ja} />
      <VeraFooter view={view} tagline={{ ja: 'Mac の上に、まだないものを。', en: 'Building what the Mac does not have yet.' }} />
    </main>
  );
}

type P = { t: (o: L) => string; ja: boolean };

/* ── Hero ────────────────────────────────────────────────── */

function Hero({ t, ja }: P) {
  const reduce = useReducedMotion();
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end start'] });
  const lift = useTransform(scrollYProgress, [0, 1], [0, reduce ? 0 : -60]);
  const fade = useTransform(scrollYProgress, [0, 0.8], [1, reduce ? 1 : 0.2]);

  return (
    <section ref={ref} id="verantyx" className="relative px-5 sm:px-6 pt-28 sm:pt-36 pb-16 sm:pb-24 scroll-mt-16">
      <div className="vx-floor" aria-hidden />
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            'radial-gradient(ellipse 70% 60% at 15% 0%, rgba(var(--accent-rgb), 0.16), transparent 60%), radial-gradient(ellipse 50% 50% at 95% 30%, rgba(var(--vx-rgb), 0.14), transparent 60%)',
        }}
      />
      <motion.div style={{ y: lift, opacity: fade }} className="relative z-10 mx-auto w-full max-w-6xl grid gap-12 lg:gap-14 lg:grid-cols-[1.02fr_1fr] items-center">
        <div>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 rounded-full px-3 py-1 mb-6"
            style={{ border: '1px solid rgba(var(--vx-rgb), 0.35)', background: 'rgba(var(--vx-rgb), 0.07)' }}
          >
            <span className="vx-pulse" aria-hidden />
            <span style={{ fontSize: '0.72rem', letterSpacing: '0.14em', color: 'var(--vx)' }} className="uppercase font-semibold">
              {t({ ja: 'VeRantyx · 毎日開発中', en: 'VeRantyx · developed every day' })}
            </span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.06 }}
            className="font-display font-extrabold"
            style={{
              fontSize: ja ? 'clamp(1.95rem, 5.2vw, 3.5rem)' : 'clamp(2.3rem, 6.6vw, 4.4rem)',
              lineHeight: ja ? 1.22 : 1.02,
              letterSpacing: ja ? '-0.01em' : '-0.035em',
            }}
          >
            {ja ? (
              <>
                <span className="whitespace-nowrap">Mac で、</span>
                <span className="whitespace-nowrap">
                  <span className="vx-gradient">PC VR</span> を。
                </span>
                <br />
                <span style={{ color: 'var(--ink-2)' }}>
                  <span className="whitespace-nowrap">Vision Pro で、</span>
                  <span className="whitespace-nowrap">手で遊ぶ。</span>
                </span>
              </>
            ) : (
              <>
                <span className="vx-gradient">PC VR</span> from a Mac.
                <br />
                <span style={{ color: 'var(--ink-2)' }}>Played with your hands.</span>
              </>
            )}
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.16 }}
            className="mt-6 max-w-xl"
            style={{ color: 'var(--ink-3)', fontSize: 'clamp(1rem, 2.3vw, 1.12rem)', lineHeight: ja ? 1.95 : 1.7 }}
          >
            {t({
              ja: 'VeRantyx は、Half-Life: Alyx、Elite Dangerous、No Man\u2019s Sky などの Windows 版 SteamVR ゲームを Apple silicon の Mac で動かし、Apple Vision Pro（VeRantyx アプリか ALVR アプリ）や Quest・Pico（ALVR）に送ります。その土台の vxrt は、Wine を使わずにゼロから書いた Windows ランタイムです。',
              en: 'VeRantyx runs Windows SteamVR games such as Half-Life: Alyx, Elite Dangerous and No Man\u2019s Sky on an Apple silicon Mac and streams them to an Apple Vision Pro (the VeRantyx app or the ALVR app) or to Quest and Pico (ALVR). Underneath is vxrt, a Windows runtime written from scratch — no Wine.',
            })}
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, delay: 0.26 }}
            className="mt-8 flex flex-wrap items-center gap-3"
          >
            <a href={DMG} download className="vx-btn rounded-xl px-6 py-3 text-sm font-semibold">
              {t({ ja: `Mac 版をダウンロード（無料・${RELEASES[0].ver}）`, en: `Download for Mac (free, ${RELEASES[0].ver})` })}
            </a>
            <a
              href="#watch"
              className="rounded-xl px-6 py-3 text-sm font-semibold"
              style={{ textDecoration: 'none', color: 'var(--ink-2)', border: '1px solid var(--line-strong)' }}
            >
              {t({ ja: 'プレイ映像', en: 'Watch it play' })}
            </a>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.32 }}
            className="mt-5 max-w-xl rounded-xl px-4 py-3"
            style={{ border: '1px solid var(--line)', background: 'var(--surface)', fontSize: '0.8rem', lineHeight: ja ? 1.85 : 1.6, color: 'var(--ink-3)' }}
          >
            <p style={{ color: 'var(--ink-2)', fontWeight: 600 }}>
              {t({ ja: 'なぜこのサイトからダウンロードするのか', en: 'Why the download is on this site' })}
            </p>
            <p className="mt-1">
              {t({
                ja: '2026年10月24日から、GitHub のアカウントが自動判定で非表示（フラグ付き）になり、リポジトリとリリースがほかの人から見えなくなっています。理由は知らされていません。Reddit への投稿で数日のうちに訪問者が急に増えたことと、AI コーディングエージェントを使った開発で、1日に数十回という頻度で push や取得をしていたことが、自動の判定に引っかかったのだと考えています。規約に反することをした心当たりはありません。それまでの間、署名・公証済みの同じアプリをここに置いています。ソースは手元で開発を続けていて、公開できるようになったら戻します。',
                en: 'Since 24 October 2026 my GitHub account has been hidden (flagged) by an automatic check, so the repository and its releases are not visible to others; no reason was given. My guess is the combination of a sudden jump in visitors after Reddit posts and very frequent pushes and fetches, dozens a day, from development with AI coding agents. Nothing against GitHub’s terms was done. Until it is back, the same signed and notarized app is served here; development continues and the source will be public again.',
              })}
            </p>
            <p className="mt-1 break-all" style={{ fontSize: '0.72rem', color: 'var(--ink-4)' }}>SHA-256: {DMG_SHA256}</p>
          </motion.div>

          <motion.dl
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.38 }}
            className="mt-10 grid grid-cols-3 gap-3 max-w-lg"
          >
            {HERO_FACTS.map((f) => (
              <div key={f.value.en} className="rounded-xl px-3 py-3" style={{ border: '1px solid var(--line)', background: 'var(--surface)' }}>
                <dt className="font-display font-bold whitespace-nowrap" style={{ fontSize: 'clamp(0.95rem, 3.4vw, 1.5rem)', color: 'var(--ink)' }}>
                  {t(f.value)}
                </dt>
                <dd style={{ color: 'var(--ink-4)', fontSize: '0.72rem', lineHeight: 1.5, marginTop: 2 }}>{t(f.label)}</dd>
              </div>
            ))}
          </motion.dl>
        </div>

        <Visor t={t} />
      </motion.div>
    </section>
  );
}

/* The latest recordings, one per game, in a frame that leans toward the pointer like a headset's view. */
const CLIPS = [
  {
    key: 'nms',
    name: "No Man's Sky",
    src: '/media/verantyx/nms-loop.mp4',
    poster: '/media/verantyx/nms-frame.jpg',
    rec: '2026-10-25',
    chips: [
      { ja: 'ALVR アプリ', en: 'ALVR app' },
      { ja: '本物の Windows 版 Steam', en: 'real Windows Steam client' },
    ],
    label: {
      ja: 'No Man\'s Sky を Apple Vision Pro の ALVR アプリで VR プレイした録画（音声なし）。最初に部屋と Mac が映る',
      en: "A recording of No Man's Sky in VR in Apple Vision Pro through the ALVR app (no sound). The room and the Mac come first",
    },
    note: {
      ja: 'MacBook Pro（M1 Max）。Windows 版 Steam から起動し、Vulkan を Metal へ。ヘッドセットではまだ 25〜40 fps',
      en: 'MacBook Pro (M1 Max). Launched from the Windows Steam client, Vulkan to Metal. Still 25–40 fps in the headset',
    },
  },
  {
    key: 'elite',
    name: 'Elite Dangerous',
    src: '/media/verantyx/elite-loop.mp4',
    poster: '/media/verantyx/elite-frame.jpg',
    rec: '2026-10-24',
    chips: [
      { ja: 'VeRantyx アプリ · 3K', en: 'VeRantyx app · 3K' },
      { ja: '公式ランチャー', en: 'official launcher' },
    ],
    label: {
      ja: 'Elite Dangerous を Apple Vision Pro で VR プレイした録画（音声なし）。最初に部屋と Mac が映る',
      en: 'A recording of Elite Dangerous in VR in Apple Vision Pro (no sound). The room and the Mac come first',
    },
    note: {
      ja: 'MacBook Pro（M1 Max）・3K 設定。Odyssey、公式ランチャーと本物の SteamVR',
      en: 'MacBook Pro (M1 Max) · 3K preset. Odyssey, the official launcher and the real SteamVR',
    },
  },
  {
    key: 'alyx',
    name: 'Half-Life: Alyx',
    src: '/media/verantyx/alyx-loop.mp4',
    poster: '/media/verantyx/alyx-frame.jpg',
    rec: '2026-10-22',
    chips: [
      { ja: 'ALVR アプリ', en: 'ALVR app' },
      { ja: 'Direct3D 11 → Metal', en: 'Direct3D 11 → Metal' },
    ],
    label: {
      ja: 'Half-Life: Alyx を Apple Vision Pro の ALVR アプリで VR プレイした録画（音声なし）。最初に部屋と Mac が映る',
      en: 'A recording of Half-Life: Alyx in VR in Apple Vision Pro through the ALVR app (no sound). The room and the Mac come first',
    },
    note: {
      ja: 'MacBook Pro（M1 Max）。vxrt 版、Direct3D 11 を一段で Metal へ',
      en: 'MacBook Pro (M1 Max). The vxrt version, Direct3D 11 to Metal in one step',
    },
  },
] as const;

function Visor({ t }: { t: (o: L) => string }) {
  const reduce = useReducedMotion();
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const rx = useSpring(useTransform(my, [-0.5, 0.5], [7, -7]), { stiffness: 120, damping: 18 });
  const ry = useSpring(useTransform(mx, [-0.5, 0.5], [-9, 9]), { stiffness: 120, damping: 18 });
  const videoRef = useRef<HTMLVideoElement>(null);
  const [pick, setPick] = useState(0);
  const clip = CLIPS[pick];

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (reduce) v.pause();
    else v.play().catch(() => {});
  }, [reduce, pick]);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.8, delay: 0.12 }}
      style={{ perspective: 1200 }}
      onPointerMove={(e) => {
        if (reduce || e.pointerType !== 'mouse') return;
        const r = e.currentTarget.getBoundingClientRect();
        mx.set((e.clientX - r.left) / r.width - 0.5);
        my.set((e.clientY - r.top) / r.height - 0.5);
      }}
      onPointerLeave={() => {
        mx.set(0);
        my.set(0);
      }}
    >
      <div className="mb-3 flex flex-wrap justify-center gap-2" role="tablist" aria-label={t({ ja: '録画するゲーム', en: 'Recordings' })}>
        {CLIPS.map((c, i) => (
          <button
            key={c.key}
            type="button"
            role="tab"
            aria-selected={i === pick}
            onClick={() => setPick(i)}
            className="vx-chip"
            style={{ cursor: 'pointer', opacity: i === pick ? 1 : 0.55, outline: i === pick ? '1px solid currentColor' : 'none' }}
          >
            {c.name}
          </button>
        ))}
      </div>
      <motion.figure className="vx-visor relative" style={{ rotateX: rx, rotateY: ry, transformStyle: 'preserve-3d' }}>
        <video
          key={clip.key}
          ref={videoRef}
          className="block w-full h-auto rounded-[22px]"
          src={clip.src}
          poster={clip.poster}
          muted
          loop
          playsInline
          autoPlay
          preload="metadata"
          aria-label={t(clip.label)}
        />
        <div className="vx-scan" aria-hidden />
        <div className="absolute left-3 top-3 flex gap-2" style={{ transform: 'translateZ(40px)' }}>
          <span className="vx-chip">● REC {clip.rec}</span>
        </div>
        <div className="absolute right-3 bottom-3 flex flex-col items-end gap-2" style={{ transform: 'translateZ(50px)' }}>
          {clip.chips.map((c) => (
            <span key={c.en} className="vx-chip">{t(c)}</span>
          ))}
        </div>
        <figcaption className="sr-only">{t(clip.note)}</figcaption>
      </motion.figure>
      <p className="mt-4 text-center" style={{ color: 'var(--ink-4)', fontSize: '0.76rem' }}>
        {t(clip.note)}
      </p>
    </motion.div>
  );
}

/* ── Pipeline ────────────────────────────────────────────── */

function Pipeline({ t, ja }: P) {
  const [active, setActive] = useState(1);
  const stage = STAGES[active];
  return (
    <Section id="how" label={t({ ja: '仕組み', en: 'How it works' })} wide>
      <H2 ja={ja}>{t({ ja: '1 フレームが、目に届くまで', en: 'How one frame reaches your eyes' })}</H2>
      <Body ja={ja} className="mb-8">
        {t({
          ja: '映像は左から右へ、頭と手の動きは右から左へ。各段を押すと中身が見えます。',
          en: 'Frames flow left to right; head and hand poses flow back. Tap a stage to see inside.',
        })}
      </Body>

      <div className="vx-rail relative">
        <div className="vx-flow vx-flow-frames" aria-hidden />
        <div className="vx-flow vx-flow-poses" aria-hidden />
        <ol className="relative grid gap-3 md:grid-cols-5 md:gap-4" role="tablist" aria-label={t({ ja: '処理の段', en: 'Stages' })}>
          {STAGES.map((s, i) => {
            const on = i === active;
            return (
              <li key={s.key} role="presentation">
                <button
                  type="button"
                  role="tab"
                  aria-selected={on}
                  aria-controls="vx-stage-panel"
                  onClick={() => setActive(i)}
                  onMouseEnter={() => setActive(i)}
                  className="vx-node w-full text-left rounded-2xl px-4 py-4"
                  data-on={on || undefined}
                >
                  <span className="font-mono" style={{ fontSize: '0.66rem', color: 'var(--ink-4)', letterSpacing: '0.12em' }}>
                    0{i + 1}
                  </span>
                  <span className="block font-display font-bold mt-1" style={{ fontSize: '1.02rem', color: on ? 'var(--ink)' : 'var(--ink-2)' }}>
                    {s.name}
                  </span>
                  <span className="block mt-1" style={{ fontSize: '0.74rem', color: 'var(--ink-4)', lineHeight: 1.5 }}>
                    {t(s.where)}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      </div>

      <motion.div
        key={stage.key}
        id="vx-stage-panel"
        role="tabpanel"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="mt-5 rounded-2xl p-5 sm:p-6"
        style={{ border: '1px solid rgba(var(--vx-rgb), 0.3)', background: 'rgba(var(--vx-rgb), 0.05)' }}
      >
        <div className="flex items-center gap-3 mb-2">
          <span className="font-display font-bold" style={{ color: 'var(--vx)' }}>
            {stage.name}
          </span>
          <span style={{ fontSize: '0.74rem', color: 'var(--ink-4)' }}>{t(stage.where)}</span>
        </div>
        <p style={{ color: 'var(--ink-2)', fontSize: '0.95rem', lineHeight: ja ? 1.95 : 1.7 }}>{t(stage.body)}</p>
      </motion.div>

      <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2" style={{ fontSize: '0.76rem', color: 'var(--ink-4)' }}>
        <span className="inline-flex items-center gap-2">
          <span className="vx-key vx-key-frames" aria-hidden /> {t({ ja: '映像（左右の目・HEVC）', en: 'frames (both eyes, HEVC)' })}
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="vx-key vx-key-poses" aria-hidden /> {t({ ja: '頭と手の姿勢', en: 'head and hand poses' })}
        </span>
      </div>
    </Section>
  );
}

/* ── Runtime ─────────────────────────────────────────────── */

function Runtime({ t, ja }: P) {
  return (
    <Section id="vxrt" label="vxrt" wide>
      <div className="grid gap-10 lg:grid-cols-[0.9fr_1.1fr] items-start">
        <div className="lg:sticky lg:top-24">
          <H2 ja={ja}>{t({ ja: 'Wine を使わない、Windows ランタイム', en: 'A Windows runtime without Wine' })}</H2>
          <Body ja={ja}>
            {t({
              ja: 'vxrt（VXRuntime）は Apple silicon の Mac のためにゼロから書いた Windows 互換層です。Windows のプログラムを読み込み、呼ばれた API を macOS の機能で直接実装します。2026 年 10 月 12 日に書き始めました。Wine では CPU が詰まり 30 fps 前後で止まっていたからです。',
              en: 'vxrt (VXRuntime) is a Windows compatibility layer written from scratch for Apple silicon Macs. It loads Windows programs and implements the API they call directly with macOS. It was started on 12 October 2026, because under Wine the CPU was the wall at about 30 fps.',
            })}
          </Body>
          <Body ja={ja} className="mt-4">
            {t({
              ja: 'すべての層が手の中にあるので、シェーダの翻訳ひとつまで直せます。フェンスや草が一色に光る不具合も、定数バッファの読み方を Direct3D に合わせて直しました。',
              en: 'With every layer in hand, it can be fixed down to a single shader translation — as with fences and plants that flashed one colour, fixed by reading constant buffers the way Direct3D does.',
            })}
          </Body>
          <p className="mt-5" style={{ fontSize: '0.8rem', color: 'var(--ink-4)', lineHeight: 1.7 }}>
            {t({
              ja: 'まだないもの: Direct3D 12（実装中。まだ動くゲームはありません）。アンチチートの回避は行いません。',
              en: 'Not there yet: Direct3D 12 (in progress; no game runs on it yet). Anti-cheat is never worked around.',
            })}
          </p>
        </div>

        <ul className="vx-stack flex flex-col gap-2.5" aria-label={t({ ja: 'vxrt が実装しているもの', en: 'What vxrt implements' })}>
          {LAYERS.map((l, i) => (
            <motion.li
              key={l.name.en}
              initial={{ opacity: 0, x: 24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: '-30px' }}
              transition={{ duration: 0.45, delay: i * 0.05 }}
              className="vx-layer rounded-xl px-4 py-3.5"
              data-state={l.state}
            >
              <div className="flex items-center justify-between gap-3">
                <span className="font-semibold" style={{ fontSize: '0.95rem', color: 'var(--ink)' }}>
                  {t(l.name)}
                </span>
                <span className="vx-state shrink-0" data-state={l.state}>
                  {l.state === 'works' ? t({ ja: '動作', en: 'works' }) : t({ ja: '開発中', en: 'in progress' })}
                </span>
              </div>
              <p className="mt-1" style={{ fontSize: '0.82rem', color: 'var(--ink-3)', lineHeight: ja ? 1.8 : 1.55 }}>
                {t(l.detail)}
              </p>
            </motion.li>
          ))}
          <li className="mt-1 pl-1" style={{ fontSize: '0.74rem', color: 'var(--ink-4)' }}>
            {t({ ja: '↓ macOS（Metal・AppKit・Core Audio）', en: '↓ macOS (Metal · AppKit · Core Audio)' })}
          </li>
        </ul>
      </div>
    </Section>
  );
}

/* ── Numbers ─────────────────────────────────────────────── */

function CountUp({ to, decimals = 0, suffix = '' }: { to: number; decimals?: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const seen = useInView(ref, { once: true, margin: '-40px' });
  const reduce = useReducedMotion();
  const [v, setV] = useState(reduce ? to : 0);
  useEffect(() => {
    if (!seen) return;
    if (reduce) {
      setV(to);
      return;
    }
    const c = animate(0, to, { duration: 1.4, ease: [0.16, 1, 0.3, 1], onUpdate: setV });
    return () => c.stop();
  }, [seen, to, reduce]);
  return (
    <span ref={ref}>
      {v.toFixed(decimals)}
      {suffix}
    </span>
  );
}

function Bar({ value, max, tone }: { value: number; max: number; tone: 'vx' | 'old' }) {
  return (
    <div className="h-2.5 rounded-full overflow-hidden" style={{ background: 'var(--line)' }}>
      <motion.div
        className="h-full rounded-full"
        initial={{ width: 0 }}
        whileInView={{ width: `${(value / max) * 100}%` }}
        viewport={{ once: true, margin: '-40px' }}
        transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
        style={{
          background:
            tone === 'vx'
              ? 'linear-gradient(90deg, var(--accent), var(--vx))'
              : 'linear-gradient(90deg, rgba(148,163,184,0.5), rgba(148,163,184,0.8))',
        }}
      />
    </div>
  );
}

function Numbers({ t, ja }: P) {
  return (
    <Section id="numbers" label={t({ ja: '実測', en: 'Measured' })} wide>
      <H2 ja={ja}>{t({ ja: '同じ Mac で、倍のフレーム', en: 'Twice the frames on the same Mac' })}</H2>
      <Body ja={ja} className="mb-8">
        {t({
          ja: 'MacBook Pro（M1 Max）での Half-Life: Alyx。どの値も記録された実行から取っています。',
          en: 'Half-Life: Alyx on a MacBook Pro (M1 Max). Every value is taken from a recorded run.',
        })}
      </Body>

      <div className="grid gap-5 lg:grid-cols-2">
        <div className="rounded-2xl p-5 sm:p-7" style={{ border: '1px solid var(--line)', background: 'var(--surface)' }}>
          <div className="flex items-baseline justify-between mb-2">
            <span className="font-semibold" style={{ color: 'var(--ink)' }}>
              vxrt
            </span>
            <span className="font-display font-extrabold vx-gradient" style={{ fontSize: 'clamp(2rem, 6vw, 2.8rem)' }}>
              <CountUp to={60} /> <span style={{ fontSize: '0.45em' }}>fps</span>
            </span>
          </div>
          <Bar value={60} max={60} tone="vx" />
          <div className="flex items-baseline justify-between mt-6 mb-2">
            <span className="font-semibold" style={{ color: 'var(--ink-3)' }}>
              Wine 11 · DXVK · MoltenVK
            </span>
            <span className="font-display font-bold" style={{ fontSize: 'clamp(1.4rem, 4.5vw, 1.9rem)', color: 'var(--ink-3)' }}>
              {t({ ja: '約 ', en: '≈ ' })}
              <CountUp to={30} /> <span style={{ fontSize: '0.5em' }}>fps</span>
            </span>
          </div>
          <Bar value={30} max={60} tone="old" />
          <p className="mt-5" style={{ fontSize: '0.78rem', color: 'var(--ink-4)', lineHeight: 1.7 }}>
            {t({
              ja: 'vxrt: 第 1 章、3K 設定（片目 1440×1620）、2026-10-17 の実行。Wine 版: ゲーム中の約 30 fps、2026-10-10。',
              en: 'vxrt: chapter 1, 3K preset (1440×1620 per eye), run of 2026-10-17. Wine version: about 30 fps in game, 2026-10-10.',
            })}
          </p>
        </div>

        <div className="rounded-2xl p-5 sm:p-7" style={{ border: '1px solid var(--line)', background: 'var(--surface)' }}>
          <div className="font-semibold mb-4" style={{ color: 'var(--ink)' }}>
            {t({ ja: 'Wine なしで動くゲーム', en: 'Games running without Wine' })}
          </div>
          <ul className="flex flex-col gap-4">
            {RUNS.map((r) => (
              <li key={r.game}>
                <div className="flex items-baseline justify-between gap-3 mb-1.5">
                  <span className="min-w-0" style={{ color: 'var(--ink-2)', fontSize: '0.92rem' }}>
                    {r.game} <span style={{ color: 'var(--ink-4)', fontSize: '0.76rem' }}>· {t(r.note)}</span>
                  </span>
                  <span className="font-mono font-semibold shrink-0" style={{ color: 'var(--ink)' }}>
                    <CountUp to={r.fps} decimals={1} />
                  </span>
                </div>
                <Bar value={r.fps} max={60} tone="vx" />
              </li>
            ))}
          </ul>
          <p className="mt-5" style={{ fontSize: '0.78rem', color: 'var(--ink-4)', lineHeight: 1.7 }}>
            {t({
              ja: 'Earth VR と The Lab は 2026-10-15、Wine を一度も使わない設定での実行（fps の中央値）。No Man\u2019s Sky は 2026-10-26（M1 Max、惑星の上）、Elite Dangerous は 2026-10-20 の約 33 fps。No Man\u2019s Sky のヘッドセットでの fps はまだ測っていません。',
              en: 'Earth VR and The Lab: runs of 2026-10-15 with Wine never used (median fps). No Man\u2019s Sky: 2026-10-26 (M1 Max, on a planet); Elite Dangerous: about 33 fps, 2026-10-20. No Man\u2019s Sky in a headset is not measured yet.',
            })}
          </p>
        </div>
      </div>
    </Section>
  );
}

/* ── Headsets ────────────────────────────────────────────── */

function Headsets({ t, ja }: P) {
  return (
    <Section id="headsets" label={t({ ja: 'ヘッドセット', en: 'Headsets' })} wide>
      <H2 ja={ja}>{t({ ja: 'VR コントローラがなくても', en: 'No VR controllers required' })}</H2>
      <div className="grid gap-5 md:grid-cols-2">
        {HEADSETS.map((h, i) => (
          <motion.div
            key={h.title.en}
            initial={{ opacity: 0, y: 18 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-30px' }}
            transition={{ duration: 0.5, delay: i * 0.08 }}
            className="vx-card rounded-2xl p-5 sm:p-7"
          >
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <span className="font-display font-bold" style={{ fontSize: '1.08rem', color: 'var(--ink)' }}>
                {t(h.title)}
              </span>
              <span className="vx-state" data-state={h.checked ? 'works' : 'growing'}>
                {t(h.tag)}
              </span>
            </div>
            <ul className="flex flex-col gap-2.5">
              {h.points.map((p) => (
                <li key={p.en} className="flex gap-3" style={{ color: 'var(--ink-3)', fontSize: '0.9rem', lineHeight: ja ? 1.85 : 1.6 }}>
                  <span aria-hidden className="mt-2.5 shrink-0 rounded-full" style={{ width: 5, height: 5, background: 'var(--vx)' }} />
                  {t(p)}
                </li>
              ))}
            </ul>
          </motion.div>
        ))}
      </div>
      <p className="mt-5" style={{ fontSize: '0.84rem', color: 'var(--ink-3)' }}>
        {t({
          ja: 'Quest や Pico をお持ちなら、動作報告が大きな助けになります → ',
          en: 'Have a Quest or Pico? A report helps a lot → ',
        })}
        <a
          href={`${REPO}/issues/new/choose`}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: 'var(--vx)', textDecoration: 'none' }}
        >
          {t({ ja: '報告フォーム', en: 'report form' })}
        </a>
      </p>
    </Section>
  );
}

/* ── Journey ─────────────────────────────────────────────── */

function Journey({ t, ja }: P) {
  const ref = useRef<HTMLOListElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start 80%', 'end 60%'] });
  const grow = useTransform(scrollYProgress, [0, 1], ['0%', '100%']);
  return (
    <Section id="journey" label={t({ ja: '経緯', en: 'Journey' })} wide>
      <H2 ja={ja}>{t({ ja: '3 つのやり方を経て', en: 'Three approaches so far' })}</H2>
      <ol ref={ref} className="relative mt-2 pl-7 sm:pl-9">
        <span className="absolute left-[9px] sm:left-[13px] top-1 bottom-1 w-px" style={{ background: 'var(--line)' }} aria-hidden />
        <motion.span
          className="absolute left-[9px] sm:left-[13px] top-1 w-px"
          style={{ height: grow, background: 'linear-gradient(var(--accent), var(--vx))' }}
          aria-hidden
        />
        {JOURNEY.map((j) => (
          <motion.li
            key={j.date}
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-40px' }}
            transition={{ duration: 0.45 }}
            className="relative pb-7 last:pb-0"
          >
            <span
              className="absolute -left-7 sm:-left-9 top-1.5 rounded-full"
              style={{
                width: 11,
                height: 11,
                marginLeft: 4,
                background: 'var(--bg)',
                border: '2px solid var(--vx)',
                boxShadow: '0 0 0 4px rgba(var(--vx-rgb), 0.12)',
              }}
              aria-hidden
            />
            <div className="flex flex-wrap items-baseline gap-x-3">
              <span className="font-mono" style={{ fontSize: '0.78rem', color: 'var(--vx)' }}>
                2026.{j.date}
              </span>
              <span className="font-semibold" style={{ color: 'var(--ink)', fontSize: '0.98rem' }}>
                {t(j.title)}
              </span>
            </div>
            <p className="mt-1" style={{ color: 'var(--ink-3)', fontSize: '0.88rem', lineHeight: ja ? 1.85 : 1.6 }}>
              {t(j.body)}
            </p>
          </motion.li>
        ))}
      </ol>
    </Section>
  );
}

/* ── Watch ───────────────────────────────────────────────── */

/* YouTube loads only when the visitor asks for it (no cookies, no requests before the click). */
function YouTube({ id, title, poster, contain }: { id: string; title: string; poster: string; contain?: boolean }) {
  const [on, setOn] = useState(false);
  return (
    <div className="relative aspect-video rounded-2xl overflow-hidden" style={{ border: '1px solid var(--line)', background: '#000' }}>
      {on ? (
        <iframe
          className="absolute inset-0 w-full h-full"
          src={`https://www.youtube-nocookie.com/embed/${id}?autoplay=1&rel=0`}
          title={title}
          allow="autoplay; encrypted-media; picture-in-picture; fullscreen"
          allowFullScreen
        />
      ) : (
        <button type="button" onClick={() => setOn(true)} className="vx-play absolute inset-0 w-full h-full" aria-label={title}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={poster} alt="" className={`absolute inset-0 w-full h-full ${contain ? 'object-contain' : 'object-cover'}`} loading="lazy" />
          <span className="vx-play-btn" aria-hidden>
            <svg viewBox="0 0 24 24" width="26" height="26" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
          </span>
          <span className="absolute left-4 bottom-3 right-4 text-left font-semibold" style={{ color: '#fff', fontSize: '0.9rem', textShadow: '0 1px 8px rgba(0,0,0,0.7)' }}>
            {title}
          </span>
        </button>
      )}
    </div>
  );
}

function Watch({ t, ja }: P) {
  return (
    <Section id="watch" label={t({ ja: '映像', en: 'Watch' })} wide>
      <H2 ja={ja}>{t({ ja: '編集なしの記録', en: 'Unedited recordings' })}</H2>
      <div className="grid gap-5 md:grid-cols-2">
        <div>
          <YouTube
            id={VIDEO_FULL}
            poster="/media/verantyx/vxrt-frame.jpg"
            title={t({ ja: 'vxrt 版・Vision Pro で 4 分', en: 'vxrt version · 4 minutes in a Vision Pro' })}
          />
          <p className="mt-2" style={{ fontSize: '0.8rem', color: 'var(--ink-4)', lineHeight: 1.7 }}>
            {t({
              ja: 'メインメニューから隔離区域、ステージを越えてピストル、指の動き、リストポケット、リロードまで。',
              en: 'From the main menu into the quarantine zone, through a level change to the pistol, fingers, the wrist pocket and reloading.',
            })}
          </p>
        </div>
        <div>
          <YouTube
            id={VIDEO_MACS}
            poster="/media/verantyx/macs-frame.jpg"
            contain
            title={t({ ja: 'MacBook Pro と MacBook Air を並べて', en: 'MacBook Pro and MacBook Air side by side' })}
          />
          <p className="mt-2" style={{ fontSize: '0.8rem', color: 'var(--ink-4)', lineHeight: 1.7 }}>
            {t({
              ja: 'M1 Max（10 コア CPU・32 コア GPU）と M5 の MacBook Air。速さ比べではなく、滑らかさの違いを見る映像です。',
              en: 'An M1 Max (10-core CPU, 32-core GPU) and an M5 MacBook Air. Not a speed test — it shows how smooth each one is.',
            })}
          </p>
        </div>
      </div>
      <p className="mt-6" style={{ fontSize: '0.74rem', color: 'var(--ink-4)', lineHeight: 1.7 }}>
        {t({
          ja: 'VeRantyx は Valve・Apple・Meta とは関係のない個人のプロジェクトです。ゲームと SteamVR は各自の Steam アカウントから入れます。',
          en: 'VeRantyx is a personal project, not affiliated with Valve, Apple or Meta. The games and SteamVR come from your own Steam account.',
        })}
      </p>
    </Section>
  );
}


/* ── Downloads: the stable version and the earlier ones ─────── */

function Downloads({ t, ja }: P) {
  return (
    <Section id="download" label={t({ ja: 'ダウンロード', en: 'Download' })} wide>
      <H2 ja={ja}>{t({ ja: 'VeRantyx for Mac のダウンロード', en: 'Download VeRantyx for Mac' })}</H2>
      <p className="mb-5 max-w-2xl" style={{ color: 'var(--ink-3)', fontSize: '0.9rem', lineHeight: ja ? 1.9 : 1.65 }}>
        {t({
          ja: 'どれも無料で、Apple の署名と公証が済んでいます。普段は一番上の安定版を使ってください。新しい版で問題が出たときのために、以前の版も置いています。',
          en: 'All free, signed and notarized by Apple. Use the stable version at the top; earlier versions stay here in case a new one causes trouble for you.',
        })}
      </p>
      <ul className="grid gap-3">
        {RELEASES.map((r, i) => (
          <li key={r.ver} className="rounded-xl px-4 py-3" style={{ border: `1px solid ${i === 0 ? 'var(--vx)' : 'var(--line)'}`, background: 'var(--surface)' }}>
            <div className="flex flex-wrap items-center gap-3">
              <span className="font-display font-bold" style={{ fontSize: '1.05rem', color: 'var(--ink)' }}>{r.ver}</span>
              {i === 0 && (
                <span className="rounded-full px-2.5 py-0.5" style={{ fontSize: '0.72rem', color: 'var(--vx)', border: '1px solid var(--vx)' }}>
                  {t({ ja: '安定版', en: 'Stable' })}
                </span>
              )}
              <span style={{ fontSize: '0.78rem', color: 'var(--ink-4)' }}>{r.date}</span>
              <a href={`/download/VeRantyx-${r.ver}.dmg`} download className={i === 0 ? 'vx-btn rounded-lg px-4 py-1.5 text-sm font-semibold' : 'rounded-lg px-4 py-1.5 text-sm font-semibold'}
                style={i === 0 ? { marginLeft: 'auto' } : { marginLeft: 'auto', textDecoration: 'none', color: 'var(--ink-2)', border: '1px solid var(--line-strong)' }}>
                {t({ ja: 'ダウンロード', en: 'Download' })}
              </a>
            </div>
            <p className="mt-1.5" style={{ fontSize: '0.84rem', color: 'var(--ink-3)', lineHeight: ja ? 1.85 : 1.6 }}>{t(r.notes)}</p>
            {r.lgpl && (
              <p className="mt-1" style={{ fontSize: '0.75rem', color: 'var(--ink-4)', lineHeight: 1.6 }}>
                {t({
                  ja: 'この版の VXRuntime には、Wine に由来する約 120 行（x64 の例外の巻き戻し処理）が含まれます。その部分は LGPL-2.1 以降で、著作権は Wine の作者にあります。0.2.1 で置き換え済みです。',
                  en: 'This version\u2019s VXRuntime contains about 120 lines derived from Wine (the x64 exception unwinder), under LGPL-2.1-or-later with Wine\u2019s copyright; replaced in 0.2.1.',
                })}
              </p>
            )}
            <p className="mt-1 break-all" style={{ fontSize: '0.7rem', color: 'var(--ink-4)' }}>SHA-256: {r.sha256}</p>
          </li>
        ))}
      </ul>

      <div id="source" className="mt-8 rounded-xl px-4 py-4" style={{ border: '1px solid var(--line)', background: 'var(--surface)' }}>
        <h3 className="font-display font-bold" style={{ fontSize: '1.05rem', color: 'var(--ink)' }}>{t({ ja: 'ソースコード（MIT）', en: 'Source code (MIT)' })}</h3>
        <p className="mt-1.5" style={{ fontSize: '0.84rem', color: 'var(--ink-3)', lineHeight: ja ? 1.85 : 1.6 }}>
          {t({
            ja: 'GitHub のリポジトリが見えない間、0.3.1 のソースをここに置いています。中身は GitHub の main（62e0488）と同じで、容量のため docs/media の動画だけを外しています。Wine のコードは含まれていません。git の bundle は版ごとのスナップショットのコミットで、git clone VeRantyx-0.3.1-source.bundle -b source-snapshot で取り出せます（0.3.0 の bundle を持っていれば git pull で更新できます）。',
            en: 'While the GitHub repository is hidden, the 0.3.1 source is here. It matches GitHub’s main (62e0488) except the videos in docs/media, left out for size. It contains no Wine code. The git bundle holds one snapshot commit per version: git clone VeRantyx-0.3.1-source.bundle -b source-snapshot (or git pull it into a clone of the 0.3.0 bundle)',
          })}
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          <a href="/source/VeRantyx-0.3.1-source.tar.gz" download className="rounded-lg px-4 py-1.5 text-sm font-semibold" style={{ textDecoration: 'none', color: 'var(--ink-2)', border: '1px solid var(--line-strong)' }}>tar.gz · 4.3 MB</a>
          <a href="/source/VeRantyx-0.3.1-source.bundle" download className="rounded-lg px-4 py-1.5 text-sm font-semibold" style={{ textDecoration: 'none', color: 'var(--ink-2)', border: '1px solid var(--line-strong)' }}>git bundle · 4.4 MB</a>
          <a href="/source/SHA256SUMS" className="rounded-lg px-4 py-1.5 text-sm" style={{ textDecoration: 'none', color: 'var(--ink-3)' }}>SHA256SUMS</a>
        </div>
      </div>
    </Section>
  );
}

/* ── Projects ────────────────────────────────────────────── */

function Projects({ t, ja }: P) {
  return (
    <Section id="projects" label={t({ ja: 'ほかのプロジェクト', en: 'Other projects' })} wide>
      <H2 ja={ja}>{t({ ja: 'Verantyx のプロジェクト', en: 'Projects at Verantyx' })}</H2>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <ProjectCard
          href={DMG}
          name="VeRantyx"
          glyph="VR"
          kind={t({ ja: 'Mac で PC VR', en: 'PC VR from a Mac' })}
          body={t({
            ja: 'このページの主役。vxrt、ネイティブ ARM 版、Wine 版のすべてのソース。',
            en: 'This page’s subject: the source of vxrt, the native ARM version and the Wine version.',
          })}
          cta={t({ ja: 'ダウンロード', en: 'Download' })}
          highlight
        />
        {PROJECTS.map((p) => (
          <ProjectCard key={p.name} href={p.href} external={p.external} name={p.name} glyph={p.glyph} kind={t(p.kind)} body={t(p.body)} cta={p.external ? 'GitHub' : t({ ja: '開く', en: 'Open' })} />
        ))}
      </div>
      <nav className="mt-6 flex flex-wrap gap-2" aria-label={t({ ja: 'そのほかのページ', en: 'More pages' })}>
        {MORE.map((m) => (
          <a key={m.href} href={m.href} className="vx-pill rounded-full px-3.5 py-1.5" style={{ fontSize: '0.8rem' }}>
            {t(m.label)}
          </a>
        ))}
      </nav>
    </Section>
  );
}

function ProjectCard({
  href,
  external,
  name,
  glyph,
  kind,
  body,
  cta,
  highlight,
}: {
  href: string;
  external?: boolean;
  name: string;
  glyph: string;
  kind: string;
  body: string;
  cta: string;
  highlight?: boolean;
}): ReactNode {
  return (
    <motion.a
      href={href}
      {...(external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
      whileHover={{ y: -4 }}
      transition={{ type: 'spring', stiffness: 300, damping: 22 }}
      className="vx-card group rounded-2xl p-5 flex flex-col"
      data-highlight={highlight || undefined}
      style={{ textDecoration: 'none' }}
    >
      <div className="flex items-center gap-3 mb-3">
        <span className="vx-glyph font-display font-bold">{glyph}</span>
        <div className="min-w-0">
          <div className="font-display font-bold" style={{ color: 'var(--ink)', fontSize: '1.02rem' }}>
            {name}
          </div>
          <div style={{ color: 'var(--ink-4)', fontSize: '0.74rem' }}>{kind}</div>
        </div>
      </div>
      <p style={{ color: 'var(--ink-3)', fontSize: '0.86rem', lineHeight: 1.75 }} className="flex-1">
        {body}
      </p>
      <span className="mt-4 inline-flex items-center gap-1.5" style={{ color: 'var(--vx)', fontSize: '0.8rem', fontWeight: 600 }}>
        {cta}
        <span className="transition-transform group-hover:translate-x-1" aria-hidden>
          →
        </span>
      </span>
    </motion.a>
  );
}
