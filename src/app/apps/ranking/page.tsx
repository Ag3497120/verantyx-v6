"use client";

import { useState, useEffect } from "react";
import { getTopRanking, type RankEntry } from "@/lib/supabase";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { PageHero } from "@/components/CinematicSection";
import { useLanguage } from "@/lib/i18n";

const getGames = (lang: 'ja' | 'en') => [
  {
    id: "pakupaku",
    label: lang === 'ja' ? "🎣 パクパク釣り" : "🎣 PakuPaku Fishing",
    modes: [{ id: "timeattack", label: lang === 'ja' ? "⏱️ タイムアタック" : "⏱️ Time Attack" }],
  },
  {
    id: "moutheat",
    label: "😋 MouthEat",
    modes: [
      { id: "eat_timeattack", label: lang === 'ja' ? "⏱️ タイムアタック" : "⏱️ Time Attack" },
      { id: "eat_survival", label: lang === 'ja' ? "❤️ サバイバル" : "❤️ Survival" },
      { id: "eat_combo", label: lang === 'ja' ? "🔥 コンボチャレンジ" : "🔥 Combo Challenge" },
      { id: "eat_judgment", label: lang === 'ja' ? "🧠 ジャッジメント" : "🧠 Judgment" },
    ],
  },
];

export default function RankingPage() {
  const { lang } = useLanguage();
  const [selectedGame, setSelectedGame] = useState<"pakupaku" | "moutheat">("pakupaku");
  const [selectedMode, setSelectedMode] = useState<string>("timeattack");
  const [rankings, setRankings] = useState<RankEntry[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const GAMES = getGames(lang);
  const currentGame = GAMES.find((g) => g.id === selectedGame);
  const currentModes = currentGame?.modes || [];

  const fetchRankings = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getTopRanking(selectedMode, 100);
      setRankings(data);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load rankings");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchRankings(); }, [selectedMode]);
  useEffect(() => {
    const interval = setInterval(() => { fetchRankings(); }, 30000);
    return () => clearInterval(interval);
  }, [selectedMode]);

  const handleGameChange = (gameId: "pakupaku" | "moutheat") => {
    setSelectedGame(gameId);
    const newGame = GAMES.find((g) => g.id === gameId);
    if (newGame && newGame.modes.length > 0) {
      setSelectedMode(newGame.modes[0].id);
    }
  };

  const filteredRankings = rankings.filter((entry) =>
    entry.nickname.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-black text-white">
      <Navbar />

      <PageHero
        icon="🏆"
        title={lang === 'ja' ? 'ワールドランキング' : 'World Rankings'}
        subtitle={lang === 'ja' ? 'リアルタイムのワールドランキング' : 'Real-time global leaderboards'}
        gradient="linear-gradient(135deg, #0EA5E9, #7C3AED, #06B6D4)"
      />

      {/* Game Selector */}
      <section className="px-6 mb-8">
        <div className="max-w-6xl mx-auto">
          <div className="flex gap-3 justify-center flex-wrap">
            {GAMES.map((game) => (
              <button
                key={game.id}
                onClick={() => handleGameChange(game.id as "pakupaku" | "moutheat")}
                className="px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-300"
                style={{
                  background: selectedGame === game.id ? 'rgba(14, 165, 233, 0.15)' : 'rgba(10, 10, 20, 0.6)',
                  border: selectedGame === game.id ? '1px solid rgba(14, 165, 233, 0.4)' : '1px solid rgba(14, 165, 233, 0.08)',
                  color: selectedGame === game.id ? '#e2e8f0' : '#9ca3af',
                  boxShadow: selectedGame === game.id ? '0 0 20px rgba(14, 165, 233, 0.1)' : 'none',
                }}
              >
                {game.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Mode Selector */}
      <section className="px-6 mb-8">
        <div className="max-w-6xl mx-auto">
          <div className="flex gap-2 justify-center flex-wrap">
            {currentModes.map((mode) => (
              <button
                key={mode.id}
                onClick={() => setSelectedMode(mode.id)}
                className="px-5 py-2 rounded-lg text-xs font-medium transition-all duration-300"
                style={{
                  background: selectedMode === mode.id ? 'rgba(14, 165, 233, 0.1)' : 'transparent',
                  border: selectedMode === mode.id ? '1px solid rgba(14, 165, 233, 0.3)' : '1px solid rgba(55, 65, 81, 0.3)',
                  color: selectedMode === mode.id ? '#e2e8f0' : '#6b7280',
                }}
              >
                {mode.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Search + Refresh */}
      <section className="px-6 mb-8">
        <div className="max-w-6xl mx-auto flex gap-3 items-center flex-wrap justify-center">
          <div className="relative flex-1 max-w-md">
            <input
              type="text"
              placeholder={lang === 'ja' ? 'ニックネームで検索...' : 'Search by nickname...'}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-4 py-3 rounded-xl text-sm text-white placeholder-gray-600 focus:outline-none transition-all duration-300"
              style={{
                background: 'rgba(10, 10, 20, 0.6)',
                border: '1px solid rgba(14, 165, 233, 0.08)',
              }}
              onFocus={(e) => { e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.3)'; }}
              onBlur={(e) => { e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.08)'; }}
            />
            <span className="absolute right-3 top-3 text-gray-600 text-sm">🔍</span>
          </div>
          <button
            onClick={fetchRankings}
            disabled={loading}
            className="px-5 py-3 rounded-xl text-xs font-medium transition-all duration-300 disabled:opacity-40"
            style={{
              background: 'rgba(10, 10, 20, 0.6)',
              border: '1px solid rgba(14, 165, 233, 0.08)',
              color: '#9ca3af',
            }}
          >
            🔄 {lang === 'ja' ? '更新' : 'Refresh'}
          </button>
        </div>
        {lastUpdated && (
          <p className="text-center text-gray-600 text-xs mt-3 tracking-wide">
            {lang === 'ja' ? '最終更新' : 'Last updated'}: {lastUpdated.toLocaleTimeString(lang === 'ja' ? 'ja-JP' : 'en-US')}
          </p>
        )}
      </section>

      {/* Ranking Table */}
      <section className="px-6 pb-24">
        <div className="max-w-6xl mx-auto">
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-2xl p-6 text-center mb-6"
              style={{
                background: 'rgba(127, 29, 29, 0.15)',
                border: '1px solid rgba(239, 68, 68, 0.2)',
              }}
            >
              <p className="text-red-400/80 mb-4 text-sm">{lang === 'ja' ? 'データの取得に失敗しました' : 'Failed to load data'}</p>
              <button
                onClick={fetchRankings}
                className="px-5 py-2 rounded-lg text-xs font-medium transition-all duration-300"
                style={{
                  background: 'rgba(239, 68, 68, 0.1)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  color: 'rgba(239, 68, 68, 0.8)',
                }}
              >
                {lang === 'ja' ? 'リトライ' : 'Retry'}
              </button>
            </motion.div>
          )}

          {loading ? (
            <RankingSkeleton />
          ) : filteredRankings.length === 0 ? (
            <div
              className="rounded-2xl p-12 text-center"
              style={{
                background: 'rgba(10, 10, 20, 0.6)',
                border: '1px solid rgba(14, 165, 233, 0.08)',
              }}
            >
              <p className="text-gray-500">{lang === 'ja' ? 'データがありません' : 'No data available'}</p>
            </div>
          ) : (
            <div
              className="overflow-x-auto rounded-2xl"
              style={{
                background: 'rgba(10, 10, 20, 0.4)',
                border: '1px solid rgba(14, 165, 233, 0.06)',
              }}
            >
              <table className="w-full border-collapse">
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(55,65,81,0.3)' }}>
                    <th className="px-6 py-4 text-left text-gray-500 font-medium text-xs tracking-[0.2em] uppercase">{lang === 'ja' ? '順位' : 'Rank'}</th>
                    <th className="px-6 py-4 text-left text-gray-500 font-medium text-xs tracking-[0.2em] uppercase">{lang === 'ja' ? 'ニックネーム' : 'Nickname'}</th>
                    <th className="px-6 py-4 text-right text-gray-500 font-medium text-xs tracking-[0.2em] uppercase">{lang === 'ja' ? 'スコア' : 'Score'}</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRankings.map((entry, idx) => {
                    const isHighlighted =
                      searchQuery.length > 0 &&
                      entry.nickname.toLowerCase().includes(searchQuery.toLowerCase());
                    return (
                      <motion.tr
                        key={entry.user_id}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3, delay: idx * 0.02 }}
                        className="transition-colors duration-200"
                        style={{
                          borderBottom: '1px solid rgba(55,65,81,0.15)',
                          background: isHighlighted ? 'rgba(14, 165, 233, 0.06)' : 'transparent',
                        }}
                        onMouseEnter={(e) => {
                          if (!isHighlighted) e.currentTarget.style.background = 'rgba(14, 165, 233, 0.03)';
                        }}
                        onMouseLeave={(e) => {
                          if (!isHighlighted) e.currentTarget.style.background = 'transparent';
                        }}
                      >
                        <td className="px-6 py-4 font-semibold text-sm">
                          {entry.world_rank === 1 ? <span className="text-xl">🥇</span> :
                           entry.world_rank === 2 ? <span className="text-xl">🥈</span> :
                           entry.world_rank === 3 ? <span className="text-xl">🥉</span> :
                           <span className="text-gray-500">{entry.world_rank}</span>}
                        </td>
                        <td className="px-6 py-4 text-gray-300 text-sm">{entry.nickname}</td>
                        <td className="px-6 py-4 text-right font-bold text-sm" style={{ color: 'rgba(14, 165, 233, 0.8)' }}>
                          {entry.score.toLocaleString()}
                        </td>
                      </motion.tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      <Footer />
    </div>
  );
}

function RankingSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 10 }).map((_, i) => (
        <div
          key={i}
          className="rounded-xl p-4 flex justify-between items-center animate-pulse"
          style={{
            background: 'rgba(10, 10, 20, 0.4)',
            border: '1px solid rgba(14, 165, 233, 0.04)',
          }}
        >
          <div className="h-4 rounded w-10" style={{ background: 'rgba(55,65,81,0.3)' }} />
          <div className="h-4 rounded w-28" style={{ background: 'rgba(55,65,81,0.3)' }} />
          <div className="h-4 rounded w-16" style={{ background: 'rgba(55,65,81,0.3)' }} />
        </div>
      ))}
    </div>
  );
}
