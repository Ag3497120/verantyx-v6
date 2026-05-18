"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowLeft, Send, Settings, Radio } from "lucide-react";

export default function TalkiePressApp() {
  const [newsText, setNewsText] = useState("");
  const [useHybrid, setUseHybrid] = useState(true);
  const [apiUrl, setApiUrl] = useState("https://temporary-est-recipients-respondents.trycloudflare.com/api/generate");
  const [showSettings, setShowSettings] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [articleHtml, setArticleHtml] = useState("");

  const generateNews = async () => {
    if (!newsText.trim()) return;
    setLoading(true);
    setError("");
    setArticleHtml("");

    try {
      const res = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ news_text: newsText, use_hybrid: useHybrid })
      });

      if (res.status === 502 || res.status === 503 || res.status === 524) {
        throw new Error("OFFLINE");
      }

      const data = await res.json();
      if (res.ok) {
        setArticleHtml(data.html);
      } else {
        setError(data.detail || "Transmission failed.");
      }
    } catch (err: any) {
      if (err.message === "OFFLINE" || err.message.includes("Failed to fetch")) {
        setError("[Notice from the Telegraph Office]\nOur reporters are currently out in full force, frantically gathering news from around the world (*The developer's PC is currently asleep or running at full capacity with other tasks).\nPlease wait patiently until our correspondents return to the newsroom.");
      } else {
        setError(`Telegraph Error: Unable to reach the backend at ${apiUrl}. Detail: ${err.message}`);
      }
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-[#1c1814] text-[#e8dfc8] font-serif selection:bg-[#8c2020] selection:text-white pb-20">
      <div className="max-w-4xl mx-auto pt-10 px-4">
        <Link href="/apps" className="inline-flex items-center text-[#a89b88] hover:text-[#e8dfc8] mb-8 transition-colors">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Apps
        </Link>

        <div className="text-center mb-12 border-b-2 border-[#e8dfc8] pb-6">
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-4 uppercase" style={{ fontFamily: 'Times New Roman, serif' }}>
            TalkiePress
          </h1>
          <p className="text-xl italic text-[#a89b88]">1930s Autonomous News Generation Engine</p>
        </div>

        <div className="bg-[#2c241b] rounded-lg p-6 md:p-8 shadow-2xl border border-[#3b2f2f] mb-12">
          <label className="block text-xl font-bold mb-4 flex items-center">
            <Radio className="w-5 h-5 mr-3 text-[#8c2020]" />
            Transmit Modern News to the Telegraph Office
          </label>
          
          <textarea 
            className="w-full h-32 bg-[#e8dfc8] text-[#1c1814] p-4 rounded-md font-mono text-lg resize-none focus:outline-none focus:ring-2 focus:ring-[#8c2020] mb-4 shadow-inner"
            placeholder="e.g. A massive solar flare disrupted internet communications across Europe today..."
            value={newsText}
            onChange={(e) => setNewsText(e.target.value)}
          />

          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <button 
              onClick={() => setShowSettings(!showSettings)}
              className="text-[#a89b88] hover:text-white flex items-center text-sm transition-colors"
            >
              <Settings className="w-4 h-4 mr-1" />
              Engine Settings
            </button>
            
            <button 
              onClick={generateNews}
              disabled={loading || !newsText.trim()}
              className="w-full md:w-auto bg-[#8c2020] hover:bg-[#a32727] text-white px-8 py-3 rounded-md font-bold text-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {loading ? (
                <span className="animate-pulse">Transmitting...</span>
              ) : (
                <>
                  <Send className="w-5 h-5 mr-2" />
                  Print the Paper
                </>
              )}
            </button>
          </div>

          {showSettings && (
            <div className="mt-6 p-4 bg-[#1c1814] rounded-md border border-[#3b2f2f] animate-fade-in">
              <div className="mb-4">
                <label className="block text-sm text-[#a89b88] mb-2">Backend API URL (Cloudflare Tunnel / Localhost)</label>
                <input 
                  type="text" 
                  className="w-full bg-[#2c241b] text-white p-2 rounded border border-[#3b2f2f] focus:outline-none font-mono text-sm"
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                />
              </div>
              <label className="flex items-center space-x-3 cursor-pointer group">
                <div className="relative">
                  <input 
                    type="checkbox" 
                    className="sr-only" 
                    checked={useHybrid}
                    onChange={(e) => setUseHybrid(e.target.checked)}
                  />
                  <div className={`block w-10 h-6 rounded-full transition-colors ${useHybrid ? 'bg-[#8c2020]' : 'bg-[#3b2f2f]'}`}></div>
                  <div className={`absolute left-1 top-1 bg-[#e8dfc8] w-4 h-4 rounded-full transition-transform ${useHybrid ? 'transform translate-x-4' : ''}`}></div>
                </div>
                <div className="text-sm">
                  <span className="block font-bold text-[#e8dfc8]">Hybrid M1 Max Tunneling</span>
                  <span className="text-[#a89b88]">Direct Verantyx MLX Native Tunneling for explosive generation speed</span>
                </div>
              </label>
            </div>
          )}
        </div>

        {error && (
          <div className="bg-red-900/30 border border-red-500/50 text-red-200 p-6 rounded-md mb-8">
            <h3 className="font-bold mb-2">Communication Severed</h3>
            <p className="font-mono text-sm whitespace-pre-wrap">{error}</p>
          </div>
        )}

        {articleHtml && (
          <div className="mt-12 bg-[#e8dfc8] text-[#1c1814] p-8 md:p-12 shadow-[0_20px_50px_rgba(0,0,0,0.5)] transform rotate-[0.5deg]">
            <div dangerouslySetInnerHTML={{ __html: articleHtml }} />
            <div className="mt-8 pt-4 border-t border-[#1c1814]/20 text-center text-sm font-mono text-[#1c1814]/60">
              Printed via Verantyx Cortex V7 — M1 Max Tunneling Architecture
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
