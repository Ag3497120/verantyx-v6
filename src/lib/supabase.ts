// Supabase REST API client (no npm package - using fetch directly)

const SUPABASE_URL = "https://zekypqjmvyxevwyujicn.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpla3lwcWptdnl4ZXZ3eXVqaWNuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM1NjI5NjcsImV4cCI6MjA4OTEzODk2N30.HzU2FjYp03xEqTISUkOvkkfzzKdMBZ2Bkw2hHE-6IH0";

// Types
export interface RankEntry {
  user_id: string;
  nickname: string;
  score: number;
  world_rank: number;
}

export interface RankResult {
  world_rank: number;
  total_players: number;
  best_score: number;
}

// Generic RPC caller
async function callRPC(fnName: string, body: object): Promise<unknown> {
  const url = `${SUPABASE_URL}/rest/v1/rpc/${fnName}`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "apikey": SUPABASE_ANON_KEY,
      "Authorization": `Bearer ${SUPABASE_ANON_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`Supabase RPC error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

// Typed wrappers
export async function getTopRanking(mode: string, limit: number = 100): Promise<RankEntry[]> {
  const result = await callRPC("get_top_ranking", {
    p_mode: mode,
    p_limit: limit,
  });
  return result as RankEntry[];
}

export async function getMyRank(userId: string, mode: string): Promise<RankResult> {
  const result = await callRPC("get_my_rank", {
    p_user_id: userId,
    p_mode: mode,
  });
  return result as RankResult;
}
