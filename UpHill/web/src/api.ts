// Typed client for the UpHill backend. One session id per browser ties the terminal and the agent together.

export interface Datum {
  label: string;
  value: number | null;
  unit: string;
  source: string;
  confidence_score: number;
  timestamp: string | null;
  provenance: string;
  note: string | null;
  display?: string;
}

export interface Security {
  symbol: string;
  exchange: string;
  currency: string;
  industry: string | null;
}

export interface GPView {
  security: Security;
  as_of: string;
  adjust_splits: boolean;
  adjust_dividends: boolean;
  dates: string[];
  open?: number[];
  high?: number[];
  low?: number[];
  close: number[];
  volume: number[];
  rsi: (number | null)[];
  macd: (number | null)[];
  signal: (number | null)[];
  summary: Datum[];
}

export interface FAView {
  security: Security;
  as_of: string;
  period_type: string;
  periods: string[];
  rows: { label: string; cells: Datum[] }[];
  ratios: { label: string; cells: Datum[] }[];
  ev_table: Datum[];
  reconciliation: { is_break: boolean }[];
  notes: string[];
}

export interface RVView {
  security: Security;
  industry: string;
  as_of: string;
  columns: string[];
  rows: { security: Security; is_subject: boolean; cells: Record<string, Datum> }[];
  medians: Record<string, Datum>;
}

export interface OptionValuation {
  npv: number;
  delta: number;
  gamma: number;
  theta_per_day: number;
  vega_per_vol_point: number;
  rho_per_pct?: number;
  volatility?: number;
}

export interface QuotedLeg {
  bid: number;
  ask: number;
  implied_vol: number | null;
}

export interface OVMEView {
  security: Security;
  expiry: string;
  exercise: string;
  inputs: Datum[];
  chain: { strike: number; call: OptionValuation; put: OptionValuation; call_quote: QuotedLeg | null; put_quote: QuotedLeg | null }[];
  calculator: OptionValuation | null;
  calculator_contract: { option_type: string; strike: number } | null;
  notes: string[];
}

export interface TerminalResult {
  kind: "VIEW" | "STATUS" | "HELP" | "WATCH" | "ERROR";
  message: string;
  function: "GP" | "FA" | "RV" | "OVME" | null;
  view: GPView | FAView | RVView | OVMEView | null;
  watch_action: "ADD" | "DEL" | null;
  context: string;
  display_name?: string;
  listing?: string;
}

export interface NewsPulse {
  listing: string;
  articles: number;
  bullish: number;
  bearish: number;
  score: number;
  label: "bullish" | "bearish" | "neutral";
  latest: string | null;
}

export interface OverviewRow {
  listing: string;
  symbol: string;
  exchange: string;
  live: boolean;
  display_name: string;
  currency?: string;
  industry?: string | null;
  last_close?: number | null;
  last_close_time?: string | null;
  change_1m?: number | null;
  rsi?: number | null;
  high_1m?: number | null;
  low_1m?: number | null;
  spark?: number[];
  error: string | null;
  news: NewsPulse;
}

export interface MarketStatus {
  live_feeds: number;
  live_symbols: string[];
  watchlist: string[];
  agent_enabled: boolean;
  news_updated: string | null;
  server_time: string;
}

export interface NewsItem {
  id: string;
  title: string;
  summary: string;
  url: string;
  image: string | null;
  source: string;
  publisher: string;
  category: "stock" | "markets" | "business" | "india" | "world";
  published: string;
  symbols: string[];
  sentiment: number;
  sentiment_label: "bullish" | "bearish" | "neutral";
}

export interface SourceStatus {
  name: string;
  ok: boolean;
  items: number;
  error: string | null;
  checked: string | null;
}

export interface CitedFact {
  id: string;
  label: string;
  display: string;
  source: string;
  confidence: number;
  timestamp: string | null;
  note: string | null;
}

export interface AgentAnswer {
  text: string;
  fallback: boolean;
  steps: number;
  tool_calls: string[];
  cited: CitedFact[];
  context: string;
}

export interface LiveTick {
  listing: string;
  price: number;
  size: number;
  timestamp: string;
  prev_close?: number | null;
}

export type Cap = "large" | "mid" | "small";

export interface ScreenerRow {
  listing: string;
  symbol: string;
  name: string;
  industry: string;
  cap: Cap;
  live: boolean;
  error: string | null;
  last_close?: number | null;
  prev_close?: number | null;
  last_date?: string | null;
  chg_1d?: number | null;
  chg_1w?: number | null;
  chg_1m?: number | null;
  chg_3m?: number | null;
  chg_6m?: number | null;
  chg_1y?: number | null;
  high_52w?: number | null;
  low_52w?: number | null;
  from_high?: number | null;
  avg_volume_20d?: number | null;
  rsi?: number | null;
  news: NewsPulse;
}

export type CandleInterval = "1" | "3" | "5" | "15" | "30" | "60" | "day";

export interface CandlesResponse {
  listing: string;
  interval: CandleInterval;
  candles: [number, number, number, number, number, number][]; // epoch s, open, high, low, close, volume
}

function sessionId(): string {
  const key = "uphill.session";
  let id = localStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}

export const SESSION_ID = sessionId();

// Empty when the backend serves this UI; set VITE_API_BASE (e.g. https://uphill-api.onrender.com) for a separate host.
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/+$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(API_BASE + path, {
    ...init,
    headers: { "Content-Type": "application/json", "X-Session-Id": SESSION_ID, ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

function query(params: Record<string, string | number | string[] | undefined>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === "") continue;
    if (Array.isArray(v)) v.forEach((x) => q.append(k, x));
    else q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

export const api = {
  status: () => request<MarketStatus>("/api/market/status"),
  overview: (symbols: string[]) => request<OverviewRow[]>(`/api/market/overview${query({ symbols })}`),
  screener: () => request<{ updated: string | null; rows: ScreenerRow[] }>("/api/market/screener"),
  candles: (symbol: string, interval: CandleInterval) =>
    request<CandlesResponse>(`/api/market/candles${query({ symbol, interval })}`),
  command: (command: string) =>
    request<TerminalResult>("/api/terminal/command", { method: "POST", body: JSON.stringify({ command }) }),
  ask: (question: string) =>
    request<AgentAnswer>("/api/agent/ask", { method: "POST", body: JSON.stringify({ question }) }),
  resetAgent: () => request<void>("/api/agent/reset", { method: "POST" }),
  news: (p: { category?: string; symbol?: string; q?: string; sentiment?: string; tagged?: string[]; limit?: number }) =>
    request<NewsItem[]>(`/api/news${query(p)}`),
  sources: () => request<{ updated: string | null; sources: SourceStatus[] }>("/api/news/sources"),
};

export function marketSocketUrl(symbols: string[]): string {
  const base = new URL(API_BASE || location.origin);
  const proto = base.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${base.host}/ws/market?symbols=${encodeURIComponent(symbols.join(","))}`;
}
