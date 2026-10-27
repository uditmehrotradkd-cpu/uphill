import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { FAView, GPView, OVMEView, RVView, ScreenerRow, TerminalResult, api } from "../api";
import { bloombergName, fmt, fmtPct, signClass, timeAgo } from "../format";
import { AgentChat } from "../useAgentChat";
import { useMarketStream } from "../useMarketStream";
import { Voice } from "../voice";
import { AgentPanel } from "./AgentPanel";
import { PageHero } from "./Editorial";
import { RangeBar, volume } from "./ScreenerPage";
import { FAPanel, GPPanel, OVMEPanel, RVPanel } from "./TerminalViews";

type ToolKey = "overview" | "chart" | "fundamentals" | "peers" | "options";
type Fn = "GP" | "FA" | "RV" | "OVME";

const TOOLS: { key: ToolKey; label: string; fn: Fn; hint: string }[] = [
  { key: "overview", label: "Overview", fn: "GP", hint: "Live price, returns, 52-week range, momentum and news pulse" },
  { key: "chart", label: "Chart & studies", fn: "GP", hint: "Candles/line, intraday, SMA/EMA/BB/VWAP/RSI/MACD/Stoch/ATR" },
  { key: "fundamentals", label: "Fundamentals", fn: "FA", hint: "Point-in-time financials, margins, reconciled EV" },
  { key: "peers", label: "Peer valuation", fn: "RV", hint: "Industry multiples vs peer medians" },
  { key: "options", label: "Options & Greeks", fn: "OVME", hint: "QuantLib option chain, premiums and Greeks" },
];
const SYMBOL = /^[A-Z0-9&.-]{1,20}$/;

type Loaded = { result: TerminalResult } | { error: string };

interface Props {
  agentEnabled: boolean;
  initialListing: string | null;
  watchlist: string[];
  chat: AgentChat;
  voice: Voice;
  onSecurity: (listing: string) => void;
  onWatch: (listing: string, action: "ADD" | "DEL") => void;
}

function resolveListing(text: string, universe: ScreenerRow[]): string | null {
  const q = text.trim().toUpperCase();
  if (!q) return null;
  if (q.includes(":")) {
    const [exchange, symbol] = q.split(":");
    return /^(NSE|BSE|NYSE|NASDAQ)$/.test(exchange) && SYMBOL.test(symbol) ? `${exchange}:${symbol}` : null;
  }
  const bySymbol = universe.find((r) => r.symbol === q);
  if (bySymbol) return bySymbol.listing;
  const byName = universe.find((r) => r.name.toUpperCase().includes(q));
  if (byName) return byName.listing;
  return SYMBOL.test(q) ? `NSE:${q}` : null;
}

export function AnalyticsPage({ agentEnabled, initialListing, watchlist, chat, voice, onSecurity, onWatch }: Props) {
  const [listing, setListing] = useState(initialListing ?? watchlist[0] ?? "NSE:RELIANCE");
  const [tool, setTool] = useState<ToolKey>("overview");
  const [query, setQuery] = useState("");
  const [universe, setUniverse] = useState<ScreenerRow[]>([]);
  const [cache, setCache] = useState<Record<string, Loaded>>({});
  const [loading, setLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const fn = TOOLS.find((t) => t.key === tool)!.fn;
  const key = `${listing}|${fn}`;
  const loaded = cache[key];

  useEffect(() => {
    api.screener().then((r) => setUniverse(r.rows)).catch(() => undefined);
  }, []);

  const load = useCallback(async (target: string, f: Fn) => {
    setLoading(true);
    try {
      const result = await api.command(`${bloombergName(target)} ${f}`);
      const entry: Loaded = result.kind === "VIEW" && result.view ? { result } : { error: result.message || "No data" };
      setCache((c) => ({ ...c, [`${target}|${f}`]: entry }));
      if (result.listing) onSecurity(result.listing);
    } catch (e) {
      setCache((c) => ({ ...c, [`${target}|${f}`]: { error: e instanceof Error ? e.message : String(e) } }));
    } finally {
      setLoading(false);
    }
  }, [onSecurity]);

  useEffect(() => {
    if (!cache[key]) void load(listing, fn);
  }, [key]);

  function search(e: FormEvent) {
    e.preventDefault();
    const target = resolveListing(query, universe);
    if (!target) {
      setSearchError(`"${query}" is not a recognised symbol`);
      return;
    }
    setSearchError(null);
    setQuery("");
    setListing(target);
  }

  function refresh() {
    setCache((c) => Object.fromEntries(Object.entries(c).filter(([k]) => !k.startsWith(`${listing}|`))));
    void load(listing, fn);
  }

  const row = useMemo(() => universe.find((r) => r.listing === listing), [universe, listing]);
  const gp = cache[`${listing}|GP`];
  const gpView = gp && "result" in gp ? (gp.result.view as GPView) : null;
  const symbol = listing.split(":")[1];
  const ask = (q: string) => void chat.ask(q);
  const hi = voice.prefs.lang === "hi";

  return (
    <main className="analytics-page">
      <PageHero index="02" label="Analytics" title="Analyse," accent="then decide." outline="QUANT">
        <p className="dim">Every analysis tool for one security in one place, with an AI analyst that answers and speaks in English or Hindi.</p>
      </PageHero>
      <div className="analytics">
        <section className="panel tools-panel">
          <header className="panel-head analytics-head">
            <form className="analytics-search" onSubmit={search}>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search symbol or company (e.g. HDFCBANK, Infosys, NASDAQ:AAPL)…"
                list="analytics-universe"
                maxLength={60}
                aria-label="Security"
              />
              <datalist id="analytics-universe">
                {universe.map((r) => <option key={r.listing} value={r.symbol}>{r.name}</option>)}
              </datalist>
              <button type="submit">Analyse</button>
            </form>
            <div className="sec-title">
              <b>{symbol}</b>
              <span className="dim small">{row ? `${row.name} · ${row.industry}` : listing}</span>
              <button className="link" onClick={() => onWatch(listing, watchlist.includes(listing) ? "DEL" : "ADD")}>
                {watchlist.includes(listing) ? "− watchlist" : "+ watchlist"}
              </button>
              <button className="link" onClick={refresh} disabled={loading}>refresh</button>
            </div>
          </header>
          {searchError && <div className="error" style={{ margin: "8px 22px 0" }}>{searchError}</div>}
          <nav className="tool-tabs" aria-label="Analysis tools">
            {TOOLS.map((t) => (
              <button key={t.key} className={tool === t.key ? "on" : ""} onClick={() => setTool(t.key)} title={t.hint}>{t.label}</button>
            ))}
            {watchlist.length > 0 && (
              <span className="quick">
                {watchlist.slice(0, 8).map((l) => (
                  <button key={l} className={`chip-btn ${l === listing ? "on" : ""}`} onClick={() => setListing(l)}>{l.split(":")[1]}</button>
                ))}
              </span>
            )}
          </nav>
          <div className="viewport analytics-body">
            {!loaded && loading && <div className="skeleton-table"><div className="skeleton-row" /><div className="skeleton-row" /><div className="skeleton-row" /></div>}
            {loaded && "error" in loaded && tool !== "overview" && (
              <div className="info">{TOOLS.find((t) => t.key === tool)!.label} is not available for {symbol}: {loaded.error}</div>
            )}
            {tool === "overview" && (
              <Overview listing={listing} row={row} gp={gpView} gpError={gp && "error" in gp ? gp.error : null} hi={hi} onAsk={ask}
                        agentEnabled={agentEnabled} />
            )}
            {loaded && "result" in loaded && tool === "chart" && <GPPanel v={loaded.result.view as GPView} />}
            {loaded && "result" in loaded && tool === "fundamentals" && <FAPanel v={loaded.result.view as FAView} />}
            {loaded && "result" in loaded && tool === "peers" && <RVPanel v={loaded.result.view as RVView} />}
            {loaded && "result" in loaded && tool === "options" && <OVMEPanel v={loaded.result.view as OVMEView} />}
          </div>
        </section>
        <AgentPanel enabled={agentEnabled} activeListing={listing} chat={chat} voice={voice} title="AI ANALYST" />
      </div>
    </main>
  );
}

interface OverviewProps {
  listing: string;
  row: ScreenerRow | undefined;
  gp: GPView | null;
  gpError: string | null;
  hi: boolean;
  agentEnabled: boolean;
  onAsk: (question: string) => void;
}

function Overview({ listing, row, gp, gpError, hi, agentEnabled, onAsk }: OverviewProps) {
  const { ticks, connection } = useMarketStream([listing]);
  const tick = ticks[listing];
  const price = tick?.price ?? row?.last_close ?? gp?.close[gp.close.length - 1] ?? null;
  const base = tick ? (tick.prev_close ?? row?.last_close) : row?.prev_close;
  const dayChg = tick && base ? (tick.price / base - 1) * 100 : (row?.chg_1d ?? null);
  const symbol = listing.split(":")[1];
  const currency = gp?.security.currency ?? "INR";
  const returns: [string, number | null | undefined][] = [
    ["1D", dayChg], ["1W", row?.chg_1w], ["1M", row?.chg_1m], ["3M", row?.chg_3m], ["6M", row?.chg_6m], ["1Y", row?.chg_1y],
  ];
  const prompts = hi
    ? [`${symbol} का ट्रेंड और मोमेंटम समझाइए`, `${symbol} की ताज़ा ख़बरों का सार बताइए`, `क्या ${symbol} पीयर्स की तुलना में महँगा है?`, `${symbol} में मुख्य जोखिम क्या हैं?`]
    : [`Explain ${symbol}'s trend and momentum`, `Summarise the latest news on ${symbol}`, `Is ${symbol} expensive vs its peers?`, `What are the key risks for ${symbol}?`];

  return (
    <div className="view overview">
      <div className="ov-quote">
        <div>
          <div className="ov-price num">{fmt(price)} <span className="dim small">{currency}</span></div>
          <div className={`num ${signClass(dayChg)}`}>{fmtPct(dayChg)} today</div>
        </div>
        <span className={`pill ${tick && connection === "open" ? "open" : "closed"}`} title={tick ? `last tick ${timeAgo(tick.timestamp)}` : "no live price"}>
          {tick && connection === "open" ? "LIVE" : "EOD"}
        </span>
      </div>
      <div className="ov-grid">
        {returns.map(([label, v]) => (
          <div key={label} className="ov-stat"><span className="dim small">{label} return</span><b className={`num ${signClass(v)}`}>{fmtPct(v)}</b></div>
        ))}
        <div className="ov-stat wide">
          <span className="dim small">52-week range · {fmt(row?.low_52w)} – {fmt(row?.high_52w)}</span>
          <RangeBar low={row?.low_52w} high={row?.high_52w} value={price} />
          <span className="dim small">{fmtPct(row?.from_high)} from high</span>
        </div>
        <div className="ov-stat"><span className="dim small">RSI(14)</span><b className="num">{fmt(row?.rsi ?? null, 1)}</b>
          <span className="dim small">{row?.rsi == null ? "" : row.rsi > 70 ? "overbought" : row.rsi < 30 ? "oversold" : "neutral"}</span></div>
        <div className="ov-stat"><span className="dim small">Avg volume 20D</span><b className="num">{volume(row?.avg_volume_20d)}</b></div>
        <div className="ov-stat">
          <span className="dim small">News pulse (heuristic)</span>
          <b className={row?.news.label === "bullish" ? "up" : row?.news.label === "bearish" ? "down" : ""}>{row ? row.news.label : "n/a"}</b>
          <span className="dim small">{row ? `${row.news.articles} articles · ${row.news.bullish}▲ ${row.news.bearish}▼` : ""}</span>
        </div>
      </div>
      {gp && (
        <div className="chips">
          {gp.summary.map((d) => (
            <span key={d.label} className="chip" title={`${d.source} · ${d.note ?? d.provenance}`}>
              {d.label}: <b>{d.value === null ? "n/a" : (d.display ?? fmt(d.value))}</b>
            </span>
          ))}
        </div>
      )}
      {gpError && <p className="dim small">Price analytics unavailable: {gpError}</p>}
      {!row && <p className="dim small">Universe statistics are available for NSE stocks in the Nifty 500 universe.</p>}
      <h4>{hi ? "विश्लेषक से पूछें" : "Ask the analyst"}</h4>
      <div className="suggestions inline">
        {prompts.map((p) => <button key={p} className="chip-btn" onClick={() => onAsk(p)} disabled={!agentEnabled}>{p}</button>)}
      </div>
    </div>
  );
}
