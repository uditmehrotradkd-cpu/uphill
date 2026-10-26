import React, { useEffect, useMemo, useState } from "react";
import { Cap, ScreenerRow, api } from "../api";
import { fmt, fmtPct, signClass, timeAgo } from "../format";
import { useMarketStream } from "../useMarketStream";
import { PageHero } from "./Editorial";

const CAPS: { key: Cap | ""; label: string; index: string }[] = [
  { key: "", label: "All", index: "Nifty 500 universe" },
  { key: "large", label: "Large Cap", index: "Nifty 100 · top 100 by market cap" },
  { key: "mid", label: "Mid Cap", index: "Nifty Midcap 150 · ranks 101-250" },
  { key: "small", label: "Small Cap", index: "Nifty Smallcap 250 · ranks 251+" },
];
const CAP_LABEL: Record<Cap, string> = { large: "LARGE", mid: "MID", small: "SMALL" };

const FILTERS: { key: string; label: string; test: (r: Live) => boolean }[] = [
  { key: "", label: "No quick filter", test: () => true },
  { key: "gainers", label: "Gainers today", test: (r) => (r.dayChg ?? 0) > 0 },
  { key: "losers", label: "Losers today", test: (r) => (r.dayChg ?? 0) < 0 },
  { key: "high", label: "Within 5% of 52W high", test: (r) => (r.from_high ?? -100) >= -5 },
  { key: "low", label: "Within 5% of 52W low", test: (r) => r.price != null && r.low_52w != null && r.price <= r.low_52w * 1.05 },
  { key: "oversold", label: "RSI oversold (< 30)", test: (r) => r.rsi != null && r.rsi < 30 },
  { key: "overbought", label: "RSI overbought (> 70)", test: (r) => r.rsi != null && r.rsi > 70 },
  { key: "uptrend", label: "3M uptrend (> +10%)", test: (r) => (r.chg_3m ?? 0) > 10 },
  { key: "newsPos", label: "Bullish news", test: (r) => r.news.articles > 0 && r.news.label === "bullish" },
  { key: "newsNeg", label: "Bearish news", test: (r) => r.news.articles > 0 && r.news.label === "bearish" },
];

type SortKey = "symbol" | "price" | "dayChg" | "chg_1w" | "chg_1m" | "chg_3m" | "chg_1y" | "from_high" | "rsi" | "avg_volume_20d" | "news";
const COLUMNS: { key: SortKey; label: string; title?: string }[] = [
  { key: "price", label: "Price" },
  { key: "dayChg", label: "1D %", title: "Live change vs previous close" },
  { key: "chg_1w", label: "1W %" },
  { key: "chg_1m", label: "1M %" },
  { key: "chg_3m", label: "3M %" },
  { key: "chg_1y", label: "1Y %" },
  { key: "from_high", label: "52W range", title: "Position between 52-week closing low and high" },
  { key: "rsi", label: "RSI" },
  { key: "avg_volume_20d", label: "Avg Vol 20D" },
  { key: "news", label: "News" },
];

interface Live extends ScreenerRow {
  price: number | null;
  dayChg: number | null;
  tickTime: string | null;
}

interface Props {
  watchlist: string[];
  onOpenSecurity: (listing: string) => void;
  onWatch: (listing: string, action: "ADD" | "DEL") => void;
}

function volume(v: number | null | undefined): string {
  if (v == null) return "n/a";
  if (v >= 1e7) return `${(v / 1e7).toFixed(2)} Cr`;
  if (v >= 1e5) return `${(v / 1e5).toFixed(2)} L`;
  return v.toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

function RangeBar({ low, high, value }: { low?: number | null; high?: number | null; value: number | null }) {
  if (low == null || high == null || value == null || high <= low) return <span className="dim">n/a</span>;
  const pos = Math.max(0, Math.min(1, (value - low) / (high - low)));
  return (
    <div className="range" title={`52W low ${fmt(low)} · high ${fmt(high)}`}>
      <span className="range-fill" style={{ width: `${pos * 100}%` }} />
      <span className="range-dot" style={{ left: `${pos * 100}%` }} />
    </div>
  );
}

export function ScreenerPage({ watchlist, onOpenSecurity, onWatch }: Props) {
  const [rows, setRows] = useState<ScreenerRow[]>([]);
  const [updated, setUpdated] = useState<string | null>(null);
  const [cap, setCap] = useState<Cap | "">("");
  const [industry, setIndustry] = useState("");
  const [filter, setFilter] = useState("");
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "dayChg", desc: true });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = () =>
      api.screener()
        .then((r) => { setRows(r.rows); setUpdated(r.updated); setError(null); })
        .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    void load();
    const t = window.setInterval(load, 120_000);
    return () => window.clearInterval(t);
  }, []);

  const inCap = useMemo(() => rows.filter((r) => !cap || r.cap === cap), [rows, cap]);
  const streamList = useMemo(() => inCap.filter((r) => r.live).map((r) => r.listing), [inCap]);
  const { ticks, connection } = useMarketStream(streamList);

  const live: Live[] = useMemo(
    () =>
      inCap.map((r) => {
        const tick = ticks[r.listing];
        const price = tick?.price ?? r.last_close ?? null;
        const base = tick ? r.last_close : r.prev_close;
        const dayChg = tick && base ? (tick.price / base - 1) * 100 : (r.chg_1d ?? null);
        return { ...r, price, dayChg, tickTime: tick?.timestamp ?? null };
      }),
    [inCap, ticks],
  );

  const industries = useMemo(() => [...new Set(inCap.map((r) => r.industry))].sort(), [inCap]);
  const counts = useMemo(() => {
    const c: Record<string, number> = { "": rows.length, large: 0, mid: 0, small: 0 };
    rows.forEach((r) => (c[r.cap] += 1));
    return c;
  }, [rows]);

  const visible = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const test = FILTERS.find((f) => f.key === filter)?.test ?? (() => true);
    const value = (r: Live): number | string | null =>
      sort.key === "symbol" ? r.symbol : sort.key === "price" ? r.price : sort.key === "dayChg" ? r.dayChg
        : sort.key === "news" ? r.news.articles : ((r as unknown as Record<string, number | null>)[sort.key] ?? null);
    return live
      .filter((r) => (!industry || r.industry === industry) && test(r)
        && (!needle || r.symbol.toLowerCase().includes(needle) || r.name.toLowerCase().includes(needle)))
      .sort((a, b) => {
        const va = value(a), vb = value(b);
        if (va == null && vb == null) return 0;
        if (va == null) return 1;
        if (vb == null) return -1;
        const cmp = typeof va === "string" ? va.localeCompare(vb as string) : (va as number) - (vb as number);
        return sort.desc ? -cmp : cmp;
      });
  }, [live, industry, filter, q, sort]);

  const breadth = useMemo(() => {
    const up = live.filter((r) => (r.dayChg ?? 0) > 0).length;
    const down = live.filter((r) => (r.dayChg ?? 0) < 0).length;
    return { up, down, flat: live.length - up - down };
  }, [live]);

  const sortBy = (key: SortKey) => setSort((s) => ({ key, desc: s.key === key ? !s.desc : key !== "symbol" }));
  const arrow = (key: SortKey) => (sort.key === key ? (sort.desc ? " ▼" : " ▲") : "");
  const active = CAPS.find((c) => c.key === cap)!;

  return (
    <section className="screener">
      <PageHero index="02" label="All stocks" title="Every stock," accent="ranked." outline="NIFTY">
        <p className="dim">{rows.length} NSE listings across large, mid and small caps. Sort, filter, open any of them in the terminal.</p>
      </PageHero>
      <div className="screener-head">
        <div className="tabs">
          {CAPS.map((c) => (
            <button key={c.key} className={cap === c.key ? "on" : ""} onClick={() => { setCap(c.key); setIndustry(""); }}>
              {c.label} <span className="count">{counts[c.key] ?? 0}</span>
            </button>
          ))}
        </div>
        <span className="dim small">{active.index}</span>
        <div className="breadth" title="Advancers / decliners on the day">
          <span className="up">▲ {breadth.up}</span>
          <span className="dim">■ {breadth.flat}</span>
          <span className="down">▼ {breadth.down}</span>
          <div className="breadth-bar">
            <span className="b-up" style={{ flex: breadth.up }} />
            <span className="b-flat" style={{ flex: breadth.flat }} />
            <span className="b-down" style={{ flex: breadth.down }} />
          </div>
        </div>
      </div>
      <div className="screener-toolbar">
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search symbol or company…" maxLength={40} />
        <select value={industry} onChange={(e) => setIndustry(e.target.value)} aria-label="Industry">
          <option value="">All industries</option>
          {industries.map((i) => <option key={i} value={i}>{i}</option>)}
        </select>
        <select value={filter} onChange={(e) => setFilter(e.target.value)} aria-label="Quick filter">
          {FILTERS.map((f) => <option key={f.key} value={f.key}>{f.label}</option>)}
        </select>
        <span className="dim small">
          {visible.length} stocks · EOD stats {updated ? `updated ${timeAgo(updated)}` : "loading…"} ·{" "}
          <span className={connection === "open" && streamList.length ? "up" : "dim"}>
            {streamList.length ? (connection === "open" ? `live ticks on ${streamList.length}` : "live: connecting") : "live feed off"}
          </span>
        </span>
      </div>
      {error && <div className="error">{error}</div>}
      <div className="screener-table">
        <table className="grid">
          <thead>
            <tr>
              <th className="sortable" onClick={() => sortBy("symbol")}>Stock{arrow("symbol")}</th>
              <th>Cap</th>
              <th>Industry</th>
              {COLUMNS.map((c) => (
                <th key={c.key} className="num sortable" title={c.title} onClick={() => sortBy(c.key)}>{c.label}{arrow(c.key)}</th>
              ))}
              <th />
            </tr>
          </thead>
          <tbody>
            {visible.map((r, index) => {
              const watched = watchlist.includes(r.listing);
              return (
                <tr key={r.listing} onClick={() => onOpenSecurity(r.listing)} className="clickable"
                  style={{ "--d": Math.min(index, 25) } as React.CSSProperties}>
                  <td>
                    <div className="stock-cell">
                      <b>{r.symbol}</b>
                      {r.live && <span className="live-dot">LIVE</span>}
                    </div>
                    <div className="dim small ellipsis" title={r.name}>{r.name}</div>
                  </td>
                  <td><span className={`cap-badge ${r.cap}`}>{CAP_LABEL[r.cap]}</span></td>
                  <td className="dim ellipsis" title={r.industry}>{r.industry}</td>
                  {r.error ? (
                    <td colSpan={COLUMNS.length} className="dim small">{r.error === "loading" ? "loading…" : `no price history (${r.error.split(":")[0]})`}</td>
                  ) : (
                    <>
                      <td key={r.tickTime ?? "c"} className={`num ${r.tickTime ? "flash-live" : ""}`}>{fmt(r.price)}</td>
                      <td className={`num ${signClass(r.dayChg)}`}>{fmtPct(r.dayChg)}</td>
                      <td className={`num ${signClass(r.chg_1w)}`}>{fmtPct(r.chg_1w, 1)}</td>
                      <td className={`num ${signClass(r.chg_1m)}`}>{fmtPct(r.chg_1m, 1)}</td>
                      <td className={`num ${signClass(r.chg_3m)}`}>{fmtPct(r.chg_3m, 1)}</td>
                      <td className={`num ${signClass(r.chg_1y)}`}>{fmtPct(r.chg_1y, 1)}</td>
                      <td><RangeBar low={r.low_52w} high={r.high_52w} value={r.price} /></td>
                      <td className={`num ${r.rsi != null && r.rsi > 70 ? "down" : r.rsi != null && r.rsi < 30 ? "up" : ""}`}>{fmt(r.rsi, 0)}</td>
                      <td className="num dim">{volume(r.avg_volume_20d)}</td>
                      <td className="num">
                        {r.news.articles > 0 ? (
                          <span className={`tag ${r.news.label === "bullish" ? "up" : r.news.label === "bearish" ? "down" : ""}`}
                            title={`${r.news.bullish} bullish / ${r.news.bearish} bearish (72h)`}>{r.news.articles}</span>
                        ) : <span className="dim">-</span>}
                      </td>
                    </>
                  )}
                  <td>
                    <button
                      className={`watch-btn ${watched ? "on" : ""}`}
                      title={watched ? "Remove from watchlist" : "Add to watchlist"}
                      onClick={(e) => { e.stopPropagation(); onWatch(r.listing, watched ? "DEL" : "ADD"); }}
                    >{watched ? "★" : "☆"}</button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {!rows.length && !error && (
          <div className="skeleton-table">
            {Array.from({ length: 10 }, (_, i) => <div key={i} className="skeleton-row" style={{ "--d": i } as React.CSSProperties} />)}
          </div>
        )}
      </div>
      <p className="dim small">Price stats use NSE end-of-day closes (bhavcopy); live prices stream during market hours. Indicators are descriptive, not investment advice.</p>
    </section>
  );
}
