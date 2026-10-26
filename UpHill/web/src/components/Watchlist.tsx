import React, { FormEvent, useEffect, useRef, useState } from "react";
import { LiveTick, OverviewRow } from "../api";
import { fmt, fmtPct, signClass, timeAgo } from "../format";
import { Connection } from "../useMarketStream";
import { Sparkline } from "./Chart";

interface Props {
  rows: OverviewRow[];
  ticks: Record<string, LiveTick>;
  connection: Connection;
  feedOn: boolean;
  active: string | null;
  onSelect: (listing: string) => void;
  onAdd: (listing: string) => void;
  onRemove: (listing: string) => void;
}

/** Plain-language cues from engine data + news pulse. Descriptive only, never a recommendation. */
export function signals(row: OverviewRow): { text: string; tone: "up" | "down" | "neutral" }[] {
  const out: { text: string; tone: "up" | "down" | "neutral" }[] = [];
  if (row.change_1m != null) {
    if (row.change_1m >= 3) out.push({ text: "1M uptrend", tone: "up" });
    else if (row.change_1m <= -3) out.push({ text: "1M downtrend", tone: "down" });
  }
  if (row.rsi != null) {
    if (row.rsi >= 70) out.push({ text: "RSI overbought", tone: "down" });
    else if (row.rsi <= 30) out.push({ text: "RSI oversold", tone: "up" });
  }
  if (row.news.articles > 0 && row.news.label !== "neutral") {
    out.push({ text: `News ${row.news.label}`, tone: row.news.label === "bullish" ? "up" : "down" });
  }
  return out;
}

export function Watchlist({ rows, ticks, connection, feedOn, active, onSelect, onAdd, onRemove }: Props) {
  const previous = useRef<Record<string, number>>({});
  const [direction, setDirection] = useState<Record<string, "up" | "down">>({});
  const [adding, setAdding] = useState("");

  useEffect(() => {
    const next: Record<string, "up" | "down"> = {};
    for (const [listing, tick] of Object.entries(ticks)) {
      const prev = previous.current[listing];
      if (prev !== undefined && tick.price !== prev) next[listing] = tick.price > prev ? "up" : "down";
      previous.current[listing] = tick.price;
    }
    if (Object.keys(next).length) setDirection((d) => ({ ...d, ...next }));
  }, [ticks]);

  function add(e: FormEvent) {
    e.preventDefault();
    const value = adding.trim().toUpperCase();
    if (/^[A-Z]{3,6}:[A-Z0-9&.-]{1,32}$/.test(value)) onAdd(value);
    setAdding("");
  }

  return (
    <section className="panel watch">
      <header className="panel-head">
        <span className="title">LIVE MARKET</span>
        <span className={`pill ${feedOn ? connection : "closed"}`} title="Live tick stream">
          {!feedOn ? "○ feed off" : connection === "open" ? "● LIVE" : connection === "connecting" ? "○ connecting" : "○ offline"}
        </span>
      </header>
      <div className="watch-rows">
        {!rows.length && [0, 1, 2, 3, 4].map((i) => <div key={i} className="skeleton-row" style={{ "--d": i } as React.CSSProperties} />)}
        {rows.map((row, index) => {
          const tick = ticks[row.listing];
          const last = tick?.price ?? row.last_close ?? null;
          const change = tick && row.last_close ? (tick.price / row.last_close - 1) * 100 : null;
          return (
            <div
              key={row.listing}
              className={`watch-row ${active === row.listing ? "active" : ""}`}
              style={{ "--d": Math.min(index, 12) } as React.CSSProperties}
              onClick={() => onSelect(row.listing)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && onSelect(row.listing)}
            >
              <div className="wr-top">
                <span className="sym">{row.symbol}</span>
                <span className="exch">{row.exchange}</span>
                {row.live && <span className="live-dot" title="Streaming live ticks">LIVE</span>}
                <span
                  key={tick?.timestamp ?? "close"}
                  className={`price num ${tick ? `flash-${direction[row.listing] ?? "none"}` : ""}`}
                >
                  {fmt(last)}
                </span>
                <button className="x" title="Remove from watchlist" onClick={(e) => { e.stopPropagation(); onRemove(row.listing); }}>×</button>
              </div>
              {row.error ? (
                <div className="wr-sub dim" title={row.error}>no data: {row.error.split(":")[0]}</div>
              ) : (
                <div className="wr-sub">
                  <span className={`num ${signClass(change)}`} title="Live change vs last close">{change === null ? "close" : fmtPct(change)}</span>
                  <span className={`num ${signClass(row.change_1m)}`} title="1-month change">1M {fmtPct(row.change_1m, 1)}</span>
                  <span className="num dim" title="RSI(14)">RSI {fmt(row.rsi, 0)}</span>
                  <Sparkline values={row.spark ?? []} />
                </div>
              )}
              <div className="wr-tags">
                {signals(row).map((s) => <span key={s.text} className={`tag ${s.tone}`}>{s.text}</span>)}
                {row.news.articles > 0 && (
                  <span className="tag neutral" title={`${row.news.bullish} bullish / ${row.news.bearish} bearish, latest ${timeAgo(row.news.latest)}`}>
                    {row.news.articles} news
                  </span>
                )}
                {tick && <span className="tag time">{new Date(tick.timestamp).toLocaleTimeString()}</span>}
              </div>
            </div>
          );
        })}
      </div>
      <form className="add" onSubmit={add}>
        <input value={adding} onChange={(e) => setAdding(e.target.value)} placeholder="Add NSE:SBIN" maxLength={40} />
        <button type="submit">+</button>
      </form>
    </section>
  );
}
