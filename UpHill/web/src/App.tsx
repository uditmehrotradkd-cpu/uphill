import React, { useCallback, useEffect, useRef, useState } from "react";
import { MarketStatus, OverviewRow, api } from "./api";
import { AgentPanel } from "./components/AgentPanel";
import { AnalyticsPage } from "./components/AnalyticsPage";
import { PageHero } from "./components/Editorial";
import { NewsPage, NewsTicker } from "./components/NewsPage";
import { ScreenerPage } from "./components/ScreenerPage";
import { TerminalHandle, TerminalPanel } from "./components/TerminalPanel";
import { Watchlist } from "./components/Watchlist";
import { useAgentChat } from "./useAgentChat";
import { useMarketStream } from "./useMarketStream";
import { useVoice } from "./voice";

type Tab = "dashboard" | "analytics" | "stocks" | "news";
const icon = (d: string) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d={d} />
  </svg>
);
const TABS: { key: Tab; label: string; icon: React.ReactNode }[] = [
  { key: "dashboard", label: "Dashboard", icon: icon("M4 4h7v7H4zM13 4h7v4h-7zM13 10h7v10h-7zM4 13h7v7H4z") },
  { key: "analytics", label: "Analytics", icon: icon("M4 20V10M10 20V4M16 20v-7M22 20H2M14.5 8.5l3-3 3 3") },
  { key: "stocks", label: "All Stocks", icon: icon("M3 20h18M5 16l4-5 4 3 6-8") },
  { key: "news", label: "News", icon: icon("M5 4h11v16H6a2 2 0 0 1-2-2V5a1 1 0 0 1 1-1zM16 8h3v10a2 2 0 0 1-2 2M8 8h5M8 12h5M8 16h3") },
];
const WATCH_KEY = "uphill.watchlist";
const OVERVIEW_REFRESH_MS = 60_000;

function nseSession(now: Date): { open: boolean; label: string } {
  const ist = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
  const minutes = ist.getHours() * 60 + ist.getMinutes();
  const weekday = ist.getDay() >= 1 && ist.getDay() <= 5;
  const open = weekday && minutes >= 555 && minutes < 930; // 09:15-15:30 IST (exchange holidays not modelled)
  return { open, label: open ? "NSE OPEN" : "NSE CLOSED" };
}

function Clock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const t = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(t);
  }, []);
  const session = nseSession(now);
  return (
    <div className="clock">
      <span className={`pill ${session.open ? "open" : "closed"}`}>{session.label}</span>
      <span className="num">{now.toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour12: false })} IST</span>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [status, setStatus] = useState<MarketStatus | null>(null);
  const [watchlist, setWatchlist] = useState<string[]>(() => JSON.parse(localStorage.getItem(WATCH_KEY) ?? "[]"));
  const [rows, setRows] = useState<OverviewRow[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const terminal = useRef<TerminalHandle>(null);
  const { ticks, connection } = useMarketStream(watchlist);
  const voice = useVoice();
  const chat = useAgentChat(voice);

  useEffect(() => {
    api.status()
      .then((s) => {
        setStatus(s);
        setWatchlist((w) => (w.length ? w : s.watchlist));
      })
      .catch((e) => setBackendError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    localStorage.setItem(WATCH_KEY, JSON.stringify(watchlist));
    if (!watchlist.length) return;
    const load = () => api.overview(watchlist).then(setRows).catch(() => undefined);
    void load();
    const t = window.setInterval(load, OVERVIEW_REFRESH_MS);
    return () => window.clearInterval(t);
  }, [watchlist]);

  const updateWatch = useCallback((listing: string, action: "ADD" | "DEL") => {
    setWatchlist((w) => (action === "ADD" ? (w.includes(listing) ? w : [...w, listing]) : w.filter((x) => x !== listing)));
  }, []);

  const openSecurity = useCallback((listing: string) => {
    setTab("dashboard");
    setActive(listing);
    window.setTimeout(() => terminal.current?.open(listing), 0);
  }, []);

  return (
    <div className="app">
      <div className="progress" aria-hidden />
      <aside className="sidebar">
        <div className="brand">
          <span className="monogram">UH</span>
          <span className="name">uphill</span>
        </div>
        <nav className="main-tabs" style={{ "--i": TABS.findIndex((t) => t.key === tab), "--n": TABS.length } as React.CSSProperties}>
          <span className="tab-indicator" aria-hidden />
          {TABS.map((t) => (
            <button key={t.key} className={tab === t.key ? "on" : ""} onClick={() => setTab(t.key)} title={t.label}>
              {t.icon}
              <span className="label">{t.label}</span>
            </button>
          ))}
        </nav>
        <div className="side-foot">
          <span className="label">Market desk</span>
          <span className="dim small">NSE · live analytics</span>
        </div>
      </aside>
      <div className="shell">
      <header className="topbar">
        <Clock />
        <div className="status">
          <span className={`pill ${status?.live_feeds ? "open" : "closed"}`} title={status?.live_symbols.join(", ")}>
            {status?.live_feeds ? `FEED ${status.live_symbols.length} symbols` : "FEED OFF"}
          </span>
          <span className={`pill silver ${status?.agent_enabled ? "open" : "closed"}`}>{status?.agent_enabled ? "AGENT ON" : "AGENT OFF"}</span>
        </div>
      </header>
      {backendError && <div className="error banner">Backend unreachable: {backendError}. Start it with <code>python -m uphill</code>.</div>}

      <main className={tab === "dashboard" ? "dashboard-page" : "hidden"}>
        <PageHero index="01" label="Dashboard" title="Markets," accent="live." outline="LIVE">
          <p className="dim">Watchlist, Bloomberg-style terminal and an AI analyst that cites every number.</p>
        </PageHero>
        <div className="dashboard">
          <Watchlist
            rows={rows}
            ticks={ticks}
            connection={connection}
            feedOn={(status?.live_feeds ?? 0) > 0}
            active={active}
            onSelect={openSecurity}
            onAdd={(l) => updateWatch(l, "ADD")}
            onRemove={(l) => updateWatch(l, "DEL")}
          />
          <TerminalPanel ref={terminal} onWatch={updateWatch} onSecurity={setActive} />
          <AgentPanel enabled={status?.agent_enabled ?? !backendError} activeListing={active} chat={chat} voice={voice} />
        </div>
      </main>
      {tab === "analytics" && (
        <AnalyticsPage
          agentEnabled={status?.agent_enabled ?? !backendError}
          initialListing={active}
          watchlist={watchlist}
          chat={chat}
          voice={voice}
          onSecurity={setActive}
          onWatch={updateWatch}
        />
      )}
      {tab === "stocks" && <ScreenerPage watchlist={watchlist} onOpenSecurity={openSecurity} onWatch={updateWatch} />}
      {tab === "news" && <NewsPage watchlist={watchlist} onOpenSecurity={openSecurity} />}

      {tab === "dashboard" && <NewsTicker onOpenNews={() => setTab("news")} />}
      <footer className="disclaimer">
        UpHill shows market data, analytics and third-party news for information only. It is not investment advice; verify before you trade.
      </footer>
      </div>
    </div>
  );
}
