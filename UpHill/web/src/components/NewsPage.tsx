import { useCallback, useEffect, useState } from "react";
import { NewsItem, SourceStatus, api } from "../api";
import { bloombergName, safeHref, timeAgo } from "../format";
import { PageHero } from "./Editorial";
import { SentimentMeter } from "./Hud";

const CATEGORIES = [
  { key: "", label: "All" },
  { key: "mine", label: "My Stocks" },
  { key: "markets", label: "Markets" },
  { key: "business", label: "Business" },
  { key: "india", label: "India" },
  { key: "world", label: "World" },
];
const SENTIMENTS = ["", "bullish", "bearish", "neutral"];
const REFRESH_MS = 120_000;

interface Props {
  watchlist: string[];
  onOpenSecurity: (listing: string) => void;
}

export function NewsPage({ watchlist, onOpenSecurity }: Props) {
  const [items, setItems] = useState<NewsItem[]>([]);
  const [sources, setSources] = useState<SourceStatus[]>([]);
  const [updated, setUpdated] = useState<string | null>(null);
  const [category, setCategory] = useState("");
  const [sentiment, setSentiment] = useState("");
  const [symbol, setSymbol] = useState("");
  const [q, setQ] = useState("");
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [news, src] = await Promise.all([
        api.news({
          category: category === "mine" ? undefined : category,
          tagged: category === "mine" ? watchlist : undefined,
          sentiment, symbol, q: q.trim(), limit: 150,
        }),
        api.sources(),
      ]);
      setItems(news);
      setSources(src.sources);
      setUpdated(src.updated);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [category, sentiment, symbol, q, watchlist]);

  useEffect(() => {
    const t = window.setTimeout(() => void load(), 250);
    const interval = window.setInterval(() => void load(), REFRESH_MS);
    return () => {
      window.clearTimeout(t);
      window.clearInterval(interval);
    };
  }, [load]);

  const counts = items.reduce(
    (acc, i) => ({ ...acc, [i.sentiment_label]: acc[i.sentiment_label] + 1 }),
    { bullish: 0, bearish: 0, neutral: 0 } as Record<string, number>,
  );

  return (
    <section className="news-page">
      <PageHero index="04" label="World & market news" title="What's moving" accent="the world." outline="NEWS">
        <div className="intel-meter">
          <SentimentMeter bullish={counts.bullish} neutral={counts.neutral} bearish={counts.bearish} />
          <div className="meter-legend">
            <span className="up">{counts.bullish} bullish</span>
            <span className="dim">{counts.neutral} neutral</span>
            <span className="down">{counts.bearish} bearish</span>
          </div>
          <p className="dim small">{items.length} headlines · updated {timeAgo(updated) || "pending"} · {sources.filter((s) => s.ok).length}/{sources.length} sources online</p>
        </div>
      </PageHero>
      <div className="news-toolbar">
        <div className="tabs">
          {CATEGORIES.map((c) => (
            <button key={c.key} className={category === c.key ? "on" : ""} onClick={() => setCategory(c.key)}>{c.label}</button>
          ))}
        </div>
        <select value={symbol} onChange={(e) => setSymbol(e.target.value)} aria-label="Filter by stock">
          <option value="">All stocks</option>
          {watchlist.map((l) => <option key={l} value={l}>{bloombergName(l)}</option>)}
        </select>
        <select value={sentiment} onChange={(e) => setSentiment(e.target.value)} aria-label="Filter by sentiment">
          {SENTIMENTS.map((s) => <option key={s} value={s}>{s ? s[0].toUpperCase() + s.slice(1) : "Any sentiment"}</option>)}
        </select>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search headlines…" maxLength={80} />
      </div>
      <div className="news-summary">
        <span className="dim small">Sentiment is a keyword heuristic over third-party headlines, not advice.</span>
      </div>
      {error && <div className="error">{error}</div>}
      <div className="news-grid">
        {items.map((item, index) => (
          <article key={item.id} className={`news-card ${item.sentiment_label}`} style={{ animationDelay: `${Math.min(index, 20) * 30}ms` }}>
            <span className="card-index">{String(index + 1).padStart(3, "0")}</span>
            {safeHref(item.image) && (
              <div className="news-img">
                <img src={safeHref(item.image)} alt="" loading="lazy" referrerPolicy="no-referrer" onError={(e) => (e.currentTarget.parentElement!.style.display = "none")} />
              </div>
            )}
            <div className="news-body">
              <div className="news-meta">
                <span className="publisher">{item.publisher}</span>
                <span className="dim">· {timeAgo(item.published)}</span>
                <span className={`tag ${item.sentiment_label === "bullish" ? "up" : item.sentiment_label === "bearish" ? "down" : "neutral"}`}>
                  {item.sentiment_label}
                </span>
              </div>
              <a href={safeHref(item.url)} target="_blank" rel="noopener noreferrer" className="headline">{item.title}</a>
              {item.summary && <p className="summary">{item.summary}</p>}
              {item.symbols.length > 0 && (
                <div className="symbols">
                  {item.symbols.map((s) => (
                    <button key={s} className="chip-btn" onClick={() => onOpenSecurity(s)} title="Open in terminal">{s.split(":")[1]}</button>
                  ))}
                </div>
              )}
            </div>
          </article>
        ))}
        {items.length === 0 && !error && <p className="dim">No headlines yet. The news desk refreshes every few minutes.</p>}
      </div>
      <footer className="sources">
        {sources.map((s) => (
          <span key={s.name} className={`src ${s.ok ? "ok" : "bad"}`} title={s.error ?? `${s.items} items`}>
            {s.ok ? "●" : "○"} {s.name}
          </span>
        ))}
      </footer>
    </section>
  );
}

export function NewsTicker({ onOpenNews }: { onOpenNews: () => void }) {
  const [items, setItems] = useState<NewsItem[]>([]);
  useEffect(() => {
    const load = () => api.news({ limit: 25 }).then(setItems).catch(() => undefined);
    void load();
    const t = window.setInterval(load, REFRESH_MS);
    return () => window.clearInterval(t);
  }, []);
  if (!items.length) return null;
  return (
    <div className="ticker" onClick={onOpenNews} title="Open the news desk">
      <span className="ticker-label">NEWS</span>
      <div className="ticker-track">
        <div className="ticker-move">
          {[...items, ...items].map((i, idx) => (
            <span key={`${i.id}-${idx}`} className={`ticker-item ${i.sentiment_label}`}>
              <b>{i.publisher}</b> {i.title}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
