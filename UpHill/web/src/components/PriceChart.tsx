import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  ChartOptions,
  ColorType,
  CrosshairMode,
  DeepPartial,
  IChartApi,
  ISeriesApi,
  LineStyle,
  UTCTimestamp,
  createChart,
} from "lightweight-charts";
import { CandlesResponse, GPView, LiveTick, api } from "../api";
import { fmt, fmtPct } from "../format";
import { Bar, PALETTE, STUDY_INFO, Study, StudyKind, StudyOutput, computeStudy, studyName } from "../studies";
import { useMarketStream } from "../useMarketStream";

type ChartType = "candles" | "line";
type Interval = "1" | "5" | "15" | "30" | "60" | "D";

const INTERVALS: { key: Interval; label: string }[] = [
  { key: "1", label: "1m" },
  { key: "5", label: "5m" },
  { key: "15", label: "15m" },
  { key: "30", label: "30m" },
  { key: "60", label: "1h" },
  { key: "D", label: "1D" },
];
const UP = "#4cc38a";
const DOWN = "#e5675b";
const IST_OFFSET = 19_800; // intraday bars are plotted on the IST wall clock
const SESSION_OPEN = 9 * 3600 + 15 * 60;
const SESSION_CLOSE = 15 * 3600 + 30 * 60;
const INTRADAY_REFRESH_MS = 20_000;
const DAY_REFRESH_MS = 60_000;
const STUDIES_KEY = "uphill.chart.studies";
const TYPE_KEY = "uphill.chart.type";
const DEFAULT_STUDIES: Study[] = [
  { id: "sma20", kind: "SMA", period: 20, color: PALETTE[0] },
  { id: "sma50", kind: "SMA", period: 50, color: PALETTE[2] },
  { id: "rsi14", kind: "RSI", period: 14, color: PALETTE[0] },
];

function loadStudies(): Study[] {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(STUDIES_KEY) ?? "null");
    const valid = (s: Study) =>
      typeof s?.id === "string" && Object.keys(STUDY_INFO).includes(s.kind) && Number.isInteger(s.period) &&
      /^#[0-9a-f]{6}$/i.test(s.color);
    if (Array.isArray(raw) && raw.every(valid)) return raw as Study[];
  } catch {
    /* corrupt storage: fall back to defaults */
  }
  return DEFAULT_STUDIES;
}

const istDate = (d: Date) => d.toLocaleDateString("en-CA", { timeZone: "Asia/Kolkata" });

function dailyBars(v: GPView): Bar[] {
  return v.dates.map((d, i) => ({
    time: Date.parse(`${d}T00:00:00Z`) / 1000,
    open: v.open?.[i] ?? v.close[i],
    high: v.high?.[i] ?? v.close[i],
    low: v.low?.[i] ?? v.close[i],
    close: v.close[i],
    volume: v.volume[i] ?? 0,
  }));
}

function toBar([epoch, open, high, low, close, volume]: CandlesResponse["candles"][number], daily: boolean): Bar {
  const t = epoch + IST_OFFSET;
  return { time: daily ? t - (t % 86400) : t, open, high, low, close, volume };
}

function mergeBar(bars: Bar[], bar: Bar): Bar[] {
  const last = bars[bars.length - 1];
  if (!last || bar.time > last.time) return [...bars, bar];
  return bar.time === last.time ? [...bars.slice(0, -1), bar] : bars;
}

/** Folds a live price into the running bar; new bars only open once the server confirmed today's session. */
function applyTick(bars: Bar[], tick: LiveTick, interval: Interval): Bar[] {
  const t = Math.floor(Date.parse(tick.timestamp) / 1000) + IST_OFFSET;
  const sec = t % 86400;
  const weekday = new Date(t * 1000).getUTCDay();
  if (weekday === 0 || weekday === 6 || sec < SESSION_OPEN || sec >= SESSION_CLOSE) return bars;
  const day = t - sec;
  const size = Number(interval) * 60;
  const time = interval === "D" ? day : day + SESSION_OPEN + Math.floor((sec - SESSION_OPEN) / size) * size;
  const last = bars[bars.length - 1];
  const price = tick.price;
  if (!last || time < last.time) return bars;
  if (time === last.time) {
    if (last.close === price && last.high >= price && last.low <= price) return bars;
    return [...bars.slice(0, -1), { ...last, close: price, high: Math.max(last.high, price), low: Math.min(last.low, price) }];
  }
  if (interval === "D" || last.time - (last.time % 86400) !== day) return bars;
  return [...bars, { time, open: price, high: price, low: price, close: price, volume: 0 }];
}

interface Handles {
  charts: IChartApi[];
  candles?: ISeriesApi<"Candlestick">;
  area?: ISeriesApi<"Area">;
  volume: ISeriesApi<"Histogram">;
  studies: { study: Study; series: ISeriesApi<"Line" | "Histogram">[] }[];
}

function chartOptions(height: number, intraday: boolean, showTime: boolean): DeepPartial<ChartOptions> {
  return {
    height,
    autoSize: true,
    layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#8f8a7f", fontSize: 11,
              fontFamily: "'JetBrains Mono Variable', monospace" },
    grid: { vertLines: { color: "rgba(255,255,255,0.03)" }, horzLines: { color: "rgba(255,255,255,0.045)" } },
    rightPriceScale: { borderColor: "rgba(255,255,255,0.08)", minimumWidth: 78 },
    timeScale: { borderColor: "rgba(255,255,255,0.08)", timeVisible: intraday, secondsVisible: false, visible: showTime,
                 rightOffset: 4 },
    crosshair: { mode: CrosshairMode.Normal, vertLine: { color: "rgba(212,178,106,0.45)" },
                 horzLine: { color: "rgba(212,178,106,0.45)" } },
  };
}

function draw(h: Handles, bars: Bar[], outputs: Map<string, StudyOutput>) {
  const time = (i: number) => bars[i].time as UTCTimestamp;
  h.candles?.setData(bars.map((b, i) => ({ time: time(i), open: b.open, high: b.high, low: b.low, close: b.close })));
  h.area?.setData(bars.map((b, i) => ({ time: time(i), value: b.close })));
  h.volume.setData(bars.map((b, i) => ({
    time: time(i), value: b.volume, color: b.close >= b.open ? "rgba(76,195,138,0.32)" : "rgba(229,103,91,0.32)",
  })));
  for (const { study, series } of h.studies) {
    outputs.get(study.id)?.lines.forEach((line, j) =>
      series[j]?.setData(line.values.map((v, i) =>
        v === null ? { time: time(i) } : { time: time(i), value: v, ...(line.histogram ? { color: v >= 0 ? `${UP}99` : `${DOWN}99` } : {}) },
      )),
    );
  }
}

export function PriceChart({ view }: { view: GPView }) {
  const listing = `${view.security.exchange}:${view.security.symbol}`;
  const liveCapable = view.security.exchange === "NSE" && istDate(new Date(view.as_of)) === istDate(new Date());
  const [chartType, setChartType] = useState<ChartType>(() => (localStorage.getItem(TYPE_KEY) === "line" ? "line" : "candles"));
  const [interval, setBarInterval] = useState<Interval>("D");
  const [studies, setStudies] = useState<Study[]>(loadStudies);
  const [newKind, setNewKind] = useState<StudyKind>("EMA");
  const [newPeriod, setNewPeriod] = useState(21);
  const [hover, setHover] = useState<number | null>(null);
  const [feedError, setFeedError] = useState<string | null>(null);
  const daily = useMemo(() => dailyBars(view), [view]);
  const [bars, setBars] = useState<Bar[]>(daily);
  const { ticks, connection } = useMarketStream(liveCapable ? [listing] : []);
  const tick = liveCapable ? ticks[listing] : undefined;
  const effectiveInterval: Interval = liveCapable ? interval : "D";
  const intraday = effectiveInterval !== "D";

  const outputs = useMemo(
    () => new Map(studies.map((s) => [s.id, computeStudy(s, bars, intraday)])),
    [studies, bars, intraday],
  );
  const panes = studies.filter((s) => STUDY_INFO[s.kind].pane);

  const mainHost = useRef<HTMLDivElement>(null);
  const paneHosts = useRef<(HTMLDivElement | null)[]>([]);
  const handles = useRef<Handles | null>(null);
  const latest = useRef({ bars, outputs });
  latest.current = { bars, outputs };
  const fitPending = useRef(true);

  useEffect(() => localStorage.setItem(STUDIES_KEY, JSON.stringify(studies)), [studies]);
  useEffect(() => localStorage.setItem(TYPE_KEY, chartType), [chartType]);

  // Bars: daily history from the GP view plus today's live candle, or today's intraday candles.
  useEffect(() => {
    let cancelled = false;
    fitPending.current = true;
    setFeedError(null);
    setBars(effectiveInterval === "D" ? daily : []);
    if (!liveCapable) return;
    const load = async () => {
      try {
        const r = await api.candles(listing, effectiveInterval === "D" ? "day" : effectiveInterval);
        if (cancelled) return;
        if (effectiveInterval === "D") {
          if (r.candles[0]) setBars((prev) => mergeBar(prev, toBar(r.candles[0], true)));
        } else {
          setBars(r.candles.map((c) => toBar(c, false)));
        }
        setFeedError(null);
      } catch (e) {
        if (!cancelled) setFeedError(e instanceof Error ? e.message : String(e));
      }
    };
    void load();
    const timer = window.setInterval(load, effectiveInterval === "D" ? DAY_REFRESH_MS : INTRADAY_REFRESH_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [daily, effectiveInterval, listing, liveCapable]);

  useEffect(() => {
    if (tick) setBars((prev) => applyTick(prev, tick, effectiveInterval));
  }, [tick, effectiveInterval]);

  const structure = JSON.stringify([chartType, intraday, studies]);
  useEffect(() => {
    if (!mainHost.current) return;
    const main = createChart(mainHost.current, chartOptions(340, intraday, panes.length === 0));
    const h: Handles = {
      charts: [main],
      volume: main.addHistogramSeries({ priceScaleId: "vol", priceFormat: { type: "volume" }, priceLineVisible: false,
                                        lastValueVisible: false }),
      studies: [],
    };
    main.priceScale("vol").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    if (chartType === "candles") {
      h.candles = main.addCandlestickSeries({ upColor: UP, downColor: DOWN, borderUpColor: UP, borderDownColor: DOWN,
                                              wickUpColor: UP, wickDownColor: DOWN });
    } else {
      h.area = main.addAreaSeries({ lineColor: "#d4b26a", topColor: "#d4b26a40", bottomColor: "#d4b26a05", lineWidth: 2 });
    }
    main.priceScale("right").applyOptions({ scaleMargins: { top: 0.08, bottom: 0.22 } });
    const lineOptions = (color: string) => ({ color, lineWidth: 1 as const, priceLineVisible: false, lastValueVisible: false,
                                              crosshairMarkerVisible: false });
    for (const study of studies) {
      const info = STUDY_INFO[study.kind];
      const lines = computeStudy(study, [], intraday).lines;
      if (!info.pane) {
        h.studies.push({ study, series: lines.map((l) => main.addLineSeries(lineOptions(l.color))) });
        continue;
      }
      const index = panes.indexOf(study);
      const host = paneHosts.current[index];
      if (!host) continue;
      const pane = createChart(host, chartOptions(120, intraday, index === panes.length - 1));
      h.charts.push(pane);
      const series = lines.map((l) =>
        l.histogram
          ? pane.addHistogramSeries({ color: l.color, priceLineVisible: false, lastValueVisible: false })
          : pane.addLineSeries({ ...lineOptions(l.color), lineWidth: 2 as const, lastValueVisible: true }),
      );
      const levels = computeStudy(study, [], intraday).levels;
      levels.forEach((price) =>
        series[series.length - 1].createPriceLine({ price, color: "rgba(212,178,106,0.4)", lineStyle: LineStyle.Dashed,
                                                    lineWidth: 1, axisLabelVisible: false, title: "" }),
      );
      h.studies.push({ study, series });
    }

    let syncing = false;
    h.charts.forEach((chart) =>
      chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (syncing || !range) return;
        syncing = true;
        h.charts.forEach((other) => other !== chart && other.timeScale().setVisibleLogicalRange(range));
        syncing = false;
      }),
    );
    main.subscribeCrosshairMove((p) => {
      const at = p.time === undefined ? -1 : latest.current.bars.findIndex((b) => b.time === p.time);
      setHover(at >= 0 ? at : null);
    });

    handles.current = h;
    fitPending.current = true;
    return () => {
      handles.current = null;
      h.charts.forEach((c) => c.remove());
    };
  }, [structure]);

  useEffect(() => {
    const h = handles.current;
    if (!h) return;
    draw(h, bars, outputs);
    if (fitPending.current && bars.length) {
      fitPending.current = false;
      h.charts[0].timeScale().fitContent();
    }
  }, [bars, outputs, structure]);

  function addStudy(e: FormEvent) {
    e.preventDefault();
    const fixed = STUDY_INFO[newKind].period;
    const period = fixed === null ? 0 : Math.round(newPeriod);
    if (fixed !== null && !(period >= 2 && period <= 400)) return;
    if (studies.some((s) => s.kind === newKind && s.period === period)) return;
    const used = new Set(studies.map((s) => s.color));
    const color = PALETTE.find((c) => !used.has(c)) ?? PALETTE[studies.length % PALETTE.length];
    setStudies([...studies, { id: crypto.randomUUID(), kind: newKind, period, color }]);
  }

  const shown = hover ?? bars.length - 1;
  const bar = bars[shown];
  const prevClose = shown > 0 ? bars[shown - 1].close : bar?.open;
  const change = bar && prevClose ? (bar.close / prevClose - 1) * 100 : null;
  const live = liveCapable && connection === "open";
  const lastValue = (s: Study, named: boolean) => {
    const lines = (outputs.get(s.id)?.lines ?? []).filter((l) => !l.histogram);
    return lines
      .map((l) => `${named || lines.length > 1 ? `${l.name} ` : ""}${fmt(l.values[shown] ?? null)}`)
      .join(" · ");
  };

  return (
    <div className="price-chart">
      <div className="chart-toolbar">
        <div className="seg" role="group" aria-label="Chart type">
          <button className={chartType === "candles" ? "on" : ""} onClick={() => setChartType("candles")}>Candles</button>
          <button className={chartType === "line" ? "on" : ""} onClick={() => setChartType("line")}>Line</button>
        </div>
        <div className="seg" role="group" aria-label="Interval">
          {INTERVALS.map((iv) => (
            <button
              key={iv.key}
              className={effectiveInterval === iv.key ? "on" : ""}
              disabled={iv.key !== "D" && !liveCapable}
              title={iv.key !== "D" && !liveCapable ? "Intraday needs a live NSE view (ASOF NOW)" : `${iv.label} bars`}
              onClick={() => setBarInterval(iv.key)}
            >
              {iv.label}
            </button>
          ))}
        </div>
        {liveCapable && (
          <span className={`pill ${live ? "open" : "closed"}`} title={feedError ?? `live ticks · ${connection}`}>
            {live ? "LIVE" : connection === "connecting" ? "CONNECTING" : "OFFLINE"}
          </span>
        )}
        <form className="study-add" onSubmit={addStudy}>
          <select value={newKind} onChange={(e) => setNewKind(e.target.value as StudyKind)} aria-label="Function">
            {(Object.keys(STUDY_INFO) as StudyKind[]).map((k) => (
              <option key={k} value={k}>{k} — {STUDY_INFO[k].label}</option>
            ))}
          </select>
          {STUDY_INFO[newKind].period !== null && (
            <input type="number" min={2} max={400} value={newPeriod} aria-label="Period"
                   onChange={(e) => setNewPeriod(Number(e.target.value))} />
          )}
          <button type="submit">Apply</button>
        </form>
      </div>
      <div className="chips">
        {studies.map((s) => (
          <span key={s.id} className="chip study" title={STUDY_INFO[s.kind].label}>
            <i style={{ background: s.color }} />
            {studyName(s)}
            <button className="x" aria-label={`Remove ${studyName(s)}`} onClick={() => setStudies(studies.filter((x) => x.id !== s.id))}>×</button>
          </span>
        ))}
        {studies.length > 0 && <button className="link" onClick={() => setStudies([])}>clear</button>}
      </div>
      <div className="chart-legend num">
        {bar ? (
          <>
            <span>O <b>{fmt(bar.open)}</b></span>
            <span>H <b>{fmt(bar.high)}</b></span>
            <span>L <b>{fmt(bar.low)}</b></span>
            <span>C <b>{fmt(bar.close)}</b></span>
            <span className={change === null ? "" : change >= 0 ? "up" : "down"}>{fmtPct(change)}</span>
            <span className="dim">Vol {fmt(bar.volume, 0)}</span>
            {studies.filter((s) => !STUDY_INFO[s.kind].pane).map((s) => (
              <span key={s.id} style={{ color: s.color }}>{lastValue(s, true)}</span>
            ))}
          </>
        ) : (
          <span className="dim">{intraday ? "No intraday candles yet today (market closed or pre-open)." : "No bars."}</span>
        )}
        {feedError && <span className="warn">live: {feedError}</span>}
      </div>
      <div ref={mainHost} className="chart" style={{ height: 340 }} />
      {panes.map((s, i) => (
        <div key={s.id} className="chart-pane">
          <div className="pane-label">
            <span style={{ color: s.color }}>{studyName(s)}</span> <span className="dim num">{lastValue(s, false)}</span>
          </div>
          <div ref={(el) => { paneHosts.current[i] = el; }} className="chart" style={{ height: 120 }} />
        </div>
      ))}
    </div>
  );
}
