// Chart studies ("functions") computed in the browser so they follow live bars without a server round trip.

export interface Bar {
  time: number; // chart time: UTC midnight for daily bars, IST wall clock as epoch seconds for intraday
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export type StudyKind = "SMA" | "EMA" | "BB" | "VWAP" | "RSI" | "MACD" | "STOCH" | "ATR";

export interface Study {
  id: string;
  kind: StudyKind;
  period: number;
  color: string;
}

export const STUDY_INFO: Record<StudyKind, { label: string; pane: boolean; period: number | null }> = {
  SMA: { label: "Simple moving average", pane: false, period: 20 },
  EMA: { label: "Exponential moving average", pane: false, period: 21 },
  BB: { label: "Bollinger Bands (2σ)", pane: false, period: 20 },
  VWAP: { label: "VWAP (session; anchored on daily)", pane: false, period: null },
  RSI: { label: "Relative strength index", pane: true, period: 14 },
  MACD: { label: "MACD (12, 26, 9)", pane: true, period: null },
  STOCH: { label: "Stochastic %K / %D", pane: true, period: 14 },
  ATR: { label: "Average true range", pane: true, period: 14 },
};

// Never gold (the price colour) or the candle up/down colours, so studies cannot be mistaken for price.
export const PALETTE = ["#6aa9d4", "#b58ad8", "#e3a04a", "#7fd1c7", "#f28fb1", "#c8c8ce", "#9db4ff", "#e8e07a"];

type Values = (number | null)[];

export interface StudyLine {
  name: string;
  color: string;
  values: Values;
  histogram?: boolean;
}

export interface StudyOutput {
  lines: StudyLine[];
  levels: number[];
}

export function studyName(s: Study): string {
  return STUDY_INFO[s.kind].period === null ? s.kind : `${s.kind} ${s.period}`;
}

function sma(values: Values, n: number): Values {
  const out: Values = values.map(() => null);
  let sum = 0;
  let count = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v === null) {
      sum = 0;
      count = 0;
      continue;
    }
    sum += v;
    count++;
    if (count > n) sum -= values[i - n] as number;
    if (count >= n) out[i] = sum / n;
  }
  return out;
}

// Exponential smoothing seeded with the SMA of the first n values (the usual terminal convention).
function smooth(values: Values, n: number, alpha: number): Values {
  const out: Values = values.map(() => null);
  const seed: number[] = [];
  let prev: number | null = null;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v === null) continue;
    if (prev === null) {
      seed.push(v);
      if (seed.length < n) continue;
      prev = seed.reduce((a, b) => a + b, 0) / n;
    } else {
      prev = alpha * v + (1 - alpha) * prev;
    }
    out[i] = prev;
  }
  return out;
}

const ema = (values: Values, n: number) => smooth(values, n, 2 / (n + 1));
const wilder = (values: Values, n: number) => smooth(values, n, 1 / n);

function rsi(close: number[], n: number): Values {
  const gains: Values = close.map((c, i) => (i ? Math.max(c - close[i - 1], 0) : null));
  const losses: Values = close.map((c, i) => (i ? Math.max(close[i - 1] - c, 0) : null));
  const g = wilder(gains, n);
  const l = wilder(losses, n);
  return g.map((gv, i) => {
    const lv = l[i];
    if (gv === null || lv === null) return null;
    return lv === 0 ? 100 : 100 - 100 / (1 + gv / lv);
  });
}

function rollingStd(values: number[], mean: Values, n: number): Values {
  return mean.map((m, i) => {
    if (m === null) return null;
    let acc = 0;
    for (let j = i - n + 1; j <= i; j++) acc += (values[j] - m) ** 2;
    return Math.sqrt(acc / n);
  });
}

function vwap(bars: Bar[], intraday: boolean): Values {
  let pv = 0;
  let vol = 0;
  let day = -1;
  return bars.map((b) => {
    const d = Math.floor(b.time / 86400);
    if (intraday && d !== day) {
      pv = 0;
      vol = 0;
      day = d;
    }
    const typical = (b.high + b.low + b.close) / 3;
    pv += typical * b.volume;
    vol += b.volume;
    return vol > 0 ? pv / vol : typical;
  });
}

function stochastic(bars: Bar[], n: number): [Values, Values] {
  const raw: Values = bars.map((b, i) => {
    if (i < n - 1) return null;
    let hh = -Infinity;
    let ll = Infinity;
    for (let j = i - n + 1; j <= i; j++) {
      hh = Math.max(hh, bars[j].high);
      ll = Math.min(ll, bars[j].low);
    }
    return hh === ll ? 50 : (100 * (b.close - ll)) / (hh - ll);
  });
  const k = sma(raw, 3);
  return [k, sma(k, 3)];
}

function atr(bars: Bar[], n: number): Values {
  const tr: Values = bars.map((b, i) =>
    i ? Math.max(b.high - b.low, Math.abs(b.high - bars[i - 1].close), Math.abs(b.low - bars[i - 1].close)) : null,
  );
  return wilder(tr, n);
}

const fade = (hex: string, alpha: string) => `${hex}${alpha}`;

export function computeStudy(s: Study, bars: Bar[], intraday: boolean): StudyOutput {
  const close = bars.map((b) => b.close);
  switch (s.kind) {
    case "SMA":
      return { lines: [{ name: studyName(s), color: s.color, values: sma(close, s.period) }], levels: [] };
    case "EMA":
      return { lines: [{ name: studyName(s), color: s.color, values: ema(close, s.period) }], levels: [] };
    case "BB": {
      const mid = sma(close, s.period);
      const sd = rollingStd(close, mid, s.period);
      const band = (sign: number) => mid.map((m, i) => (m === null ? null : m + sign * 2 * (sd[i] as number)));
      return {
        lines: [
          { name: `${studyName(s)} upper`, color: fade(s.color, "99"), values: band(1) },
          { name: `${studyName(s)} mid`, color: s.color, values: mid },
          { name: `${studyName(s)} lower`, color: fade(s.color, "99"), values: band(-1) },
        ],
        levels: [],
      };
    }
    case "VWAP":
      return { lines: [{ name: "VWAP", color: s.color, values: vwap(bars, intraday) }], levels: [] };
    case "RSI":
      return { lines: [{ name: studyName(s), color: s.color, values: rsi(close, s.period) }], levels: [30, 70] };
    case "MACD": {
      const fast = ema(close, 12);
      const slow = ema(close, 26);
      const macd = fast.map((f, i) => (f === null || slow[i] === null ? null : f - (slow[i] as number)));
      const signal = ema(macd, 9);
      const hist = macd.map((m, i) => (m === null || signal[i] === null ? null : m - (signal[i] as number)));
      return {
        lines: [
          { name: "Histogram", color: "#5f5c55", values: hist, histogram: true },
          { name: "MACD", color: s.color, values: macd },
          { name: "Signal", color: "#c8c8ce", values: signal },
        ],
        levels: [0],
      };
    }
    case "STOCH": {
      const [k, d] = stochastic(bars, s.period);
      return {
        lines: [
          { name: `%K ${s.period}`, color: s.color, values: k },
          { name: "%D", color: "#c8c8ce", values: d },
        ],
        levels: [20, 80],
      };
    }
    case "ATR":
      return { lines: [{ name: studyName(s), color: s.color, values: atr(bars, s.period) }], levels: [] };
  }
}
