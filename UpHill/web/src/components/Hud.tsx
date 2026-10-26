/** Arc-reactor style emblem: concentric rings, counter-rotating segments, pulsing core. */
export function Reactor({ size = 34 }: { size?: number }) {
  return (
    <svg className="reactor" viewBox="0 0 100 100" width={size} height={size} aria-hidden>
      <circle cx="50" cy="50" r="46" className="r-outer" />
      <g className="r-spin">
        <circle cx="50" cy="50" r="38" className="r-dash" />
      </g>
      <g className="r-spin-rev">
        <circle cx="50" cy="50" r="29" className="r-dash2" />
      </g>
      <circle cx="50" cy="50" r="17" className="r-core" />
      <circle cx="50" cy="50" r="8" className="r-hot" />
    </svg>
  );
}

/** Bullish / neutral / bearish split as a segmented HUD bar. */
export function SentimentMeter({ bullish, neutral, bearish }: { bullish: number; neutral: number; bearish: number }) {
  const total = bullish + neutral + bearish || 1;
  const pct = (n: number) => `${(n / total) * 100}%`;
  return (
    <div className="meter" role="img" aria-label={`${bullish} bullish, ${neutral} neutral, ${bearish} bearish`}>
      <span className="m-up" style={{ width: pct(bullish) }} />
      <span className="m-mid" style={{ width: pct(neutral) }} />
      <span className="m-down" style={{ width: pct(bearish) }} />
    </div>
  );
}
