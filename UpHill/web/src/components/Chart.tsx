import { useEffect, useId, useRef } from "react";
import { ColorType, createChart, IChartApi, LineStyle, Time } from "lightweight-charts";

export interface Series {
  name: string;
  kind: "line" | "area" | "histogram";
  color: string;
  points: { time: string; value: number | null }[];
}

interface Props {
  series: Series[];
  height: number;
  levels?: number[];
}

/** Thin wrapper over TradingView lightweight-charts; null points are gaps (e.g. RSI warm-up). */
export function Chart({ series, height, levels = [] }: Props) {
  const host = useRef<HTMLDivElement>(null);
  // Recreate the chart only when the data actually changes, not on every parent render.
  const signature = JSON.stringify([series, levels, height]);

  useEffect(() => {
    if (!host.current) return;
    const chart: IChartApi = createChart(host.current, {
      height,
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#8f8a7f", fontSize: 11,
                fontFamily: "'JetBrains Mono Variable', monospace" },
      grid: { vertLines: { color: "rgba(255,255,255,0.03)" }, horzLines: { color: "rgba(255,255,255,0.045)" } },
      rightPriceScale: { borderColor: "rgba(255,255,255,0.08)" },
      timeScale: { borderColor: "rgba(255,255,255,0.08)" },
      crosshair: { mode: 0, vertLine: { color: "rgba(212,178,106,0.45)" }, horzLine: { color: "rgba(212,178,106,0.45)" } },
    });
    series.forEach((s, index) => {
      const data = s.points.filter((p) => p.value !== null).map((p) => ({ time: p.time as Time, value: p.value as number }));
      const target =
        s.kind === "histogram"
          ? chart.addHistogramSeries({ color: s.color, priceLineVisible: false })
          : s.kind === "area"
            ? chart.addAreaSeries({
                lineColor: s.color,
                topColor: `${s.color}55`,
                bottomColor: `${s.color}05`,
                lineWidth: 2,
                priceLineVisible: false,
              })
            : chart.addLineSeries({ color: s.color, lineWidth: 2, priceLineVisible: false });
      target.setData(data);
      if (index === 0) {
        levels.forEach((price) =>
          target.createPriceLine({ price, color: "rgba(212,178,106,0.5)", lineStyle: LineStyle.Dashed, lineWidth: 1, axisLabelVisible: true, title: "" }),
        );
      }
    });
    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [signature]);

  return <div ref={host} className="chart" style={{ height }} />;
}

export function Sparkline({ values, width = 90, height = 26 }: { values: number[]; width?: number; height?: number }) {
  const id = useId();
  if (values.length < 2) return <svg width={width} height={height} />;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const coords = values.map((v, i) => [(i / (values.length - 1)) * width, height - 2 - ((v - min) / span) * (height - 4)]);
  const line = coords.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  const up = values[values.length - 1] >= values[0];
  const color = up ? "var(--up)" : "var(--down)";
  return (
    <svg width={width} height={height} className="spark" aria-hidden>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity="0.35" />
          <stop offset="1" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`${line} L${width} ${height} L0 ${height} Z`} fill={`url(#${id})`} className="spark-area" />
      <path d={line} fill="none" stroke={color} strokeWidth="1.6" pathLength={1} className="spark-line" />
    </svg>
  );
}
