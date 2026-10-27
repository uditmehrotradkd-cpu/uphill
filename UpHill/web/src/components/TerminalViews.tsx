import { Datum, FAView, GPView, OVMEView, OptionValuation, QuotedLeg, RVView } from "../api";
import { fmt } from "../format";
import { PriceChart } from "./PriceChart";

const LOW_CONFIDENCE = 0.8;
const WARN_CONFIDENCE = 0.5;

function confidenceClass(d: Datum): string {
  if (d.value === null) return "dim";
  if (d.confidence_score >= LOW_CONFIDENCE) return "";
  return d.confidence_score >= WARN_CONFIDENCE ? "warn" : "bad";
}

export function DatumCell({ d }: { d: Datum }) {
  const title = `${d.source} | conf ${d.confidence_score.toFixed(2)}${d.timestamp ? ` | ${new Date(d.timestamp).toLocaleString()}` : ""}${d.note ? ` | ${d.note}` : ""}`;
  return (
    <td className={`num ${confidenceClass(d)}`} title={title}>
      {d.value === null ? "n/a" : (d.display ?? fmt(d.value))}
    </td>
  );
}

function ProvenanceTable({ rows }: { rows: Datum[] }) {
  return (
    <table className="grid">
      <thead>
        <tr><th>Item</th><th className="num">Value</th><th>Source</th><th className="num">Conf</th><th>Knowledge time</th><th>Note</th></tr>
      </thead>
      <tbody>
        {rows.map((d) => (
          <tr key={d.label}>
            <td>{d.label}</td>
            <DatumCell d={d} />
            <td className="dim">{d.source}</td>
            <td className="num">{d.confidence_score.toFixed(2)}</td>
            <td className="dim">{d.timestamp ? new Date(d.timestamp).toLocaleString() : "-"}</td>
            <td className="dim">{d.note ?? ""}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function GPPanel({ v }: { v: GPView }) {
  const basis = v.adjust_dividends ? "split + dividend adjusted" : v.adjust_splits ? "split adjusted" : "raw";
  return (
    <div className="view">
      <div className="chips">
        <span className="chip strong">{v.security.symbol} GP · {basis}</span>
        {v.summary.map((d) => (
          <span key={d.label} className={`chip ${confidenceClass(d)}`} title={d.note ?? d.provenance}>
            {d.label}: <b>{d.value === null ? "n/a" : (d.display ?? fmt(d.value))}</b>
          </span>
        ))}
      </div>
      <PriceChart view={v} />
    </div>
  );
}

export function FAPanel({ v }: { v: FAView }) {
  const breaks = v.reconciliation.filter((r) => r.is_break).length;
  return (
    <div className="view">
      <h4>Point-in-time fundamentals · {v.security.symbol} ({v.period_type})</h4>
      <div className="scroll-x">
        <table className="grid">
          <thead>
            <tr><th>Metric</th>{v.periods.map((p) => <th key={p} className="num">{p}</th>)}</tr>
          </thead>
          <tbody>
            {v.rows.map((r) => (
              <tr key={r.label}><td>{r.label}</td>{r.cells.map((c, i) => <DatumCell key={i} d={c} />)}</tr>
            ))}
            {v.ratios.map((r) => (
              <tr key={r.label} className="italic"><td>{r.label}</td>{r.cells.map((c, i) => <DatumCell key={i} d={c} />)}</tr>
            ))}
          </tbody>
        </table>
      </div>
      <h4>Reconciled enterprise value</h4>
      <ProvenanceTable rows={v.ev_table} />
      <p className={breaks ? "warn" : "ok"}>Reconciliation: {v.reconciliation.length} items, {breaks} breaks.</p>
      {v.notes.map((n) => <p key={n} className="warn">! {n}</p>)}
    </div>
  );
}

export function RVPanel({ v }: { v: RVView }) {
  return (
    <div className="view">
      <h4>Relative valuation · industry {v.industry} ({v.rows.length} listings)</h4>
      <div className="scroll-x">
        <table className="grid">
          <thead>
            <tr><th>Security</th>{v.columns.map((c) => <th key={c} className="num">{c}</th>)}</tr>
          </thead>
          <tbody>
            {v.rows.map((r) => (
              <tr key={`${r.security.exchange}:${r.security.symbol}`} className={r.is_subject ? "subject" : ""}>
                <td>{r.security.symbol}</td>
                {v.columns.map((c) => <DatumCell key={c} d={r.cells[c]} />)}
              </tr>
            ))}
            <tr className="median">
              <td>Peer median</td>
              <td />
              {v.columns.slice(1).map((c) => <DatumCell key={c} d={v.medians[c]} />)}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function legCells(o: OptionValuation, q: QuotedLeg | null) {
  const market = q
    ? [fmt(q.bid), fmt(q.ask), q.implied_vol === null ? "n/a" : `${(q.implied_vol * 100).toFixed(2)}%`]
    : ["-", "-", "-"];
  return [...market, fmt(o.npv), o.delta.toFixed(3), o.gamma.toFixed(4), o.theta_per_day.toFixed(3), o.vega_per_vol_point.toFixed(3)];
}

export function OVMEPanel({ v }: { v: OVMEView }) {
  const spot = v.inputs[0]?.value ?? null;
  const atm = spot === null ? null : v.chain.reduce((best, r) => (Math.abs(r.strike - spot) < Math.abs(best - spot) ? r.strike : best), v.chain[0]?.strike ?? 0);
  const sides = ["Bid", "Ask", "Mkt IV", "Model", "Delta", "Gamma", "Theta", "Vega"];
  return (
    <div className="view">
      <h4>Market inputs · {v.security.symbol} expiry {v.expiry} ({v.exercise.toLowerCase()})</h4>
      <ProvenanceTable rows={v.inputs} />
      <h4>Chain (calls | strike | puts)</h4>
      <div className="scroll-x">
        <table className="grid chain">
          <thead>
            <tr>
              {sides.map((s) => <th key={`c${s}`} className="num">C {s}</th>)}
              <th className="num">Strike</th>
              {[...sides].reverse().map((s) => <th key={`p${s}`} className="num">P {s}</th>)}
            </tr>
          </thead>
          <tbody>
            {v.chain.map((r) => (
              <tr key={r.strike} className={r.strike === atm ? "subject" : ""}>
                {legCells(r.call, r.call_quote).map((t, i) => <td key={`c${i}`} className="num">{t}</td>)}
                <td className="num strike">{r.strike}</td>
                {legCells(r.put, r.put_quote).reverse().map((t, i) => <td key={`p${i}`} className="num">{t}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {v.calculator && v.calculator_contract && (
        <p className="calc">
          Calculator {v.calculator_contract.option_type} {v.calculator_contract.strike}: premium {fmt(v.calculator.npv, 4)} · delta{" "}
          {v.calculator.delta.toFixed(4)} · gamma {v.calculator.gamma.toFixed(5)} · theta/day {v.calculator.theta_per_day.toFixed(4)} · vega/pt{" "}
          {v.calculator.vega_per_vol_point.toFixed(4)}
        </p>
      )}
      {v.notes.map((n) => <p key={n} className="dim">{n}</p>)}
    </div>
  );
}
