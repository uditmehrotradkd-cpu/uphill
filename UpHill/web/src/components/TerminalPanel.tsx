import { FormEvent, KeyboardEvent, forwardRef, useImperativeHandle, useRef, useState } from "react";
import { FAView, GPView, OVMEView, RVView, TerminalResult, api } from "../api";
import { bloombergName } from "../format";
import { FAPanel, GPPanel, OVMEPanel, RVPanel } from "./TerminalViews";

export interface TerminalHandle {
  run: (command: string) => void;
  open: (listing: string) => void;
}

interface Props {
  onWatch: (listing: string, action: "ADD" | "DEL") => void;
  onSecurity: (listing: string) => void;
}

const FUNCTIONS = ["GP", "FA", "RV", "OVME"] as const;
const EXAMPLES = ["RELIANCE IN Equity GP", "INFY IN Equity FA Q 8", "HDFCBANK IN Equity RV", "TCS IN Equity OVME", "GP 6M NODIV", "ASOF 2026-06-30"];

export const TerminalPanel = forwardRef<TerminalHandle, Props>(function TerminalPanel({ onWatch, onSecurity }, ref) {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<TerminalResult | null>(null);
  const [context, setContext] = useState("no security | - | live");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const history = useRef<string[]>([]);
  const cursor = useRef(-1);

  async function run(command: string) {
    const text = command.trim();
    if (!text) return;
    history.current = [text, ...history.current.filter((h) => h !== text)].slice(0, 50);
    cursor.current = -1;
    setBusy(true);
    setError(null);
    try {
      const r = await api.command(text);
      setContext(r.context);
      setInfo(r.kind === "STATUS" || r.kind === "WATCH" ? r.message : null);
      if (r.kind === "ERROR") setError(r.message);
      if (r.kind === "WATCH" && r.listing && r.watch_action) onWatch(r.listing, r.watch_action);
      if (r.kind === "VIEW" || r.kind === "HELP") setResult(r);
      if (r.listing && r.kind !== "WATCH") onSecurity(r.listing);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  useImperativeHandle(ref, () => ({
    run: (c: string) => void run(c),
    // Loading a security keeps the active function (Bloomberg behaviour); default to GP on first load.
    open: (listing: string) => void run(`${bloombergName(listing)}${result?.function ? "" : " GP"}`),
  }));

  function submit(e: FormEvent) {
    e.preventDefault();
    const text = input;
    setInput("");
    void run(text);
  }

  function onKey(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault();
      const next = Math.max(-1, Math.min(history.current.length - 1, cursor.current + (e.key === "ArrowUp" ? 1 : -1)));
      cursor.current = next;
      setInput(next === -1 ? "" : history.current[next]);
    }
  }

  return (
    <section className="panel terminal">
      <header className="panel-head">
        <span className="title">BLOOMBERG TERMINAL</span>
        <span className="context">{context}</span>
      </header>
      <form className="command" onSubmit={submit}>
        <span className="prompt">&gt;</span>
        <input
          autoFocus
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKey}
          placeholder="<TICKER> <CODE> Equity [GP|FA|RV|OVME]  ·  ASOF yyyy-mm-dd  ·  WATCH ADD ...  ·  HELP"
          spellCheck={false}
          autoComplete="off"
          maxLength={200}
        />
        <button type="submit" disabled={busy}>{busy ? "…" : "GO"}</button>
      </form>
      <div className="fkeys">
        {FUNCTIONS.map((f) => (
          <button key={f} onClick={() => void run(f)} disabled={busy}>{f}</button>
        ))}
        <button onClick={() => void run("BACK")} disabled={busy}>BACK</button>
        <button onClick={() => void run("ASOF NOW")} disabled={busy}>LIVE</button>
        <button onClick={() => void run("HELP")} disabled={busy}>HELP</button>
      </div>
      {error && <div className="error">{error}</div>}
      {info && <div className="info">{info}</div>}
      <div className="viewport">
        {!result && (
          <div className="welcome">
            <p>Type a security and a function, or click a watchlist row. Examples:</p>
            <div className="examples">
              {EXAMPLES.map((ex) => (
                <button key={ex} className="link" onClick={() => void run(ex)}>{ex}</button>
              ))}
            </div>
          </div>
        )}
        {result?.kind === "HELP" && <pre className="help">{result.message}</pre>}
        {result?.kind === "VIEW" && result.function === "GP" && <GPPanel v={result.view as GPView} />}
        {result?.kind === "VIEW" && result.function === "FA" && <FAPanel v={result.view as FAView} />}
        {result?.kind === "VIEW" && result.function === "RV" && <RVPanel v={result.view as RVView} />}
        {result?.kind === "VIEW" && result.function === "OVME" && <OVMEPanel v={result.view as OVMEView} />}
      </div>
    </section>
  );
});
