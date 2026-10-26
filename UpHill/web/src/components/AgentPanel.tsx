import { FormEvent, useEffect, useRef, useState } from "react";
import { AgentAnswer, api } from "../api";

interface Message {
  role: "user" | "agent" | "error";
  text: string;
  answer?: AgentAnswer;
}

interface Props {
  enabled: boolean;
  activeListing: string | null;
}

export function AgentPanel({ enabled, activeListing }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const log = useRef<HTMLDivElement>(null);
  const symbol = activeListing?.split(":")[1];

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  async function ask(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    setMessages((m) => [...m, { role: "user", text: q }]);
    setBusy(true);
    try {
      const answer = await api.ask(q);
      setMessages((m) => [...m, { role: "agent", text: answer.text, answer }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "error", text: e instanceof Error ? e.message : String(e) }]);
    } finally {
      setBusy(false);
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    const q = input;
    setInput("");
    void ask(q);
  }

  async function reset() {
    await api.resetAgent().catch(() => undefined);
    setMessages([]);
  }

  const suggestions = symbol
    ? [`Should I consider buying ${symbol}? Weigh trend, valuation and news.`, `What is the latest news on ${symbol}?`, `How volatile is ${symbol} vs the market (beta)?`]
    : ["Compare RELIANCE IN and TCS IN momentum", "What world news could move Indian markets today?", "Is INFY IN overbought?"];

  return (
    <section className="panel agent">
      <header className="panel-head">
        <span className="title">AI STOCK AGENT</span>
        <button className="link" onClick={() => void reset()}>new chat</button>
      </header>
      <div className="agent-log" ref={log}>
        {!enabled && (
          <div className="warn small">
            Agent disabled. Add <code>QT_AGENT_API_KEY</code> (Gemini key) to <code>.env</code> and restart UpHill.
          </div>
        )}
        {enabled && messages.length === 0 && (
          <div className="dim small">
            Every number the agent cites comes from the engine with source, confidence and knowledge time; headlines come from the news desk.
            {symbol && <> It knows the terminal's active security (<b>{symbol}</b>).</>}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">{m.text}</div>
            {m.answer && (
              <div className="meta">
                {m.answer.fallback && <span className="warn">model answer withheld (grounding) · </span>}
                {m.answer.steps} steps · tools: {m.answer.tool_calls.join(", ") || "none"} · {m.answer.cited.length} facts cited
              </div>
            )}
          </div>
        ))}
        {busy && (
          <div className="msg agent">
            <div className="bubble typing" aria-label="analysing"><span /><span /><span /></div>
          </div>
        )}
      </div>
      <div className="suggestions">
        {suggestions.map((s) => (
          <button key={s} className="chip-btn" onClick={() => void ask(s)} disabled={!enabled || busy}>{s}</button>
        ))}
      </div>
      <form className="ask" onSubmit={submit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={enabled ? "Ask about a stock, the market or the news…" : "Agent disabled"}
          disabled={!enabled}
          maxLength={1000}
        />
        <button type="submit" disabled={!enabled || busy}>Ask</button>
      </form>
    </section>
  );
}
