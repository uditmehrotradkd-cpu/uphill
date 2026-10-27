import { FormEvent, useEffect, useRef, useState } from "react";
import { AnswerLanguage } from "../api";
import { AgentChat } from "../useAgentChat";
import { Voice } from "../voice";

interface Props {
  enabled: boolean;
  activeListing: string | null;
  chat: AgentChat;
  voice: Voice;
  title?: string;
}

const TEXT = {
  en: {
    placeholder: "Ask me anything: a stock, the markets, world news, this site…",
    ask: "Ask",
    newChat: "new chat",
    greet: ["Good morning", "Good afternoon", "Good evening"],
    hello: "I'm your UpHill AI analyst. Ask me about any stock or the markets, what's happening in the world, how to use this site, or anything else on your mind.",
    intro: "Market figures are verified: every number comes from the engine with source, confidence and time.",
    general: "General knowledge · not engine-verified",
    knows: "It knows the active security",
    listen: "Listen",
    stop: "Stop",
    voice: "Voice",
    autoRead: "Read answers aloud",
    speaker: "Voice",
    best: "Best available",
    speed: "Speed",
    test: "Test voice",
    sample: "Hello, I am your UpHill market analyst.",
    noVoice: "No English voice is installed in this browser.",
    unsupported: "This browser cannot speak answers (Web Speech API unavailable).",
    withheld: "model answer withheld (grounding)",
    steps: "steps",
    facts: "facts cited",
  },
  hi: {
    placeholder: "कुछ भी पूछें: शेयर, बाज़ार, दुनिया की ख़बरें, यह साइट…",
    ask: "पूछें",
    newChat: "नई चैट",
    greet: ["सुप्रभात", "नमस्ते", "शुभ संध्या"],
    hello: "मैं आपका UpHill AI विश्लेषक हूँ। किसी भी शेयर या बाज़ार, दुनिया की ख़बरों, इस साइट के इस्तेमाल — या किसी भी विषय पर पूछें।",
    intro: "बाज़ार के आंकड़े सत्यापित हैं: हर संख्या इंजन से स्रोत, भरोसे और समय के साथ आती है।",
    general: "सामान्य जानकारी · इंजन से सत्यापित नहीं",
    knows: "इसे सक्रिय शेयर पता है",
    listen: "सुनें",
    stop: "रोकें",
    voice: "आवाज़",
    autoRead: "उत्तर बोलकर सुनाएँ",
    speaker: "आवाज़ चुनें",
    best: "सबसे अच्छी उपलब्ध",
    speed: "गति",
    test: "आवाज़ जाँचें",
    sample: "नमस्ते, मैं आपका UpHill बाज़ार विश्लेषक हूँ।",
    noVoice: "इस ब्राउज़र में हिंदी आवाज़ नहीं मिली। Chrome या Edge इस्तेमाल करें, या Windows/Android सेटिंग्स में हिंदी स्पीच इंस्टॉल करें।",
    unsupported: "यह ब्राउज़र उत्तर बोल नहीं सकता (Web Speech API उपलब्ध नहीं)।",
    withheld: "मॉडल का उत्तर रोका गया (सत्यापन)",
    steps: "चरण",
    facts: "आंकड़े उद्धृत",
  },
} as const;

function suggestions(lang: AnswerLanguage, symbol: string | undefined): string[] {
  if (lang === "hi") {
    return symbol
      ? [`क्या ${symbol} खरीदना ठीक रहेगा? ट्रेंड, वैल्यूएशन और ख़बरें देखें।`, `${symbol} की ताज़ा ख़बरें क्या हैं?`, "आज दुनिया में क्या हो रहा है?", "चार्ट पर RSI कैसे जोड़ें?"]
      : ["नमस्ते! आप क्या-क्या कर सकते हैं?", "आज दुनिया में क्या हो रहा है?", "क्या INFY IN ओवरबॉट है?", "Analytics पेज कैसे इस्तेमाल करें?"];
  }
  return symbol
    ? [`Should I consider buying ${symbol}? Weigh trend, valuation and news.`, `What is the latest news on ${symbol}?`, "What's happening in the world today?", "How do I add RSI to the chart?"]
    : ["Hi! What can you do?", "What's happening in the world today?", "Is INFY IN overbought?", "How do I use the Analytics page?"];
}

function greeting(lang: AnswerLanguage): string {
  const hour = Number(new Date().toLocaleString("en-US", { timeZone: "Asia/Kolkata", hour: "numeric", hour12: false }));
  const t = TEXT[lang];
  return `${t.greet[hour < 12 ? 0 : hour < 17 ? 1 : 2]}! ${t.hello}`;
}

const SpeakerIcon = ({ on }: { on: boolean }) => (
  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M4 9h4l5-4v14l-5-4H4z" />
    {on ? <path d="M16 9a4 4 0 0 1 0 6M18.5 6.5a8 8 0 0 1 0 11" /> : <path d="M17 9l5 6M22 9l-5 6" />}
  </svg>
);

export function AgentPanel({ enabled, activeListing, chat, voice, title = "AI STOCK AGENT" }: Props) {
  const [input, setInput] = useState("");
  const [showVoice, setShowVoice] = useState(false);
  const log = useRef<HTMLDivElement>(null);
  const symbol = activeListing?.split(":")[1];
  const lang = voice.prefs.lang;
  const t = TEXT[lang];
  const { messages, busy } = chat;

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  function submit(e: FormEvent) {
    e.preventDefault();
    const q = input;
    setInput("");
    void chat.ask(q);
  }

  const voices = voice.voicesFor(lang);
  const speakable = voice.canSpeak(lang);

  return (
    <section className="panel agent" lang={lang}>
      <header className="panel-head">
        <span className="title">{title}</span>
        <div className="seg lang-seg" role="group" aria-label="Answer language">
          <button className={lang === "en" ? "on" : ""} onClick={() => voice.setPrefs({ lang: "en" })} title="Answer in English">EN</button>
          <button className={lang === "hi" ? "on" : ""} onClick={() => voice.setPrefs({ lang: "hi" })} title="हिंदी में उत्तर">हिं</button>
        </div>
        <button
          className={`icon-btn ${voice.prefs.autoSpeak && speakable ? "on" : ""}`}
          onClick={() => {
            if (voice.prefs.autoSpeak) voice.stop();
            voice.setPrefs({ autoSpeak: !voice.prefs.autoSpeak });
          }}
          title={t.autoRead}
          aria-pressed={voice.prefs.autoSpeak}
          disabled={!voice.supported}
        >
          <SpeakerIcon on={voice.prefs.autoSpeak && speakable} />
        </button>
        <button className={`icon-btn ${showVoice ? "on" : ""}`} onClick={() => setShowVoice((s) => !s)} title={t.voice} aria-expanded={showVoice}>
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden>
            <path d="M4 7h10M18 7h2M4 17h4M12 17h8M14 4v6M8 14v6" />
          </svg>
        </button>
        <button className="link" onClick={() => void chat.reset()}>{t.newChat}</button>
      </header>
      {showVoice && (
        <div className="voice-settings">
          {!voice.supported ? (
            <span className="warn small">{t.unsupported}</span>
          ) : (
            <>
              <label className="check">
                <input type="checkbox" checked={voice.prefs.autoSpeak} onChange={(e) => voice.setPrefs({ autoSpeak: e.target.checked })} />
                {t.autoRead}
              </label>
              <label>
                <span>{t.speaker}</span>
                <select
                  value={voice.prefs.voice[lang]}
                  onChange={(e) => voice.setPrefs({ voice: { ...voice.prefs.voice, [lang]: e.target.value } })}
                  disabled={!speakable}
                >
                  <option value="">{t.best}</option>
                  {voices.map((v) => <option key={v.voiceURI} value={v.voiceURI}>{v.name} ({v.lang})</option>)}
                </select>
              </label>
              <label>
                <span>{t.speed} {voice.prefs.rate.toFixed(1)}×</span>
                <input type="range" min={0.6} max={1.6} step={0.1} value={voice.prefs.rate}
                       onChange={(e) => voice.setPrefs({ rate: Number(e.target.value) })} />
              </label>
              <button className="chip-btn" onClick={() => voice.speak("sample", t.sample, lang)} disabled={!speakable}>{t.test}</button>
              {!speakable && <span className="warn small">{t.noVoice}</span>}
            </>
          )}
        </div>
      )}
      <div className="agent-log" ref={log}>
        {!enabled && (
          <div className="warn small">
            Agent disabled. Add <code>QT_AGENT_API_KEY</code> (Gemini key) to <code>.env</code> and restart UpHill.
          </div>
        )}
        {enabled && messages.length === 0 && (
          <div className="msg agent greeting" lang={lang}>
            <div className="bubble">{greeting(lang)}</div>
            <div className="meta">
              {t.intro}
              {symbol && <> {t.knows} (<b>{symbol}</b>).</>}
            </div>
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`msg ${m.role}`} lang={m.lang}>
            <div className="bubble">{m.text}</div>
            {m.answer && (
              <div className="meta">
                {voice.canSpeak(m.lang) && (
                  <button className="speak-btn" onClick={() => (voice.speakingId === m.id ? voice.stop() : voice.speak(m.id, m.text, m.lang))}>
                    <SpeakerIcon on={voice.speakingId !== m.id} /> {voice.speakingId === m.id ? TEXT[m.lang].stop : TEXT[m.lang].listen}
                  </button>
                )}
                {m.answer.fallback && <span className="warn">{TEXT[m.lang].withheld} · </span>}
                {m.answer.general ? (
                  <span className="general-tag">{TEXT[m.lang].general}</span>
                ) : (
                  <>{m.answer.steps} {TEXT[m.lang].steps} · tools: {m.answer.tool_calls.join(", ") || "none"} · {m.answer.cited.length} {TEXT[m.lang].facts}</>
                )}
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
        {suggestions(lang, symbol).map((s) => (
          <button key={s} className="chip-btn" onClick={() => void chat.ask(s)} disabled={!enabled || busy}>{s}</button>
        ))}
      </div>
      <form className="ask" onSubmit={submit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={enabled ? t.placeholder : "Agent disabled"}
          disabled={!enabled}
          maxLength={1000}
        />
        <button type="submit" disabled={!enabled || busy}>{t.ask}</button>
      </form>
    </section>
  );
}
