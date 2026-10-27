import { useCallback, useRef, useState } from "react";
import { AgentAnswer, AnswerLanguage, api } from "./api";
import { Voice } from "./voice";

export interface ChatMessage {
  id: string;
  role: "user" | "agent" | "error";
  text: string;
  lang: AnswerLanguage;
  answer?: AgentAnswer;
}

export interface AgentChat {
  messages: ChatMessage[];
  busy: boolean;
  ask: (question: string) => Promise<void>;
  reset: () => Promise<void>;
}

/** One conversation per browser session, shared by every chat panel (the server keeps the agent memory). */
export function useAgentChat(voice: Voice): AgentChat {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);

  const ask = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || inFlight.current) return;
      inFlight.current = true;
      const lang = voice.prefs.lang;
      voice.stop();
      setMessages((m) => [...m, { id: crypto.randomUUID(), role: "user", text: q, lang }]);
      setBusy(true);
      try {
        const answer = await api.ask(q, lang);
        const id = crypto.randomUUID();
        setMessages((m) => [...m, { id, role: "agent", text: answer.text, lang: answer.language, answer }]);
        if (voice.prefs.autoSpeak) voice.speak(id, answer.text, answer.language);
      } catch (e) {
        setMessages((m) => [...m, { id: crypto.randomUUID(), role: "error", text: e instanceof Error ? e.message : String(e), lang }]);
      } finally {
        inFlight.current = false;
        setBusy(false);
      }
    },
    [voice],
  );

  const reset = useCallback(async () => {
    voice.stop();
    await api.resetAgent().catch(() => undefined);
    setMessages([]);
  }, [voice]);

  return { messages, busy, ask, reset };
}
