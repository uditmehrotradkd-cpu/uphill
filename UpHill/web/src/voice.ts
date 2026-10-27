import { useCallback, useEffect, useRef, useState } from "react";
import { AnswerLanguage } from "./api";

// Text-to-speech through the browser's Web Speech API: free, on-device or vendor voices, no audio upload.

export interface VoicePrefs {
  lang: AnswerLanguage;
  autoSpeak: boolean;
  rate: number;
  voice: Record<AnswerLanguage, string>; // chosen voiceURI per language ("" = best available)
}

export interface Voice {
  supported: boolean;
  prefs: VoicePrefs;
  setPrefs: (update: Partial<VoicePrefs>) => void;
  voicesFor: (lang: AnswerLanguage) => SpeechSynthesisVoice[];
  canSpeak: (lang: AnswerLanguage) => boolean;
  speak: (id: string, text: string, lang: AnswerLanguage) => void;
  stop: () => void;
  speakingId: string | null;
}

const PREFS_KEY = "uphill.voice";
const DEFAULT_PREFS: VoicePrefs = { lang: "en", autoSpeak: true, rate: 1, voice: { en: "", hi: "" } };
const LANG_TAG: Record<AnswerLanguage, string> = { en: "en-IN", hi: "hi-IN" };
const CHUNK = 180; // Chrome stops long utterances after ~15 s, so speak sentence-sized pieces

function loadPrefs(): VoicePrefs {
  try {
    const raw = JSON.parse(localStorage.getItem(PREFS_KEY) ?? "null");
    if (raw && (raw.lang === "en" || raw.lang === "hi") && typeof raw.autoSpeak === "boolean"
        && typeof raw.rate === "number" && raw.rate >= 0.5 && raw.rate <= 2
        && typeof raw.voice?.en === "string" && typeof raw.voice?.hi === "string") {
      return raw as VoicePrefs;
    }
  } catch {
    /* corrupt storage: defaults */
  }
  return DEFAULT_PREFS;
}

function rank(voices: SpeechSynthesisVoice[], lang: AnswerLanguage): SpeechSynthesisVoice[] {
  const score = (v: SpeechSynthesisVoice) =>
    (v.lang.toLowerCase().replace("_", "-").endsWith("-in") ? 2 : 0) + (/natural|online|google/i.test(v.name) ? 1 : 0);
  return voices
    .filter((v) => v.lang.toLowerCase().startsWith(lang))
    .sort((a, b) => score(b) - score(a) || a.name.localeCompare(b.name));
}

/** What the voice should read: the answer body without provenance brackets or the source table. */
export function speechText(text: string, lang: AnswerLanguage): string {
  const hi = lang === "hi";
  return text
    .split(/\n\s*Sources:\s*\n/)[0]
    .replace(/\s*\[[^\]]*\]/g, "")
    .replace(/[*_#`>|]+/g, " ")
    .replace(/\bINR\b/g, hi ? "रुपये" : "rupees")
    .replace(/\bUSD\b/g, hi ? "डॉलर" : "dollars")
    .replace(/\s+/g, " ")
    .trim();
}

function chunks(text: string): string[] {
  const out: string[] = [];
  for (const sentence of text.split(/(?<=[.!?।])\s+/)) {
    let rest = sentence.trim();
    while (rest.length > CHUNK) {
      let cut = rest.lastIndexOf(",", CHUNK);
      if (cut < CHUNK / 2) cut = rest.lastIndexOf(" ", CHUNK);
      if (cut <= 0) cut = CHUNK;
      out.push(rest.slice(0, cut + 1));
      rest = rest.slice(cut + 1).trim();
    }
    if (!rest) continue;
    const last = out.length - 1;
    if (last >= 0 && out[last].length + rest.length < CHUNK) out[last] += ` ${rest}`;
    else out.push(rest);
  }
  return out;
}

export function useVoice(): Voice {
  const supported = typeof window !== "undefined" && "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [prefs, setPrefsState] = useState<VoicePrefs>(loadPrefs);
  const [speakingId, setSpeakingId] = useState<string | null>(null);
  const run = useRef(0);

  useEffect(() => {
    if (!supported) return;
    const load = () => setVoices(window.speechSynthesis.getVoices());
    load();
    window.speechSynthesis.addEventListener("voiceschanged", load);
    return () => {
      window.speechSynthesis.removeEventListener("voiceschanged", load);
      window.speechSynthesis.cancel();
    };
  }, [supported]);

  useEffect(() => localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)), [prefs]);

  const setPrefs = useCallback((update: Partial<VoicePrefs>) => setPrefsState((p) => ({ ...p, ...update })), []);
  const voicesFor = useCallback((lang: AnswerLanguage) => rank(voices, lang), [voices]);
  const canSpeak = useCallback((lang: AnswerLanguage) => supported && rank(voices, lang).length > 0, [supported, voices]);

  const stop = useCallback(() => {
    run.current += 1;
    if (supported) window.speechSynthesis.cancel();
    setSpeakingId(null);
  }, [supported]);

  const speak = useCallback(
    (id: string, text: string, lang: AnswerLanguage) => {
      stop();
      const candidates = rank(voices, lang);
      if (!supported || !candidates.length) return; // never read Hindi with an English voice
      const voice = candidates.find((v) => v.voiceURI === prefs.voice[lang]) ?? candidates[0];
      const parts = chunks(speechText(text, lang));
      if (!parts.length) return;
      const token = run.current;
      const done = () => {
        if (run.current === token) setSpeakingId(null);
      };
      setSpeakingId(id);
      window.speechSynthesis.resume(); // Chrome can be left paused by an earlier tab switch
      parts.forEach((part, i) => {
        const u = new SpeechSynthesisUtterance(part);
        u.voice = voice;
        u.lang = voice.lang || LANG_TAG[lang];
        u.rate = prefs.rate;
        u.onerror = done;
        if (i === parts.length - 1) u.onend = done;
        window.speechSynthesis.speak(u);
      });
    },
    [prefs, stop, supported, voices],
  );

  return { supported, prefs, setPrefs, voicesFor, canSpeak, speak, stop, speakingId };
}
