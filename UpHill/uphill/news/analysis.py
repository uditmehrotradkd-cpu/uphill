"""Headline heuristics: listing tagging and a finance lexicon sentiment score. Indicative only, not advice."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from uphill.news.models import SentimentLabel

_POSITIVE = {
    "surge", "surges", "surged", "rally", "rallies", "rallied", "gain", "gains", "gained", "jump", "jumps", "jumped",
    "soar", "soars", "soared", "rise", "rises", "rose", "climb", "climbs", "climbed", "beat", "beats", "record",
    "upgrade", "upgrades", "upgraded", "outperform", "outperforms", "bullish", "growth", "strong", "stronger",
    "rebound", "rebounds", "recovery", "profit", "profits", "boost", "boosts", "boosted", "approval", "approves",
    "win", "wins", "expands", "expansion", "dividend", "buyback", "upbeat", "optimism", "peace",
    "ceasefire", "easing", "stimulus", "accelerates", "robust", "tops", "exceeds",
}
_NEGATIVE = {
    "fall", "falls", "fell", "slump", "slumps", "slumped", "plunge", "plunges", "plunged", "drop", "drops",
    "dropped", "decline", "declines", "declined", "miss", "misses", "missed", "downgrade", "downgrades",
    "downgraded", "loss", "losses", "bearish", "weak", "weaker", "crash", "crashes", "selloff", "sell-off",
    "tumble", "tumbles", "tumbled", "sink", "sinks", "sank", "slide", "slides", "war", "sanctions", "probe",
    "fraud", "default", "recession", "layoffs", "layoff", "tariff", "tariffs", "inflation", "lawsuit", "penalty",
    "fine", "fined", "strike", "attack", "conflict", "crisis", "slowdown", "concern", "concerns", "fears", "fear",
    "risk", "risks", "warning", "warns", "lowest", "volatile", "turmoil", "uncertainty",
}
_NEGATORS = {"no", "not", "never", "without", "despite", "fails", "failed"}
_WORD = re.compile(r"[a-z][a-z\-']*")


def sentiment(text: str) -> tuple[float, SentimentLabel]:
    words = _WORD.findall(text.lower())
    pos = neg = 0
    for i, w in enumerate(words):
        flip = i > 0 and words[i - 1] in _NEGATORS
        if w in _POSITIVE:
            neg, pos = (neg + 1, pos) if flip else (neg, pos + 1)
        elif w in _NEGATIVE:
            pos, neg = (pos + 1, neg) if flip else (pos, neg + 1)
    if pos + neg == 0:
        return 0.0, "neutral"
    score = (pos - neg) / (pos + neg)
    return score, label_for(score)


def label_for(score: float) -> SentimentLabel:
    return "bullish" if score >= 0.2 else "bearish" if score <= -0.2 else "neutral"


_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[&.\-][A-Za-z0-9]+)*")


class SymbolTagger:
    """Tags text with EXCHANGE:SYMBOL listings by ticker (exact case) or company alias (case-insensitive).

    Word n-gram dictionary lookups (not one regex per listing) keep tagging cheap for ~500 listings.
    """

    def __init__(self, aliases: Mapping[str, Iterable[str]]) -> None:
        self._tickers: dict[str, set[str]] = {}
        self._aliases: dict[str, set[str]] = {}
        self._max_words = 1
        for listing, names in aliases.items():
            ticker = listing.partition(":")[2]
            if len(ticker) >= 3:
                self._tickers.setdefault(ticker, set()).add(listing)
            for name in names:
                words = _TOKEN.findall(name)
                if words:
                    self._aliases.setdefault(" ".join(words).lower(), set()).add(listing)
                    self._max_words = max(self._max_words, len(words))

    def tag(self, text: str) -> list[str]:
        words = _TOKEN.findall(text)
        found: set[str] = set()
        for i, word in enumerate(words):
            found |= self._tickers.get(word, set())
            for n in range(1, min(self._max_words, len(words) - i) + 1):
                found |= self._aliases.get(" ".join(words[i:i + n]).lower(), set())
        return sorted(found)
