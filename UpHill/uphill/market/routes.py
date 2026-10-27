from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status

from quantterm.live import parse_listing
from quantterm.models import Exchange
from quantterm.streaming import tick_channel
from quantterm.terminal.command_router import display_name
from uphill.state import State, UpHillState

log = logging.getLogger(__name__)

router = APIRouter(tags=["market"])

MAX_SYMBOLS = 50
MAX_STREAM_SYMBOLS = 600  # the all-stocks screener streams a whole cap bucket
SNAPSHOT_TTL_SECONDS = 60.0
CandleInterval = Literal["1", "3", "5", "15", "30", "60", "day"]


def _listings(refs: list[str] | None, state: UpHillState, limit: int = MAX_SYMBOLS) -> list[tuple[Exchange, str]]:
    if not refs:
        return state.watchlist
    out = []
    for ref in refs[:limit]:
        try:
            out.append(parse_listing(ref.upper()))
        except ValueError:
            continue
    return list(dict.fromkeys(out))


async def _snapshot(state: UpHillState, exchange: Exchange, symbol: str, live: set[str]) -> dict[str, Any]:
    listing = f"{exchange.value}:{symbol}"
    hit = state.snapshots.get(listing)
    if hit is not None and time.monotonic() - hit[0] < SNAPSHOT_TTL_SECONDS:
        return hit[1]
    row: dict[str, Any] = {"listing": listing, "symbol": symbol, "exchange": exchange.value, "live": listing in live}
    try:
        sec = await state.workspace.resolve(symbol, (exchange,))
        view = await state.workspace.gp(sec, state.clock(), lookback_days=31, adjust_splits=True,
                                        adjust_dividends=False)
        summary = {d.label: d for d in view.summary}
        last = summary["Last close"]
        row |= {
            "display_name": display_name(sec), "currency": sec.currency, "industry": sec.industry,
            "last_close": last.value, "last_close_time": last.timestamp.isoformat() if last.timestamp else None,
            "change_1m": summary["Period change"].value, "rsi": summary["RSI(14)"].value,
            "high_1m": summary["Period high"].value, "low_1m": summary["Period low"].value,
            "spark": view.close, "error": None,
        }
    except Exception as exc:  # one listing without data must not break the board
        row |= {"display_name": listing, "error": f"{type(exc).__name__}: {exc}"[:200]}
    state.snapshots[listing] = (time.monotonic(), row)
    return row


def _live(state: UpHillState) -> set[str]:
    """Listings with live prices: streamed tick feeds plus the polled universe quotes."""
    streamed = (set(state.settings.upstox_symbols) | set(state.settings.kite_symbols)) if state.live_feeds else set()
    polled = set(state.quotes.quotes) if state.quotes is not None else set()
    return streamed | polled


@router.get("/api/market/overview")
async def overview(
    state: State, symbols: Annotated[list[str] | None, Query(max_length=MAX_SYMBOLS)] = None,
) -> list[dict[str, Any]]:
    listings = _listings(symbols, state)
    live = _live(state)
    rows = await asyncio.gather(*(_snapshot(state, e, s, live) for e, s in listings))
    pulses = state.news.pulse([r["listing"] for r in rows])
    return [r | {"live": r["listing"] in live, "news": pulses[r["listing"]].model_dump(mode="json")} for r in rows]


@router.get("/api/market/screener")
async def screener(state: State) -> dict[str, Any]:
    """Every large/mid/small-cap stock with EOD stats; filtering and sorting happen in the browser."""
    if state.screener is None:
        return {"updated": None, "rows": []}
    rows = list(state.screener.rows.values())
    pulses = state.news.pulse([r["listing"] for r in rows])
    live = _live(state)
    return {
        "updated": state.screener.updated,
        "rows": [r | {"live": r["listing"] in live, "news": pulses[r["listing"]].model_dump(mode="json")}
                 for r in rows],
    }


@router.get("/api/market/candles")
async def candles(state: State, symbol: str, interval: CandleInterval = "5") -> dict[str, Any]:
    """Today's live candles [epoch s, o, h, l, c, v] (oldest first); "day" = the running session candle."""
    if state.quotes is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "live quotes are not configured")
    try:
        exchange, sym = parse_listing(symbol.upper())
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "symbol must look like NSE:RELIANCE") from None
    try:
        rows = (await state.quotes.intraday(sym, 30 if interval == "day" else int(interval))
                if exchange is Exchange.NSE else None)
    except Exception as exc:
        log.warning("intraday candles %s failed: %s", sym, exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "intraday candles unavailable") from None
    if rows is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no live data for {symbol}")
    if interval == "day" and rows:
        rows = [(rows[0][0], rows[0][1], max(r[2] for r in rows), min(r[3] for r in rows), rows[-1][4],
                 sum(r[5] for r in rows))]
    return {"listing": f"{exchange.value}:{sym}", "interval": interval, "candles": rows}


@router.get("/api/market/status")
async def market_status(state: State) -> dict[str, Any]:
    return {
        "live_feeds": state.live_feeds,
        "live_symbols": sorted(set(state.settings.upstox_symbols) | set(state.settings.kite_symbols)),
        "live_quotes": len(state.quotes.quotes) if state.quotes else 0,
        "live_quotes_updated": state.quotes.updated if state.quotes else None,
        "watchlist": [f"{e.value}:{s}" for e, s in state.watchlist],
        "agent_enabled": state.agent_enabled,
        "news_updated": state.news.updated,
        "eod_last_session": state.eod.last_session if state.eod else None,
        "eod_checked": state.eod.checked if state.eod else None,
        "server_time": state.clock(),
    }


@router.websocket("/ws/market")
async def market_stream(ws: WebSocket, symbols: str = "") -> None:
    state: UpHillState = ws.app.state.uphill
    listings = _listings([s for s in symbols.split(",") if s], state, MAX_STREAM_SYMBOLS)
    if not listings:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    channels = {tick_channel(e, s): f"{e.value}:{s}" for e, s in listings}
    try:
        # Subscribe before accepting so no tick published after the handshake can be missed.
        async with state.broker.subscribe(list(channels)) as messages:
            await ws.accept()
            if state.quotes is not None:  # latest polled prices immediately, not after the next change
                for listing in channels.values():
                    if (q := state.quotes.quotes.get(listing)) is not None:
                        await ws.send_json({"listing": listing, "price": q["price"], "size": 0,
                                            "timestamp": q["timestamp"], "prev_close": q["prev_close"]})

            async def forward() -> None:
                async for channel, data in messages:
                    listing = channels.get(channel)
                    if listing is None:
                        continue
                    tick = json.loads(data)["tick"]
                    prev = tick.get("prev_close")
                    if prev is None and state.quotes is not None:  # streamed ticks: borrow the polled prev close
                        prev = (state.quotes.quotes.get(listing) or {}).get("prev_close")
                    await ws.send_json({"listing": listing, "price": tick["price"], "size": tick["size"],
                                        "timestamp": tick["timestamp"], "prev_close": prev})

            # TaskGroup guarantees the forwarder never outlives the connection, including on cancellation.
            try:
                async with asyncio.TaskGroup() as tg:
                    forwarder = tg.create_task(forward())
                    while (await ws.receive())["type"] != "websocket.disconnect":
                        pass
                    forwarder.cancel()
            except* (WebSocketDisconnect, RuntimeError, OSError) as group:
                log.info("market stream closed: %s", [type(e).__name__ for e in group.exceptions])
    except (WebSocketDisconnect, OSError):
        return
