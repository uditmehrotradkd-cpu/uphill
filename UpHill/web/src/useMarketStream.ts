import { useEffect, useRef, useState } from "react";
import { LiveTick, marketSocketUrl } from "./api";

export type Connection = "connecting" | "open" | "closed";

/** Live validated ticks for the given listings; reconnects with backoff when the server or network drops. */
export function useMarketStream(symbols: string[]): { ticks: Record<string, LiveTick>; connection: Connection } {
  const [ticks, setTicks] = useState<Record<string, LiveTick>>({});
  const [connection, setConnection] = useState<Connection>("connecting");
  const key = symbols.join(",");
  const backoff = useRef(1000);

  useEffect(() => {
    if (!key) return;
    let socket: WebSocket | null = null;
    let timer: number | undefined;
    let stopped = false;

    const connect = () => {
      setConnection("connecting");
      socket = new WebSocket(marketSocketUrl(key.split(",")));
      socket.onopen = () => {
        backoff.current = 1000;
        setConnection("open");
      };
      socket.onmessage = (event) => {
        const tick = JSON.parse(event.data) as LiveTick;
        setTicks((prev) => ({ ...prev, [tick.listing]: tick }));
      };
      socket.onclose = () => {
        setConnection("closed");
        if (!stopped) {
          timer = window.setTimeout(connect, backoff.current);
          backoff.current = Math.min(backoff.current * 2, 30000);
        }
      };
    };
    connect();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      socket?.close();
    };
  }, [key]);

  return { ticks, connection };
}
