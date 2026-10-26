"""Seed the demo database (idempotent) and open the QuantTerm terminal on it.

    python seed_and_launch.py [--database quantterm_demo] [--as-of ...]

Try:  RELIANCE IN Equity GP   |   FA   |   AAPL US Equity OVME   |   TCS IN Equity FA   |   ASOF 2026-05-01
"""

from __future__ import annotations

import sys

from seed_database import seed

from quantterm.terminal.app import main

if __name__ == "__main__":
    main(seed(sys.argv[1:]))
