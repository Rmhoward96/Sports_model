"""Every number the early-week monitor's spec fixes, in one place. PURE -- no IO.

Copied from docs/superpowers/specs/2026-10-13-early-week-monitor-design.md. The pass-test numbers live in
`passtest.py` (frozen there); everything the polling, flagging and grading code shares lives here.
"""
from __future__ import annotations

# -- flag rule (spec section 1)
FLAG_EV_MIN = 0.015                  # +1.5% EV
FLAG_EV_EPS = 1e-12                  # float noise only: an EV of 0.015 - 1e-13 still flags
MIN_HOURS_TO_KICK = 24.0             # kickoff must be at least 24 h after the poll
FAIR_MIN, FAIR_MAX = 0.15, 0.85      # Pinnacle fair outside this range: no ticket
MAX_LINE_GAP = 3.0                   # |book line - Pinnacle line| above this many points: no ticket

# -- landing tables (spec section 1)
WINDOW_NARROW = 1.0                  # games whose closing line is within +/-1.0 point of L
WINDOW_WIDE = 2.0                    # ...widened to +/-2.0 when that gives fewer than MIN_GAMES
MIN_GAMES = 300
GRID_STEP = 0.5                      # half-point grid of line values
LINE_EPS = 1e-4                      # odds_snapshot.line is a float32 REAL: compare lines with a tolerance
GRID_EPS = 1e-6

# -- books and markets (spec sections 1 and 2)
PINNACLE = "pinnacle"
US_BOOKS = ("draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "fanatics", "hardrockbet",
            "ballybet", "betrivers")        # williamhill_us is Caesars' Odds API key
BOOKMAKERS = US_BOOKS + (PINNACLE,)         # the 10 bookmaker keys: one billed region
SPORT_ORDER = ("cfb", "nfl")
MARKETS = {"cfb": ("spread", "total"), "nfl": ("spread",)}          # NFL totals and all moneylines are excluded
API_MARKETS = {"cfb": ("spreads", "totals"), "nfl": ("spreads",)}   # The Odds API market keys per poll
SIDES = {"spread": ("home", "away"), "total": ("over", "under")}
LANDING_TABLES = (("cfb", "spread"), ("cfb", "total"), ("nfl", "spread"))

# -- credits (spec section 2)
MONITOR_FLOOR = 1500                 # remaining credits below this: the monitor skips its remaining calls

# -- grading (spec section 3)
NEXT_POLL_MIN_MINUTES = 30.0         # available_next_poll: the next poll must be 30 min ...
NEXT_POLL_MAX_MINUTES = 180.0        # ... to 3 h after the ticket, else NULL
SETTLE_HOURS = 4                     # grade a ticket only this long after kickoff (finals have landed)
