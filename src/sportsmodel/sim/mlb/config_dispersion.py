"""Production scoring-channel rates and dispersion σ's for the sim kernel.

`p_roe` is measured from Statcast (assets/scoring_rates.json); `p_wp` is a
literature constant in that same asset (this backfill doesn't carry WP/PB
reliably). The dispersion σ's are tuned to match real MLB's outlier rates
(scripts/tune_dispersion.py) -- zeros until that tuning is committed.
"""
from __future__ import annotations

import json

from sportsmodel import config
from .kernel import Dispersion

_RATES_PATH = config.PROJECT_ROOT / "assets" / "scoring_rates.json"


def load_rates() -> dict:
    """{"p_roe":..,"p_wp":..} from the committed asset (zeros if absent)."""
    if _RATES_PATH.exists():
        return json.loads(_RATES_PATH.read_text())
    return {"p_roe": 0.0, "p_wp": 0.0}


# Tuned in scripts/tune_dispersion.py (Task 7). Zeros = no dispersion until then.
DISPERSION = Dispersion(sigma_shared=0.0, sigma_team=0.0, sigma_pitcher=0.0)
