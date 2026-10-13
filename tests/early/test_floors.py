"""The credit floors, as credits fall: regular prop pulls stop first, then the early-week monitor, then regular game-line
pulls, then close-capture props, then all close capture. The monitor must never starve the close capture. No network."""
import pytest

from sportsmodel.early import polling
from sportsmodel.early.constants import MONITOR_FLOOR
from sportsmodel.ingest import odds
from sportsmodel.ledger import close


def test_the_floors_are_strictly_ordered():
    assert odds.INGEST_PROPS_FLOOR == 2500 and MONITOR_FLOOR == 1500 and odds.INGEST_LINES_FLOOR == 800
    assert close.BUDGET_PROPS_FLOOR == 600 and close.BUDGET_ALL_FLOOR == 200
    assert (odds.INGEST_PROPS_FLOOR > MONITOR_FLOOR > odds.INGEST_LINES_FLOOR > close.BUDGET_PROPS_FLOOR
            > close.BUDGET_ALL_FLOOR)


@pytest.mark.parametrize("remaining, ingest_props, monitor, ingest_lines, close_props, close_lines", [
    (20000, True, True, True, True, True),
    (2500, True, True, True, True, True),
    (2499, False, True, True, True, True),             # regular prop pulls stop first
    (1500, False, True, True, True, True),
    (1499, False, False, True, True, True),            # then this monitor
    (800, False, False, True, True, True),
    (799, False, False, False, True, True),            # then regular game-line pulls
    (600, False, False, False, True, True),
    (599, False, False, False, False, True),           # then close-capture props
    (200, False, False, False, False, True),
    (199, False, False, False, False, False),          # then all close capture
])
def test_what_each_job_may_still_spend_at_each_balance(remaining, ingest_props, monitor, ingest_lines, close_props, close_lines):
    lines_ok, props_ok, _ = odds.ingest_budget_decision(remaining)
    capture = close.budget_decision(remaining)
    assert (props_ok, polling.budget_decision(remaining).ok, lines_ok, capture.props, capture.lines) == (
        ingest_props, monitor, ingest_lines, close_props, close_lines)
