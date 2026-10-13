"""Ensure the src layout is importable in tests even if the editable install is
rebuilt in a broken state (the project path contains a space, which makes the
hatchling editable .pth fragile). Belt-and-suspenders alongside `uv sync`.
"""
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The local .env carries a real DATABASE_URL: no test may reach the pick ledger (or the capture / grading jobs
# built on it) through the production database. Tests that exercise the ledger inject a fake store instead.
os.environ["SM_LEDGER_OFF"] = "1"
