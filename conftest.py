"""Ensure the src layout is importable in tests even if the editable install is
rebuilt in a broken state (the project path contains a space, which makes the
hatchling editable .pth fragile). Belt-and-suspenders alongside `uv sync`.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
