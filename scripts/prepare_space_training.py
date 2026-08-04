#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.prepare import main  # noqa: E402


if __name__ == "__main__":
    main()
