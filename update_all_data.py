#!/usr/bin/env python3
"""兼容入口：转到 scripts/update_all_data.py。"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "update_all_data.py"
    runpy.run_path(str(target), run_name="__main__")
